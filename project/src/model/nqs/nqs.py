import dataclasses
import logging
from dataclasses import dataclass
from functools import partial
from typing import List

import hydra
import jax
import jax.numpy as jnp
import jax.random as jrnd
import netket as nk
import optax
from netket.optimizer import identity_preconditioner
from src.model.NN.interface import NNConfig, NNType
from src.model.NN.utils import apply_fun, init_fun_tr_learning, save_model, type2nn
from src.model.nqs.callback.utils import type2callback
from src.model.operator import get_model_netket_op
from src.model.operator.interface import PreconditionerType
from src.model.operator.utils import type2operator
from src.model.optimizer.interface import OptimizerConfig, OptimizerType
from src.model.optimizer.utils import type2optimizer
from src.model.sampler.interface import SamplerConfig, SamplerType
from src.model.sampler.utils import type2sampler
from src.model.struct import ChainConfig
from src.result.struct import Result
from src.utils import report_name_with_data
from tqdm import tqdm


@dataclass
class ModelNQSConfig:
    chain: ChainConfig
    min_n_samples: int
    scale_n_samples: int
    preconditioner: bool
    sr_diag_shift: float
    model_config: NNConfig
    tr_learning: bool
    noise_scale: float
    callbacks: List[str]
    preconditioner_operator: str
    preconditioner_operator_eps: float
    sampler: SamplerConfig
    optimizer: OptimizerConfig

    def __post_init__(self):
        self.dtype = self.model_config.dtype
        self.rnd_seed = self.model_config.seed
        self.save_model_path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )


class ModelNQS:
    def __init__(self, cfg: ModelNQSConfig, cfg_exp, logger: logging.Logger):
        self.cfg = cfg
        self.cfg_exp = cfg_exp
        self.logger = logger
        self.graph = nk.graph.Chain(length=self.cfg.chain.n, pbc=self.cfg.chain.pbc)
        self.hilbert = nk.hilbert.Spin(N=self.cfg.chain.n, s=self.cfg.chain.spin)
        self.key = jrnd.PRNGKey(self.cfg.rnd_seed)
        self.preconditioner_operator = type2operator[
            self.cfg.preconditioner_operator
        ]

        self.model_file = None

        self._set_sampler()

        self._init_state()

    def _init_state(self):
        self._set_ops()

        self._set_machine()

        self._set_optimizer()

        self._set_vmc(ham=self.ham)

    def _set_ops(self):
        self.ham = get_model_netket_op(
            self.cfg.chain,
            self.hilbert,
        )

    def _set_machine(self):
        self.nn = type2nn[self.cfg.model_config.nntype](self.cfg.model_config)

    def _set_sampler(self):
        self.sampler = type2sampler[self.cfg.sampler.type](
            hilbert=self.hilbert,
            graph=self.graph,
            cfg=self.cfg.sampler,
        )

    def _set_optimizer(self):
        self.optimizer = type2optimizer[self.cfg.optimizer.type](self.cfg.optimizer)
        nn = self.cfg.model_config.nntype
        if nn == NNType.PHASE_TRANSFORMER and (
            self.cfg.model_config.pqc or self.cfg.model_config.gcnn
        ):
            optimizer = dataclasses.replace(
                self.cfg.optimizer, type=OptimizerType.ADAM_ZERO_TR
            )
            self.optimizer_pqc = type2optimizer[self.cfg.optimizer.type](optimizer)

    def _set_vmc(self, ham):
        self.n_samples = max(
            [self.cfg.min_n_samples, self.cfg.chain.n * self.cfg.scale_n_samples]
        )
        training_kwargs = dict()
        nn = self.cfg.model_config.nntype
        if nn in (NNType.TRANSFORMER, NNType.PHASE_TRANSFORMER):
            training_kwargs = {
                "generate": False,
            }

        self.mcstate = nk.vqs.MCState(
            sampler=self.sampler,
            n_samples=self.n_samples,
            init_fun=partial(
                init_fun_tr_learning,
                model=self.nn,
                tr_learning=self.cfg.tr_learning,
                model_file=self.model_file,
                noise_scale=self.cfg.noise_scale,
            ),
            apply_fun=partial(
                apply_fun, model=self.nn, nntype=self.cfg.model_config.nntype, rnd_seed= self.cfg.rnd_seed
            ),
            seed=self.key,
            sampler_seed=self.key,
            training_kwargs=training_kwargs,
        )

        sr = identity_preconditioner
        if self.cfg.preconditioner:
            sr = nk.optimizer.SR(
                diag_shift=self.cfg.sr_diag_shift, solver_restart=False
            )

        self.vmc = nk.driver.VMC(
            hamiltonian=ham,
            optimizer=self.optimizer,
            variational_state=self.mcstate,
            preconditioner=sr,
        )

    def set_h(self, h: float):
        nn = self.cfg.model_config.nntype
        self.last_h = self.cfg.chain.h
        chain = dataclasses.replace(self.cfg.chain, h=h)
        self.cfg.chain = chain
        if nn == NNType.PHASE_TRANSFORMER:
            tr_config = dataclasses.replace(
                self.cfg.model_config.tr_config, chain=chain
            )
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, tr_config=tr_config
            )
        else:
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, chain=chain
            )
        self._init_state()

    def _get_current_exp_hash(self):
        return report_name_with_data(self.cfg_exp, f"{self.nn.config.chain.h:.5f}")

    def train(self):
        logger_runtime = nk.logging.RuntimeLog()
        logger_tensorboard = nk.logging.TensorBoardLog(
            comment="_" + self._get_current_exp_hash()
        )
        logger_out = [logger_runtime, logger_tensorboard]
        callback = [lambda *x: True] + [type2callback[callback]() for callback in self.cfg.callbacks]

        nn = self.cfg.model_config.nntype
        res = self.get_result()
        self.logger.info(res)
        if nn == NNType.PHASE_TRANSFORMER:
            n_ter = self.cfg.n_iter
            cfg = self.cfg.model_config
            if cfg.pqc or cfg.gcnn:
                n_ter = int(self.cfg.n_iter * (1 - cfg.phase_fine_ratio))
            self.vmc.run(
                n_iter=n_ter,
                out=logger_out,
                callback=callback,
            )
            if cfg.pqc or cfg.gcnn:
                if cfg.transformer_sampler:
                    self.sampler = type2sampler[SamplerType.TRANSFORMER](
                        hilbert=self.hilbert,
                        graph=self.graph,
                        cfg=self.cfg.sampler,
                    )
                n_ter = self.cfg.optimizer.n_iter - n_ter
                self.cfg.model_config = dataclasses.replace(cfg, phase_train=True)
                self.vmc.optimizer = self.optimizer_pqc
                self._set_vmc(ham=self.ham)
                self.vmc.run(
                    n_iter=n_ter,
                    out=logger_out,
                    callback=callback,
                )
            res = self.get_result()
            self.logger.info(res)
        else:
            ham = self.ham

            self._set_vmc(ham=ham)
            self.vmc.run(
                n_iter=self.cfg.optimizer.n_iter,
                out=logger_out,
                callback=callback,
            )
            res = self.get_result()
            self.logger.info(res)

            if self.cfg.preconditioner_operator != PreconditionerType.Exact:
                en = jnp.real(self.get_ops_vals(ham)[Result.ENERGY].mean)
                delta_en = float("inf")
                while delta_en > self.cfg.preconditioner_operator_eps:
                    ham = self.preconditioner_operator(ham=self.ham, sigma=en)
                    self._set_vmc(ham=ham)
                    self.vmc.run(
                        n_iter=self.cfg.n_iter,
                        out=logger_out,
                        callback=callback,
                    )
                    res = self.get_result()
                    self.logger.info(res)

                    en_new = jnp.real(self.get_ops_vals(ham)[Result.ENERGY].mean)
                    delta_en = abs(en_new - en)
                    en = en_new

        self.model_file = save_model(
            parameters=self.mcstate.parameters,
            save_model_path=self.cfg.save_model_path,
            name_model=self.nn.__class__.__name__,
            postfix=self._get_current_exp_hash(),
        )
        self.logger.info(f"Model is saved: {self.model_file}")

        return res

    def get_ops_vals(self, ham: nk.operator.AbstractOperator):
        train = self.nn.config.training
        self.nn.config = dataclasses.replace(self.nn.config, training=False)

        n_samples = max(
            [self.cfg.min_n_samples, self.cfg.chain.n * self.cfg.scale_n_samples]
        )
        chain_length = int(n_samples / self.cfg.sampler.n_chains)
        discard_by_chain = int(chain_length * 0.3)
        self.mcstate.n_samples = n_samples
        self.mcstate.sample(
            chain_length=chain_length, n_discard_per_chain=discard_by_chain
        )

        ops = Result.get_spin_operators(self.cfg.chain, ham, self.hilbert)
        ops_vals = self.vmc.estimate(ops)
        self.nn.config = dataclasses.replace(self.nn.config, training=train)

        return ops_vals

    def get_result(self) -> Result:
        ops_vals = self.get_ops_vals(self.ham)
        res = Result(self.cfg.chain)
        res.update(Result.ops_vals_to_res_data(ops_vals))

        self.mcstate.n_samples = self.n_samples
        return res


class ModelCustomNQS(ModelNQS):
    def __init__(self, cfg: ModelNQSConfig):
        super().__init__(cfg)

    def _set_ops(self):
        res = super()._set_ops()
        self.ham_jax = self.ham.to_pauli_strings().to_jax_operator()
        return res

    def to_array(self, parameters):
        all_configurations = self.hilbert.all_states()
        all_configurations = all_configurations.reshape(-1, self.n)

        logpsi = self.nn.apply(parameters, all_configurations)
        if isinstance(logpsi, tuple):
            logpsi = logpsi[0]
        psi = jnp.exp(logpsi)
        psi = psi / jnp.linalg.norm(psi)
        return psi

    def compute_energy(self, parameters, hamiltonian_sparse):
        psi_gs = self.to_array(parameters)
        return psi_gs.conj().T @ (hamiltonian_sparse @ psi_gs)

    def compute_energy_and_gradient(self, parameters, hamiltonian_sparse):
        grad_fun = jax.value_and_grad(self.compute_energy, argnums=1)
        return grad_fun(self.nn, parameters, hamiltonian_sparse)

    def compute_local_energies(self, parameters, hamiltonian_jax, sigma):
        eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

        logpsi_sigma = self.nn.apply(parameters, sigma)
        energy, psi_2_intermediate = None, None
        if isinstance(logpsi_sigma, tuple):
            energy, psi_2_intermediate = logpsi_sigma[1:]
            logpsi_sigma = logpsi_sigma[0]
        eta = eta.reshape(-1, eta.shape[-1])
        logpsi_eta = self.nn.apply(parameters, eta)
        if isinstance(logpsi_eta, tuple):
            logpsi_eta = logpsi_eta[0]
        logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
        logpsi_eta = logpsi_eta.reshape(-1, H_sigmaeta.shape[-1])

        res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

        return res, energy, psi_2_intermediate

    def compute_prob_energy_based_model(self, psi_2_intermediate):
        psi_2_intermediate = jnp.exp(-1 * psi_2_intermediate)

        psi_2_intermediate = psi_2_intermediate / jnp.sum(
            psi_2_intermediate, axis=1, keepdims=True
        )

        psi_2_intermediate *= psi_2_intermediate.shape[1]

        return psi_2_intermediate[:, -1]

    def estimate_energy(self, parameters, hamiltonian_jax, sigma):
        E_loc, _, _ = self.compute_local_energies(parameters, hamiltonian_jax, sigma)

        E_average = jnp.mean(E_loc)
        E_variance = jnp.var(E_loc)
        E_error = jnp.sqrt(E_variance / E_loc.size)

        return nk.stats.Stats(
            mean=E_average, error_of_mean=E_error, variance=E_variance
        )

    def estimate_energy_and_gradient(self, parameters, hamiltonian_jax, sigma):
        sigma = sigma.reshape(-1, sigma.shape[-1])
        E_loc, energy, psi_2_intermediate = self.compute_local_energies(
            parameters, hamiltonian_jax, sigma
        )
        E_loc = E_loc.astype(jnp.float64)

        E_average = jnp.mean(E_loc)
        E_variance = jnp.var(E_loc)
        E_error = jnp.sqrt(E_variance / E_loc.size)
        E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)

        def logpsi_sigma_fun(pars):
            ap = self.nn.apply(pars, sigma)
            if isinstance(ap, tuple):
                ap = ap[0]

            return ap

        _, vjpfun = jax.vjp(logpsi_sigma_fun, parameters)
        E_grad = vjpfun((E_loc - E_average) / E_loc.size)[0]

        def compute_energy_based_model_loss(params, s, eloc, en):
            probs = self.compute_prob_energy_based_model(psi_2_intermediate)

            eloc = eloc[:, None, None]
            psi_loc, _, _ = self.nn.apply(params, s, eloc)
            psi, _, _ = self.nn.apply(params, s, en)

            return jnp.mean(probs * (psi**2 - psi_loc**2))

        energy_based_model_grad = jax.grad(compute_energy_based_model_loss)
        energy_based_model_grad = energy_based_model_grad(
            parameters, sigma, E_loc, energy
        )

        return E, E_grad, energy_based_model_grad

    def train(self, n_iter: int = 100):
        chain_length = 1000 // self.sampler.n_chains

        key = jrnd.PRNGKey(42)
        inp = jnp.ones(16 * self.n)
        inp = inp.at[jrnd.randint(key, (16, self.n), 0, 16 * self.n)].set(-1)
        inp = inp.reshape(16, self.n)
        init_rngs = {"params": key, "dropout": key}
        parameters = self.nn.init(init_rngs, inp)
        sampler_state = self.sampler.init_state(self.nn, parameters, seed=1)
        optimizer_state = self.optimizer.init(parameters)

        logger = nk.logging.RuntimeLog()

        for i in tqdm(range(n_iter)):
            sampler_state = self.sampler.reset(self.nn, parameters, state=sampler_state)
            samples, sampler_state = self.sampler.sample(
                self.nn, parameters, state=sampler_state, chain_length=chain_length
            )

            E, E_grad, enBased_grad = self.estimate_energy_and_gradient(
                parameters, self.ham_jax, samples
            )

            updates, optimizer_state = self.optimizer.update(
                E_grad, optimizer_state, parameters
            )
            parameters = optax.apply_updates(parameters, updates)

            if i % 3 == 0:
                updates, optimizer_state = self.optimizer.update(
                    enBased_grad, optimizer_state, parameters
                )
                parameters = optax.apply_updates(parameters, updates)

            logger(step=i, item={"Energy": E})
            self.logger.info(E)
