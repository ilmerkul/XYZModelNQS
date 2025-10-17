import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrnd
import netket as nk
import optax
from flax.core import freeze, unfreeze
from flax.training import checkpoints
from netket.optimizer import identity_preconditioner
from tqdm import tqdm

from src.model.NN import NNConfig, NNType
from src.model.NN.convolution import CNN
from src.model.NN.feedforward import FF
from src.model.NN.feedforward.transfer_learning import NN
from src.model.NN.graph import GCNN
from src.model.NN.transformer import PhaseTransformer, Transformer
from src.model.nqs.operator import get_model_netket_op
from src.model.optimizer import NQSOptimizer
from src.model.sampler import SamplerType, TransformerSampler
from src.model.struct import ChainConfig
from src.result.struct import Result


@dataclass
class ModelNQSConfig:
    chain: ChainConfig
    nn: str
    sampler: str
    optimizer: str
    n_iter: int
    n_chains: int
    lr: float
    min_n_samples: int
    scale_n_samples: int
    preconditioner: bool
    dtype: jnp.dtype
    rnd_seed: int
    sr_diag_shift: float
    model_config: NNConfig
    tr_learning: bool
    save_model_path: str


class ModelNQS:
    def __init__(self, cfg: ModelNQSConfig):
        self.cfg = cfg
        self.graph = nk.graph.Chain(length=self.cfg.chain.n, pbc=self.cfg.chain.pbc)
        self.hilbert = nk.hilbert.Spin(N=self.cfg.chain.n, s=self.cfg.chain.spin)

        self._set_sampler()

        self._init_state()

    def _init_state(self):
        self._set_ops()

        self._set_machine()

        self._set_optimizer()

        self._set_vmc()

    def _set_ops(self):
        self.ham = get_model_netket_op(
            self.cfg.chain,
            self.hilbert,
        )
        self.ham_jax = self.ham.to_pauli_strings().to_jax_operator()

    def _set_machine(self):
        if self.cfg.nn == NNType.PHASE_TRANSFORMER:
            self.nn = PhaseTransformer(self.cfg.model_config)
        elif self.cfg.nn == NNType.TRANSFORMER:
            self.nn = Transformer(self.cfg.model_config)
        elif self.cfg.nn == NNType.CNN:
            self.nn = CNN(self.cfg.model_config)
        elif self.cfg.nn == NNType.GCNN:
            self.nn = GCNN(self.cfg.model_config)
        elif self.cfg.nn == NNType.FFN:
            self.nn = FF(self.cfg.model_config)
        else:
            raise ValueError("error nn")

    def _set_sampler(self):
        if self.cfg.sampler == SamplerType.TRANSFORMER:
            pass
        elif self.cfg.sampler == SamplerType.METROPOLIS:
            self.sampler = nk.sampler.MetropolisSampler(
                self.hilbert,
                rule=nk.sampler.rules.ExchangeRule(graph=self.graph, d_max=2),
                n_chains=self.cfg.n_chains,
                dtype=self.cfg.dtype,
            )
        else:
            raise ValueError("error sampler")

    def _set_optimizer(self):
        self.optimizer = NQSOptimizer.get_optimizer(
            self.cfg.optimizer, self.cfg.lr, self.cfg.n_iter
        )
        if self.cfg.nn == NNType.PHASE_TRANSFORMER and (
            self.cfg.model_config.pqc or self.cfg.model_config.gcnn
        ):
            self.optimizer_pqc = NQSOptimizer.get_optimizer(
                NQSOptimizer.ADAM_ZERO_PQC, self.cfg.lr, self.cfg.n_iter
            )

    def _set_vmc(self):
        self.n_samples = max(
            [self.cfg.min_n_samples, self.cfg.chain.n * self.cfg.scale_n_samples]
        )
        key = jax.random.PRNGKey(self.cfg.rnd_seed)
        self.mcstate = nk.vqs.MCState(
            sampler=self.sampler,
            n_samples=self.n_samples,
            init_fun=self.init_fun,
            apply_fun=self.apply_fun,
            seed=key,
        )

        sr = identity_preconditioner
        if self.cfg.preconditioner:
            sr = nk.optimizer.SR(
                diag_shift=self.cfg.sr_diag_shift, solver_restart=False
            )

        self.vmc = nk.driver.VMC(
            hamiltonian=self.ham,
            optimizer=self.optimizer,
            variational_state=self.mcstate,
            preconditioner=sr,
        )

    def set_h(self, h: float):
        if self.cfg.nn == NNType.TRANSFORMER:
            chain = dataclasses.replace(self.cfg.chain, h=h)
            self.cfg.chain = chain
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, chain=chain
            )
        elif self.cfg.nn == NNType.PHASE_TRANSFORMER:
            chain = dataclasses.replace(self.cfg.chain, h=h)
            self.cfg.chain = chain
            tr_config = dataclasses.replace(
                self.cfg.model_config.tr_config, chain=chain
            )
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, tr_config=tr_config
            )
        elif self.cfg.nn == NNType.CNN:
            chain = dataclasses.replace(self.cfg.chain, h=h)
            self.cfg.chain = chain
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, chain=chain
            )
        self._init_state()

    def train(self):
        logger_runtime = nk.logging.RuntimeLog()
        logger_tensorboard = nk.logging.TensorBoardLog()
        logger_out = [logger_runtime, logger_tensorboard]

        if self.cfg.nn == NNType.PHASE_TRANSFORMER:
            n_ter = self.cfg.n_iter
            if self.cfg.model_config.pqc or self.cfg.model_config.gcnn:
                n_ter //= 2
            self.vmc.run(
                n_iter=n_ter,
                out=logger_out,
                callback=lambda *x: True,
            )
            # self.sampler = TransformerSampler(self.hilbert)
            if self.cfg.model_config.pqc or self.cfg.model_config.gcnn:
                self.cfg.model_config.phase_train = True
                self.vmc.optimizer = self.optimizer_pqc
                self._set_vmc()
                self.vmc.run(
                    n_iter=n_ter,
                    out=logger_out,
                    callback=lambda *x: True,
                )
        else:
            self.vmc.run(
                n_iter=self.cfg.n_iter,
                out=logger_out,
                callback=lambda *x: True,
            )

        if self.cfg.tr_learning:
            self.save_model()

    def get_result(self) -> Result:
        train = self.nn.config.training
        self.nn.config.training = False

        n_samples = max(
            [self.cfg.min_n_samples, self.cfg.chain.n * self.cfg.scale_n_samples]
        )
        chain_length = int(n_samples / self.cfg.n_chains)
        discard_by_chain = int(chain_length * 0.3)
        self.mcstate.n_samples = n_samples
        self.mcstate.sample(
            chain_length=chain_length, n_discard_per_chain=discard_by_chain
        )

        ops = Result.get_spin_operators(self.cfg.chain, self.hilbert)
        ops_vals = self.vmc.estimate(ops)
        res = Result(self.cfg.chain)
        res.update(Result.ops_vals_to_res_data(ops_vals))

        self.mcstate.n_samples = self.n_samples
        self.nn.config.training = train
        return res

    def save_model(self):
        concrete_params = jax.tree_map(
            lambda x: x.copy() if hasattr(x, "copy") else x, self.mcstate.parameters
        )

        file_name = checkpoints.save_checkpoint(
            ckpt_dir=self.cfg.save_model_path,
            target=concrete_params,
            step=0,
            prefix=self.nn.__class__.__name__,
            overwrite=True,
        )
        print(f"model is saved in {file_name}")

    def apply_fun(
        self,
        params,
        variables,
        batch_dim: int = 16,
        mutable: bool = False,
        generate: bool = False,
        **kwargs,
    ):
        if self.cfg.nn == NNType.PHASE_TRANSFORMER:
            return self.nn.apply(
                params,
                variables,
                generate=generate,
                n_chains=batch_dim,
                rngs={"dropout": jrnd.PRNGKey(self.cfg.rnd_seed)},
                mutable=mutable,
                **kwargs,
            )
        elif self.cfg.nn == NNType.TRANSFORMER:
            return self.nn.apply(
                params,
                variables,
                generate=generate,
                n_chains=batch_dim,
                rngs={"dropout": jrnd.PRNGKey(self.cfg.rnd_seed)},
                mutable=mutable,
                **kwargs,
            )
        elif self.cfg.nn == NNType.CNN:
            return self.nn.apply(params, variables)
        elif self.cfg.nn == NNType.GCNN:
            return self.nn.apply(params, variables)
        elif self.cfg.nn == NNType.FFN:
            return self.nn.apply(params, variables)
        else:
            raise ValueError("error apply function")

    def init_fun(self, params, inp):
        rng_key = params["params"]
        init_rng = {"params": rng_key, "dropout": rng_key}

        variables = self.nn.init(init_rng, inp)
        pam_count = sum(x.size for x in jax.tree_util.tree_leaves(variables))
        print(f"Parameters count: {pam_count}")

        if self.cfg.tr_learning:
            try:
                restored = checkpoints.restore_checkpoint(
                    ckpt_dir=self.cfg.save_model_path,
                    target=variables,
                    step=0,
                    prefix=self.nn.__class__.__name__,
                )
                if restored is not None:
                    variables = restored
                    print(f"Loaded model from {self.cfg.save_model_path}")
            except (ValueError, FileNotFoundError):
                print(
                    f"Params model in {self.cfg.save_model_path} not found, using random initialization"
                )

        return variables


class ModelCustomNQS(ModelNQS):
    def __init__(self, cfg: ModelNQSConfig):
        super().__init__(cfg)

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
            print(E)


class ModelTLNQS(ModelNQS):
    def __init__(self, cfg: ModelNQSConfig):
        super().__init__(cfg)

    def set_machine(self, base_weights: Optional[Path] = None):
        nn = NN(self.n, alpha=5)

        variables = nn.init(jrnd.PRNGKey(self.cfg.rnd_seed), jnp.ones(self.cfg.n))
        if base_weights is not None:
            kernel = jnp.load(base_weights.joinpath("kernel.npy"))
            bias = jnp.load(base_weights.joinpath("bias.npy"))

            updatable = unfreeze(variables)
            updatable["params"]["base"]["kernel"] = kernel
            updatable["params"]["base"]["bias"] = bias
            variables = freeze(updatable)

        self.nn = nn
        self.variables = variables

    def set_optimizer(self, lr: float = 3e-2, train_base=True):
        self.train_base = train_base
        opt = optax.sgd(
            learning_rate=optax.exponential_decay(lr, 10, 0.97, 1000, end_value=1e-2),
            momentum=0.9,
            nesterov=True,
        )

        def zero_grads():
            def init_fn(_):
                return ()

            def update_fn(updates, state, params=None):
                return jax.tree_map(jnp.zeros_like, updates), ()

            return optax.GradientTransformation(init_fn, update_fn)

        if not train_base:
            self.optimizer = optax.multi_transform(
                freeze(
                    {
                        "sgd": opt,
                        "zero": zero_grads(),
                    }
                ),
                freeze({"head": "sgd", "base": "zero"}),
            )
        else:
            self.optimizer = opt

    def set_vmc(self):
        n_samples = max([2000, self.n * 45])
        self.mcstate = nk.vqs.MCState(
            sampler=self.sampler,
            model=self.nn,
            n_samples=n_samples,
            variables=self.variables,
        )
        sr = nk.optimizer.SR(solver_restart=False)
        self.vmc = nk.driver.VMC(
            hamiltonian=self.ham,
            optimizer=self.optimizer,
            variational_state=self.mcstate,
            preconditioner=sr,
        )

    def print_base_mean(self):
        return jnp.mean(self.mcstate.parameters["base"]["kernel"])

    def print_head_mean(self):
        return jnp.mean(self.mcstate.parameters["head"]["kernel"])

    def save_base_weights(self, prefix: Path):
        kernel = self.mcstate.parameters["base"]["kernel"]
        bias = self.mcstate.parameters["base"]["bias"]

        jnp.save(file=prefix.joinpath("kernel.npy"), arr=kernel)
        jnp.save(file=prefix.joinpath("bias.npy"), arr=bias)
