import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import pickle
import hydra
import jax
import jax.numpy as jnp
import jax.random as jrnd
import netket as nk
import optax
from flax.core import freeze, unfreeze
from netket.optimizer import identity_preconditioner
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
from src.utils import report_name
from tqdm import tqdm

from .callbacks import AdaptiveMomentumCallback, ParametersPrint, VarianceCallback


@dataclass
class ModelNQSConfig:
    chain: ChainConfig
    sampler: str
    optimizer: str
    n_iter: int
    n_chains: int
    lr: float
    min_n_samples: int
    scale_n_samples: int
    preconditioner: bool
    sr_diag_shift: float
    model_config: NNConfig
    tr_learning: bool

    def __post_init__(self):
        self.dtype = self.model_config.dtype
        self.rnd_seed = self.model_config.seed
        self.save_model_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


class ModelNQS:
    def __init__(self, cfg: ModelNQSConfig):
        self.cfg = cfg
        self.graph = nk.graph.Chain(length=self.cfg.chain.n, pbc=self.cfg.chain.pbc)
        self.hilbert = nk.hilbert.Spin(N=self.cfg.chain.n, s=self.cfg.chain.spin)
        self.key = jrnd.PRNGKey(self.cfg.rnd_seed)

        self.last_h = cfg.chain.h

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
        nn = self.cfg.model_config.nntype
        if nn == NNType.PHASE_TRANSFORMER:
            self.nn = PhaseTransformer(self.cfg.model_config)
        elif nn == NNType.TRANSFORMER:
            self.nn = Transformer(self.cfg.model_config)
        elif nn == NNType.CNN:
            self.nn = CNN(self.cfg.model_config)
        elif nn == NNType.GCNN:
            self.nn = GCNN(self.cfg.model_config)
        elif nn == NNType.FFN:
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
        nn = self.cfg.model_config.nntype
        if nn == NNType.PHASE_TRANSFORMER and (
            self.cfg.model_config.pqc or self.cfg.model_config.gcnn
        ):
            self.optimizer_pqc = NQSOptimizer.get_optimizer(
                NQSOptimizer.ADAM_PQC, self.cfg.lr, self.cfg.n_iter
            )

    def _set_vmc(self):
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
            init_fun=self.init_fun,
            apply_fun=self.apply_fun,
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
            hamiltonian=self.ham,
            optimizer=self.optimizer,
            variational_state=self.mcstate,
            preconditioner=sr,
        )

    def set_h(self, h: float):
        nn = self.cfg.model_config.nntype
        self.last_h = self.cfg.chain.h
        if nn == NNType.TRANSFORMER:
            chain = dataclasses.replace(self.cfg.chain, h=h)
            self.cfg.chain = chain
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, chain=chain
            )
        elif nn == NNType.PHASE_TRANSFORMER:
            chain = dataclasses.replace(self.cfg.chain, h=h)
            self.cfg.chain = chain
            tr_config = dataclasses.replace(
                self.cfg.model_config.tr_config, chain=chain
            )
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, tr_config=tr_config
            )
        elif nn == NNType.CNN:
            chain = dataclasses.replace(self.cfg.chain, h=h)
            self.cfg.chain = chain
            self.cfg.model_config = dataclasses.replace(
                self.cfg.model_config, chain=chain
            )
        self._init_state()

    def train(self):
        logger_runtime = nk.logging.RuntimeLog()
        logger_tensorboard = nk.logging.TensorBoardLog(
            comment=report_name(self.nn.config.chain)
        )
        logger_out = [logger_runtime, logger_tensorboard]

        callback = [
            lambda *x: True,
            # ParametersPrint(),
            AdaptiveMomentumCallback(lr=self.cfg.lr, n_iter=self.cfg.n_iter),
            VarianceCallback(),
        ]
        nn = self.cfg.model_config.nntype
        if nn == NNType.PHASE_TRANSFORMER:
            n_ter = self.cfg.n_iter
            if self.cfg.model_config.pqc or self.cfg.model_config.gcnn:
                n_ter //= 2
            self.vmc.run(
                n_iter=n_ter,
                out=logger_out,
                callback=callback,
            )
            # self.sampler = TransformerSampler(self.hilbert)
            if self.cfg.model_config.pqc or self.cfg.model_config.gcnn:
                self.cfg.model_config = dataclasses.replace(
                    self.cfg.model_config, phase_train=True
                )
                self.vmc.optimizer = self.optimizer_pqc
                self._set_vmc()
                self.vmc.run(
                    n_iter=n_ter,
                    out=logger_out,
                    callback=callback,
                )
        else:
            self.vmc.run(
                n_iter=self.cfg.n_iter,
                out=logger_out,
                callback=callback,
            )

        self.save_model()

    def get_result(self) -> Result:
        train = self.nn.config.training
        self.nn.config = dataclasses.replace(self.nn.config, training=False)

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
        self.nn.config = dataclasses.replace(self.nn.config, training=train)
        return res

    def save_model(self):
        try:
            concrete_params = jax.tree_map(
                lambda x: x.copy() if hasattr(x, "copy") else x, self.mcstate.parameters
            )

            os.makedirs(self.cfg.save_model_path, exist_ok=True)

            model_file = os.path.join(
                self.cfg.save_model_path, f"{self.nn.__class__.__name__}_params_h({self.cfg.chain.h}).pkl"
            )

            with open(model_file, "wb") as f:
                pickle.dump(concrete_params, f)

            print(f"Model parameters saved in {model_file}")

        except Exception as e:
            print(f"Error saving model: {e}")
            self._save_model_simple()

    def _save_model_simple(self):
        try:

            params = self.mcstate.parameters
            os.makedirs(self.cfg.save_model_path, exist_ok=True)

            model_file = os.path.join(
                self.cfg.save_model_path, f"{self.nn.__class__.__name__}_simple_h({self.cfg.chain.h}).pkl"
            )

            with open(model_file, "wb") as f:
                pickle.dump(params, f)

            print(f"Model saved simply in {model_file}")

        except Exception as e:
            print(f"Even simple save failed: {e}")

    def load_model(self):
        try:

            model_file = os.path.join(
                self.cfg.save_model_path, f"{self.nn.__class__.__name__}_params_h({self.last_h}).pkl"
            )

            if os.path.exists(model_file):
                with open(model_file, "rb") as f:
                    loaded_params = pickle.load(f)
                return loaded_params
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def apply_fun(
        self,
        params,
        variables,
        **kwargs,
    ):
        x_hash = jnp.abs(jnp.sum(variables)).astype(jnp.int32)
        base_key = jrnd.PRNGKey(self.cfg.rnd_seed)
        key = jrnd.fold_in(base_key, x_hash)
        dropout_key, symmetry_key, output_key = jrnd.split(key, num=3)
        nn = self.cfg.model_config.nntype
        if nn == NNType.PHASE_TRANSFORMER:
            output = self.nn.apply(
                params,
                variables,
                rngs={
                    "dropout": dropout_key,
                    "symmetry": symmetry_key,
                    "output": output_key,
                },
                **kwargs,
            )
        elif nn == NNType.TRANSFORMER:
            output = self.nn.apply(
                params,
                variables,
                rngs={
                    "dropout": dropout_key,
                    "symmetry": symmetry_key,
                    "output": output_key,
                },
                **kwargs,
            )
        elif nn == NNType.CNN:
            output = self.nn.apply(
                params,
                variables,
                **kwargs,
            )
        elif nn == NNType.GCNN:
            output = self.nn.apply(
                params,
                variables,
                **kwargs,
            )
        elif nn == NNType.FFN:
            output = self.nn.apply(
                params,
                variables,
                **kwargs,
            )
        else:
            raise ValueError("error apply function")

        return output

    def init_fun(self, params_rng_key, inp):
        rng_key = params_rng_key["params"]
        init_rng = {
            "params": rng_key,
            "dropout": rng_key,
            "symmetry": rng_key,
            "output": rng_key,
        }

        variables = self.nn.init(init_rng, inp)
        pam_count = sum(x.size for x in jax.tree_util.tree_leaves(variables))
        print(f"Parameters count: {pam_count}")

        if self.cfg.tr_learning:
            try:
                loaded_params = self.load_model()
                if loaded_params is not None:

                    def add_adaptive_noise(params, key, noise_scale):
                        def add_noise_to_leaf(param, leaf_key):
                            param_std = jnp.std(param)
                            noise = (
                                jrnd.normal(leaf_key, param.shape)
                                * param_std
                                * noise_scale
                            )
                            return param + noise

                        keys_tree = jax.tree_util.tree_map(
                            lambda param: jrnd.fold_in(
                                key, jnp.sum(param).astype(jnp.int32)
                            ),
                            params,
                        )

                        return jax.tree_util.tree_map(
                            add_noise_to_leaf, params, keys_tree
                        )

                    noisy_params = add_adaptive_noise(
                        loaded_params, rng_key, noise_scale=1.0
                    )

                    variables = {"params": noisy_params}
                    print(f"Loaded model from {self.cfg.save_model_path}")
                else:
                    print("No saved model found, using random initialization")
            except Exception as e:
                print(f"Error loading model: {e}, using random initialization")

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
