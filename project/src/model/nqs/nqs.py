from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.random
import netket as nk
import optax
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from netket.optimizer import identity_preconditioner
from src.model.NN import CNN, GCNN, CNNConfig, PhaseTransformer, TransformerConfig
from src.model.NN.feedforward.transfer_learning import NN
from src.model.sampler import TransformerSampler
from src.result.struct import Results
from tqdm import tqdm

from .operators import get_model_netket_op, get_spin_operators


@dataclass
class ModelNQSConfig:
    nn: str = "phase_transformer"
    sampler: str = "transformer"
    optimizer: str = "adam"
    pbc: bool = False
    spin: float = 1 / 2
    gamma: float = 1.0
    lam: float = 1.0
    j: float = 1.0
    n: int = 10
    h: float = 1.0
    n_chains: int = 16
    lr: float = 1e-3
    min_n_samples: int = 200
    scale_n_samples: int = 45
    preconditioner: bool = False
    dtype: jnp.dtype = jnp.float64


class ModelNQS:
    def __init__(self, cfg: ModelNQSConfig):
        self.cfg = cfg
        self.graph = nk.graph.Chain(length=self.cfg.n, pbc=self.cfg.pbc)
        self.hilbert = nk.hilbert.Spin(N=self.cfg.n, s=self.cfg.spin)

        self._set_ops()

        self._set_machine()

        self._set_sampler()

        self._set_optimizer()

        self._set_vmc()

    def _set_machine(self):
        if self.cfg.nn == "phase_transformer":
            self.nn = PhaseTransformer(
                TransformerConfig(
                    length=self.cfg.n,
                    training=False,
                    dtype=self.cfg.dtype,
                    symm=True,
                    automorphisms=self.graph.automorphisms(),
                )
            )
        elif self.cfg.nn == "cnn":
            self.nn = CNN(CNNConfig())
        elif self.cfg.nn == "gcnn":
            self.nn = GCNN(self.cfg.n)
        else:
            raise ValueError("error nn")

    def _set_sampler(self):
        if self.cfg.sampler == "transformer":
            self.sampler = TransformerSampler(self.hilbert)
        elif self.cfg.sampler == "metropolis":
            self.sampler = nk.sampler.MetropolisSampler(
                self.hilbert,
                rule=nk.sampler.rules.ExchangeRule(graph=self.graph, d_max=3),
                n_chains=self.cfg.n_chains,
                dtype=jnp.float64,
            )
        else:
            raise ValueError("error sampler")

    def _set_optimizer(self):
        lr = self.cfg.lr

        def zero_grads():
            def init_fn(_):
                return ()

            def update_fn(updates, state, params=None):
                return jax.tree_map(jnp.zeros_like, updates), ()

            return optax.GradientTransformation(init_fn, update_fn)

        def map_nested_fn(fn):
            def map_fn(nested_dict):
                p = {
                    k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                    for k, v in nested_dict.items()
                }
                return p

            return map_fn

        label_fn = map_nested_fn(lambda k, _: "tr" if k.startswith("tr") else "pqc")
        self.optimizer = optax.multi_transform(
            {
                "tr": optax.adam(
                    learning_rate=optax.join_schedules(
                        [
                            optax.constant_schedule(lr * 10),
                            optax.exponential_decay(
                                init_value=lr * 10,
                                transition_steps=10,
                                decay_rate=0.8,
                                end_value=lr,
                            ),
                        ],
                        boundaries=[25],
                    )
                ),
                "pqc": zero_grads(),
            },
            label_fn,
        )
        if self.cfg.optimizer == "adam":
            self.optimizer = nk.optimizer.Adam(1e-3)
        elif self.cfg.optimizer == "adam_cos":
            self.optimizer = optax.adam(
                learning_rate=optax.cosine_decay_schedule(
                    init_value=1e-2, decay_steps=10
                )
            )
        elif self.cfg.optimizer == "adam_exp":
            self.optimizer = optax.adam(
                learning_rate=optax.join_schedules(
                    [
                        optax.constant_schedule(lr * 10),
                        optax.exponential_decay(
                            init_value=lr * 10,
                            transition_steps=40,
                            decay_rate=0.8,
                            end_value=lr,
                        ),
                    ],
                    boundaries=[25],
                )
            )
        elif self.cfg.optimizer == "sgd_lin_exp":
            self.optimizer = optax.sgd(
                learning_rate=optax.join_schedules(
                    [
                        optax.linear_schedule(
                            init_value=lr, end_value=lr * 10, transition_steps=10
                        ),
                        optax.constant_schedule(lr * 10),
                        optax.exponential_decay(
                            init_value=lr * 10,
                            transition_steps=35,
                            decay_rate=0.8,
                            end_value=lr / 10,
                        ),
                    ],
                    boundaries=[10, 15],
                )
            )
        else:
            self.optimizer = optax.sgd(
                learning_rate=optax.exponential_decay(
                    init_value=1e-3,
                    transition_steps=10,
                    decay_rate=0.9,
                    transition_begin=50,
                    end_value=1e-4,
                ),
                momentum=0.9,
                nesterov=True,
            )
        if self.cfg.nn == "phase_transformer":
            self.optimizer_pqc = nk.optimizer.Adam(1e-3)

    def _set_vmc(self):
        self.n_samples = max(
            [self.cfg.min_n_samples, self.cfg.n * self.cfg.scale_n_samples]
        )

        if self.cfg.nn == "phase_transformer":
            self.mcstate = nk.vqs.MCState(
                sampler=self.sampler,
                n_samples=self.n_samples,
                init_fun=self.init_fun,
                apply_fun=self.apply_fun,
            )
        else:
            self.mcstate = nk.vqs.MCState(
                model=self.nn,
                sampler=self.sampler,
                n_samples=self.n_samples,
            )
        sr = identity_preconditioner
        if self.cfg.preconditioner:
            sr = nk.optimizer.SR(diag_shift=0.01, solver_restart=False)

        self.vmc = nk.driver.VMC(
            hamiltonian=self.ham,
            optimizer=self.optimizer,
            variational_state=self.mcstate,
            preconditioner=sr,
        )

    def set_h(self, h: float):
        self.cfg.h = h
        self._set_ops()

    def _set_ops(self):
        self.ham = get_model_netket_op(
            self.cfg.n,
            self.cfg.j,
            self.cfg.h,
            self.cfg.lam,
            self.cfg.gamma,
            self.hilbert,
        )
        self.ham_jax = self.ham.to_pauli_strings().to_jax_operator()

        ops = get_spin_operators(self.cfg.n, self.hilbert)
        ops["e"] = self.ham
        self.ops = ops

    def train(self, n_iter: int = 500):
        logger = nk.logging.RuntimeLog()
        assert self.ops is not None

        if self.cfg.nn == "phase_transformer":
            self.vmc.run(
                n_iter=n_iter // 2,
                out=logger,
                callback=lambda *x: True,
            )
            self.vmc.optimizer = self.optimizer_pqc
            self.vmc.run(
                n_iter=n_iter // 2,
                out=logger,
                callback=lambda *x: True,
            )
        else:
            self.vmc.run(
                n_iter=n_iter,
                out=logger,
                callback=lambda *x: True,
            )

    def get_hilbert(self):
        return self.hilbert

    def get_j(self):
        return self.j

    def get_h(self):
        return self.h

    def get_analytical(self) -> Results:
        res = Results(self.cfg.h, self.cfg.j, self.cfg.n)
        res.analytical()

        return res

    def get_results(
        self, min_n_samples: int = 6000, scale_n_samples: int = 200
    ) -> Results:
        # train = self.nn.config.training
        # self.nn.config.training = False

        n_samples = max([min_n_samples, self.cfg.n * scale_n_samples])
        chain_length = int(n_samples / self.cfg.n_chains)
        discard_by_chain = int(chain_length * 0.3)
        self.mcstate.n_samples = n_samples
        self.mcstate.sample(
            chain_length=chain_length, n_discard_per_chain=discard_by_chain
        )
        ops_vals = self.vmc.estimate(self.ops)

        res = self.get_analytical()

        res.update(
            energy=jnp.real(ops_vals["e"].mean),
            var=jnp.real(ops_vals["e"].variance),
            spins=jnp.array(
                [
                    jnp.real(val.mean)
                    for op, val in ops_vals.items()
                    if op.startswith("s_")
                ]
            ),
            xx=jnp.real(ops_vals["xx"].mean),
            yy=jnp.real(ops_vals["yy"].mean),
            zz=jnp.real(ops_vals["zz"].mean),
            zz_mid=jnp.real(ops_vals["zz_mid"].mean),
        )

        self.mcstate.n_samples = self.n_samples
        # self.nn.config.training = train
        return res

    def save_model(self, path: Path):
        self.nn.save(str(path.absolute()))

    def apply_fun(
        self,
        params,
        variables,
        batch_dim: int = 16,
        mutable: bool = False,
        generate: bool = False,
        **kwargs
    ):
        if self.cfg.nn == "phase_transformer":
            return self.nn.apply(
                params,
                variables,
                generate=generate,
                n_chains=batch_dim,
                rngs={"dropout": jax.random.PRNGKey(42)},
                mutable=mutable,
                **kwargs
            )
        elif self.cfg.nn == "cnn":
            return self.nn.apply(params, variables)
        elif self.cfg.nn == "gcnn":
            return self.nn.apply(params, variables)
        else:
            raise ValueError("error apply function")

    def init_fun(self, params, inp):
        rng_key = params["params"]
        init_rng = {"params": rng_key, "dropout": rng_key}

        params = self.nn.init(init_rng, inp)

        return params


class ModelCustomNQS(ModelNQS):
    def __init__(self, cfg):
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

        key = jax.random.PRNGKey(42)
        inp = jnp.ones(16 * self.n)
        inp = inp.at[jax.random.randint(key, (16, self.n), 0, 16 * self.n)].set(-1)
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
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_machine(self, base_weights: Optional[Path] = None):
        nn = NN(self.n, alpha=5)

        variables = nn.init(PRNGKey(0), jnp.ones(self.n))
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
