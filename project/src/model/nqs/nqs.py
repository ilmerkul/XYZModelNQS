from functools import partial
from pathlib import Path
from typing import Dict

import jax.random
import netket as nk
import optax
from jax import numpy as jnp
from tqdm import tqdm

from ...result.struct import Results
from ..NN import (CNN, CNNConfig, EnergyBasedTransformer,
                  EnergyBasedTransformerConfig, EnergyOptimModel, Transformer,
                  TransformerConfig, PhaseTransformer, GCNN,
                  TransformerSampler)
from .callbacks import RHatStop, VarianceCallback
from .operators import get_model_netket_op


class Model:
    dtype: jnp.dtype = jnp.float64

    def __init__(self, n: int, h: float, j: float,
                 lam: float, gamma: float,
                 pbc: bool = False, spin: float = 1 / 2):
        self.n = n
        self.graph = nk.graph.Chain(length=n, pbc=pbc)
        self.hilbert = nk.hilbert.Spin(N=n, s=spin)

        self.h = h
        self.j = j
        self.lam = lam
        self.gamma = gamma

        self.ham = get_model_netket_op(n, j, h, lam, gamma, self.hilbert)
        self.ham_jax = self.ham.to_pauli_strings().to_jax_operator()

        self.nn_cfg = None
        self.nn = None

        self.n_chains = None
        self.sampler = None

        self.optimizer = None

        self.n_samples = None
        self.mcstate = None
        self.vmc = None

        self.ops = None

    def set_machine(self):
        self.nn_cfg = TransformerConfig(length=self.n,
                                        training=False,
                                        dtype=self.dtype,
                                        symm=True,
                                        automorphisms=self.graph.automorphisms(),
                                        )
        self.nn = PhaseTransformer(self.nn_cfg)
        # self.nn = CNN(CNNConfig())
        # self.nn = GCNN(self.n)

    def set_sampler(self, n_chains: int = 16):
        self.n_chains = n_chains
        # self.sampler = nk.sampler.MetropolisSampler(
        #    self.hilbert,
        #    rule=nk.sampler.rules.ExchangeRule(graph=self.graph, d_max=5),
        #    n_chains=n_chains,
        #    dtype=jnp.float64,
        # )
        self.sampler = TransformerSampler(self.hilbert)

    def set_optimizer(self, lr: float = 1e-3):
        def zero_grads():
            def init_fn(_):
                return ()

            def update_fn(updates, state, params=None):
                return jax.tree_map(jnp.zeros_like, updates), ()

            return optax.GradientTransformation(init_fn, update_fn)

        def map_nested_fn(fn):
            def map_fn(nested_dict):
                p = {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                     for k, v in nested_dict.items()}
                return p

            return map_fn

        label_fn = map_nested_fn(
            lambda k, _: "tr" if k.startswith("tr") else "pqc")
        self.optimizer = optax.multi_transform(
            {"tr": optax.adam(learning_rate=optax.join_schedules([
                optax.constant_schedule(lr * 10),
                optax.exponential_decay(init_value=lr * 10,
                                        transition_steps=10,
                                        decay_rate=0.8, end_value=lr)
            ], boundaries=[25])),
                "pqc": zero_grads()},
            label_fn)
        self.optimizer_pqc = nk.optimizer.Adam(1e-3)
        self.optimizer = nk.optimizer.Adam(1e-3)

        # self.optimizer = optax.adam(learning_rate=optax.cosine_decay_schedule(init_value=1e-2,
        #                                                                      decay_steps=10))
        # self.optimizer = optax.sgd(learning_rate=optax.join_schedules(
        #    [optax.linear_schedule(init_value=lr, end_value=lr * 10,
        #                           transition_steps=10),
        #     optax.constant_schedule(lr * 10),
        #     optax.exponential_decay(init_value=lr * 10, transition_steps=35,
        #                             decay_rate=0.8, end_value=lr / 10)],
        #    boundaries=[10, 15]))
        # self.optimizer = optax.adam(learning_rate=optax.join_schedules([
        #    optax.constant_schedule(lr * 10),
        #    optax.exponential_decay(init_value=lr * 10, transition_steps=40,
        #                            decay_rate=0.8, end_value=lr)
        # ], boundaries=[25]))
        # self.optimizer = optax.sgd(
        #    learning_rate=optax.exponential_decay(init_value=1e-3,
        #                                          transition_steps=10,
        #                                          decay_rate=0.9,
        #                                          transition_begin=50,
        #                                          end_value=1e-4),
        #    momentum=0.9,
        #    nesterov=True,
        # )

    def set_vmc(self, min_n_samples: int = 2000, scale_n_samples: int = 45):
        # self.n_samples = max([min_n_samples, self.n * scale_n_samples])
        self.n_samples = min_n_samples

        self.mcstate = nk.vqs.MCState(
            sampler=self.sampler,
            n_samples=self.n_samples,
            init_fun=self.init_fun,
            apply_fun=self.apply_fun,
        )
        sr = nk.optimizer.SR(diag_shift=0.1, solver_restart=False)
        self.vmc = nk.driver.VMC(
            hamiltonian=self.ham,
            optimizer=self.optimizer,
            variational_state=self.mcstate,
            preconditioner=sr,
        )

    def set_ops(self, ops: Dict[str, nk.operator.LocalOperator]):
        ops["e"] = self.ham
        self.ops = ops

    def train(self, n_iter: int = 500):
        logger = nk.logging.RuntimeLog()

        self.vmc.run(
            n_iter=n_iter // 2,
            out=logger,
            callback=[
                # RHatStop(base=1.3, stop_cnt=30),
                # VarianceCallback(base=5.0e-20),
            ],
        )

        self.vmc.optimizer = self.optimizer_pqc

        self.vmc.run(
            n_iter=n_iter // 2,
            out=logger,
            callback=[
                # RHatStop(base=1.3, stop_cnt=30),
                # VarianceCallback(base=5.0e-20),
            ],
        )

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

    def compute_local_energies(self, parameters,
                               hamiltonian_jax, sigma):
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

        psi_2_intermediate = psi_2_intermediate / jnp.sum(psi_2_intermediate,
                                                          axis=1,
                                                          keepdims=True)

        psi_2_intermediate *= psi_2_intermediate.shape[1]

        return psi_2_intermediate[:, -1]

    def estimate_energy(self, parameters, hamiltonian_jax, sigma):
        E_loc, _, _ = self.compute_local_energies(
            parameters,
            hamiltonian_jax,
            sigma)

        E_average = jnp.mean(E_loc)
        E_variance = jnp.var(E_loc)
        E_error = jnp.sqrt(E_variance / E_loc.size)

        return nk.stats.Stats(mean=E_average,
                              error_of_mean=E_error,
                              variance=E_variance)

    def estimate_energy_and_gradient(self, parameters,
                                     hamiltonian_jax, sigma):
        sigma = sigma.reshape(-1, sigma.shape[-1])
        E_loc, energy, psi_2_intermediate = self.compute_local_energies(
            parameters,
            hamiltonian_jax,
            sigma)
        E_loc = E_loc.astype(jnp.float64)

        E_average = jnp.mean(E_loc)
        E_variance = jnp.var(E_loc)
        E_error = jnp.sqrt(E_variance / E_loc.size)
        E = nk.stats.Stats(mean=E_average,
                           error_of_mean=E_error,
                           variance=E_variance)

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

            return jnp.mean(probs * (psi ** 2 - psi_loc ** 2))

        energy_based_model_grad = jax.grad(compute_energy_based_model_loss)
        energy_based_model_grad = energy_based_model_grad(parameters, sigma,
                                                          E_loc, energy)

        return E, E_grad, energy_based_model_grad

    def custom_train(self, n_iter: int = 100):
        chain_length = 1000 // self.sampler.n_chains

        key = jax.random.PRNGKey(42)
        inp = jnp.ones(16 * self.n)
        inp = inp.at[
            jax.random.randint(key, (16, self.n), 0, 16 * self.n)].set(-1)
        inp = inp.reshape(16, self.n)
        init_rngs = {'params': key,
                     'dropout': key}
        parameters = self.nn.init(init_rngs, inp)
        sampler_state = self.sampler.init_state(self.nn, parameters, seed=1)
        optimizer_state = self.optimizer.init(parameters)

        logger = nk.logging.RuntimeLog()

        for i in tqdm(range(n_iter)):
            sampler_state = self.sampler.reset(self.nn, parameters,
                                               state=sampler_state)
            samples, sampler_state = self.sampler.sample(self.nn, parameters,
                                                         state=sampler_state,
                                                         chain_length=chain_length)

            E, E_grad, enBased_grad = self.estimate_energy_and_gradient(
                parameters,
                self.ham_jax,
                samples)

            updates, optimizer_state = self.optimizer.update(E_grad,
                                                             optimizer_state,
                                                             parameters)
            parameters = optax.apply_updates(parameters, updates)

            if i % 3 == 0:
                updates, optimizer_state = self.optimizer.update(enBased_grad,
                                                                 optimizer_state,
                                                                 parameters)
                parameters = optax.apply_updates(parameters, updates)

            logger(step=i, item={'Energy': E})
            print(E)

    def get_hilbert(self):
        return self.hilbert

    def get_j(self):
        return self.j

    def get_h(self):
        return self.h

    def get_analytical(self) -> Results:
        res = Results(self.h, self.j, self.n, 'xy')  # TODO
        res.analytical()

        return res

    def get_results(self, min_n_samples: int = 6000,
                    scale_n_samples: int = 200) -> Results:
        # state = self.mcstate.sampler_state.σ.copy()
        # self.mcstate.sampler = nk.sampler.MetropolisSamplerNumpy(
        #    self.hilbert,
        #    rule=nk.sampler.rules.HamiltonianRuleNumpy(operator=self.ham),
        #    n_chains=self.n_chains,
        #    dtype=jnp.float64,
        # )
        # self.mcstate.sampler_state.σ = jnp.array(state)

        # train = self.nn.config.training
        # self.nn.config.training = False

        n_samples = max([min_n_samples, self.n * scale_n_samples])
        chain_length = int(n_samples / self.n_chains)
        discard_by_chain = int(chain_length * 0.3)
        self.mcstate.n_samples = n_samples
        self.mcstate.sample(chain_length=chain_length,
                            n_discard_per_chain=discard_by_chain)
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

        # self.mcstate.sampler = self.sampler
        self.mcstate.n_samples = self.n_samples
        # self.nn.config.training = train
        return res

    def save_model(self, path: Path):
        self.nn.save(str(path.absolute()))

    def apply_fun(self, params, variables, mutable=False, generate=False,
                  n_chains: int = 16):
        return self.nn.apply(params, variables,
                             generate=generate,
                             n_chains=n_chains,
                             rngs={'dropout': jax.random.PRNGKey(42)},
                             mutable=mutable)

    def init_fun(self, params, inp):
        rng_key = params['params']
        init_rng = {'params': rng_key,
                    'dropout': rng_key}

        params = self.nn.init(init_rng, inp)

        return params
