from pathlib import Path
from typing import Dict

import jax.random
import netket as nk
from jax import numpy as jnp
import optax
from ...result.struct import Results

from .callbacks import RHatStop, VarianceCallback
from ..NN import Transformer, Config
from .operators import get_model_netket_op


class Model:
    dtype: jnp.dtype = jnp.complex64

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
        self.nn_cfg = Config(length=self.n, training=False,
                             dtype=self.dtype, symm=True)
        self.nn = Transformer(self.nn_cfg)

    def set_sampler(self, n_chains: int = 16):
        self.n_chains = n_chains
        self.sampler = nk.sampler.MetropolisSampler(
            self.hilbert,
            rule=nk.sampler.rules.LocalRule(),
            n_chains=n_chains,
            dtype=jnp.float64,
        )

    def set_optimizer(self, lr: float = 1e-4):
        # self.optimizer = nk.optimizer.Adam(lr)
        # self.optimizer = nk.optimizer.Sgd(0.001)
        self.optimizer = optax.adam(learning_rate=optax.join_schedules(
            [optax.linear_schedule(init_value=lr, end_value=lr * 10,
                                   transition_steps=10),
             optax.constant_schedule(lr * 10),
             optax.exponential_decay(init_value=lr * 10, transition_steps=35,
                                     decay_rate=0.8, end_value=lr/10)],
            boundaries=[10, 15]))
        # self.optimizer = optax.sgd(
        #    learning_rate=optax.exponential_decay(init_value=lr,
        #                                          transition_steps=10,
        #                                          decay_rate=0.8,
        #                                          transition_begin=0,
        #                                          end_value=1e-4),
        #    momentum=0.9,
        #    nesterov=True,
        # )

    def set_vmc(self, min_n_samples: int = 2000, scale_n_samples: int = 45):
        self.n_samples = max([min_n_samples, self.n * scale_n_samples])

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
            # preconditioner=sr,
        )

    def set_ops(self, ops: Dict[str, nk.operator.LocalOperator]):
        ops["e"] = self.ham
        self.ops = ops

    def train(self, n_iter: int = 50):
        logger = nk.logging.RuntimeLog()
        self.vmc.run(
            n_iter=n_iter,
            out=logger,
            # callback=[
            # RHatStop(base=1.3, stop_cnt=30),
            # VarianceCallback(base=5.0e-20),
            # ],
        )

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
        return res

    def save_model(self, path: Path):
        self.nn.save(str(path.absolute()))

    def apply_fun(self, variables, params, mutable=False):
        return self.nn.apply(variables, params,
                             rngs={'dropout': jax.random.PRNGKey(42)},
                             mutable=mutable)

    def init_fun(self, params, inp):
        rng_key = params['params']
        init_rng = {'params': rng_key,
                    'dropout': rng_key}

        params = self.nn.init(init_rng, inp)

        return params
