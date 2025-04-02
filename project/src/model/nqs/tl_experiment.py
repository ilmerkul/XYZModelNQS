from pathlib import Path
from typing import Dict, Optional

import jax
import netket as nk
import numpy as np
import optax
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from jax.random import PRNGKey

from ...result.struct import Results
from .callbacks import VarianceCallback
from .transfer_learning import NN


class Model:
    def __init__(self, n: int, pbc: bool = False):
        self.n = n
        self.graph = nk.graph.Hypercube(length=n, n_dim=1, pbc=pbc)
        self.hilbert = nk.hilbert.Spin(N=n, s=0.5)

    def set_meta_params(self, h: float, j: float):
        self.h = h
        self.j = j

    def set_ham(self, op: nk.operator.LocalOperator):
        self.ham = op

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

    def set_sampler(self, n_chains: int = 16):
        self.n_chains = n_chains
        self.sampler = nk.sampler.MetropolisSampler(
            self.hilbert,
            rule=nk.sampler.rules.LocalRule(),
            n_chains=n_chains,
            dtype=np.float32,
        )

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

    def set_ops(self, ops: Dict[str, nk.operator.LocalOperator]):
        self.ops = ops
        ops["e"] = self.ham

    def train(self, n_iter: int = 4000):
        self.logger = nk.logging.RuntimeLog()
        self.vmc.run(
            n_iter=n_iter,
            out=self.logger,
            callback=[
                VarianceCallback(base=5.0e-7),
            ],
        )

    def get_hilbert(self):
        return self.hilbert

    def get_j(self):
        return self.j

    def get_h(self):
        return self.h

    def get_analytical(self) -> Results:
        res = Results(self.h, self.j, self.n)
        res.analytical()

        return res

    def get_results(self, is_xx: bool = True) -> Results:
        res = Results(self.h, self.j, self.n, type="xx" if is_xx else "xy")
        res.analytical()
        state = self.mcstate.sampler_state.σ.to_py()
        self.mcstate.sampler = nk.sampler.MetropolisSamplerNumpy(
            self.hilbert,
            rule=nk.sampler.rules.HamiltonianRuleNumpy(operator=self.ham),
            n_chains=self.n_chains,
            dtype=np.float32,
        )
        self.mcstate.sampler_state.σ = np.array(state)
        n_samples = max([6000, self.n * 200])
        chain_length = int(n_samples / self.n_chains)
        discard_by_chain = int(chain_length * 0.3)
        self.mcstate.sample(
            chain_length=chain_length, n_discard_per_chain=discard_by_chain
        )
        ops_vals = self.vmc.estimate(self.ops)

        res.update(
            energy=np.real(ops_vals["e"].mean),
            var=np.real(ops_vals["e"].variance),
            spins=np.array(
                [
                    np.real(val.mean)
                    for op, val in ops_vals.items()
                    if op.startswith("s_")
                ]
            ),
            xx=np.real(ops_vals["xx"].mean),
            yy=np.real(ops_vals["yy"].mean),
            zz=np.real(ops_vals["zz"].mean),
            zz_mid=np.real(ops_vals["zz_mid"].mean),
        )

        return res

    def save_model(self, path: Path):
        self.nn.save(str(path.absolute()))

    def print_base_mean(self):
        return jnp.mean(self.mcstate.parameters["base"]["kernel"])

    def print_head_mean(self):
        return jnp.mean(self.mcstate.parameters["head"]["kernel"])

    def save_base_weights(self, prefix: Path):
        kernel = self.mcstate.parameters["base"]["kernel"]
        bias = self.mcstate.parameters["base"]["bias"]

        jnp.save(file=prefix.joinpath("kernel.npy"), arr=kernel)
        jnp.save(file=prefix.joinpath("bias.npy"), arr=bias)
