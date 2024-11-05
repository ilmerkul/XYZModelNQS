from pathlib import Path
from typing import Dict

import netket as nk
import numpy as np
import optax
from src.struct import Results

from .callbacks import RHatStop, VarianceCallback
from .machine import FF


class Model:
    def __init__(self, n: int, pbc: bool = False, type: str = "xx"):
        self.n = n
        self.type = type
        self.graph = nk.graph.Hypercube(length=n, n_dim=1, pbc=pbc)
        self.hilbert = nk.hilbert.Spin(N=n, s=0.5)

    def set_meta_params(self, h: float, j: float):
        self.h = h
        self.j = j

    def set_ham(self, op: nk.operator.LocalOperator):
        self.ham = op

    def set_machine(self):
        self.nn = FF(n=self.n, alpha=5, dtype=np.float32)

    def set_sampler(self, n_chains: int = 16):
        self.n_chains = n_chains
        self.sampler = nk.sampler.MetropolisSampler(
            self.hilbert,
            rule=nk.sampler.rules.LocalRule(),
            n_chains=n_chains,
            dtype=np.float32,
        )

    def set_optimizer(self, lr: float = 3e-2):
        self.optimizer = optax.sgd(
            learning_rate=optax.exponential_decay(lr, 10, 0.97, 1000, end_value=1e-2),
            momentum=0.9,
            nesterov=True,
        )

    def set_vmc(self):
        n_samples = max([2000, self.n * 45])
        self.mcstate = nk.vqs.MCState(
            sampler=self.sampler,
            model=self.nn,
            n_samples=n_samples,
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
        logger = nk.logging.RuntimeLog()
        self.vmc.run(
            n_iter=n_iter,
            out=logger,
            callback=[
                RHatStop(base=1.3, stop_cnt=30),
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
        res = Results(self.h, self.j, self.n, self.type)
        res.analytical()

        return res

    def get_results(self) -> Results:
        res = Results(self.h, self.j, self.n, self.type)
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
