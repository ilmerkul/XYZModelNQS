from dataclasses import dataclass

import numpy as np


@dataclass
class VarianceCallback:
    base: float = 1.0e-6

    def __call__(self, step, log_data, driver):
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        return var >= self.base


@dataclass
class SigmaCallback:
    base: float = 1.0e-5

    def __call__(self, step, log_data, driver):
        var = np.real(getattr(log_data[driver._loss_name], "error_of_mean"))
        return var >= self.base


@dataclass
class ChainReset:
    base: float = 1.2

    def __call__(self, step, log_data, driver):
        var = np.real(getattr(log_data[driver._loss_name], "R_hat"))

        if var > self.base:
            driver._variational_state.sampler = driver._variational_state.sampler

        return True


@dataclass
class RHatStop:
    base: float = 1.3
    _cnt: int = 0
    stop_cnt: int = 10

    def __call__(self, step, log_data, driver):
        var = np.real(getattr(log_data[driver._loss_name], "R_hat"))

        if var > self.base:
            self._cnt += 1
            if self._cnt >= self.stop_cnt:
                return False

        else:
            self._cnt = 0

        return True
