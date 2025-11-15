from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax


@dataclass
class VarianceCallback:
    base: float = 5e-3

    def __call__(self, step, log_data, driver):
        var = jnp.real(getattr(log_data[driver._loss_name], "variance"))
        return var >= self.base or step < 500


@dataclass
class SigmaCallback:
    base: float = 1.0e-5

    def __call__(self, step, log_data, driver):
        var = jnp.real(getattr(log_data[driver._loss_name], "error_of_mean"))
        return var >= self.base


@dataclass
class ChainReset:
    base: float = 1.2

    def __call__(self, step, log_data, driver):
        var = jnp.real(getattr(log_data[driver._loss_name], "R_hat"))

        if var > self.base:
            driver._variational_state.sampler = driver._variational_state.sampler

        return True


@dataclass
class RHatStop:
    base: float = 1.3
    _cnt: int = 0
    stop_cnt: int = 10

    def __call__(self, step, log_data, driver):
        var = jnp.real(getattr(log_data[driver._loss_name], "R_hat"))

        if var > self.base:
            self._cnt += 1
            if self._cnt >= self.stop_cnt:
                return False

        else:
            self._cnt = 0

        return True


@dataclass
class ParametersPrint:
    print_step: int = 200
    clear_cache_step: int = 2000

    def __call__(self, step, log_data, driver):
        if step % self.clear_cache_step == 0:
            jax.clear_caches()

        if step % self.print_step == 0:
            params = driver.state.parameters
            print(f"\n=== Step {step} ===")

            for module_name, module_params in params.items():
                print(f"Module: {module_name}")
                for param_name, param_value in module_params.items():
                    if hasattr(param_value, "shape"):
                        print(
                            f"  {param_name}: shape={param_value.shape}, "
                            f"mean={jnp.mean(param_value):.6f}, "
                            f"std={jnp.std(param_value):.6f}"
                        )
                    else:
                        print(f"  {param_name}: {param_value}")
            print("=================\n")
        return True


class AdaptiveMomentumCallback:
    def __init__(self, lr, n_iter, patience=35, min_improvement=1e-2):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = float("inf")
        self.wait = 0
        self.momentum_enabled = True

        self.lr = lr
        self.n_iter = n_iter

    def __call__(self, step, log_data, driver):
        current_loss = log_data["Energy"].mean.real

        if current_loss < self.best_loss - self.min_improvement:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience and self.momentum_enabled:
            print(f"Step {step}: Loss not improving, disabling momentum")
            self._disable_momentum(driver)
            self.momentum_enabled = False
            return True

        return True

    def _disable_momentum(self, driver):
        new_optimizer = optax.sgd(
            learning_rate=optax.exponential_decay(
                init_value=self.lr * 10,
                transition_steps=(self.n_iter - self.n_iter // 4) // (4 / 3),
                decay_rate=0.9,
                transition_begin=self.n_iter // 4,
                end_value=self.lr,
            ),
            nesterov=False,
        )
        new_optimizer = optax.sgd(learning_rate=self.lr)

        driver.optimizer.optimizer = new_optimizer
        driver.state.optimizer_state = new_optimizer.init(driver.state.parameters)
