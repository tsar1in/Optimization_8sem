import jax
import jax.numpy as jnp

from typing import Callable, NamedTuple

class SGDState(NamedTuple):
    pass

class SGD:
    def __init__(self,
                 loss_fn: Callable,
                 lr: float,
                 need_jit: bool = True):
        self.lr = lr
        self.loss_fn = jax.jit(loss_fn) if need_jit else loss_fn
        self.update_fn = jax.jit(self._make_update_fn()) if need_jit else self._make_update_fn()

    def init(self, params: dict) -> SGDState:
        return SGDState()
    
    def _make_update_fn(self) -> Callable:
        def update_fn(params: dict,
                   batch: tuple[jnp.ndarray, jnp.ndarray],
                   state: SGDState) -> tuple[float, tuple[dict, SGDState]]:
            grads = jax.grad(self.loss_fn)(params, batch)
            new_params = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, params, grads)
            return new_params, state
        return update_fn

    def update(self,
               params: dict,
               batch: tuple[jnp.ndarray, jnp.ndarray],
               state: SGDState) -> tuple[float, tuple[dict, SGDState]]:
        return self.update_fn(params, batch, state)
