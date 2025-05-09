import jax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict

from typing import Callable, NamedTuple

class SGDState(NamedTuple):
    params: FrozenDict
    batch_stats: FrozenDict

class SGD:
    def __init__(self,
                 loss_fn: Callable,
                 lr: float,
                 need_jit: bool = True):
        self.lr = lr
        self.loss_fn = jax.jit(loss_fn) if need_jit else loss_fn
        self.update_fn = jax.jit(self._make_update_fn()) if need_jit else self._make_update_fn()

        self.computed_grad_count = 0

    def init(self, variables: dict) -> SGDState:
        return SGDState(params=FrozenDict(variables["params"]), batch_stats=FrozenDict(variables.get("batch_stats", {})))
    
    def _make_update_fn(self) -> Callable:
        def update_fn(state: SGDState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, SGDState]:
            loss_fn = lambda p: self.loss_fn({"params": p, "batch_stats": state.batch_stats}, batch)
            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            new_params = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, state.params, grads)
            return loss, SGDState(params=FrozenDict(new_params), batch_stats=FrozenDict(new_batch_stats))
        return update_fn

    def update(self,
               state: SGDState,
               batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, SGDState]:
        self.computed_grad_count += batch[0].shape[0]
        return self.update_fn(state, batch)
