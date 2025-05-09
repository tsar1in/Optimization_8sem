import jax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict

from typing import Any, Callable, NamedTuple

class PAGEState(NamedTuple):
    params: FrozenDict
    batch_stats: FrozenDict = FrozenDict({})
    g: Any = None
    key: jax.random.PRNGKey = jax.random.PRNGKey(42)

class PAGE:
    def __init__(self,
                 loss_fn: Callable,
                 eval_loss_fn: Callable,
                 p: float,
                 lr: float,
                 bs: int,
                 bs_hat: int,
                 need_jit: bool = True):
        self.lr = lr
        self.p = p
        self.bs = bs
        self.bs_hat = bs_hat
        self.loss_fn = jax.jit(loss_fn) if need_jit else loss_fn
        self.eval_loss_fn = jax.jit(eval_loss_fn) if need_jit else eval_loss_fn
        self.update_fn = jax.jit(self._make_update_fn()) if need_jit else self._make_update_fn()
        self._compute_full_grad_fn = jax.jit(self._make_full_grad_fn()) if need_jit else self._make_full_grad_fn()
        self._compute_partial_grad_fn = jax.jit(self._make_partial_grad_fn()) if need_jit else self._make_partial_grad_fn()

    def init(self, variables: dict, init_batch: tuple[jnp.ndarray, jnp.ndarray]) -> PAGEState:
        state = PAGEState(params=FrozenDict(variables["params"]), batch_stats=FrozenDict(variables.get("batch_stats", {})))
        _, g, _ = self._compute_full_grad_fn(state, init_batch)
        return PAGEState(g=g, params=FrozenDict(variables["params"]), batch_stats=FrozenDict(variables.get("batch_stats", {})))
    
    def _make_full_grad_fn(self) -> Callable:
        def compute_full_grad(state: PAGEState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, jnp.ndarray, FrozenDict]:
            loss_fn = lambda p: self.loss_fn({"params": p, "batch_stats": state.batch_stats}, batch)
            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return loss, grads, FrozenDict(new_batch_stats)
        return compute_full_grad
    
    def _make_partial_grad_fn(self) -> Callable:
        def compute_partial_grad(new_state: PAGEState, old_state: PAGEState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, jnp.ndarray, FrozenDict]:
            loss_fn = lambda p: self.loss_fn({"params": p, "batch_stats": new_state.batch_stats}, batch)
            (loss, new_batch_stats), new_f_grad = jax.value_and_grad(loss_fn, has_aux=True)(new_state.params)

            loss_fn = lambda p: self.eval_loss_fn({"params": p, "batch_stats": old_state.batch_stats}, batch)
            old_f_grad, _ = jax.grad(loss_fn, has_aux=True)(old_state.params)
            delta_g = jax.tree_util.tree_map(lambda new, old: new - old, new_f_grad, old_f_grad)
            return loss, delta_g, FrozenDict(new_batch_stats)
        return compute_partial_grad
    
    def _make_update_fn(self) -> Callable:
        def update_fn(state: PAGEState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, PAGEState]:
            key, subkey = jax.random.split(state.key)
            new_params = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, state.params, state.g)
            new_state = PAGEState(params=FrozenDict(new_params),
                                  batch_stats=state.batch_stats)

            do_full_update = jax.random.bernoulli(subkey, p=self.p)

            def full_update(batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, FrozenDict]:
                return self._compute_full_grad_fn(new_state, batch)
            
            def partial_update(batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, jnp.ndarray, FrozenDict]:
                idxs = jax.random.choice(subkey,
                                         jnp.arange(batch[0].shape[0]), 
                                         shape=(self.bs_hat,),
                                         replace=False)
                sub_batch = (batch[0][idxs], batch[1][idxs])
                loss, delta_g, new_batch_stats = self._compute_partial_grad_fn(new_state, state, sub_batch)
                return loss, jax.tree_util.tree_map(lambda g, dg: g + dg, state.g, delta_g), FrozenDict(new_batch_stats)
            
            loss, new_g, new_batch_stats = jax.lax.cond(do_full_update,
                                                        full_update,
                                                        partial_update,
                                                        batch)

            return loss, PAGEState(g=new_g,
                                   key=key,
                                   params=FrozenDict(new_params),
                                   batch_stats=FrozenDict(new_batch_stats))
        return update_fn

    def update(self,
               state: PAGEState,
               batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, PAGEState]:
        return self.update_fn(state, batch)
