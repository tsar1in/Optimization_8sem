import jax
import jax.numpy as jnp

from typing import Any, Callable, NamedTuple

class PAGEState(NamedTuple):
    g: Any
    key: jax.random.PRNGKey

class PAGE:
    def __init__(self,
                 loss_fn: Callable,
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
        self.update_fn = jax.jit(self._make_update_fn()) if need_jit else self._make_update_fn()
        self._compute_full_grad_fn = jax.jit(self._make_full_grad_fn()) if need_jit else self._make_full_grad_fn()
        self._compute_partial_grad_fn = jax.jit(self._make_partial_grad_fn()) if need_jit else self._make_partial_grad_fn()

    def init(self, params: dict, init_batch: tuple[jnp.ndarray, jnp.ndarray]) -> PAGEState:
        g = self._compute_full_grad_fn(params, init_batch)
        return PAGEState(g=g, key=jax.random.PRNGKey(0))
    
    def _make_full_grad_fn(self) -> Callable:
        def compute_full_grad(params: dict, batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            return jax.grad(self.loss_fn)(params, batch)
        return compute_full_grad
    
    def _make_partial_grad_fn(self) -> Callable:
        def compute_partial_grad(new_params: dict, params: dict, batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            new_f_grad = jax.grad(self.loss_fn)(new_params, batch)
            old_f_grad = jax.grad(self.loss_fn)(params, batch)
            return jax.tree_util.tree_map(lambda new, cur: new - cur, new_f_grad, old_f_grad)
        return compute_partial_grad
    
    def _make_update_fn(self) -> Callable:
        def update_fn(params: dict, batch: tuple[jnp.ndarray, jnp.ndarray], state: PAGEState) -> tuple[dict, PAGEState]:
            key, subkey = jax.random.split(state.key)
            new_params = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, params, state.g)

            do_full_update = jax.random.bernoulli(subkey, p=self.p)

            def full_update(batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
                return self._compute_full_grad_fn(new_params, batch)
            
            def partial_update(batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
                idxs = jax.random.choice(subkey,
                                         jnp.arange(batch[0].shape[0]), 
                                         shape=(self.bs_hat,),
                                         replace=False)
                sub_batch = (batch[0][idxs], batch[1][idxs])
                delta_g = self._compute_partial_grad_fn(new_params, params, sub_batch)
                return jax.tree_util.tree_map(lambda g, dg: g + dg, state.g, delta_g)
            
            new_g = jax.lax.cond(do_full_update,
                                 full_update,
                                 partial_update,
                                 batch)

            return new_params, PAGEState(g=new_g, key=key)
        return update_fn

    def update(self,
               params: dict,
               batch: tuple[jnp.ndarray, jnp.ndarray],
               state: PAGEState) -> tuple[float, tuple[dict, PAGEState]]:
        return self.update_fn(params, batch, state)
