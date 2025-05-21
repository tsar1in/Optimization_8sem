import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from typing import Callable, NamedTuple, Optional
import numpy as np


class SVRGState(NamedTuple):
    params: FrozenDict 
    batch_stats: FrozenDict
    snapshot_params: FrozenDict  
    mu: Optional[FrozenDict]
    inner_step: int
    rng_key: jax.random.PRNGKey


class SVRG:
    def __init__(
        self,
        loss_fn: Callable,
        lr: float,
        inner_iters: int = 1000,
        need_jit: bool = True
    ):
        self.lr = lr
        self.inner_iters = inner_iters
        self.loss_fn = jax.jit(loss_fn) if need_jit else loss_fn
        self.computed_grad_count = 0

        self._update_fn = jax.jit(self._make_update_fn()) if need_jit else self._make_update_fn()
        self._single_grad_fn = jax.jit(self._make_single_grad_fn()) if need_jit else self._make_single_grad_fn()

    def init(self, variables: dict, rng_key: jax.random.PRNGKey) -> SVRGState:
        snapshot_params = FrozenDict(variables["params"])
        return SVRGState(
            params=FrozenDict(variables["params"]),
            batch_stats=FrozenDict(variables.get("batch_stats", {})),
            snapshot_params=snapshot_params,
            mu=None,
            inner_step=0,
            rng_key=rng_key
        )

    def full_grad(self, params: FrozenDict, batch_stats: FrozenDict, full_ds) -> FrozenDict:
        def loss_wrap(p, x, y):
            return self.loss_fn({"params": p, "batch_stats": batch_stats}, (x, y))[0]
        
        grads = []
        for x, y in zip(full_ds[0], full_ds[1]):
            g = jax.grad(loss_wrap)(params, x, y)
            grads.append(g)
        
        mean_grad = jax.tree_util.tree_map(lambda *arrays: jnp.mean(jnp.stack(arrays), axis=0), *grads)
        return mean_grad

    def set_snapshot(self, state: SVRGState, full_ds) -> SVRGState:
        mu = self.full_grad(state.params, state.batch_stats, full_ds)
        return state._replace(
            snapshot_params=state.params,
            mu=mu,
            inner_step=0
        )

    def _make_single_grad_fn(self) -> Callable:
        def single_grad_fn(params, batch_stats, snapshot_params, x, y):
            def loss(p):
                return self.loss_fn({"params": p, "batch_stats": batch_stats}, (x, y))[0]
            
            grad = jax.grad(loss)(params)
            snapshot_grad = jax.grad(loss)(snapshot_params)
            return grad, snapshot_grad
        
        return single_grad_fn

    def _make_update_fn(self) -> Callable:
        def update_fn(state: SVRGState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, SVRGState]:
            # Выбираем случайный пример из батча
            rng_key, subkey = jax.random.split(state.rng_key)
            idx = jax.random.randint(subkey, (1,), 0, batch[0].shape[0])[0]
            x, y = batch[0][idx], batch[1][idx]

            # Вычисляем градиенты для одного примера
            grad, snapshot_grad = self._single_grad_fn(
                state.params, state.batch_stats, state.snapshot_params, x, y
            )

            # SVRG-поправка
            svrg_grad = jax.tree_util.tree_map(lambda g, sg, mu: g - sg + mu, grad, snapshot_grad, state.mu)
            new_params = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, state.params, svrg_grad)

            # Обновляем состояние
            loss = self.loss_fn({"params": new_params, "batch_stats": state.batch_stats}, (x, y))[0]
            new_state = state._replace(
                params=FrozenDict(new_params),
                inner_step=state.inner_step + 1,
                rng_key=rng_key
            )
            return loss, new_state
        
        return update_fn

    def update(self, state: SVRGState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[float, SVRGState]:
        self.computed_grad_count += 2
        return self._update_fn(state, batch)