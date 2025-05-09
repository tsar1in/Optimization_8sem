import jax
import jax.numpy as jnp

from flax import linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential([nn.Dense(512),
                              nn.relu,
                              nn.Dense(512),
                              nn.relu,
                              nn.Dense(512),
                              nn.relu,
                              nn.Dense(1)])(x)
