import jax
import jax.numpy as jnp

def tf_to_jax(batch) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Конвертирует TF-батч в JAX-совместимый формат."""
    images, labels = batch[0]._numpy(), batch[1]._numpy()
    return jax.device_put(images), jax.device_put(labels)