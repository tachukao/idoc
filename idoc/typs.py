import jax.numpy as jnp
import flax


class State(flax.struct.PyTreeNode):
    X: jnp.ndarray
    U: jnp.ndarray
    Nu: jnp.ndarray
