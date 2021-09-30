import jax.numpy as jnp
import flax
from dataclasses import dataclass
from typing import Callable


class State(flax.struct.PyTreeNode):
    X: jnp.ndarray
    U: jnp.ndarray
    Nu: jnp.ndarray


@dataclass
class Solver:
    direct: Callable
    implicit: Callable
    kkt: Callable
