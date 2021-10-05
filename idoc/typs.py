import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, NamedTuple


class State(NamedTuple):
    X: jnp.ndarray
    U: jnp.ndarray
    Nu: jnp.ndarray


@dataclass
class Solver:
    direct: Callable
    implicit: Callable
    kkt: Callable
