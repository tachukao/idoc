import jax
import jax.numpy as jnp
from typing import Callable, Any
from jax import flatten_util
import scipy
from . import typs


def init_stable(key, state_dim):
    """Initialize a stable matrix with dimensions `state_dim`."""
    R = jax.random.normal(key, (state_dim, state_dim))
    A, _ = jnp.linalg.qr(R)
    return 0.5 * A


def check_kkt(kkt: Callable, s: typs.State, theta: Any, thres=1e-5) -> None:
    """Check KKT outputs."""
    kkt_state = kkt(s, theta)
    x = jnp.mean(jnp.abs(kkt_state.X))
    u = jnp.mean(jnp.abs(kkt_state.U))
    nu = jnp.mean(jnp.abs(kkt_state.Nu))
    print(f"dLdX: {x}")
    print(f"dLdU: {u}")
    print(f"dLdNu: {nu}")
    assert x < thres
    assert u < thres
    assert nu < thres


def relative_difference(z1, z2):
    az1 = jnp.abs(z1)
    az2 = jnp.abs(z2)
    err = jnp.mean(jnp.abs(z1 - z2) / (az1 + az2 + 1e-9))
    return err


def print_and_check(err, thres=1e-4):

    print(err)
    assert err < thres


def finite_difference_grad(f, x, eps=1e-8):
    flat_x, unravel = flatten_util.ravel_pytree(x)

    def flat_f(flat_theta):
        theta = unravel(flat_theta)
        return f(theta)

    d = jnp.ones_like(flat_x) * eps
    flat_g = scipy.optimize.approx_fprime(flat_x, flat_f, d)
    g = unravel(flat_g)
    return g
