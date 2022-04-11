"""Differentiable quadratic programming
"""
import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from . import typs
from typing import NamedTuple


class QP(NamedTuple):
    """QP specs"""

    Q: jnp.ndarray
    c: jnp.ndarray
    E: jnp.ndarray
    d: jnp.ndarray


def loss(z, theta: QP):
    """QP loss"""
    return 0.5 * jnp.dot(jnp.dot(theta.Q, z), z) + jnp.dot(theta.c, z)


dloss = jax.grad(loss)


def feasibility(z, theta):
    """QP constraint"""
    E, d = theta.E, theta.d
    return E @ z - d


def kkt(x, theta):
    """QP KKT conditions"""
    z, nu = x
    _, feasibility_vjp = jax.vjp(feasibility, z, theta)
    stationarity = dloss(z, theta) + feasibility_vjp(nu)[0]
    primal_feasability = feasibility(z, theta)
    return stationarity, primal_feasability


def _direct_solver(_, theta):
    Q, c, E, d = theta.Q, theta.c, theta.E, theta.d
    Q = 0.5 * (Q + Q.T)

    A1 = jnp.concatenate((Q, E.T), axis=1)
    A2 = jnp.concatenate((E, jnp.zeros((E.shape[0], E.shape[0]))), axis=1)
    A = jnp.concatenate((A1, A2), axis=0)
    y = jnp.concatenate((-c, d), axis=0)
    x = jax.scipy.linalg.solve(A, y)
    dim = Q.shape[0]
    z = x[0:dim]
    nu = x[dim:]
    return z, nu


_implicit_solver = implicit_diff.custom_root(kkt)(_direct_solver)

direct_solver = lambda theta: _direct_solver(None, theta)
implicit_solver = lambda theta: _implicit_solver(None, theta)

solver = typs.Solver(direct=direct_solver, implicit=implicit_solver, kkt=kkt)
