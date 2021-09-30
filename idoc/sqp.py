"""Differentiable sequential quadratic programming
"""
import jax
import jax.numpy as jnp
from jax import lax
from . import qp
from jaxopt import implicit_diff
from typing import Callable, Any
from dataclasses import dataclass
from . import typs


@dataclass
class SQP:
    f: Callable[[jnp.ndarray, Any], jnp.ndarray]
    g: Callable[[jnp.ndarray, Any], jnp.ndarray]

    def approximate(self, z, theta) -> qp.QP:
        Q = jax.jacfwd(jax.grad(self.f, argnums=0), argnums=0)(z, theta)
        Q = 0.5 * (Q + Q.T)
        c = jax.grad(self.f, argnums=0)(z, theta) - Q @ z
        E = jax.jacobian(self.g, argnums=0)(z, theta)
        d = self.g(z, theta) - E @ z
        return qp.QP(Q, c, E, d)


def build(sqp: SQP, iterations: int) -> typs.Solver:
    def f(z, p: qp.QP):
        return 0.5 * jnp.dot(jnp.dot(p.Q, z), z) + jnp.dot(p.c, z)

    df = jax.grad(f)

    def H(z, p: qp.QP):
        E, d = p.E, p.d
        return jnp.dot(E, z) - d

    def kkt(x, theta):
        z, nu = x
        p = sqp.approximate(z, theta)
        _, H_vjp = jax.vjp(H, z, p)
        stationarity = df(z, p) + H_vjp(nu)[0]
        primal_feasability = H(z, p)
        return stationarity, primal_feasability

    def step(p):
        Q, c, E, d = p.Q, p.c, p.E, p.d

        A1 = jnp.concatenate((Q, E.T), axis=1)
        A2 = jnp.concatenate((E, jnp.zeros((E.shape[0], E.shape[0]))), axis=1)
        A = jnp.concatenate((A1, A2), axis=0)
        y = jnp.concatenate((-c, d), axis=0)
        x = jnp.linalg.solve(A, y)
        dim = Q.shape[0]
        z = x[0:dim]
        nu = x[dim:]
        return z, nu

    def solver(x, theta):
        def loop(x, _):
            z, _ = x
            p = sqp.approximate(z, theta)
            return step(p), None

        x, _ = lax.scan(loop, x, jnp.arange(iterations))
        return x

    return typs.Solver(direct=solver, kkt=kkt, implicit=implicit_diff.custom_root(kkt)(solver))
