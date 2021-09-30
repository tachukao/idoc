"""Differentiable iLQR
"""

import jax
from jax import lax, vmap
import jax.numpy as jnp
from jaxopt import implicit_diff
from typing import Callable, Any, Optional
import flax
from . import lqr, typs
import functools
from dataclasses import dataclass


@dataclass
class Problem:
    cost: Callable[[int, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray]
    costf: Callable[[jnp.ndarray, Optional[Any]], jnp.ndarray]
    dynamics: Callable[[int, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray]
    horizon: int
    state_dim: int
    control_dim: int


class Params(flax.struct.PyTreeNode):
    """iLQR Params"""

    x0: jnp.ndarray
    theta: Any


def make_lqr_approx(cs: Problem, params: Params, flag=False) -> Callable:
    T = cs.horizon
    x0, theta = params.x0, params.theta
    mm = vmap(jnp.matmul)

    @jax.vmap
    def approx_timestep(t, x, u):
        M = jax.jacfwd(jax.grad(cs.cost, argnums=1), argnums=2)(t, x, u, theta)
        Q = jax.jacfwd(jax.grad(cs.cost, argnums=1), argnums=1)(t, x, u, theta)
        R = jax.jacfwd(jax.grad(cs.cost, argnums=2), argnums=2)(t, x, u, theta)
        q, r = jax.grad(cs.cost, argnums=(1, 2))(t, x, u, theta)
        if flag:
            q = q - (Q @ x + M @ u)
            r = r - (R @ u + M.T @ x)
        A, B = jax.jacobian(cs.dynamics, argnums=(1, 2))(t, x, u, theta)
        return Q, q, R, r, M, A, B

    def approx(X, U):
        sX = jnp.concatenate((x0[None, ...], X[:-1]))
        xf = X[-1]
        Q, q, R, r, M, A, B = approx_timestep(jnp.arange(T), sX, U)
        Qf = jax.jacfwd(jax.grad(cs.costf, argnums=0), argnums=0)(xf, theta)
        qf = jax.grad(cs.costf, argnums=0)(xf, theta)
        if flag:
            qf = qf - Qf @ xf
            d = X - (mm(A, sX) + mm(B, U))
        else:
            d = jnp.zeros((cs.horizon, cs.state_dim))
        return lqr.LQR(Q=Q, q=q, R=R, r=r, M=M, A=A, B=B, Qf=Qf, qf=qf, d=d)

    return approx


def simulate(cs: Problem, U: jnp.ndarray, params: Params) -> jnp.ndarray:
    x0 = params.x0
    T = U.shape[0]

    def fwd(state, inp):
        t, u = inp
        x = state
        nx = cs.dynamics(t, x, u, params.theta)
        return nx, nx

    inps = jnp.arange(T), U
    _, X = lax.scan(fwd, x0, inps)
    return X


def build(cs: Problem, iterations: int):
    T = cs.horizon

    def kkt(s: typs.State, params: Params) -> typs.State:
        p = make_lqr_approx(cs, params, True)(s.X, s.U)
        return lqr.kkt(s, lqr.Params(x0=params.x0, lqr=p))

    def update(
        X: jnp.ndarray,
        U: jnp.ndarray,
        gains: lqr.Gains,
        params: Params,
    ):
        x0, theta = params.x0, params.theta
        sX = jnp.concatenate((x0[None, ...], X[:-1]))

        def fwd(state, inp):
            xhat, l = state
            t, gain, x, u = inp
            dx = xhat - x
            du = gain.K @ dx + gain.k
            uhat = u + du
            nl = l + cs.cost(t, xhat, uhat, theta)
            nxhat = cs.dynamics(t, xhat, uhat, theta)
            return (nxhat, nl), (nxhat, uhat)

        inps = jnp.arange(T), gains, sX, U
        (xf, nl), (X, U) = lax.scan(fwd, (x0, 0), inps)
        l = nl + cs.costf(xf, theta)
        return X, U, l

    def ilqr(
        init: typs.State,
        params: Params,
    ) -> typs.State:
        lqr_approx = make_lqr_approx(cs, params, False)

        def loop(z, _):
            X, U = z
            p = lqr_approx(X, U)
            gains = lqr.backward(p, T)
            nX, nU, l = update(X, U, gains, params)
            return (nX, nU), l

        (X, U), L = lax.scan(loop, (init.X, init.U), jnp.arange(iterations))
        lqr_approx = make_lqr_approx(cs, params, True)
        p = lqr_approx(X, U)
        Nu = lqr.adjoint(X, U, p, T)
        return typs.State(X=X, U=U, Nu=Nu)

    implicit_ilqr = implicit_diff.custom_root(kkt)(ilqr)

    return typs.Solver(
        direct=ilqr,
        kkt=kkt,
        implicit=implicit_ilqr,
    )
