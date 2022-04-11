"""Differentiable iLQR
"""

import jax
from jax import lax, vmap
import jax.numpy as jnp
from jaxopt import implicit_diff, linear_solve
from typing import Callable, Any, Optional, NamedTuple
import flax
from . import lqr, typs
import functools
from dataclasses import dataclass

mm = jax.vmap(jnp.matmul)


@dataclass
class Problem:
    """iLQR Problem

    cost : Callable
        running cost l(t, x, u)
    costf : Callable
        final state cost lf(xf)
    dynamics : Callable
        dynamical update f(t, x, u, theta)
    horizon : int
        horizon of the problem
    state_dim : int
        dimensionality of the state
    control_dim : int
        dimensionality of the control inputs
    """

    cost: Callable[[int, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray]
    costf: Callable[[jnp.ndarray, Optional[Any]], jnp.ndarray]
    dynamics: Callable[[int, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray]
    horizon: int
    state_dim: int
    control_dim: int


class Params(NamedTuple):
    """iLQR Params"""

    x0: jnp.ndarray
    theta: Any


def make_lqr_approx(cs: Problem, params: Params, local=True) -> Callable:
    """Create LQR approximation function"""
    T = cs.horizon
    x0, theta = params.x0, params.theta

    @jax.vmap
    def approx_timestep(t, x, u):
        M = jax.jacfwd(jax.grad(cs.cost, argnums=1), argnums=2)(t, x, u, theta)
        Q = jax.jacfwd(jax.grad(cs.cost, argnums=1), argnums=1)(t, x, u, theta)
        R = jax.jacfwd(jax.grad(cs.cost, argnums=2), argnums=2)(t, x, u, theta)
        q, r = jax.grad(cs.cost, argnums=(1, 2))(t, x, u, theta)
        A, B = jax.jacobian(cs.dynamics, argnums=(1, 2))(t, x, u, theta)
        if not local:
            q = q - Q @ x - M @ u
            r = r - R @ u - M.T @ x
            d = cs.dynamics(t, x, u, theta) - A @ x - B @ u
        else:
            d = jnp.zeros(cs.state_dim)

        return Q, q, R, r, M, A, B, d

    def approx(X, U) -> lqr.LQR:
        sX = jnp.concatenate((x0[None, ...], X[:-1]))
        xf = X[-1]
        Q, q, R, r, M, A, B, d = approx_timestep(jnp.arange(T), sX, U)
        Qf = jax.jacfwd(jax.grad(cs.costf, argnums=0), argnums=0)(xf, theta)
        qf = jax.grad(cs.costf, argnums=0)(xf, theta)
        if not local:
            qf = qf - Qf @ xf
        return lqr.LQR(Q=Q, q=q, R=R, r=r, M=M, A=A, B=B, Qf=Qf, qf=qf, d=d)

    return approx


def simulate(cs: Problem, U: jnp.ndarray, params: Params) -> jnp.ndarray:
    """Simulates state trajectory"""
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


def make_line_search(*, lower=1e-3, upper=2.0, rho=0.5, maxiter=8):
    def backtrack(f, c_old, expected_change, unroll, jit):
        def check(p):
            carry_on = jnp.logical_not(jnp.logical_and(p >= lower, p <= upper))
            return carry_on

        def update(alpha):
            X, U, c_new = f(alpha)
            p = (c_new - c_old) / expected_change(alpha)
            carry_on = check(p)
            return X, U, c_new, carry_on

        def body_fun(val):
            _, _, _, alpha, it, _ = val
            alpha *= rho
            X, U, c_new, carry_on = update(alpha)
            return X, U, c_new, alpha, it + 1, carry_on

        X, U, c, carry_on = update(1.0)
        X, U, c_new, _, _, _ = jaxopt.loop.while_loop(
            lambda v: v[-1],
            body_fun,
            (X, U, c, 1.0, 0, carry_on),
            maxiter=maxiter,
            jit=jit,
            unroll=unroll,
        )
        return X, U, c_new

    return backtrack


def build(
    cs: Problem, thres: float = 1e-8, iterations: int = 100, linesearch=None
) -> typs.Solver:
    """Build iLQR solver"""
    T = cs.horizon

    def kkt(s: typs.State, params: Params) -> typs.State:
        p = make_lqr_approx(cs, params, local=False)(s.X, s.U)
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
        lqr_approx_global = make_lqr_approx(cs, params, local=False)
        lqr_approx_local = make_lqr_approx(cs, params, local=True)

        def loop(z, _):
            X, U = z
            p = lqr_approx_local(X, U)
            gains = lqr.backward(p, T)
            nX, nU, l = update(X, U, gains, params)
            return (nX, nU), l

        (X, U), L = lax.scan(loop, (init.X, init.U), jnp.arange(iterations))
        print(L)
        p = lqr_approx_global(X, U)
        Nu = lqr.adjoint(X, U, p, T)
        return typs.State(X=X, U=U, Nu=Nu)

    implicit_ilqr = implicit_diff.custom_root(kkt, solve=linear_solve.solve_cg)(ilqr)

    return typs.Solver(
        direct=ilqr,
        kkt=kkt,
        implicit=implicit_ilqr,
    )
