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


def make_lqr_approx(cs: Problem, params: Params) -> Callable:
    T = cs.horizon
    x0, theta = params.x0, params.theta
    mm = vmap(jnp.matmul)

    @jax.vmap
    def approx_timestep(t, x, u):
        M = jax.jacfwd(jax.grad(cs.cost, argnums=2), argnums=1)(t, x, u, theta).T
        Q = jax.jacfwd(jax.grad(cs.cost, argnums=1), argnums=1)(t, x, u, theta)
        R = jax.jacfwd(jax.grad(cs.cost, argnums=2), argnums=2)(t, x, u, theta)
        q, r = jax.grad(cs.cost, argnums=(1, 2))(t, x, u, theta)
        q = q - Q @ x - M @ u
        r = r - R @ u - M.T @ x
        A, B = jax.jacobian(cs.dynamics, argnums=(1, 2))(t, x, u, theta)
        return Q, q, R, r, M, A, B

    def approx(X, U):
        sX = jnp.concatenate((x0[None, ...], X[:-1]))
        xf = X[-1]
        Q, q, R, r, M, A, B = approx_timestep(jnp.arange(T), sX, U)
        Qf = jax.jacfwd(jax.grad(cs.costf, argnums=0), argnums=0)(xf, theta)
        qf = jax.grad(cs.costf, argnums=0)(xf, theta)
        qf = qf - Qf @ xf
        d = X - (mm(A, sX) + mm(B, U))
        return lqr.LQR(Q=Q, q=q, R=R, r=r, M=M, A=A, B=B, Qf=Qf, qf=qf, d=d)

    return approx


def build(cs: Problem, iterations: int):
    T = cs.horizon

    def kkt(s: typs.State, params: Params) -> typs.State:
        p = make_lqr_approx(cs, params)(s.X, s.U)
        return lqr.kkt(s, lqr.Params(x0=params.x0, lqr=p))

    def update(
        gains: lqr.Gains,
        params: Params,
    ):
        x0, theta = params.x0, params.theta

        def fwd(state, inp):
            xhat, l = state
            t, gain = inp
            uhat = gain.K @ xhat + gain.k
            nl = l + cs.cost(t, xhat, uhat, theta)
            nxhat = cs.dynamics(t, xhat, uhat, theta)
            return (nxhat, nl), (nxhat, uhat)

        inps = jnp.arange(T), gains
        (xf, nl), (X, U) = lax.scan(fwd, (x0, 0), inps)
        l = nl + cs.costf(xf, theta)
        return X, U, l

    def ilqr(
        init: typs.State,
        params: Params,
    ) -> typs.State:
        x0 = params.x0
        assert x0.ndim == 1 and x0.shape[0] == cs.state_dim
        assert init.U.ndim > 0 and init.U.shape[0] == cs.horizon
        lqr_approx = make_lqr_approx(cs, params)

        def loop(z, _):
            X, U = z
            p_lqr = lqr_approx(X, U)
            gains = lqr.backward(p_lqr, T)
            X, U, l = update(gains, params)
            return (X, U), l

        (X, U), L = lax.scan(loop, (init.X, init.U), jnp.arange(iterations))
        print(L)
        lqr_approx = make_lqr_approx(cs, params)
        Nu = lqr.adjoint(X, U, lqr_approx(X, U), T)
        return typs.State(X=X, U=U, Nu=Nu)

    def simulate(U: jnp.ndarray, params: Params) -> jnp.ndarray:
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

    implicit_ilqr = implicit_diff.custom_root(kkt)(ilqr)

    def solve(ilqr, U, params):
        assert U.shape == (T, cs.control_dim)
        X = simulate(U, params)
        Nu = jnp.zeros_like(X)
        return ilqr(typs.State(X=X, U=U, Nu=Nu), params)

    return (
        functools.partial(solve, ilqr),
        kkt,
        functools.partial(solve, implicit_ilqr),
        simulate,
    )


# def mpc_predict(
#    solver,
#    cs: ILQR,
#    x0: jnp.ndarray,
#    U: jnp.ndarray,
#    params: Any,
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#    assert x0.ndim == 1 and x0.shape[0] == cs.state_dim
#    T = cs.horizon
#
#    def zero_padded_controls_window(U, t):
#        U_pad = jnp.vstack((U, jnp.zeros(U.shape)))
#        return lax.dynamic_slice_in_dim(U_pad, t, T, axis=0)
#
#    def loop(t, state):
#        cost = lambda t_, x, u, params: cs.cost(t + t_, x, u, params)
#        dyns = lambda t_, x, u, params: cs.dynamics(t + t_, x, u, params)
#
#        X, U = state
#        p_ = ILQR(
#            cost=cost,
#            costf=cs.costf,
#            dynamics=dyns,
#            horizon=T,
#            state_dim=cs.state_dim,
#            control_dim=cs.control_dim,
#        )
#        xt = X[t]
#        U_rem = zero_padded_controls_window(U, t)
#        _, U_, _ = solver(p_, xt, U_rem, params)
#        ut = U_[0]
#        x = cs.dynamics(t, xt, ut, params)
#        X = jo.index_update(X, jo.index[t + 1], x)
#        U = jo.index_update(U, jo.index[t], ut)
#        return X, U
#
#    X = jnp.zeros((T + 1, cs.state_dim))
#    X = jo.index_update(X, jo.index[0], x0)
#    return fori_loop(0, T, loop, (X, U))
#
