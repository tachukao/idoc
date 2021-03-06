"""Differentiable iLQR
"""

import jax
from jax import lax, vmap
import jax.numpy as jnp
import jaxopt
from jaxopt import implicit_diff, linear_solve
from typing import Callable, Any, Optional, NamedTuple
import flax
from . import lqr, typs
import functools
from dataclasses import dataclass
import logging

mm = jax.vmap(jnp.matmul)


def backward(
    p: lqr.LQR,
    horizon: int,
    *,
    return_expected_change: bool = False,
    unroll=False,
    jit=True,
) -> lqr.Gains:
    """iLQR backward pass

    Returns Gains used in the forward pass
    """
    A, B, d = p.A, p.B, p.d
    Q, q, Qf, qf = p.Q, p.q, p.Qf, p.qf
    R, r = p.R, p.r
    M = p.M
    AT = A.transpose(0, 2, 1)
    BT = B.transpose(0, 2, 1)
    EPS = 1e-8
    jitter = EPS * jnp.eye(R.shape[-1])
    symmetrize = lambda x: 0.5 * (x + x.T)

    _, n, m = B.shape
    Ks = jnp.zeros((horizon, m, n))
    ks = jnp.zeros((horizon, m))

    def loop(val):
        state, t, _ = val
        V, v, dC, dc, Ks, ks = state
        Gxx = symmetrize(Q[t] + AT[t] @ V @ A[t])
        Guu = symmetrize(R[t] + BT[t] @ V @ B[t])
        Gxu = M[t] + AT[t] @ V @ B[t]
        Vd = V @ d[t]
        gx = q[t] + AT[t] @ v + AT[t] @ Vd
        gu = r[t] + BT[t] @ v + BT[t] @ Vd
        evals, _ = jnp.linalg.eigh(Guu)
        min_eval = evals[0]
        #print(f"minimum eigenvalue: {min_eval: .05f}")
        Gtuu = Guu + jnp.maximum(0.0, EPS - min_eval) * jnp.eye(Guu.shape[0])
        K = -jax.scipy.linalg.solve(Gtuu, Gxu.T)
        k = -jax.scipy.linalg.solve(Gtuu, gu)
        V = symmetrize(Gxx + Gxu @ K + K.T @ Gxu.T + K.T @ Guu @ K)
        v = gx + Gxu @ k + K.T @ gu + K.T @ Guu @ k
        dC = dC + 0.5 * jnp.dot(jnp.dot(Guu, k), k)
        dc = dc + jnp.dot(gu, k)
        Ks = jax.ops.index_update(Ks, t, K)
        ks = jax.ops.index_update(ks, t, k)
        val = (V, v, dC, dc, Ks, ks), t - 1, True
        return val

    initial_val = (Qf, qf, 0.0, 0.0, Ks, ks), horizon - 1, True

    (_, _, dC, dc, Ks, ks), _, _ = jaxopt.loop.while_loop(
        lambda v: v[-1],
        loop,
        initial_val,
        maxiter=horizon,
        unroll=unroll,
        jit=jit,
    )

    gains = lqr.Gains(K=Ks, k=ks)
    if not return_expected_change:
        return gains

    def expected_change(alpha):
        """Expected change as predicted by quadratic model

        This should be a negative number.
        """
        return ((alpha ** 2) * dC) + (alpha * dc)

    return gains, expected_change


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
        c, x = state
        c = c + cs.cost(t, x, u, params.theta)
        nx = cs.dynamics(t, x, u, params.theta)
        return (c, nx), nx

    inps = jnp.arange(T), U
    (c, xf), X = lax.scan(fwd, (0.0, x0), inps)
    c = c + cs.costf(xf, params.theta)
    return X, c


def build(
    cs: Problem,
    *,
    maxiter: int = 100,
    thres: float = 1e-8,
    line_search=None,
    unroll: bool = False,
    jit: bool = True,
    verbose: bool=False,
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
        alpha: float = 1.0,
    ):
        x0, theta = params.x0, params.theta
        sX = jnp.concatenate((x0[None, ...], X[:-1]))

        def fwd(state, inp):
            xhat, l = state
            t, gain, x, u = inp
            dx = xhat - x
            du = gain.K @ dx + alpha * gain.k
            uhat = u + du
            nl = l + cs.cost(t, xhat, uhat, theta)
            nxhat = cs.dynamics(t, xhat, uhat, theta)
            return (nxhat, nl), (nxhat, uhat)

        inps = jnp.arange(T), gains, sX, U
        (xf, nl), (X, U) = lax.scan(fwd, (x0, 0.), inps)
        l = nl + cs.costf(xf, theta)
        return X, U, l

    def ilqr(
        init: typs.State,
        params: Params,
    ) -> typs.State:
        lqr_approx_global = make_lqr_approx(cs, params, local=False)
        lqr_approx_local = make_lqr_approx(cs, params, local=True)

        def loop(val):
            X_old, U_old, c_old, iteration, _ = val
            p = lqr_approx_local(X_old, U_old)
            gains, expected_change = lqr.backward(
                p, T, return_expected_change=True
            )

            def f(alpha):
                return update(X_old, U_old, gains, params, alpha=alpha)

            if line_search is None:
                (nX, nU, nc) = f(1.0)
                line_search_passes = True
            else:
                (nX, nU, nc, line_search_passes) = line_search(
                    f, (X_old, U_old, c_old), expected_change, unroll=unroll, jit=jit
                )

            pct_change = abs((c_old - nc) / c_old)
            if verbose:
                print(f"[{iteration}] nc {nc:.05f} pct_change {pct_change:.09f}")
            if not line_search_passes:
                logging.warning("Linear search did not pass!")
            carry_on = pct_change > thres
            new_val = nX, nU, nc, iteration + 1, carry_on
            return new_val

        U = init.U
        _, c_old = simulate(cs, U, params)
        init_val = (init.X, init.U, c_old, 0, True)
        X, U, c, _, _ = jaxopt.loop.while_loop(
            lambda v: v[-1],
            loop,
            init_val,
            maxiter=maxiter,
            unroll=unroll,
            jit=jit,
        )
        p = lqr_approx_global(X, U)
        Nu = lqr.adjoint(X, U, p, T)
        return typs.State(X=X, U=U, Nu=Nu)

    solve = functools.partial(linear_solve.solve_cg, tol=1e-8)

    implicit_ilqr = implicit_diff.custom_root(kkt, solve=solve)(ilqr)

    return typs.Solver(
        direct=ilqr,
        kkt=kkt,
        implicit=implicit_ilqr,
    )
