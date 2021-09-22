"""
iLQR implementation adapted from 

    https://github.com/google/jax/blob/main/examples/control.py

Original license:

Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import jax
from jax import lax
import jax.numpy as jnp
import jax.ops as jo
from flax import struct
from typing import Callable, Tuple, Any, Optional
from dataclasses import dataclass

# Specifies a general finite-horizon, time-varying control problem. Given cost
# function `c`, transition function `f`, and initial state `x0`, the goal is to
# compute:
#
#   argmin(lambda X, U: c(T, X[T]) + sum(c(t, X[t], U[t]) for t in range(T)))
#
# subject to the constraints that `X[0] == x0` and that:
#
#   all(X[t + 1] == f(X[t], U[t]) for t in range(T)) .
#
# The special case in which `c` is quadratic and `f` is linear is the
# linear-quadratic regulator (LQR) problem, and can be specified explicitly
# further below.
#
@dataclass
class ControlSpec:
    cost: Callable[[int, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray]
    dynamics: Callable[[int, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray]
    horizon: int
    state_dim: int
    control_dim: int


# Specifies a finite-horizon, time-varying LQR problem. Notation:
#
#   cost(t, x, u) = sum(
#       dot(x.T, Q[t], x) + dot(q[t], x) +
#       dot(u.T, R[t], u) + dot(r[t], u) +
#       dot(x.T, M[t], u)
#
#   dynamics(t, x, u) = dot(A[t], x) + dot(B[t], u)
#
class LQRSpec(struct.PyTreeNode):
    Q: jnp.ndarray
    q: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    M: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray


def mv(mat, vec):
    """Matrix-vector product helper function"""
    assert mat.ndim == 2
    assert vec.ndim == 1
    return jnp.dot(mat, vec)


def fori_loop(low, high, loop, init):
    """For loop helper function"""

    def scan_f(x, t):
        return loop(t, x), ()

    x, _ = lax.scan(scan_f, init, jnp.arange(low, high))
    return x


def trajectory(
    dynamics: Callable, U: jnp.ndarray, x0: jnp.ndarray, params: Any
) -> jnp.ndarray:
    """Unrolls `X[t+1] = dynamics(t, X[t], U[t])`, where `X[0] = x0`."""
    T, _ = U.shape
    (d,) = x0.shape

    X = jnp.zeros((T + 1, d))
    X = jo.index_update(X, jo.index[0], x0)

    def loop(t, X):
        x = dynamics(t, X[t], U[t], params)
        X = jo.index_update(X, jo.index[t + 1], x)
        return X

    return fori_loop(0, T, loop, X)


def make_lqr_approx(
    p: ControlSpec, params: Any
) -> Callable[[jnp.ndarray, jnp.ndarray], LQRSpec]:
    T = p.horizon

    def approx_timestep(t, x, u):
        M = jax.jacfwd(jax.grad(p.cost, argnums=2), argnums=1)(t, x, u, params).T
        Q = jax.jacfwd(jax.grad(p.cost, argnums=1), argnums=1)(t, x, u, params)
        R = jax.jacfwd(jax.grad(p.cost, argnums=2), argnums=2)(t, x, u, params)
        q, r = jax.grad(p.cost, argnums=(1, 2))(t, x, u, params)
        A, B = jax.jacobian(p.dynamics, argnums=(1, 2))(t, x, u, params)
        return Q, q, R, r, M, A, B

    _approx = jax.vmap(approx_timestep)

    def approx(X, U):
        assert X.shape[0] == T + 1 and U.shape[0] == T
        U_pad = jnp.vstack((U, jnp.zeros((1,) + U.shape[1:])))
        Q, q, R, r, M, A, B = _approx(jnp.arange(T + 1), X, U_pad)
        return LQRSpec(Q=Q, q=q, R=R[:T], r=r[:T], M=M[:T], A=A[:T], B=B[:T])

    return approx


def lqr_solve(spec: LQRSpec):
    EPS = 1e-7
    T, control_dim, _ = spec.R.shape
    _, state_dim, _ = spec.Q.shape

    K = jnp.zeros((T, control_dim, state_dim))
    k = jnp.zeros((T, control_dim))

    def rev_loop(t_, state):
        t = T - t_ - 1
        spec, P, p, K, k = state

        Q, q = spec.Q[t], spec.q[t]
        R, r = spec.R[t], spec.r[t]
        M = spec.M[t]
        A, B = spec.A[t], spec.B[t]

        AtP = jnp.matmul(A.T, P)
        BtP = jnp.matmul(B.T, P)
        G = R + jnp.matmul(BtP, B)
        H = jnp.matmul(BtP, A) + M.T
        h = r + mv(B.T, p)
        K_ = -jnp.linalg.solve(G + EPS * jnp.eye(G.shape[0]), H)
        k_ = -jnp.linalg.solve(G + EPS * jnp.eye(G.shape[0]), h)
        P_ = Q + jnp.matmul(AtP, A) + jnp.matmul(K_.T, H)
        p_ = q + mv(A.T, p) + mv(K_.T, h)

        K = jo.index_update(K, jo.index[t], K_)
        k = jo.index_update(k, jo.index[t], k_)
        return spec, P_, p_, K, k

    _, P, p, K, k = fori_loop(
        0, T, rev_loop, (spec, spec.Q[T + 1], spec.q[T + 1], K, k)
    )

    return K, k


def lqr_predict(spec: LQRSpec, x0: jnp.ndarray):
    T, control_dim, _ = spec.R.shape
    _, state_dim, _ = spec.Q.shape

    K, k = lqr_solve(spec)

    def fwd_loop(t, state):
        spec, X, U = state
        A, B = spec.A[t], spec.B[t]
        u = mv(K[t], X[t]) + k[t]
        x = mv(A, X[t]) + mv(B, u)
        X = jo.index_update(X, jo.index[t + 1], x)
        U = jo.index_update(U, jo.index[t], u)
        return spec, X, U

    U = jnp.zeros((T, control_dim))
    X = jnp.zeros((T + 1, state_dim))
    X = jo.index_update(X, jo.index[0], x0)
    _, X, U = fori_loop(0, T, fwd_loop, (spec, X, U))
    return X, U


def ilqr(
    iterations: int,
    p: ControlSpec,
    x0: jnp.ndarray,
    U: jnp.ndarray,
    params: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert x0.ndim == 1 and x0.shape[0] == p.state_dim
    assert U.ndim > 0 and U.shape[0] == p.horizon

    lqr_approx = make_lqr_approx(p, params)

    def loop(_, state):
        X, U = state
        p_lqr = lqr_approx(X, U)
        dX, dU = lqr_predict(p_lqr, jnp.zeros_like(x0))
        U = U + dU
        X = trajectory(p.dynamics, U, X[0] + dX[0], params)
        return X, U

    X = trajectory(p.dynamics, U, x0, params)
    return fori_loop(0, iterations, loop, (X, U))


def mpc_predict(
    solver,
    p: ControlSpec,
    x0: jnp.ndarray,
    U: jnp.ndarray,
    params: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert x0.ndim == 1 and x0.shape[0] == p.state_dim
    T = p.horizon

    def zero_padded_controls_window(U, t):
        U_pad = jnp.vstack((U, jnp.zeros(U.shape)))
        return lax.dynamic_slice_in_dim(U_pad, t, T, axis=0)

    def loop(t, state):
        cost = lambda t_, x, u, params: p.cost(t + t_, x, u, params)
        dyns = lambda t_, x, u, params: p.dynamics(t + t_, x, u, params)

        X, U = state
        p_ = ControlSpec(
            cost=cost,
            dynamics=dyns,
            horizon=T,
            state_dim=p.state_dim,
            control_dim=p.control_dim,
        )
        xt = X[t]
        U_rem = zero_padded_controls_window(U, t)
        _, U_ = solver(p_, xt, U_rem, params)
        ut = U_[0]
        x = p.dynamics(t, xt, ut, params)
        X = jo.index_update(X, jo.index[t + 1], x)
        U = jo.index_update(U, jo.index[t], ut)
        return X, U

    X = jnp.zeros((T + 1, p.state_dim))
    X = jo.index_update(X, jo.index[0], x0)
    return fori_loop(0, T, loop, (X, U))
