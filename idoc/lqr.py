"""Differentiable LQR (finite horizon, discrete time, time-invariant) 
"""

import flax
import jax
import jax.numpy as jnp
from jax import lax
from absl import app
from jaxopt import implicit_diff
from typing import Callable
from . import typs


mm = jax.vmap(jnp.matmul)


class Gains(flax.struct.PyTreeNode):
    """LQR gains"""

    K: jnp.ndarray
    k: jnp.ndarray


class LQR(flax.struct.PyTreeNode):
    """LQR specs"""

    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.ndarray
    qf: jnp.ndarray
    M: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray
    d: jnp.ndarray

    def symm(self):
        Q = 0.5 * (self.Q + self.Q.transpose(0, 2, 1))
        Qf = 0.5 * (self.Qf + self.Qf.T)
        R = 0.5 * (self.R + self.R.transpose(0, 2, 1))
        return LQR(
            Q=Q,
            Qf=Qf,
            R=R,
            A=self.A,
            B=self.B,
            M=self.M,
            q=self.q,
            qf=self.qf,
            r=self.r,
            d=self.d,
        )


class Params(flax.struct.PyTreeNode):
    """LQR parameters"""

    x0: jnp.ndarray
    lqr: LQR


def backward(lqr: LQR, horizon: int) -> Gains:
    """LQR backward pass

    Returns Gains used in the forward pass
    """
    A, B, d = lqr.A, lqr.B, lqr.d
    Q, q, Qf, qf = lqr.Q, lqr.q, lqr.Qf, lqr.qf
    R, r = lqr.R, lqr.r
    M = lqr.M
    AT = A.transpose(0, 2, 1)
    BT = B.transpose(0, 2, 1)

    def bwd(state, inps):
        EPS = 1e-12
        jitter = EPS * jnp.eye(R.shape[-1])
        t = inps
        V, v = state
        Gxx = Q[t] + AT[t] @ V @ A[t]
        Guu = R[t] + BT[t] @ V @ B[t]
        Gxu = M[t] + AT[t] @ V @ B[t]
        gx = q[t] + AT[t] @ v + AT[t] @ V @ d[t]
        gu = r[t] + BT[t] @ v + BT[t] @ V @ d[t]
        Gtuu = Guu + jitter
        K = -jax.scipy.linalg.solve(Gtuu, Gxu.T)
        k = -jax.scipy.linalg.solve(Gtuu, gu)
        V = Gxx + Gxu @ K + K.T @ Gxu.T + K.T @ Guu @ K
        v = gx + Gxu @ k + K.T @ gu + K.T @ Guu @ k
        return (V, v), (K, k)

    _, (Ks, ks) = lax.scan(bwd, (Qf, qf), jnp.flip(jnp.arange(horizon)))
    return Gains(K=jnp.flip(Ks, axis=0), k=jnp.flip(ks, axis=0))


def adjoint(X, U, lqr: LQR, horizon: int) -> jnp.ndarray:
    """Computes LQR adjoints 
    """
    A = lqr.A
    Q, q, Qf, qf = lqr.Q, lqr.q, lqr.Qf, lqr.qf
    M = lqr.M
    AT = A.transpose(0, 2, 1)

    def adj(_nu, _t):
        t = horizon - _t
        nu = AT[t] @ _nu + Q[t] @ X[t - 1] + q[t] + M[t] @ U[t]
        return nu, _nu

    nuf = Qf @ X[-1] + qf
    nu0, _Nu = lax.scan(adj, nuf, jnp.arange(1, horizon))
    Nu = jnp.concatenate((_Nu, nu0[None, ...]), axis=0)
    return jnp.flip(Nu, axis=0)


def kkt(s: typs.State, params: Params) -> typs.State:
    """LQR KKT conditions"""
    x0, lqr = params.x0, params.lqr.symm()
    X, U, Nu = s.X, s.U, s.Nu
    A, B, d = lqr.A, lqr.B, lqr.d
    Q, q, Qf, qf = lqr.Q, lqr.q, lqr.Qf, lqr.qf
    R, r = lqr.R, lqr.r
    M = lqr.M
    AT = A.transpose(0, 2, 1)
    BT = B.transpose(0, 2, 1)
    MT = M.transpose(0, 2, 1)

    dLdX = jnp.concatenate(
        (
            q[1:] + mm(M[1:], U[1:]) + mm(Q[1:], X[:-1]) + mm(AT[1:], Nu[1:]) - Nu[:-1],
            (qf + Qf @ X[-1] - Nu[-1])[None, ...],
        ),
        axis=0,
    )
    sX = jnp.concatenate((x0[None, ...], X[:-1]), axis=0)
    dLdU = mm(BT, Nu) + mm(R, U) + r + mm(MT, sX)
    dLdNu = d + mm(A, sX) + mm(B, U) - X
    return typs.State(X=dLdX, U=dLdU, Nu=dLdNu)


def forward(lqr: LQR, x0: jnp.ndarray, gains: Gains):
    """Simulates forward dynamics"""
    A, B, d = lqr.A, lqr.B, lqr.d

    def dyn(x, inps):
        A, B, d, gain = inps
        u = gain.K @ x + gain.k
        nx = A @ x + B @ u + d
        return nx, (nx, u)

    _, (X, U) = lax.scan(dyn, x0, (A, B, d, gains))
    return X, U


def build(horizon: int) -> typs.Solver:
    """Build a LQR differentiable solver"""

    def direct(_, params: Params):
        x0, lqr = params.x0, params.lqr
        gains = backward(lqr, horizon)
        X, U = forward(lqr, x0, gains)
        Nu = adjoint(X, U, lqr, horizon)
        return typs.State(X, U, Nu)

    implicit = implicit_diff.custom_root(kkt)(direct)
    return typs.Solver(
        direct=lambda x: direct(None, x), kkt=kkt, implicit=lambda x: implicit(None, x)
    )
