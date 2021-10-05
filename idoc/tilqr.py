"""Differentiable LQR (finite horizon, discrete time, time-invariant) 
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxopt import implicit_diff
from . import typs
from typing import NamedTuple


class TILQR(NamedTuple):
    """time-invariant LQR specs"""

    Q: jnp.ndarray
    Qf: jnp.ndarray
    R: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray

    def symm(self):
        Q = 0.5 * (self.Q + self.Q.T)
        Qf = 0.5 * (self.Qf + self.Qf.T)
        R = 0.5 * (self.R + self.R.T)
        return TILQR(Q=Q, Qf=Qf, R=R, A=self.A, B=self.B)


class Params(NamedTuple):
    """time-invariant LQR parameters"""

    x0: jnp.ndarray
    lqr: TILQR


def build(horizon: int) -> typs.Solver:
    """Build a time-invariant LQR differentiable solver"""

    def kkt(s: typs.State, theta: Params) -> typs.State:
        x0, lqr = theta.x0, theta.lqr.symm()
        X, U, Nu = s.X, s.U, s.Nu
        A, B, Q, Qf, R = lqr.A, lqr.B, lqr.Q, lqr.Qf, lqr.R
        AT = A.T
        BT = B.T

        dLdX = jnp.concatenate(
            (
                (X[:-1] @ Q.T) + (Nu[1:] @ A) - Nu[:-1],
                X[-1][None, ...] @ Qf.T - Nu[-1][None, ...],
            ),
            axis=0,
        )
        dLdU = Nu @ B + U @ R.T
        dLdNu = jnp.concatenate((x0[None, ...], X[:-1]), axis=0) @ AT + U @ BT - X
        return typs.State(X=dLdX, U=dLdU, Nu=dLdNu)

    def direct(_, theta: Params):
        x0, lqr = theta.x0, theta.lqr.symm()
        A, B, Q, R, Qf = lqr.A, lqr.B, lqr.Q, lqr.R, lqr.Qf
        AT = A.T
        BT = B.T

        def forward(x0: jnp.ndarray, Ks: jnp.ndarray):
            def dyn(x, K):
                u = K @ x
                nx = A @ x + B @ u
                return nx, (nx, u)

            _, (X, U) = lax.scan(dyn, x0, Ks)
            return X, U

        def gain(P):
            return -jax.scipy.linalg.solve(R + BT @ P @ B, BT @ P.T @ A, sym_pos=True)

        def backward():
            def bwd(P, _):
                K = gain(P)
                P = Q + AT @ P @ A + AT @ P @ B @ K
                return P, K

            _, Ks = lax.scan(bwd, Qf, jnp.arange(horizon))
            return jnp.flip(Ks, axis=0)

        Ks = backward()
        X, U = forward(x0, Ks)

        def adjoint(X):
            Xflip = jnp.flip(X, axis=0)

            def adj(_nu, x):
                nu = AT @ _nu + Q @ x
                return nu, _nu

            nuf = Qf @ Xflip[0]
            nu0, _Nu = lax.scan(adj, nuf, Xflip[1:])
            Nu = jnp.concatenate((_Nu, nu0[None, ...]), axis=0)
            return jnp.flip(Nu, axis=0)

        Nu = adjoint(X)
        return typs.State(X, U, Nu)

    implicit = implicit_diff.custom_root(kkt)(direct)
    return typs.Solver(
        direct=lambda x: direct(None, x), kkt=kkt, implicit=lambda x: implicit(None, x)
    )
