"""Differentiable batch LQR (finite horizon, discrete time, time-invariant) 
"""

import flax
import jax
import jax.numpy as jnp
import jax.scipy as sp
from jax import lax
from absl import app
from jaxopt import implicit_diff, linear_solve
from typing import Callable, NamedTuple
from . import typs

mm = jax.vmap(jnp.matmul)


class Gains(NamedTuple):
    """LQR gains"""

    K: jnp.ndarray
    k: jnp.ndarray


class BLQR(NamedTuple):
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
        Q = 0.5 * (self.Q + self.Q.transpose(0, 1, 3, 2))
        Qf = 0.5 * (self.Qf + self.Qf.transpose(0, 2, 1))
        R = 0.5 * (self.R + self.R.transpose(0, 2, 1))
        return BLQR(
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


class Params(NamedTuple):
    """LQR parameters"""

    x0: jnp.ndarray
    blqr: BLQR


def batch_lqr_step(V, v, dC, dc, Q, q, R, r, M, A, B, d, delta=1e-8):
    """Single Batch LQR Step.
    Args:
    V: [batch_size, n, batch_size, n] numpy array.
    v: [batch_size, n] numpy array.
    dC: float
    dc: float
    Q: [batch_size, n, n] numpy array.
    q: [batch_size, n] numpy array.
    R: [m, m] numpy array.
    r: [m] numpy array.
    M: [batch_size, m, n] numpy array.
    A: [batch_size, n, n] numpy array.
    B: [batch_size, n, m] numpy array.
    d: [batch_size, n] numpy array.
    delta: Enforces positive definiteness by ensuring smallest eigenval > delta.
    Returns:
    V, v: updated matrices encoding quadratic value function.
    dC, dc: values used in expected_change computation
    K, k: state feedback gain and affine term.
    """
    batch_size, n, m = B.shape
    symmetrize = lambda x: (x + x.T) / 2
    symmetrize_full = lambda x: (x + x.transpose(2, 3, 0, 1)) / 2

    AtV = jnp.einsum("...ji,...jkl", A, V)
    AtVA = symmetrize_full(jnp.einsum("ai...j,...jk->ai...k", AtV, A))
    BtV = jnp.einsum("ijk,ijlm", B, V)  # (m, batch_size, n)
    BtVA = jnp.einsum("i...k,...km->...im", BtV, A)
    BtVB = jnp.einsum("ijk,jkl", BtV, B)
    G = symmetrize(R + jnp.einsum("ijk,jkl", BtV, B))
    # make G positive definite so that smallest eigenvalue > delta.
    S, _ = jnp.linalg.eigh(G)
    G_ = G + jnp.maximum(0.0, delta - S[0]) * jnp.eye(G.shape[0])

    H = BtVA + M.transpose(0, 2, 1)  # (batch_size, m, n)
    h = jnp.einsum("ijk,ij", B, v) + jnp.einsum("ijk,jk", BtV, d) + r

    vlinsolve = jax.vmap(
        lambda x, y: sp.linalg.solve(x, y, sym_pos=True), in_axes=(None, 0)
    )
    K = -vlinsolve(G_, H)  # (batch_size, m, n)
    k = -sp.linalg.solve(G_, h, sym_pos=True)  # (m, )

    H_GK = H + jnp.einsum("ij,ajk->aik", G, K)  # (batch_size, m, n)
    V = symmetrize_full(
        sp.linalg.block_diag(*Q).reshape((batch_size, n, batch_size, n))
        + AtVA
        + jnp.einsum("ijk,mjo->ikmo", H_GK, K)
        + jnp.einsum("ijk,mjo->ikmo", K, H)
    )

    v = (
        q
        + jax.vmap(jnp.matmul)(A.transpose(0, 2, 1), v)
        + jnp.einsum("ijkl,kl", AtV, d)
        + jnp.matmul(H_GK.transpose(0, 2, 1), k)
        + jnp.matmul(K.transpose(0, 2, 1), h)
    )

    dC = dC + 0.5 * jnp.dot(jnp.dot(G, k), k)
    dc = dc + jnp.dot(h, k)
    return (V, v, dC, dc), (K, k)


def backward(lqr: BLQR, horizon: int, *, return_expected_change: bool = False) -> Gains:
    """LQR backward pass

    Returns Gains used in the forward pass
    """
    A, B, d = lqr.A, lqr.B, lqr.d
    Q, q, Qf, qf = lqr.Q, lqr.q, lqr.Qf, lqr.qf
    R, r = lqr.R, lqr.r
    M = lqr.M

    def bwd(state, inps):
        t = inps
        V, v, dC, dc = state
        return batch_lqr_step(
            V, v, dC, dc, Q[t], q[t], R[t], r[t], M[t], A[t], B[t], d[t], delta=1e-8
        )

    _, batch_size, n, _ = B.shape
    Qf = sp.linalg.block_diag(*Qf).reshape((batch_size, n, batch_size, n))
    (_, _, dC, dc), (Ks, ks) = lax.scan(
        bwd, (Qf, qf, 0.0, 0.0), jnp.flip(jnp.arange(horizon))
    )
    gains = Gains(K=jnp.flip(Ks, axis=0), k=jnp.flip(ks, axis=0))
    if not return_expected_change:
        return gains

    def expected_change(alpha):
        return ((alpha ** 2) * dC) + (alpha * dc)

    return gains, expected_change


def adjoint(X, U, lqr: BLQR, horizon: int) -> jnp.ndarray:
    """Computes LQR adjoints"""
    A = lqr.A
    Q, q, Qf, qf = lqr.Q, lqr.q, lqr.Qf, lqr.qf
    M = lqr.M

    # we have A of size TxBxnxn
    # want a final nu of the same size, or might have to reshape but will deal with that after
    def adj(_nu, _t):
        t = horizon - _t
        nu = (
            jnp.einsum("ijk,ij-> ik", A[t], _nu)
            + jnp.einsum("ijk,ik -> ij", Q[t], X[t - 1])
            + q[t]
            + jnp.einsum("ijk,k->ij", M[t], U[t])
        )
        # nu = AT[t] @ _nu + Q[t] @ X[t - 1] + q[t] + M[t] @ U[t]
        return nu, _nu

    nuf = jax.vmap(jnp.matmul, in_axes=(0, 0))(Qf, X[-1]) + qf
    nu0, _Nu = lax.scan(adj, nuf, jnp.arange(1, horizon))
    Nu = jnp.concatenate((_Nu, nu0[None, ...]), axis=0)
    return jnp.flip(Nu, axis=0)


def kkt(s: typs.State, params: Params) -> typs.State:
    """LQR KKT conditions"""
    x0, lqr = params.x0, params.blqr.symm()
    X, U, Nu = s.X, s.U, s.Nu
    A, B, d = lqr.A, lqr.B, lqr.d
    Q, q, Qf, qf = lqr.Q, lqr.q, lqr.Qf, lqr.qf
    R, r = lqr.R, lqr.r
    M = lqr.M
    BT = B.transpose(0, 1, 3, 2)
    MT = M.transpose(0, 1, 3, 2)
    # passing those as TxBxnxn (or TxBxmxm)
    dLdX = jnp.concatenate(
        (
            q[1:]
            + jnp.einsum("ijkl,il->ijk", M[1:], U[1:])
            + jnp.einsum("ijkl,ijl->ijk", Q[1:], X[:-1])
            + jnp.einsum("ijlk,ijl->ijk", A[1:], Nu[1:])
            - Nu[:-1],
            (qf + jnp.einsum("ijk,ik->ij", Qf, X[-1]) - Nu[-1])[None, ...],
        ),
        axis=0,
    )
    sX = jnp.concatenate((x0[None, ...], X[:-1]), axis=0)
    batch_size = (M.shape)[1]
    # print(
    #     jnp.einsum("ijkl,ijl->ijk", BT, Nu).shape,
    #     jnp.einsum("ijkl,ijl->ijk", MT, sX).shape,
    #     jnp.einsum("ijk,ik->ij", R, U).shape,
    #     r.shape,
    # )
    # dLdU = mm(BT, Nu) + mm(R, U) + r + mm(MT, sX)
    dLdU = (
        jnp.einsum("ijkl,ijl->ijk", BT, Nu).sum(1)
        + jnp.einsum("ijkl,ijl->ijk", MT, sX).sum(1)
        + jnp.einsum("ijk,ik->ij", R, U)
        + r
    )
    # dLdNu = d + mm(A, sX) + mm(B, U) - X
    dLdNu = (
        d + jnp.einsum("ijkl,ijl->ijk", A, sX) + jnp.einsum("ijkl,il->ijk", B, U) - X
    )
    return typs.State(X=dLdX, U=dLdU, Nu=dLdNu)


def forward(lqr: BLQR, x0: jnp.ndarray, gains: Gains):
    """Simulates forward dynamics"""
    A, B, d = lqr.A, lqr.B, lqr.d
    T, batch_size, m, n = gains.K.shape
    U = jnp.zeros((T, m))

    def dyn(x, inps):
        A, B, d, gain = inps
        u = jnp.einsum("ijk,ik", gain.K, x) + gain.k
        nx = jax.vmap(jnp.matmul)(A, x) + jnp.matmul(B, u) + d
        return nx, (nx, u)

    _, (X, U) = lax.scan(dyn, x0, (A, B, d, gains))
    return X, U


def build(horizon: int) -> typs.Solver:
    """Build a LQR differentiable solver"""

    def direct(_, params: Params):
        x0, blqr = params.x0, params.blqr
        gains = backward(blqr, horizon)
        X, U = forward(blqr, x0, gains)
        Nu = adjoint(X, U, blqr, horizon)
        shapex = X.shape
        X = X.reshape((shapex[0], -1, shapex[-1]))
        Nu = Nu.reshape((shapex[0], -1, shapex[-1]))
        return typs.State(X, U, Nu)

    implicit = implicit_diff.custom_root(kkt, solve=linear_solve.solve_cg)(direct)
    return typs.Solver(
        direct=lambda x: direct(None, x), kkt=kkt, implicit=lambda x: implicit(None, x)
    )
