"""Differentiable iLQR
"""

import jax
from jax import lax, vmap
import jax.numpy as jnp
from jaxopt import implicit_diff, linear_solve
from typing import Callable, Any, Optional, NamedTuple
import flax
from . import lqr, typs, batch_lqr
import functools
from dataclasses import dataclass

mm = jax.vmap(jnp.matmul)
def vmap_grad(f,argnums=0):
    return jax.vmap(jax.grad(f,argnums=argnums))

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
        #here : add the vmaps : x is of size BxN
        M = jax.vmap(jax.jacfwd(jax.grad(cs.cost, argnums=2), argnums=1),in_axes=(None,0,None,None))(t, x, u, theta)
        R = jnp.sum(jax.vmap(jax.jacfwd(jax.grad(cs.cost, argnums=2), argnums=2), in_axes=(None,0,None,None))(t, x, u, theta),axis=0)
        q = jax.vmap(jax.grad(cs.cost, argnums=1), in_axes = (None,0,None,None))(t, x, u, theta)
        Q = jax.vmap(jax.jacfwd(jax.grad(cs.cost, argnums=1), argnums=1),in_axes=(None,0,None,None))(t, x, u, theta)
        r = jax.vmap(jax.grad(cs.cost, argnums=2), in_axes = (None,0,None,None))(t, x, u, theta)
        A = jax.vmap(jax.jacobian(cs.dynamics, argnums=(1)),in_axes = (None,0,None,None))(t, x, u, theta)
        B = jax.vmap(jax.jacobian(cs.dynamics, argnums=2),in_axes = (None,0,None,None))(t, x, u, theta)
        if not local:
            q = q - jnp.einsum("ijk,ik->ij", Q, x) - jnp.einsum("ijk,j", M, u) #shapes will be wrong, need to print to be sure
            #want something of size TxBxn : might need to specify axes of None for some things
            r = r - R @ u - jnp.einsum("ijk,ik -> j", M, x)
            #so actually we need r of size Bxm
            d = jax.vmap(cs.dynamics, in_axes = (None,0,None,None))(t, x, u, theta) - jnp.einsum("ijk,ik->ij", A, x) - jnp.einsum("ijk,k->ij", B, u)
        else:
            d = jnp.zeros(x.shape)

        return Q, q, R, r, M, A, B, d

    def approx(X, U) -> batch_lqr.BLQR:
        sX = jnp.concatenate((x0[None, ...], X[:-1]))
        xf = X[-1]
        Q, q, R, r, M, A, B, d = approx_timestep(jnp.arange(T), sX, U)
        Qf = jax.vmap(jax.jacfwd(jax.grad(cs.costf, argnums=0), argnums=0), in_axes=(0,None))(xf, theta)
        qf = jax.vmap(jax.grad(cs.costf, argnums=0), in_axes=(0,None))(xf, theta)
        if not local:
            qf = qf - jax.vmap(jnp.matmul,in_axes = (0,0))(Qf,xf)
        return batch_lqr.BLQR(Q=Q, q=q, R=R, r=r, M=M, A=A, B=B, Qf=Qf, qf=qf, d=d)

    return approx


def simulate(cs: Problem, U: jnp.ndarray, params: Params) -> jnp.ndarray:
    """Simulates state trajectory"""
    x0 = params.x0
    T = U.shape[0]
    @functools.partial(jax.vmap,in_axes=(None,0,None))
    def batch_dyn(t,x,u):
        return cs.dynamics(t, x, u, params.theta)
    def fwd(state, inp):
        t, u = inp
        x = state
        nx = batch_dyn(t, x, u)
        return nx, nx

    inps = jnp.arange(T), U
    _, X = lax.scan(fwd, x0, inps)
    return X


def build(cs: Problem, iterations: int) -> typs.Solver:
    """Build iLQR solver"""
    T = cs.horizon

    def kkt(s: typs.State, params: Params) -> typs.State:
        p = make_lqr_approx(cs, params, local=False)(s.X, s.U)
        return batch_lqr.kkt(s, batch_lqr.Params(x0=params.x0, blqr=p))

    def update(
        X: jnp.ndarray,
        U: jnp.ndarray,
        gains: batch_lqr.Gains,
        params: Params,
    ):
        x0, theta = params.x0, params.theta
        sX = jnp.concatenate((x0[None, ...], X[:-1]))

        def fwd(state, inp):
            xhat, l = state
            t, gain, x, u = inp
            dx = xhat - x
            du =jnp.sum((mm(gain.K, dx)),axis=0) + gain.k #check as well
            uhat = u + du
            nl = l + jnp.sum(jax.vmap(cs.cost, in_axes=(None,0,None,None))(t, xhat, uhat, theta), axis=0)
            nxhat = jax.vmap(cs.dynamics, in_axes = (None,0,None,None))(t, xhat, uhat, theta)
            return (nxhat, nl), (nxhat, uhat)

        inps = jnp.arange(T), gains, sX, U
        (xf, nl), (X, U) = lax.scan(fwd, (x0, 0), inps)
        l = nl + jnp.sum(jax.vmap(cs.costf, in_axes=(0,None))(xf, theta), axis=0)
        return X, U, l

    def ilqr(
        init: typs.State,
        params: Params,
    ) -> typs.State:
        lqr_approx_global = make_lqr_approx(cs, params, local=False)
        lqr_approx_local= make_lqr_approx(cs, params, local=True)

        def loop(z, _):
            X, U = z
            p = lqr_approx_local(X, U)
            gains = batch_lqr.backward(p, T)
            nX, nU, l = update(X, U, gains, params)
            return (nX, nU), l

        (X, U), L = lax.scan(loop, (init.X, init.U), jnp.arange(iterations))
        print(L)
        p = lqr_approx_global(X, U)
        Nu = batch_lqr.adjoint(X, U, p, T)
        return typs.State(X=X, U=U, Nu=Nu)

    implicit_ilqr = implicit_diff.custom_root(kkt, solve=linear_solve.solve_cg)(ilqr)

    return typs.Solver(
        direct=ilqr,
        kkt=kkt,
        implicit=implicit_ilqr,
    )
