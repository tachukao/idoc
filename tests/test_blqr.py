"""Test for LQR solver"""

import idoc
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.test_util import check_grads


def init_blqr(
    key, state_dim: int, control_dim: int, horizon: int, batch_size: int
) -> idoc.blqr.BLQR:
    """Initialize a random LQR spec."""
    Q = jnp.tile(jnp.eye(state_dim), (horizon, batch_size, 1, 1))
    q = 0.2 * jnp.tile(jnp.ones(state_dim), (horizon, batch_size, 1))
    Qf = jnp.tile(jnp.eye(state_dim), (batch_size, 1, 1))
    qf = 0.2 * jnp.ones((batch_size, state_dim))
    R = 1e-4 * jnp.tile(jnp.eye(control_dim), (horizon, 1, 1))
    r = 1e-4 * jnp.ones((horizon, control_dim))
    M = 1e-4 * jnp.ones((horizon, batch_size, state_dim, control_dim))
    key, subkey = jr.split(key)
    A = jnp.tile(idoc.utils.init_stable(subkey, state_dim), (horizon, batch_size, 1, 1))
    key, subkey = jr.split(key)
    B = jnp.tile(
        jr.normal(subkey, (state_dim, control_dim)), (horizon, batch_size, 1, 1)
    )
    d = jnp.tile(jnp.ones(state_dim), (horizon, batch_size, 1))
    return idoc.blqr.BLQR(Q=Q, q=q, Qf=Qf, qf=qf, R=R, r=r, A=A, B=B, d=d, M=M)


def init_params(key, state_dim, control_dim, horizon, batch_size) -> idoc.blqr.Params:
    """Initialize random parameters."""
    key, subkey = jr.split(key)
    x0 = jr.normal(subkey, (batch_size, state_dim))
    key, subkey = jr.split(key)
    blqr = init_blqr(subkey, state_dim, control_dim, horizon, batch_size)
    return idoc.blqr.Params(x0, blqr)


def test_blqr():
    jax.config.update("jax_enable_x64", True)
    batch_size = 30
    # problem dimensions
    state_dim, control_dim, T = 3, 2, 5
    # random key
    key = jr.PRNGKey(42)
    # initialize solvers
    solver = idoc.blqr.build(T)
    # initialize parameters
    params = init_params(key, state_dim, control_dim, T, batch_size)
    # check that both solvers give the same solution
    for k, solve in [("direct", solver.direct), ("implicit", solver.implicit)]:
        print(k)
        s = solve(params)
        idoc.utils.check_kkt(solver.kkt, s, params)

    # check that the gradients match between two solvers
    def loss(s, params):
        return (
            0.5 * jnp.sum(s.X ** 2)
            + 0.5 * jnp.sum(s.U ** 2)
            + jnp.sum(params.x0 ** 2)
            + jnp.sum(params.blqr.A ** 2)
        )

    def direct_loss(params):
        params = idoc.blqr.Params(params.x0, params.blqr.symm())
        s = solver.direct(params)
        return loss(s, params)

    def implicit_loss(params):
        params = idoc.blqr.Params(params.x0, params.blqr.symm())
        s = solver.implicit(params)
        return loss(s, params)

    # check along one random direction

    direct = jax.grad(direct_loss)(params)
    implicit = jax.grad(implicit_loss)(params)

    pc = idoc.utils.print_and_check
    rd = idoc.utils.relative_difference

    print("Direct v implicit")
    pc(rd(direct.x0, implicit.x0))
    pc(rd(direct.blqr.A, implicit.blqr.A))
    pc(rd(direct.blqr.B, implicit.blqr.B))
    pc(rd(direct.blqr.d, implicit.blqr.d))
    pc(rd(direct.blqr.M, implicit.blqr.M))
    pc(rd(direct.blqr.Q, implicit.blqr.Q))
    pc(rd(direct.blqr.q, implicit.blqr.q))
    pc(rd(direct.blqr.R, implicit.blqr.R))
    pc(rd(direct.blqr.r, implicit.blqr.r))
    pc(rd(direct.blqr.Qf, implicit.blqr.Qf))
    pc(rd(direct.blqr.qf, implicit.blqr.qf))


if __name__ == "__main__":
    test_blqr()
