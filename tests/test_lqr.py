"""Test for LQR solver"""

import idoc
import jax
import jax.numpy as jnp
from jax.test_util import check_grads


def init_lqr(key, state_dim: int, control_dim: int, horizon: int) -> idoc.lqr.LQR:
    """Initialize a random LQR spec."""
    Q = jnp.stack(horizon * (jnp.eye(state_dim),))
    q = 0.2 * jnp.stack(horizon * (jnp.ones(state_dim),))
    Qf = jnp.eye(state_dim)
    qf = 0.2 * jnp.ones((state_dim,))
    R = 1e-4 * jnp.stack(horizon * (jnp.eye(control_dim),))
    r = 1e-4 * jnp.stack(horizon * (jnp.ones(control_dim),))
    M = 1e-4 * jnp.stack(horizon * (jnp.ones((state_dim, control_dim)),))
    key, subkey = jax.random.split(key)
    A = jnp.stack(horizon * (idoc.utils.init_stable(subkey, state_dim),))
    key, subkey = jax.random.split(key)
    B = jnp.stack(horizon * (jax.random.normal(subkey, (state_dim, control_dim)),))
    d = jnp.stack(horizon * (jnp.ones(state_dim),))
    return idoc.lqr.LQR(Q=Q, q=q, Qf=Qf, qf=qf, R=R, r=r, A=A, B=B, d=d, M=M)


def init_params(key, state_dim, control_dim, horizon) -> idoc.lqr.Params:
    """Initialize random parameters."""
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (state_dim,))
    key, subkey = jax.random.split(key)
    lqr = init_lqr(subkey, state_dim, control_dim, horizon)
    return idoc.lqr.Params(x0, lqr)


def test_lqr():
    jax.config.update("jax_enable_x64", True)
    # problem dimensions
    state_dim, control_dim, T = 3, 2, 5
    # random key
    key = jax.random.PRNGKey(42)
    # initialize solvers
    solver = idoc.lqr.build(T)
    # initialize parameters
    params = init_params(key, state_dim, control_dim, T)
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
            + jnp.sum(params.lqr.A ** 2)
        )

    def direct_loss(params):
        params = idoc.lqr.Params(params.x0, params.lqr.symm())
        s = solver.direct(params)
        return loss(s, params)

    def implicit_loss(params):
        params = idoc.lqr.Params(params.x0, params.lqr.symm())
        s = solver.implicit(params)
        return loss(s, params)

    # check along one random direction
    # check_grads(implicit_loss, (params,), 1, modes=("rev",))

    direct = jax.grad(direct_loss)(params)
    implicit = jax.grad(implicit_loss)(params)

    pc = idoc.utils.print_and_check
    rd = idoc.utils.relative_difference

    print("Direct v implicit")
    pc(rd(direct.x0, implicit.x0))
    pc(rd(direct.lqr.A, implicit.lqr.A))
    pc(rd(direct.lqr.B, implicit.lqr.B))
    pc(rd(direct.lqr.d, implicit.lqr.d))
    pc(rd(direct.lqr.M, implicit.lqr.M))
    pc(rd(direct.lqr.Q, implicit.lqr.Q))
    pc(rd(direct.lqr.q, implicit.lqr.q))
    pc(rd(direct.lqr.R, implicit.lqr.R))
    pc(rd(direct.lqr.r, implicit.lqr.r))
    pc(rd(direct.lqr.Qf, implicit.lqr.Qf))
    pc(rd(direct.lqr.qf, implicit.lqr.qf))


if __name__ == "__main__":
    test_lqr()
