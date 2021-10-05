"""Tests for time-invariant LQR solver"""

import jax
import jax.numpy as jnp
import idoc


def lqr_cost(X, U, theta: idoc.tilqr.Params):
    Q = theta.lqr.Q
    x0 = theta.x0
    lq = jnp.sum(X * jnp.dot(X, Q)) + jnp.sum(x0 * jnp.dot(Q, x0))
    lr = jnp.sum(U * jnp.dot(U, theta.lqr.R))
    return 0.5 * (lq + lr)


def init_lqr(key, state_dim, control_dim) -> idoc.tilqr.TILQR:
    Q = jnp.eye(state_dim)
    Qf = jnp.eye(state_dim)
    R = jnp.eye(control_dim) * 0.01
    key, subkey = jax.random.split(key)
    A = idoc.utils.init_stable(subkey, state_dim)
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (state_dim, control_dim))
    return idoc.tilqr.TILQR(Q, Qf, R, A, B)


def init_params(key, state_dim, control_dim) -> idoc.tilqr.Params:
    """Initialize random parameters."""
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (state_dim,))
    key, subkey = jax.random.split(key)
    lqr = init_lqr(subkey, state_dim, control_dim)
    return idoc.tilqr.Params(x0, lqr)


def test_tilqr():
    jax.config.update("jax_enable_x64", True)
    # problem dimensions
    state_dim, control_dim, T = 3, 2, 40
    # random key
    key = jax.random.PRNGKey(42)
    # initialize solvers
    solver = idoc.tilqr.build(T)
    # initialize parameters
    theta = init_params(key, state_dim, control_dim)
    # check that both solvers give the same solution
    for k, solve in [("direct", solver.direct), ("implicit", solver.implicit)]:
        print(k)
        s = solve(theta)
        print(lqr_cost(s.X, s.U, theta))
        idoc.utils.check_kkt(solver.kkt, s, theta)

    # check that the gradients match between two solvers
    def loss(s, theta):
        return (
            1.0 * jnp.sum(s.X ** 2)
            + 0.5 * jnp.sum(s.U ** 2)
            + jnp.sum(theta.x0 ** 2)
            + jnp.sum(theta.lqr.A ** 2)
        )

    def loss_direct(theta):
        return loss(solver.direct(theta), theta)

    def loss_implicit(theta):
        return loss(solver.implicit(theta), theta)

    direct = jax.grad(loss_direct)(theta)
    implicit = jax.grad(loss_implicit)(theta)

    thres = 1e-4
    pc = idoc.utils.print_and_check
    rd = idoc.utils.relative_difference

    pc(rd(direct.x0, implicit.x0))
    pc(rd(direct.lqr.A, implicit.lqr.A))
    pc(rd(direct.lqr.B, implicit.lqr.B))
    pc(rd(direct.lqr.Q, implicit.lqr.Q))
    pc(rd(direct.lqr.Qf, implicit.lqr.Qf))
    pc(rd(direct.lqr.R, implicit.lqr.R))


if __name__ == "__main__":
    test_tilqr()
