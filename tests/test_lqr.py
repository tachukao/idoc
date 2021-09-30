import idoc
import jax
import jax.numpy as jnp


def lqr_cost(X, U, theta: idoc.lqr.Params):
    Q = theta.lqr.Q
    x0 = theta.x0
    lq = jnp.sum(X * jnp.dot(X, Q)) + jnp.sum(x0 * jnp.dot(Q, x0))
    lr = jnp.sum(U * jnp.dot(U, theta.lqr.R))
    return 0.5 * (lq + lr)


def init_lqr(key, state_dim: int, control_dim: int, horizon: int) -> idoc.lqr.LQR:
    """Initialize a random LQR spec."""
    Q = jnp.stack(horizon * (jnp.eye(state_dim),))
    q = 0.2 * jnp.stack(horizon * (jnp.ones(state_dim),))
    Qf = jnp.eye(state_dim)
    qf = 0.2 * jnp.ones((state_dim,))
    R = jnp.stack(horizon * (jnp.eye(control_dim),)) * 0.01
    r = 0.01 * jnp.stack(horizon * (jnp.ones(control_dim),))
    M = 0.02 * jnp.stack(horizon * (jnp.ones((state_dim, control_dim)),))
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
    theta = init_params(key, state_dim, control_dim, T)
    # check that both solvers give the same solution
    for k, solve in [("direct", solver.direct), ("implicit", solver.implicit)]:
        print(k)
        s = solve(theta)
        # print(lqr_cost(s.X, s.U, theta))
        idoc.utils.check_kkt(solver.kkt, s, theta)

    # check that the gradients match between two solvers
    def loss(s, theta):
        return (
            1.0 * jnp.sum(s.X ** 2)
            + 0.5 * jnp.sum(s.U ** 2)
            + jnp.sum(theta.x0 ** 2)
            + jnp.sum(theta.lqr.A ** 2)
        )

    def direct_loss(theta):
        return loss(solver.direct(theta), theta)

    def implicit_loss(theta):
        return loss(solver.implicit(theta), theta)

    direct = jax.grad(direct_loss)(theta)
    implicit = jax.grad(implicit_loss)(theta)

    pc = idoc.utils.print_and_check
    rd = idoc.utils.relative_difference

    print("Direct v implicit")

    pc(rd(direct.x0, implicit.x0))
    pc(rd(direct.lqr.A, implicit.lqr.A))
    pc(rd(direct.lqr.B, implicit.lqr.B))
    pc(rd(direct.lqr.d, implicit.lqr.d))
    pc(rd(direct.lqr.M, implicit.lqr.M))
    pc(rd(direct.lqr.Q, implicit.lqr.Q))
    pc(rd(direct.lqr.Qf, implicit.lqr.Qf))
    pc(rd(direct.lqr.q, implicit.lqr.q))
    pc(rd(direct.lqr.qf, implicit.lqr.qf))
    pc(rd(direct.lqr.R, implicit.lqr.R))
    pc(rd(direct.lqr.r, implicit.lqr.r))

    # findiff = idoc.utils.finite_difference_grad(lambda theta: direct_loss(theta), theta)
    # print("Implicit v Finite Difference")
    # pc(rd(findiff.x0, implicit.x0))
    # pc(rd(findiff.lqr.A, implicit.lqr.A))
    # pc(rd(findiff.lqr.B, implicit.lqr.B))
    # pc(rd(findiff.lqr.Q, implicit.lqr.Q))
    # pc(rd(findiff.lqr.Qf, implicit.lqr.Qf))
    # pc(rd(findiff.lqr.R, implicit.lqr.R))


if __name__ == "__main__":
    test_lqr()
