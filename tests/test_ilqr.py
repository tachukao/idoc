import jax.numpy as jnp
import jax
from flax import struct
import idoc


class Params(struct.PyTreeNode):
    Q: jnp.ndarray
    Qf: jnp.ndarray
    R: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray


def init_theta(key, state_dim, control_dim) -> Params:
    Q = jnp.eye(state_dim)
    Qf = jnp.eye(state_dim)
    R = jnp.eye(control_dim) * 0.01
    key, subkey = jax.random.split(key)
    A = idoc.utils.init_stable(subkey, state_dim)
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (state_dim, control_dim))
    return Params(Q=Q, Qf=Qf, R=R, A=A, B=B)


def init_ilqr_problem(
    state_dim: int, control_dim: int, horizon: int
) -> idoc.ilqr.Problem:
    def dynamics(_, x, u, theta):
        return theta.A @ x + theta.B @ u

    def cost(_, x, u, theta):
        lQ = 0.5 * jnp.dot(jnp.dot(theta.Q, x), x)
        lR = 0.5 * jnp.dot(jnp.dot(theta.R, u), u)
        Z = jnp.outer(x, u)
        return lQ + lR

    def costf(xf, theta):
        return 0.5 * jnp.dot(jnp.dot(theta.Qf, xf), xf)

    return idoc.ilqr.Problem(
        cost=cost,
        costf=costf,
        dynamics=dynamics,
        horizon=horizon,
        state_dim=state_dim,
        control_dim=control_dim,
    )


def init_params(key, state_dim, control_dim) -> idoc.ilqr.Params:
    """Initialize random parameters."""
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (state_dim,))
    key, subkey = jax.random.split(key)
    theta = init_theta(subkey, state_dim, control_dim)
    return idoc.ilqr.Params(x0, theta=theta)


def test_ilqr():
    jax.config.update("jax_enable_x64", True)
    # problem dimensions
    state_dim, control_dim, T, iterations = 3, 3, 10, 3
    # random key
    key = jax.random.PRNGKey(42)
    # initialize ilqr
    ilqr_problem = init_ilqr_problem(state_dim, control_dim, T)
    # initialize solvers
    solver = idoc.ilqr.build(ilqr_problem, iterations)
    # initialize parameters
    params = init_params(key, state_dim, control_dim)
    # initialize state
    Uinit = jnp.zeros((T, control_dim))
    Xinit = idoc.ilqr.simulate(ilqr_problem, Uinit, params)
    sinit = idoc.typs.State(X=Xinit, U=Uinit, Nu=jnp.zeros_like(Xinit))
    # check that both solvers give the same solution
    def check_solution():
        for k, solve in [("direct", solver.direct), ("implicit", solver.implicit)]:
            print(k)
            s = solve(sinit, params)
            idoc.utils.check_kkt(solver.kkt, s, params)

    check_solution()

    # check that the gradients match between two solvers
    def loss(s, params):
        return 1.0 * jnp.sum(s.X ** 2) + 0.5 * jnp.sum(s.U ** 2)

    def loss_direct(params):
        s = solver.direct(sinit, params)
        return loss(s, params)

    def loss_implicit(params):
        s = solver.implicit(sinit, params)
        return loss(s, params)

    direct = jax.grad(loss_direct)(params)
    implicit = jax.grad(loss_implicit)(params)
    thres = 1e-4

    print(direct.x0)
    print(implicit.x0)
    print(direct.theta.A)
    print(implicit.theta.A)

    idoc.utils.print_and_check(
        idoc.utils.relative_difference(direct.theta.A, implicit.theta.A)
    )
    idoc.utils.print_and_check(
        idoc.utils.relative_difference(direct.theta.B, implicit.theta.B)
    )
    idoc.utils.print_and_check(
        idoc.utils.relative_difference(direct.theta.Q, implicit.theta.Q)
    )
    idoc.utils.print_and_check(
        idoc.utils.relative_difference(direct.theta.Qf, implicit.theta.Qf)
    )
    idoc.utils.print_and_check(
        idoc.utils.relative_difference(direct.theta.R, implicit.theta.R)
    )
    idoc.utils.print_and_check(idoc.utils.relative_difference(direct.x0, implicit.x0))


if __name__ == "__main__":
    test_ilqr()
