import jax.numpy as jnp
import jax
from flax import struct
import dilqr
from typing import Callable
from absl import app


class Params(struct.PyTreeNode):
    Q: jnp.ndarray
    Qf: jnp.ndarray
    R: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray


def init_stable(key, state_dim):
    """Initialize a stable matrix with dimensions `state_dim`."""
    R = jax.random.normal(key, (state_dim, state_dim))
    A, _ = jnp.linalg.qr(R)
    return 0.5 * A


def init_theta(key, state_dim, control_dim) -> Params:
    Q = jnp.eye(state_dim)
    Qf = jnp.eye(state_dim)
    R = jnp.eye(control_dim) * 0.01
    key, subkey = jax.random.split(key)
    A = init_stable(subkey, state_dim)
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (state_dim, control_dim))
    return Params(Q=Q, Qf=Qf, R=R, A=A, B=B)


def init_ilqr_problem(
    state_dim: int, control_dim: int, horizon: int
) -> dilqr.ilqr.Problem:
    def dynamics(_, x, u, theta):
        return jnp.tanh(theta.A @ x) + theta.B @ u + 0.5

    def cost(_, x, u, theta):
        lQ = 0.5 * jnp.dot(jnp.dot(theta.Q, x), x)
        lR = 0.5 * jnp.dot(jnp.dot(theta.R, u), u)
        Z = jnp.outer(x, u)
        return lQ + lR + jnp.sum(Z ** 2)

    def costf(xf, theta):
        return 0.5 * jnp.dot(jnp.dot(theta.Qf, xf), xf)

    return dilqr.ilqr.Problem(
        cost=cost,
        costf=costf,
        dynamics=dynamics,
        horizon=horizon,
        state_dim=state_dim,
        control_dim=control_dim,
    )


def init_params(key, state_dim, control_dim) -> dilqr.ilqr.Params:
    """Initialize random parameters."""
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (state_dim,))
    key, subkey = jax.random.split(key)
    theta = init_theta(subkey, state_dim, control_dim)
    return dilqr.ilqr.Params(x0, theta=theta)


def check_kkt(kkt: Callable, s: dilqr.typs.State, theta: Params) -> None:
    kkt_state = kkt(s, theta)
    print(f"dLdX: {jnp.mean(jnp.abs(kkt_state.X))}")
    print(f"dLdU: {jnp.mean(jnp.abs(kkt_state.U))}")
    print(f"dLdNu: {jnp.mean(jnp.abs(kkt_state.Nu))}")


def main(argv):
    # problem dimensions
    state_dim, control_dim, T, iterations = 3, 2, 40, 10
    # random key
    key = jax.random.PRNGKey(42)
    # initialize ilqr
    ilqr_problem = init_ilqr_problem(state_dim, control_dim, T)
    # initialize solvers
    solve_direct, kkt, solve_implicit, simulate = dilqr.ilqr.build(
        ilqr_problem, iterations
    )
    # initialize parameters
    params = init_params(key, state_dim, control_dim)
    # initialize U
    Uinit = jnp.zeros((T, control_dim))
    # check that both solvers give the same solution
    for k, solve in [("direct", solve_direct), ("implicit", solve_implicit)]:
        print(k)
        s = solve(Uinit, params)
        check_kkt(kkt, s, params)

    # check that the gradients match between two solvers
    def loss(s, params):
        return (
            1.0 * jnp.sum(s.X ** 2)
            + 0.5 * jnp.sum(s.U ** 2)
            + jnp.sum(params.x0 ** 2)
            + jnp.sum(params.theta.A ** 2)
        )

    def loss_direct(params):
        return loss(solve(Uinit, params), params)

    def loss_implicit(params):
        return loss(solve_implicit(Uinit, params), params)

    direct = jax.grad(loss_direct)(params)
    implicit = jax.grad(loss_implicit)(params)

    def compare(z1, z2):
        err = jnp.mean((z1 - z2) ** 2) / jnp.mean((z1 + z2) ** 2 + 1e-9)
        return err

    print(f"Gradient difference for [x0]: {compare(direct.x0, implicit.x0)}")
    print(f"Gradient difference for [A]: {compare(direct.theta.A, implicit.theta.A)}")
    print(f"Gradient difference for [B]: {compare(direct.theta.B, implicit.theta.B)}")
    print(f"Gradient difference for [Q]: {compare(direct.theta.Q, implicit.theta.Q)}")
    print(
        f"Gradient difference for [Qf]: {compare(direct.theta.Qf, implicit.theta.Qf)}"
    )
    print(f"Gradient difference for [R]: {compare(direct.theta.R, implicit.theta.R)}")


if __name__ == "__main__":
    app.run(main)
