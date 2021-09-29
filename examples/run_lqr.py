import dilqr
import jax
import jax.numpy as jnp
from absl import app
from typing import Callable


def lqr_cost(X, U, theta: dilqr.lqr.Params):
    Q = theta.lqr.Q
    x0 = theta.x0
    lq = jnp.sum(X * jnp.dot(X, Q)) + jnp.sum(x0 * jnp.dot(Q, x0))
    lr = jnp.sum(U * jnp.dot(U, theta.lqr.R))
    return 0.5 * (lq + lr)


def init_stable(key, state_dim):
    """Initialize a stable matrix with dimensions `state_dim`."""
    R = jax.random.normal(key, (state_dim, state_dim))
    A, _ = jnp.linalg.qr(R)
    return 0.5 * A


def init_lqr(key, state_dim: int, control_dim: int, horizon: int) -> dilqr.lqr.LQR:
    """Initialize a random LQR spec."""
    Q = jnp.stack(horizon * (jnp.eye(state_dim),))
    q = 0.2 * jnp.stack(horizon * (jnp.ones(state_dim),))
    Qf = jnp.eye(state_dim)
    qf = 0.2 * jnp.ones((state_dim,))
    R = jnp.stack(horizon * (jnp.eye(control_dim),)) * 0.01
    r = 0.01 * jnp.stack(horizon * (jnp.ones(control_dim),))
    M = 0.02 * jnp.stack(horizon * (jnp.ones((state_dim, control_dim)),))
    key, subkey = jax.random.split(key)
    A = jnp.stack(horizon * (init_stable(subkey, state_dim),))
    key, subkey = jax.random.split(key)
    B = jnp.stack(horizon * (jax.random.normal(subkey, (state_dim, control_dim)),))
    d = jnp.stack(horizon * (jnp.ones(state_dim),))
    return dilqr.lqr.LQR(Q=Q, q=q, Qf=Qf, qf=qf, R=R, r=r, A=A, B=B, d=d, M=M)


def init_params(key, state_dim, control_dim, horizon) -> dilqr.lqr.Params:
    """Initialize random parameters."""
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (state_dim,))
    key, subkey = jax.random.split(key)
    lqr = init_lqr(subkey, state_dim, control_dim, horizon)
    return dilqr.lqr.Params(x0, lqr)


def check_kkt(kkt: Callable, s: dilqr.typs.State, params: dilqr.lqr.Params) -> None:
    kkt_state = kkt(s, params)
    print(f"dLdX: {jnp.mean(jnp.abs(kkt_state.X))}")
    print(f"dLdU: {jnp.mean(jnp.abs(kkt_state.U))}")
    print(f"dLdNu: {jnp.mean(jnp.abs(kkt_state.Nu))}")


def main(argv):
    # problem dimensions
    state_dim, control_dim, T = 3, 2, 10
    # random key
    key = jax.random.PRNGKey(42)
    # initialize solvers
    solve_direct, kkt, solve_implicit = dilqr.lqr.build(T)
    # initialize parameters
    theta = init_params(key, state_dim, control_dim, T)
    # check that both solvers give the same solution
    for k, solve in [("direct", solve_direct), ("implicit", solve_implicit)]:
        print(k)
        s = solve(theta)
        # print(lqr_cost(s.X, s.U, theta))
        check_kkt(kkt, s, theta)

    # check that the gradients match between two solvers
    def loss(s, theta):
        return (
            1.0 * jnp.sum(s.X ** 2)
            + 0.5 * jnp.sum(s.U ** 2)
            + jnp.sum(theta.x0 ** 2)
            + jnp.sum(theta.lqr.A ** 2)
        )

    def loss_direct(theta):
        return loss(solve(theta), theta)

    def loss_implicit(theta):
        return loss(solve_implicit(theta), theta)

    direct = jax.grad(loss_direct)(theta)
    implicit = jax.grad(loss_implicit)(theta)

    def compare(z1, z2):
        err = jnp.mean((z1 - z2) ** 2) / jnp.mean((z1 + z2) ** 2 + 1e-9)
        return err

    print(f"Gradient difference for [x0]: {compare(direct.x0, implicit.x0)}")
    print(f"Gradient difference for [A]: {compare(direct.lqr.A, implicit.lqr.A)}")
    print(f"Gradient difference for [B]: {compare(direct.lqr.B, implicit.lqr.B)}")
    print(f"Gradient difference for [Q]: {compare(direct.lqr.Q, implicit.lqr.Q)}")
    print(f"Gradient difference for [Qf]: {compare(direct.lqr.Qf, implicit.lqr.Qf)}")
    print(f"Gradient difference for [R]: {compare(direct.lqr.R, implicit.lqr.R)}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    app.run(main)
