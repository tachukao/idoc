"""Test for iLQR solver"""

import jax.numpy as jnp
import jax
import jax.random as jr
import jax.scipy as sp
from typing import NamedTuple
import idoc
from jax.test_util import check_grads
import os
from idoc import bilqr

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Params(NamedTuple):
    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray


def system_dimensions():
    n = 10
    m = 10
    T = 30
    return n, m, T


dims = system_dimensions()
batch_size = 1
system_key = jr.PRNGKey(10000)


def init_theta() -> Params:
    key = system_key
    state_dim, control_dim, _ = dims
    Q = jnp.eye(state_dim)
    q = jnp.ones(state_dim) * 0.01
    Qf = jnp.eye(state_dim)
    R = jnp.eye(control_dim)
    r = jnp.ones(control_dim)
    key, subkey = jax.random.split(key)
    A = idoc.utils.init_stable(subkey, state_dim)
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (state_dim, control_dim))
    return Params(Q=Q, q=q, Qf=Qf, R=R, r=r, A=A, B=B)


def init_ilqr_problem(state_dim: int, control_dim: int, horizon: int) -> bilqr.Problem:
    phi = lambda x: jax.nn.relu(x)

    def dynamics(_, x, u, theta):
        return phi(theta.A @ x) + theta.B @ u + 0.5

    def cost(_, x, u, theta):
        n = x.shape[-1]
        m = u.shape[-1]
        lQ = 0.5 * jnp.dot(jnp.dot(theta.Q, x), x)
        lq = jnp.dot(theta.q, x)
        lR = 1e-4 * jnp.dot(jnp.dot(theta.R, u), u)
        lM = -1e-4 * jnp.dot(jnp.dot(jnp.ones((n, m)), u), x)
        lr = 1e-4 * jnp.dot(theta.r, u)
        x = x + 0.001
        return lQ + lq + lR + lr + lM + jnp.sum(x*jnp.log(x*x))/batch_size

    def costf(xf, theta):
        return 0.5 * jnp.dot(jnp.dot(theta.Qf, xf), xf)

    return bilqr.Problem(
        cost=cost,
        costf=costf,
        dynamics=dynamics,
        horizon=horizon,
        state_dim=state_dim,
        control_dim=control_dim,
    )


def init_params(key) -> idoc.ilqr.Params:
    state_dim, control_dim, _ = dims
    """Initialize random parameters."""
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (batch_size, state_dim))
    key, subkey = jax.random.split(key)
    theta = init_theta()
    return bilqr.Params(x0, theta=theta)


def test_ilqr():
    jax.config.update("jax_enable_x64", True)
    # problem dimensions
    state_dim, control_dim, T = dims
    maxiter = 30
    # random key
    key = jax.random.PRNGKey(42)
    # initialize ilqr
    ilqr_problem = init_ilqr_problem(state_dim, control_dim, T)
    line_search = idoc.make_line_search()
    # initialize solvers
    solver = bilqr.build(
        ilqr_problem, maxiter=maxiter, thres=1e-8, line_search=line_search
    )
    # initialize parameters
    params = init_params(key)
    # initialize state
    Uinit = jnp.zeros((T, control_dim))
    Xinit, _ = bilqr.simulate(ilqr_problem, Uinit, params)
    sinit = idoc.typs.State(X=Xinit, U=Uinit, Nu=jnp.zeros_like(Xinit))

    # check that both solvers give the same solution
    def check_solution():
        for k, solve in [("direct", solver.direct), ("implicit", solver.implicit)]:
            # print(k)
            s = solve(sinit, params)
            idoc.utils.check_kkt(solver.kkt, s, params)

    check_solution()

    # check that the gradients match between two solvers
    def loss(s, params):
        return jnp.sum(s.X ** 2) + 0.5 * jnp.sum(s.U ** 2)

    def direct_loss(params):
        s = solver.direct(sinit, params)
        return loss(s, params)

    def implicit_loss(params):
        s = solver.implicit(sinit, params)
        return loss(s, params)

    # check along one random direction
    check_grads(implicit_loss, (params,), 1, modes=("rev",))

    # LONG checks

    # direct = jax.grad(direct_loss)(params)
    # implicit = jax.grad(implicit_loss)(params)

    # pc = idoc.utils.print_and_check
    # rd = idoc.utils.relative_difference

    # print("Direct v implicit")

    # pc(rd(direct.x0, implicit.x0))
    # pc(rd(direct.theta.A, implicit.theta.A))
    # pc(rd(direct.theta.B, implicit.theta.B))
    # pc(rd(direct.theta.Q, implicit.theta.Q))
    # pc(rd(direct.theta.Qf, implicit.theta.Qf))
    # pc(rd(direct.theta.q, implicit.theta.q))
    # pc(rd(direct.theta.R, implicit.theta.R))
    # pc(rd(direct.theta.r, implicit.theta.r))

    # findiff = idoc.utils.finite_difference_grad(direct_loss, params)
    # print("Implicit v Finite Difference")
    # pc(rd(findiff.x0, implicit.x0))
    # pc(rd(findiff.theta.A, implicit.theta.A))
    # pc(rd(findiff.theta.B, implicit.theta.B))
    # pc(rd(findiff.theta.Q, implicit.theta.Q))
    # pc(rd(findiff.theta.Qf, implicit.theta.Qf))
    # pc(rd(findiff.theta.q, implicit.theta.q))
    # pc(rd(findiff.theta.R, implicit.theta.R))
    # pc(rd(findiff.theta.r, implicit.theta.r))


if __name__ == "__main__":
    test_ilqr()
