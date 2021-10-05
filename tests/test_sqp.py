"""Tests for SQP solver"""
import jax
import jax.numpy as jnp
from jax import flatten_util
import idoc
import scipy


def init_theta(key, dim):
    Q = jnp.eye(dim) * 10
    c = jnp.ones((dim,))
    key, subkey = jax.random.split(key)
    E = jax.random.normal(key, (dim, dim))
    h = jax.random.normal(subkey, (dim,))
    d = E @ h
    return idoc.qp.QP(Q, c, E, d)


def init_nonlinear_sqp():
    def f(z, p: idoc.qp.QP):
        h = jax.nn.relu(z)
        return 0.5 * jnp.dot(jnp.dot(p.Q, h), h) + (jnp.dot(p.c, h) ** 2)

    def g(z, p: idoc.qp.QP):
        E, d = p.E, p.d
        return jnp.tanh(jnp.dot(E, z) - d)

    def init_state(p):
        z = jnp.linalg.solve(p.E, p.d)
        nu = jnp.zeros((p.E.shape[0],))
        return z, nu

    return idoc.sqp.SQP(f=f, g=g), init_state


def init_linear_sqp():
    def f(z, p: idoc.qp.QP):
        return 0.5 * jnp.dot(jnp.dot(p.Q, z), z) + jnp.dot(p.c, z)

    def g(z, p: idoc.qp.QP):
        E, d = p.E, p.d
        return jnp.dot(E, z) - d

    def init_state(p):
        x = jnp.linalg.solve(p.E, p.d)
        nu = jnp.zeros((p.E.shape[0],))
        return x, nu

    return idoc.sqp.SQP(f=f, g=g), init_state


def test_sqp():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(42)
    dim = 3
    iterations = 100

    for label, (sqp, init_state) in [
        ("linear", init_linear_sqp()),
        ("nonlinear", init_nonlinear_sqp()),
    ]:
        print(label)
        solver = idoc.sqp.build(sqp, iterations)

        def _loss(z, nu):
            return jnp.sum(z ** 2) + jnp.sum(nu ** 2)

        def direct_loss(x, theta):
            z, nu = solver.direct(x, theta)
            return _loss(z, nu)

        def implicit_loss(x, theta):
            z, nu = solver.implicit(x, theta)
            return _loss(z, nu)

        key, subkey = jax.random.split(key)
        theta = init_theta(subkey, dim)
        key, subkey = jax.random.split(key)
        init_x = init_state(theta)
        print(sqp.g(init_x[0], theta))
        print(sqp.g(solver.direct(init_x, theta)[0], theta))
        print(sqp.g(solver.implicit(init_x, theta)[0], theta))
        print("KKT")
        print(solver.kkt(solver.direct(init_x, theta), theta))
        print(solver.kkt(solver.implicit(init_x, theta), theta))
        direct = jax.grad(direct_loss, argnums=1)(init_x, theta)
        implicit = jax.grad(implicit_loss, argnums=1)(init_x, theta)

        print("Direct v Implicit")
        idoc.utils.print_and_check(idoc.utils.relative_difference(direct.Q, implicit.Q))
        idoc.utils.print_and_check(idoc.utils.relative_difference(direct.c, implicit.c))
        idoc.utils.print_and_check(idoc.utils.relative_difference(direct.d, implicit.d))
        idoc.utils.print_and_check(idoc.utils.relative_difference(direct.E, implicit.E))

        print("Implicit v Finite Difference")
        findiff = idoc.utils.finite_difference_grad(
            lambda theta: direct_loss(init_x, theta), theta
        )

        idoc.utils.print_and_check(
            idoc.utils.relative_difference(implicit.Q, findiff.Q)
        )
        idoc.utils.print_and_check(
            idoc.utils.relative_difference(implicit.c, findiff.c)
        )
        idoc.utils.print_and_check(
            idoc.utils.relative_difference(implicit.d, findiff.d)
        )
        idoc.utils.print_and_check(
            idoc.utils.relative_difference(implicit.E, findiff.E)
        )


if __name__ == "__main__":
    test_sqp()
