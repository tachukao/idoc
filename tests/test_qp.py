import jax
import jax.numpy as jnp
import idoc


def outer_loss(z, nu):
    return jnp.sum(z ** 2) + jnp.sum(nu ** 2)


def direct_loss(theta):
    return outer_loss(*idoc.qp.solver.direct(theta))


def implicit_loss(theta):
    return outer_loss(*idoc.qp.solver.implicit(theta))


def test_qp():
    jax.config.update("jax_enable_x64", True)

    def init_theta(key, dim):
        Q = jnp.eye(dim) * 10
        c = jnp.ones((dim,))
        E = jax.random.normal(key, (dim, dim))
        d = jnp.ones((dim,))
        return idoc.qp.QP(Q, c, E, d)

    key = jax.random.PRNGKey(42)
    subkey, key = jax.random.split(key)
    dim = 3
    theta = init_theta(subkey, dim)
    direct = jax.grad(direct_loss)(theta)
    implicit = jax.grad(implicit_loss)(theta)
    assert idoc.utils.relative_difference(direct.Q, implicit.Q)
    assert idoc.utils.relative_difference(direct.c, implicit.c)
    assert idoc.utils.relative_difference(direct.E, implicit.E)
    assert idoc.utils.relative_difference(direct.d, implicit.d)


if __name__ == "__main__":
    test_qp()
