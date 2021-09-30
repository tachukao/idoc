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

    print("KKT")
    print(idoc.qp.kkt(idoc.qp.solver.direct(theta), theta))
    print(idoc.qp.kkt(idoc.qp.solver.implicit(theta), theta))
    direct = jax.grad(direct_loss)(theta)
    implicit = jax.grad(implicit_loss)(theta)

    pc = idoc.utils.print_and_check
    rd = idoc.utils.relative_difference

    print("Implicit v Finite Difference")
    findiff = idoc.utils.finite_difference_grad(lambda theta: direct_loss(theta), theta)

    pc(rd(implicit.Q, findiff.Q))
    pc(rd(implicit.c, findiff.c))
    pc(rd(implicit.d, findiff.d))
    pc(rd(implicit.E, findiff.E))

    print("Direct v Implicit")
    pc(rd(direct.Q, implicit.Q))
    pc(rd(direct.c, implicit.c))
    pc(rd(direct.d, implicit.d))
    pc(rd(direct.E, implicit.E))


if __name__ == "__main__":
    test_qp()
