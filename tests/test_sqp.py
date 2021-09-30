import jax
import jax.numpy as jnp
import idoc


def init_theta(key, dim):
    Q = jnp.eye(dim) * 10
    c = jnp.ones((dim,))
    E = jax.random.normal(key, (dim, dim))
    d = jnp.ones((dim,))
    return idoc.qp.QP(Q, c, E, d)


def init_state(key, dim):
    x = jax.random.normal(key, (dim,))
    nu = jnp.zeros((dim,))
    return x, nu


def init_nonlinear_sqp():
    def f(z, p: idoc.qp.QP):
        h = jax.nn.relu(z)
        return 0.5 * jnp.dot(jnp.dot(p.Q, h), h) + (jnp.dot(p.c, h) ** 2)

    def g(z, p: idoc.qp.QP):
        E, d = p.E, p.d
        return (jnp.dot(E, z) - d) ** 2 + (z ** 2)

    return idoc.sqp.SQP(f=f, g=g)


def init_linear_sqp():
    def f(z, p: idoc.qp.QP):
        return 0.5 * jnp.dot(jnp.dot(p.Q, z), z) + jnp.dot(p.c, z)

    def g(z, p: idoc.qp.QP):
        E, d = p.E, p.d
        return jnp.dot(E, z) - d

    return idoc.sqp.SQP(f=f, g=g)


def test_sqp():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(42)
    dim = 3
    iterations = 100

    for sqp in [init_linear_sqp(), init_nonlinear_sqp()]:
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
        init_x = init_state(subkey, dim)
        print(solver.direct(init_x, theta))
        print(solver.implicit(init_x, theta))
        print(solver.kkt(solver.implicit(init_x, theta), theta))
        direct_grad = jax.grad(direct_loss, argnums=1)(init_x, theta)
        implicit_grad = jax.grad(implicit_loss, argnums=1)(init_x, theta)
        thres = 1e-4
        assert idoc.utils.relative_difference(direct_grad.Q, implicit_grad.Q) < thres
        assert idoc.utils.relative_difference(direct_grad.c, implicit_grad.c) < thres
        assert idoc.utils.relative_difference(direct_grad.E, implicit_grad.E) < thres
        assert idoc.utils.relative_difference(direct_grad.d, implicit_grad.d) < thres


if __name__ == "__main__":
    test_sqp()
