import jax
import jax.numpy as jnp
from absl import app
import dilqr


def main(argv):
    def init_theta(key, dim):
        Q = jnp.eye(dim) * 10
        c = jnp.ones((dim,))
        E = jax.random.normal(key, (dim, dim))
        d = jnp.ones((dim,))
        return dilqr.qp.QP(Q, c, E, d)

    def outer_loss(z, nu):
        return jnp.sum(z ** 2) + jnp.sum(nu ** 2)

    def direct_loss(theta):
        return outer_loss(*dilqr.qp.direct_solver(theta))

    def implicit_loss(theta):
        return outer_loss(*dilqr.qp.implicit_solver(theta))

    key = jax.random.PRNGKey(42)
    subkey, key = jax.random.split(key)
    dim = 3
    theta = init_theta(subkey, dim)
    direct = jax.grad(direct_loss)(theta)
    implicit= jax.grad(implicit_loss)(theta)
    print(f"Gradient Error in Q\n------\n{direct.Q - implicit.Q}")
    print(f"Gradient Error in c\n------\n{direct.c - implicit.c}")
    print(f"Gradient Error in E\n------\n{direct.E - implicit.E}")
    print(f"Gradient Error in d\n------\n{direct.d - implicit.d}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    app.run(main)
