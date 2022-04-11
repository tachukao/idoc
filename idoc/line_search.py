import jax.numpy as jnp
import jaxopt


def make_line_search(*, lower=1e-3, upper=2.0, rho=0.5, maxiter=8):
    def backtrack(f, c_old, expected_change, unroll, jit):
        def check(p):
            carry_on = jnp.logical_not(jnp.logical_and(p >= lower, p <= upper))
            return carry_on

        def update(alpha):
            X, U, c_new = f(alpha)
            p = (c_new - c_old) / expected_change(alpha)
            carry_on = check(p)
            return X, U, c_new, carry_on

        def body_fun(val):
            _, _, _, alpha, it, _ = val
            alpha *= rho
            X, U, c_new, carry_on = update(alpha)
            return X, U, c_new, alpha, it + 1, carry_on

        X, U, c, carry_on = update(1.0)
        X, U, c_new, _, _, _ = jaxopt.loop.while_loop(
            lambda v: v[-1],
            body_fun,
            (X, U, c, 1.0, 0, carry_on),
            maxiter=maxiter,
            jit=jit,
            unroll=unroll,
        )
        return X, U, c_new

    return backtrack
