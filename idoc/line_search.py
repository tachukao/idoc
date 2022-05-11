import jax
import jax.numpy as jnp
import jaxopt
import logging


def make_line_search(*, lower:float=1e-3, upper:float=2.0, rho:float=0.5, maxiter:int=8, verbose:bool=False):
    """Makes line search function for ilqr and bilqr."""
    def backtrack(f, initial, expected_change, unroll, jit):
        _, _, c_initial = initial

        def update(alpha):
            X, U, c = f(alpha)
            ec = expected_change(alpha)
            change = c_initial - c
            p = jnp.abs((change + 1e-10) / (1e-10 + ec))
            carry_on = jnp.logical_or((p<=lower), p>=upper)
            return (X, U, c), p, carry_on


        def loop(val):
            _, alpha, it, _ = val
            (X, U, c), p, carry_on = update(alpha)
            if verbose:
                print(f"line search ({it}) c {c: .05f}, alpa {alpha: .04f}, pct change {p: .3f}")
            alpha *= rho
            return (X, U, c), alpha, it + 1, carry_on

        (X, U, c), _, _, carry_on = jaxopt.loop.while_loop(
            lambda v: v[-1],
            loop,
            (initial, 1.0, 0, True),
            maxiter=maxiter,
            jit=jit,
            unroll=unroll,
        )

        line_search_passes = jnp.logical_not(carry_on)

        return X, U, c, line_search_passes

    return backtrack
