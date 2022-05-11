import jax
import jax.numpy as jnp
import jaxopt
import logging


def make_line_search(*, lower:float=1e-3, upper:float=2.0, rho:float=0.5, maxiter:int=10, verbose:bool=False):
    """Makes line search function for ilqr and bilqr."""
    def backtrack(f, c_old, expected_change, unroll, jit):
        def update(alpha):
            X, U, c_new = f(alpha)
            ec = expected_change(alpha)
            change = (c_new - c_old) 
            c1 = (lower * ec  <= change)
            c2 = (upper * ec >= change )
            carry_on = jnp.logical_not(jnp.logical_and(c1, c2))
            return X, U, c_new, ec, carry_on

        def loop(val):
            _, _, _, alpha, it, _ = val
            alpha *= rho
            X, U, c_new, ec, carry_on = update(alpha)
            if verbose:
                print(f"line search ({it}) cnew {c_new: .05f}, alpa {alpha: .04f}, ec {ec: .3f}")
            return X, U, c_new, alpha, it + 1, carry_on

        X, U, c, _, carry_on = update(1.)
        X, U, c_new, _, _, carry_on = jaxopt.loop.while_loop(
            lambda v: v[-1],
            loop,
            (X, U, c, 1.0, 0, carry_on),
            maxiter=maxiter,
            jit=jit,
            unroll=unroll,
        )

        line_search_passes = jnp.logical_not(carry_on)

        return X, U, c_new, line_search_passes

    return backtrack
