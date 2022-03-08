"""Differentiable sequential quadratic programming

This does not work at the moment
"""
# import jax
# import jax.numpy as jnp
# from jax import lax
# from . import qp
# from jaxopt import implicit_diff
# from typing import Callable, Any
# from dataclasses import dataclass
# from . import typs
# 
# 
# @dataclass
# class SQP:
#     f: Callable[[jnp.ndarray, Any], jnp.ndarray]
#     g: Callable[[jnp.ndarray, Any], jnp.ndarray]
# 
#     def lagrangian(self, z, nu, theta):
#         return self.f(z, theta) - jnp.dot(nu, self.g(z, theta))
# 
#     def approx(self, x, theta) -> qp.QP:
#         z, nu = x
#         Q = jax.jacfwd(jax.grad(self.lagrangian, argnums=0), argnums=0)(z, nu, theta)
#         c = jax.grad(self.f, argnums=0)(z, theta)
#         E = jax.jacobian(self.g, argnums=0)(z, theta)
#         d = -self.g(z, theta)
#         return qp.QP(Q, c, E, d)
# 
# 
# def build(sqp: SQP, iterations: int) -> typs.Solver:
#     def kkt(x, theta):
#         """SQP KKT conditions"""
#         z, nu = x
#         stationarity = jax.grad(sqp.lagrangian, argnums=0)(z, nu, theta)
#         primal_feasability = sqp.g(z, theta)
#         return stationarity, primal_feasability
# 
#     def solver(x, theta):
#         def loop(x, _):
#             z, _ = x
#             p = sqp.approx(x, theta)
#             dz, nu = qp.direct_solver(p)
#             return (z + dz, -nu), None
# 
#         x, _ = lax.scan(loop, x, jnp.arange(iterations))
#         return x
# 
#     return typs.Solver(
#         direct=solver, kkt=kkt, implicit=implicit_diff.custom_root(kkt)(solver)
#     )
