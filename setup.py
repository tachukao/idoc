from setuptools import setup
from setuptools import find_packages

setup(
    name="dilqr",
    author="Ta-Chu Kao",
    version="0.0.1",
    description="Jax implementation of differentiable iLQR",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxopt",
        "optax",
        "flax",
        "scikit-learn",
    ],
    packages=find_packages(),
)
