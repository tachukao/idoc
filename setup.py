from setuptools import setup
from setuptools import find_packages

setup(
    name="idoc",
    author="Ta-Chu Kao",
    version="0.0.1",
    description="Implicit Differentiable Optimal Control with Jax",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxopt",
        "optax",
        "scikit-learn",
    ],
    packages=find_packages(),
)
