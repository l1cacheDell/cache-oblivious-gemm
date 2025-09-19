from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import sysconfig

import numpy as np

cxx_std = 17
extra_compile_args = []
extra_link_args = []

# linux environment, default
extra_compile_args += ["-O3", "-ffast-math", "-march=native", "-fopenmp", "-Wall", "-Wextra", "-Wpedantic", "-mavx2", "-mfma"]
extra_link_args += ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "cogemm",                           # co -> cache-oblivious
        sources=["cogemm.cpp"],
        include_dirs=[np.get_include()],
        cxx_std=cxx_std,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="cogemm",
    version="0.1.0",
    description="Minimal cache oblivious gemm environment field",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
