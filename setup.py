from setuptools import setup, Extension
import numpy


ext = Extension(
    "healsparse._healsparse_lib",
    [
        "healsparse/healsparse_lib.c",
    ],
)

setup(
    ext_modules=[ext],
    include_dirs=numpy.get_include(),
)
