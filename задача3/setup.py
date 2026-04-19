from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

ext = Extension(
    "cy_verlet",
    ["cy_verlet.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    include_dirs=[np.get_include()]
)

setup(ext_modules=cythonize([ext], compiler_directives={'language_level': "3"}))