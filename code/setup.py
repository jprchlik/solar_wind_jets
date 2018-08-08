from distutils.core import setup
from Cython.Build import cythonize

setup(name="multi_dtw", ext_modules=cythonize('multi_dtw.pyx'),)