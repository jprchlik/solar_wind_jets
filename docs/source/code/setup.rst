setup module
============

.. automodule:: setup
    :members:
    :undoc-members:
    :show-inheritance:

A quick script to compile the custom mult_dtw function used in this program.

from distutils.core import setup
from Cython.Build import cythonize

setup(name="multi_dtw", ext_modules=cythonize('multi_dtw.pyx'),)
