#!/usr/bin/env python

import numpy

from setuptools import setup, find_packages

from distutils import core
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


extensions = [Extension('config.Config', sources=['config/Config.pyx'], language="c++", extra_compile_args=["-std=c++11"])]

core.setup(
  ext_modules=cythonize(extensions),
  include_dirs=[numpy.get_include()]
)

requires = [
]

setup(
  name='openke',
  version='0.0.1',
  author='take',
  url='',
  packages=find_packages(),
  scripts=[
  ],
  install_requires=requires,
  license='MIT',
  test_suite='test',
  classifiers=[
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Programming Language :: Python',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: KE',
  ],
  data_files=[
  ]
)
