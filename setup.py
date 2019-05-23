#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This is for setting up libEnsemble, license and details can be
# found at https://github.com/Libensemble/libensemble/

from setuptools import setup
from setuptools.command.test import test as TestCommand


class Run_TestSuite(TestCommand):
    def run_tests(self):
        import os
        import sys
        py_version = sys.version_info[0]
        print('Python version from setup.py is', py_version)
        run_string = "code/tests/run-tests.sh -p " + str(py_version)
        os.system(run_string)


class ToxTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def run_tests(self):
        import tox
        tox.cmdline()


setup(
    name='libensemble',
    version='0.5.0',
    description='Library for managing ensemble-like collections of computations',
    url='https://github.com/Libensemble/libensemble',
    author='Jeffrey Larson, Stephen Hudson, Stefan M. Wild, and David Bindel',
    author_email='libensemble@lists.mcs.anl.gov',
    license='BSD 2-clause',

    packages=['libensemble',
              'libensemble.gen_funcs',
              'libensemble.sim_funcs',
              'libensemble.sim_funcs.branin',
              'libensemble.alloc_funcs',
              'libensemble.tests',
              'libensemble.comms',
              'libensemble.util',
              'libensemble.tests.unit_tests',
              'libensemble.tests.regression_tests'],

    package_data={'libensemble.sim_funcs.branin': ['known_minima_and_func_values']},

    install_requires=['numpy'],

    # If run tests through setup.py - downloads these but does not install
    tests_require=['pytest>=3.1',
                   'pytest-cov>=2.5',
                   'pytest-pep8>=1.0',
                   'pytest-timeout',
                   ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    cmdclass={'test': Run_TestSuite, 'tox': ToxTest}
)
