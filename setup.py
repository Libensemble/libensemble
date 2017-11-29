#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.command.test import test as TestCommand

class Run_TestSuite(TestCommand):
    def run_tests(self):
        import os
        import sys
        py_version=sys.version_info[0]
        print('Python version from setup.py is', py_version)
        run_string="code/tests/run-tests.sh -p " + str(py_version)
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
    version='0.1.0',    
    description='Library for managing ensemble-like collections of computations',
    url='https://github.com/Libensemble/libensemble',
    author='Jeffrey Larson',
    author_email='libensemble@lists.mcs.anl.gov',
    license='BSD 2-clause',
    packages=['libensemble'],
    package_dir={'libensemble'  : 'code/src'},

    install_requires=['mpi4py>=2.0',
                      'numpy',
                      'scipy',                      
                      #'petsc>=3.5',
                      'petsc4py>=3.5'
                      ],

    #If run tests through setup.py - downloads these but does not install
    tests_require=['pytest>=3.1',
                   'pytest-cov>=2.5',
                   'pytest-pep8>=1.0',
                   'tox>=2.7'
                  ],
        
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',   
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',         
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'  
    ],

    cmdclass = {'test': Run_TestSuite,
                'tox': ToxTest}
)
