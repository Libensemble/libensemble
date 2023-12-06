#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This is for setting up libEnsemble, license and details can be
# found at https://github.com/Libensemble/libensemble/

"""libEnsemble

libEnsemble is a Python toolkit for coordinating workflows of asynchronous
and dynamic ensembles of calculations.

libEnsemble can help users take advantage of massively parallel resources to
solve design, decision, and inference problems and expand the class of
problems that can benefit from increased parallelism.

"""

from pathlib import Path

from setuptools import setup
from setuptools.command.test import test as TestCommand

exec(open("libensemble/version.py").read())


class Run_TestSuite(TestCommand):
    def run_tests(self):
        import os
        import sys

        py_version = sys.version_info[0]
        print("Python version from setup.py is", py_version)
        run_string = "libensemble/tests/run-tests.sh -p " + str(py_version)
        os.system(run_string)


class ToxTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def run_tests(self):
        import tox

        tox.cmdline()


setup(
    name="libensemble",
    version=__version__,
    description="Library to coordinate the concurrent evaluation of dynamic ensembles of calculations",
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    url="https://github.com/Libensemble/libensemble",
    author="Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro",
    author_email="libensemble@lists.mcs.anl.gov",
    license="BSD 3-clause",
    packages=[
        "libensemble",
        "libensemble.gen_funcs",
        "libensemble.sim_funcs",
        "libensemble.sim_funcs.branin",
        "libensemble.alloc_funcs",
        "libensemble.tests",
        "libensemble.comms",
        "libensemble.utils",
        "libensemble.tools",
        "libensemble.executors",
        "libensemble.resources",
        "libensemble.tests.unit_tests",
        "libensemble.tests.regression_tests",
    ],
    install_requires=["numpy>=1.21", "psutil>=5.9.4", "pydantic>=1.10", "tomli>=1.2.1", "pyyaml>=6.0"],
    # numpy - oldest working version. psutil - oldest working version.
    # pyyaml - oldest working version.
    # If run tests through setup.py - downloads these but does not install
    tests_require=[
        "pytest>=3.1",
        "pytest-cov>=2.5",
        "pytest-pep8>=1.0",
        "pytest-timeout",
        "mock",
    ],
    extras_require={
        "docs": [
            "autodoc_pydantic",
            "sphinx<8",
            "sphinx_design",
            "sphinx_rtd_theme",
            "sphinxcontrib-bibtex",
            "sphinx-copybutton",
        ],
    },
    scripts=[
        "scripts/liberegister",
        "scripts/libesubmit",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={"test": Run_TestSuite, "tox": ToxTest},
)
