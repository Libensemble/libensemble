#! /usr/bin/env python

# Configuring Balsam for libEnsemble consists of installing the latest version
#   of Balsam from source and moving Balsam-relevant tests to the regression_tests
#   directory. This way, run-tests won't run a non-relevant test if Balsam or the
#   necessary Python version aren't installed.

import sys
import os
import subprocess


def install_balsam():
    here = os.getcwd()
    os.chdir('../balsam/balsam-0.3.5.1/balsam/scripts')

    # Replace old version of balsamactivate
    os.remove('balsamactivate')
    subprocess.check_call('wget https://raw.githubusercontent.com/balsam-alcf/balsam/master/balsam/scripts/balsamactivate'.split())

    # Pip install Balsam
    os.chdir('../..')
    subprocess.check_call('pip install -e .'.split())

    os.chdir(here)


def move_test_balsam(balsam_test):
    # Moves specified test from /conda to /regression_tests
    reg_dir_with_btest = './libensemble/tests/regression_tests/' + balsam_test
    if not os.path.isfile(reg_dir_with_btest):
        os.rename('./conda/' + balsam_test, reg_dir_with_btest)


def configure_coverage():
    # Enables coverage of balsam_controller.py if running test
    coveragerc = './libensemble/tests/.coveragerc'
    with open(coveragerc, 'r') as f:
        lines = f.readlines()

    newlines = [i for i in lines if i != '    */balsam_controller.py\n']

    with open(coveragerc, 'w') as f:
        for line in newlines:
            f.write(line)


if int(sys.version[2]) >= 6:  # Balsam only supports Python 3.6+
    install_balsam()
    move_test_balsam('test_balsam_hworld.py')
    configure_coverage()
    subprocess.run('./conda/configure-balsam-test.sh'.split())
