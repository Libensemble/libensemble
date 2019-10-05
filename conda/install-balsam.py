#! /usr/bin/env python

# Configuring Balsam for libEnsemble consists of installing the latest version
#   of Balsam from source and moving Balsam-relevant tests to the regression_tests
#   directory. This way, run-tests won't run a non-relevant test if Balsam or the
#   necessary Python version aren't installed.

import sys
import os
import subprocess


def install_balsam():
    # Installs Balsam in a directory on the same level as the current directory.
    here = os.getcwd()
    balsamclone = 'git clone https://github.com/balsam-alcf/balsam.git ../balsam'
    subprocess.check_call(balsamclone.split())
    os.chdir('../balsam')
    subprocess.check_call('pip install -e .'.split())
    os.chdir(here)


def move_test_balsam(balsam_test):
    # Moves specified test from /conda to /regression_tests
    reg_dir_with_btest = os.path.join('./libensemble/tests/regression_tests', balsam_test)
    if not os.path.isfile(reg_dir_with_btest):
        os.rename('./conda/{}'.format(balsam_test), reg_dir_with_btest)


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
