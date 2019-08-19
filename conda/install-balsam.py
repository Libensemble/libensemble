#! /usr/bin/env python

# Installing Balsam consists of both cloning and pip installing the latest version
#   of Balsam from source but also moving the Balsam-relevant test to the regression_tests
#   directory. This way, run-tests won't run a non-relevant test if Balsam isn't
#   installed or the necessary Python version isn't installed.

import sys
import os
import subprocess

balsamclone = 'git clone https://github.com/balsam-alcf/balsam.git ../balsam'

if int(sys.version[2]) >= 6: # Balsam only supports Python 3.6+
    subprocess.check_call(balsamclone.split())
    os.chdir('../balsam')
    subprocess.check_call('pip install -e .'.split())
    os.chdir('../libensemble')
    if not os.path.isfile('./libensemble/tests/regression_tests/test_balsam.py'):
        os.rename('./conda/test_balsam.py',
                  './libensemble/tests/regression_tests/test_balsam.py')
    # os.system('./conda/configure-balsam-test.sh')
    subprocess.run('./conda/configure-balsam-test.sh'.split())
