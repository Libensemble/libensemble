import sys
import os
import subprocess

balsamclone = 'git clone https://github.com/balsam-alcf/balsam.git ../balsam'

if sys.version[2] >= 6:
    subprocess.check_call(balsamclone.split())
    os.chdir('../balsam')
    subprocess.check_call('pip install -e .'.split())
    os.chdir('../libensemble')
    os.rename('./conda/test_balsam.py', 
              './libensemble/tests/regression_tests/test_balsam.py')
    subprocess.check_call('./conda/configure-balsam-test.sh')
