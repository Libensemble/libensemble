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
    os.chdir("../balsam/balsam-0.5.0")
    subprocess.check_call("pip install -e .".split())
    os.chdir(here)


def move_test_balsam():
    current_dir_with_btest = "./libensemble/tests/regression_tests/scripts_used_by_reg_tests/test_balsam_hworld.py"
    reg_dir_with_btest = "./libensemble/tests/regression_tests/test_balsam_hworld.py"
    if not os.path.isfile(reg_dir_with_btest):
        os.rename(current_dir_with_btest, reg_dir_with_btest)


def configure_coverage():
    # Enables coverage of balsam_controller.py if running test
    coveragerc = "./libensemble/tests/.coveragerc"
    with open(coveragerc, "r") as f:
        lines = f.readlines()

    newlines = [i for i in lines if i != "    */legacy_balsam_executor.py\n"]

    print("New libensemble/tests/.coveragerc: \n")
    with open(coveragerc, "w") as f:
        for line in newlines:
            print(line)
            f.write(line)


if int(sys.version[2]) >= 6:  # Balsam only supports Python 3.6+
    install_balsam()
    move_test_balsam()
    configure_coverage()
