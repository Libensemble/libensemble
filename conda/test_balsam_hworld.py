import subprocess
import os
import time
import sys
import libensemble
from libensemble.tests.regression_tests.common import modify_Balsam_worker

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3

# This test is NOT submitted as a job to Balsam. Instead, script_test_balsam_hworld.py
#   This test executes that job through the 'runstr' line in run_Balsam_job()


def run_Balsam_job():
    runstr = 'balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1'
    print('Executing Balsam job with command: {}'.format(runstr))
    subprocess.Popen(runstr.split())


def wait_for_job_dir(basedb):
    sleeptime = 0

    while not os.path.isdir(basedb) and sleeptime < 15:
        time.sleep(1)
        sleeptime += 1

    print('Waiting for Job Directory'.format(sleeptime))
    while len(os.listdir(basedb)) == 0 and sleeptime < 15:
        print('{}'.format(sleeptime), end=" ")
        sys.stdout.flush()
        time.sleep(1)
        sleeptime += 1

    jobdirname = os.listdir(basedb)[0]
    jobdir = os.path.join(basedb, jobdirname)
    return jobdir


def wait_for_job_output(jobdir):
    sleeptime = 0

    output = os.path.join(jobdir, 'job_script_test_balsam_hworld.out')
    print('Checking for Balsam output file: {}'.format(output))

    while not os.path.isfile(output) and sleeptime < 30:
        print('{}'.format(sleeptime), end=" ")
        sys.stdout.flush()
        time.sleep(2)
        sleeptime += 2

    return output


def print_job_output(outscript):
    sleeptime = 0

    print('Output file found. Waiting for complete Balsam Job Output.')
    lastlines = ['Job 4 done on worker 1\n', 'Job 4 done on worker 2\n']
    lastposition = 0

    while sleeptime < 60:
        with open(outscript, 'r') as f:
            f.seek(lastposition)
            new = f.read()
            lastposition = f.tell()

        if len(new) > 0:
            print(new)
        else:
            print('{}'.format(sleeptime), end=" ")
        sys.stdout.flush()

        if any(new.endswith(line) for line in lastlines):
            break

        time.sleep(1)
        sleeptime += 1


def move_job_coverage(jobdir):
    # Move coverage files from Balsam DB to ./regression_tests (for concatenation)
    here = os.getcwd()
    covname = '.cov_reg_out.'

    for file in os.listdir(jobdir):
        if file.startswith(covname):
            balsam_cov = os.path.join(jobdir, file)
            here_cov = os.path.join(here, file)
            os.rename(balsam_cov, here_cov)


if __name__ == '__main__':

    # For Balsam-specific Coverage config file, to not evaluate Balsam data dir
    libepath = os.path.dirname(libensemble.__file__)
    os.environ['LIBE_PATH'] = libepath
    home = os.environ['HOME']

    basedb = home + '/test-balsam/data/libe_test-balsam'

    modify_Balsam_worker()
    run_Balsam_job()

    jobdir = wait_for_job_dir(basedb)
    output = wait_for_job_output(jobdir)
    print_job_output(output)
    move_job_coverage(jobdir)

    print('Test complete.')
