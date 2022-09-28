import subprocess
import os
import time
import libensemble
from libensemble.tests.regression_tests.common import modify_Balsam_worker, modify_Balsam_JobEnv

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3
# TESTSUITE_EXTRA: true

# This test is NOT submitted as a job to Balsam. script_test_balsam_hworld.py is
#   the executable submitted to Balsam as a job. This test executes that job
#   through the 'runstr' line in run_Balsam_job()


def run_Balsam_job():
    runstr = "balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1"
    print(f"Executing Balsam job with command: {runstr}")
    subprocess.Popen(runstr.split())


def build_simfunc():
    buildstring = "mpicc -o my_simtask.x ../unit_tests/simdir/my_simtask.c"
    subprocess.check_call(buildstring.split())


def wait_for_job_dir(basedb):
    sleeptime = 0
    limit = 15

    # Stop sleeping once database directory detected
    print("Waiting for Balsam Database directory.")
    while sleeptime < limit:
        if os.path.isdir(basedb):
            break
        time.sleep(1)
        sleeptime += 1

    assert sleeptime < limit, f"Balsam Database directory not created within {limit} seconds."

    # Stop sleeping once job directory detected within database directory
    print(f"Waiting for Job Directory {sleeptime}")
    while sleeptime < limit:
        if len(os.listdir(basedb)) > 0:
            break
        print(sleeptime, end=" ", flush=True)
        time.sleep(1)
        sleeptime += 1

    assert sleeptime < limit, f"Balsam Job directory not created within {limit} seconds."

    # Assumes database dir was empty, now contains single job dir
    jobdir = os.path.join(basedb, os.listdir(basedb)[0])
    return jobdir


def wait_for_job_output(jobdir):
    sleeptime = 0
    limit = 60

    output = os.path.join(jobdir, "job_script_test_balsam_hworld.out")
    print(f"Checking for Balsam output file: {output}")

    while sleeptime < limit:
        if os.path.isfile(output):
            break
        print(sleeptime, end=" ", flush=True)
        print(os.listdir(jobdir), flush=True)
        time.sleep(1)
        sleeptime += 1

    assert sleeptime < limit, f"Balsam output file not created within {limit} seconds."

    return output


def print_job_output(outscript):
    sleeptime = 0
    limit = 90

    print("Blank output file found. Waiting for expected complete Balsam Job Output.")
    succeed_line = "Received:  [34 34 31 31 34 34 32 32 33 33]\n"

    while sleeptime < limit:
        with open(outscript, "r") as f:
            lines = f.readlines()

        print(sleeptime, end=" ", flush=True)

        if succeed_line in lines:
            print("Success. Received task statuses match expected.")
            break

        time.sleep(1)
        sleeptime += 1

    assert sleeptime < limit, f"Expected Balsam Job output-file contents not detected after {limit} seconds."


def move_job_coverage(jobdir):
    # Move coverage files from Balsam DB to ./regression_tests (for concatenation)
    print("Moving job coverage results.")
    here = os.getcwd()
    covname = ".cov_reg_out."

    assert any(
        [file.startswith(covname) for file in os.listdir(jobdir)]
    ), "Coverage results not detected in Balsam Job directory."

    for file in os.listdir(jobdir):
        if file.startswith(covname):
            balsam_cov = os.path.join(jobdir, file)
            here_cov = os.path.join(here, file)
            os.rename(balsam_cov, here_cov)


if __name__ == "__main__":

    # Used by Balsam Coverage config file. Dont evaluate Balsam data dir
    libepath = os.path.dirname(libensemble.__file__)
    os.environ["LIBE_PATH"] = libepath
    # os.environ['BALSAM_DB_PATH'] = '~/test-balsam'

    basedb = os.environ["HOME"] + "/test-balsam/data/libe_test-balsam"

    subprocess.run("./scripts_used_by_reg_tests/configure-balsam-test.sh".split())

    if not os.path.isfile("./my_simtask.x"):
        build_simfunc()

    modify_Balsam_worker()
    modify_Balsam_JobEnv()
    run_Balsam_job()

    jobdir = wait_for_job_dir(basedb)
    output = wait_for_job_output(jobdir)
    print_job_output(output)
    move_job_coverage(jobdir)

    print("Test complete.")
    subprocess.run("./scripts_used_by_reg_tests/cleanup-balsam-test.sh".split())
