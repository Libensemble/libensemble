# Test of executor running multiple tasks for libensemble. Could support
# hybrid mode - including, eg. running multi tasks per node (launched locally),
# or simply sharing burden on central system/consecutive pipes to balsam
# database - could enable use of threads if supply run-directories rather than
# assuming in-place runs etc....

# Test does not require running full libensemble
import os
from libensemble.executors.executor import Executor


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simtask.x my_simtask.f90' # On cray need to use ftn
    buildstring = "mpicc -o my_simtask.x simdir/my_simtask.c"
    # subprocess.run(buildstring.split(),check=True) # Python3.5+
    subprocess.check_call(buildstring.split())


# --------------- Calling script ---------------------------------------------------------------
# sim_app = 'simdir/my_simtask.x'
# gen_app = 'gendir/my_genjob.x'

# temp
sim_app = "./my_simtask.x"

if not os.path.isfile(sim_app):
    build_simfunc()

USE_BALSAM = False  # Take as arg
# USE_BALSAM = True # Take as arg

# Create and add exes to registry
if USE_BALSAM:
    from libensemble.baslam_executor import LegacyBalsamMPIExecutor

    exctr = LegacyBalsamMPIExecutor()
else:
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()

exctr.register_app(full_path=sim_app, calc_type="sim")

# Alternative to IF could be using eg. fstring to specify: e.g:
# EXECUTOR = 'Balsam'
# registry = f"{EXECUTOR}Register()"

# --------------- Worker: sim func -------------------------------------------------------------
# Should work with Balsam or not

# Can also use an internal iterable list of tasks in EXECUTOR - along with all_done func etc...


def polling_loop(exctr, task_list, timeout_sec=40.0, delay=1.0):
    import time

    start = time.time()

    while time.time() - start < timeout_sec:

        # Test all done - (return list of not-finished tasks and test if empty)
        active_list = [task for task in task_list if not task.finished]
        if not active_list:
            break

        for task in task_list:
            if not task.finished:
                time.sleep(delay)
                print(f"Polling task {task.id} at time {time.time() - start}")
                task.poll()

                if task.finished:
                    continue
                elif task.state == "WAITING":
                    print(f"Task {task.id} waiting to execute")
                elif task.state == "RUNNING":
                    print(f"Task {task.id} still running ....")

                # Check output file for error
                if task.stdout_exists():
                    if "Error" in task.read_stdout():
                        print(f"Found (deliberate) Error in output file - cancelling task {task.id}")
                        exctr.kill(task)
                        time.sleep(delay)  # Give time for kill
                        continue

                # But if I want to do something different -
                #  I want to make a file - no function for THAT!
                # But you can get all the task attributes!
                # Uncomment to test
                # path = os.path.join(task.workdir,'newfile'+str(time.time()))
                # open(path, 'a')

    print("Loop time", time.time() - start)

    for task in task_list:
        if task.finished:
            if task.state == "FINISHED":
                print(f"Task {task.id} finished successfully. Status: {task.state}")
            elif task.state == "FAILED":
                print(f"Task {task.id} failed. Status: {task.state}")
            elif task.state == "USER_KILLED":
                print(f"Task {task.id} has been killed. Status: {task.state}")
            else:
                print(f"Task {task.id} status: {task.state}")
        else:
            print(f"Task {task.id} timed out. Status: {task.state}")
            exctr.kill(task)
            if task.finished:
                print(f"Task {task.id} Now killed. Status: {task.state}")
                # double check
                task.poll()
                print(f"Task {task.id} state is {task.state}")


# Tests

# From worker call Executor by different name to ensure getting registered
# app from Executor
exctr = Executor.executor


print("\nTest 1 - 3 tasks should complete successfully with status FINISHED :\n")

task_list = []
cores = 4

for j in range(3):
    # Could allow submission to generate outfile names based on task.id
    # outfilename = 'out_' + str(j) + '.txt'
    sleeptime = 6 + j * 3  # Change args
    args_for_sim = "sleep" + " " + str(sleeptime)
    rundir = "run_" + str(sleeptime)
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task_list.append(task)


polling_loop(exctr, task_list)
