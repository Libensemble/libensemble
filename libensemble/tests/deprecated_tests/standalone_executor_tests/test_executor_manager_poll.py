#!/usr/bin/env python

# ********************* NOT YET IMPLEMENTED ***********************
# ********************* Interface demo: see exctr.manager_poll(task)

# Test of executor module for libensemble:
#  Detecting manager kill signal
#  This keeps MPI out of user code and is portable across different
#  worker concurrency schemes (MPI/threading/multiprocessing)

# Test does not require running full libensemble

import os
from libensemble.executors.executor import Executor
from libensemble.message_numbers import MAN_SIGNAL_KILL


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simtask.x my_simtask.f90' # On cray need to use ftn
    buildstring = "mpicc -o my_simtask.x simdir/my_simtask.c"
    # subprocess.run(buildstring.split(),check=True) # Python3.5+
    subprocess.check_call(buildstring.split())


# --------------- Calling script ------------------------------------------

# sim_app = 'simdir/my_simtask.x'
# gen_app = 'gendir/my_gentask.x'

# temp
sim_app = "./my_simtask.x"

if not os.path.isfile(sim_app):
    build_simfunc()

USE_BALSAM = False  # Take as arg
# USE_BALSAM = True # Take as arg

# Create and add exes to registry
if USE_BALSAM:
    from libensemble.executors.balsam_executors import LegacyBalsamMPIExecutor

    exctr = LegacyBalsamMPIExecutor()
else:
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()

exctr.register_app(full_path=sim_app, calc_type="sim")

# Alternative to IF could be using eg. fstring to specify: e.g:
# EXECUTOR = 'Balsam'
# registry = f"{EXECUTOR}Register()"


# --------------- Worker: sim func ----------------------------------------
# Should work with Balsam or not


def polling_loop(exctr, task, timeout_sec=20.0, delay=2.0):
    import time

    start = time.time()

    while time.time() - start < timeout_sec:

        exctr.manager_poll(task)

        if task.manager_signal == MAN_SIGNAL_KILL:
            print("Manager has sent kill signal - killing task")
            exctr.kill(task)

        # In future might support other manager signals eg:
        # elif task.manager_signal == "pause":
        #     checkpoint_task()
        #     pass

        time.sleep(delay)
        print("Polling at time", time.time() - start)
        task.poll()
        if task.finished:
            break
        elif task.state == "WAITING":
            print("Task waiting to execute")
        elif task.state == "RUNNING":
            print("Task still running ....")

    if task.finished:
        if task.state == "FINISHED":
            print("Task finished successfully. Status:", task.state)
        elif task.state == "FAILED":
            print("Task failed. Status:", task.state)
        elif task.state == "USER_KILLED":
            print("Task has been killed. Status:", task.state)
        else:
            print("Task status:", task.state)
    else:
        print("Task timed out")
        exctr.kill(task)
        if task.finished:
            print("Now killed")
            # double check
            task.poll()
            print("Task state is", task.state)


# Tests
# ********************* NOT YET IMPLEMENTED ***********************

# From worker call Executor by different name to ensure getting registered app from Executor
exctr = Executor.executor

print("\nTest 1 - should complete successfully with status FINISHED :\n")
cores = 4
args_for_sim = "sleep 5"

task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
polling_loop(exctr, task)

print("\nTest 2 - Task should be MANAGER_KILLED \n")
cores = 4
args_for_sim = "sleep 5"

task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
polling_loop(exctr, task)
