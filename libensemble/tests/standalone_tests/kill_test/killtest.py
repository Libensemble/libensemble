import os
import subprocess
import sys
import time
import signal


# Does not kill, even on laptop
# kill 1
def kill_task_1(process):
    process.kill()
    process.wait()


# kill 2 - with  preexec_fn=os.setsid in subprocess
def kill_task_2(process):
    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.wait()


if len(sys.argv) != 4:
    raise Exception("Usage: python killtest.py <kill_type> <num_nodes> <num_procs_per_node>")

# user_code = "./burn_time.x"
user_code = "./sleep_and_print.x"

# sys.argv[0] is python exe.
kill_type = int(sys.argv[1])  # 1, 2
num_nodes = int(sys.argv[2])
num_procs_per_node = int(sys.argv[3])
num_procs = num_nodes * num_procs_per_node

print("Running Kill test with program", user_code)
print(f"Kill type: {kill_type}   num_nodes: {num_nodes}   procs_per_node: {num_procs_per_node}")


# Create common components of submit line (currently all of it)

# Am I in an aprun environment
launcher = "mpich"  # Includes mpich based - eg. intelmpi
try:
    subprocess.check_call(["aprun", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except OSError:
    launcher = "mpich"
else:
    launcher = "aprun"

if launcher == "mpich":
    mpicmd_launcher = "mpirun"
    mpicmd_numprocs = "-np"
    mpicmd_ppn = "-ppn"
elif launcher == "aprun":
    mpicmd_launcher = "aprun"
    mpicmd_numprocs = "-n"
    mpicmd_ppn = "-N"

# As runline common to tasks currently - construct here.
runline = []  # E.g: 2 nodes run
runline.append(mpicmd_launcher)  # mpirun
runline.append(mpicmd_numprocs)  # mpirun -np
runline.append(str(num_procs))  # mpirun -np 8
runline.append(mpicmd_ppn)  # mpirun -np 8 --ppn
runline.append(str(num_procs_per_node))  # mpirun -np 8 --ppn 4
runline.append(user_code)  # mpirun -np 8 --ppn 4 ./burn_time.x


# print("Running killtest.py with task size {} procs".format(num_procs))
total_start_time = time.time()

for run_num in range(2):
    time.sleep(4)  # Show gap where none should be running
    stdout = "out_" + str(run_num) + ".txt"
    # runline = ['mpirun', '-np', str(num_procs), user_code]
    print("---------------------------------------------------------------")
    print(f"\nRun num: {run_num}   Runline: {' '.join(runline)}\n")

    if kill_type == 1:
        process = subprocess.Popen(runline, cwd="./", stdout=open(stdout, "w"), shell=False)  # with kill 1
    elif kill_type == 2:
        process = subprocess.Popen(
            runline, cwd="./", stdout=open(stdout, "w"), shell=False, preexec_fn=os.setsid
        )  # kill 2
    else:
        raise Exception("kill_type not recognized")

    time_limit = 4
    start_time = time.time()
    finished = False
    state = "Not started"
    while not finished:

        time.sleep(2)
        poll = process.poll()
        if poll is None:
            state = "RUNNING"
            print("Running....")

        else:
            finished = True
            if process.returncode == 0:
                state = "PASSED"
            else:
                state = "FAILED"

        if time.time() - start_time > time_limit:
            print("Killing task", run_num)
            # kill_task(process, user_code)

            if kill_type == 1:
                kill_task_1(process)
            elif kill_type == 2:
                kill_task_2(process)
            state = "KILLED"
            finished = True

    # Assert task killed
    assert state == "KILLED", "Task not registering as killed. State is: " + state

    # Checking if processes still running and producing output
    grace_period = 1  # Seconds after kill when first read last line
    recheck_period = 2  # Recheck last line after this many seconds
    num_rechecks = 2  # Number of times to check for new output

    time.sleep(grace_period)  # Give chance to kill

    # Test if task is still producing output
    with open(stdout, "rb") as fh:
        line_on_kill = fh.readlines()[-1].decode().rstrip()
    print(f"Last line after task kill:  {line_on_kill}")

    if "has finished" in line_on_kill:
        raise Exception("Task may have already finished - test invalid")

    for recheck in range(1, num_rechecks + 1):
        time.sleep(recheck_period)
        with open(stdout, "rb") as fh:
            lastline = fh.readlines()[-1].decode().rstrip()
        print(f"Last line after {recheck_period * recheck} seconds: {lastline}")

        if lastline != line_on_kill:
            print(f"Task {run_num} still producing output")
            # print("Last line check 1:", line_on_kill)
            # print("Last line check 2:", lastline)
            assert 0

total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"\nTask kill test completed in {total_time} seconds\n")
