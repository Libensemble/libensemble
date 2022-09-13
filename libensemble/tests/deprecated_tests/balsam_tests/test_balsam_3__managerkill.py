#!/usr/bin/env python
import os  # for adding to path
import time
from mpi4py import MPI

import balsam.launcher.dag as dag


def poll_until_state(job, state, timeout_sec=120.0, delay=2.0):
    start = time.time()
    while time.time() - start < timeout_sec:
        time.sleep(delay)
        job.refresh_from_db()
        if job.state == state:
            return True
        elif job.state == "USER_KILLED":
            return False
        raise RuntimeError(f"Task {job.cute_id} failed to reach state {state} in {timeout_sec:.1f} seconds")


myrank = MPI.COMM_WORLD.Get_rank()
steps = 3
sleep_time = 3  # + myrank

# Create output dir
script_name = os.path.splitext(os.path.basename(__file__))[0]
sim_input_dir = "simdir_" + script_name.split("test_", 1).pop()
dir_path = os.path.dirname(os.path.realpath(__file__))
sim_path = os.path.join(dir_path, sim_input_dir)

if myrank == 0:
    if not os.path.isdir(sim_path):
        try:
            os.mkdir(sim_path)
        except Exception as e:
            print(e)
            raise (f"Cannot make simulation directory {sim_path}")
MPI.COMM_WORLD.Barrier()  # Ensure output dir created

print("Host job rank is %d Output dir is %s" % (myrank, sim_input_dir))

start = time.time()
for sim_id in range(steps):
    jobname = "outfile_t3_" + "for_sim_id_" + str(sim_id) + "_ranks_" + str(myrank) + ".txt"

    current_job = dag.add_job(
        name=jobname,
        workflow="libe_workflow",
        application="helloworld",
        application_args=str(sleep_time),
        num_nodes=1,
        procs_per_node=8,
        stage_out_url="local:" + sim_path,
        stage_out_files=jobname + ".out",
    )

    # Kill only from manager - pending and running jobs of given ID
    if myrank == 0:
        if sim_id == 1:
            # kill all sim_id 1 pending jobs in database
            time.sleep(0.5)

            BalsamJob = dag.BalsamJob

            # If job already finished will stage out results
            # pending_sim1_jobs =
            # BalsamJob.objects.filter(name__contains='t3_for_sim_id_1').exclude(state='JOB_FINISHED')

            # If job already finished will NOT stage out results - once classed as USER_KILLED
            pending_sim1_jobs = BalsamJob.objects.filter(name__contains="t3_for_sim_id_1")

            num_pending = pending_sim1_jobs.count()  # will only kill if already in database

            # Iterate over the jobs and kill:
            for sim in pending_sim1_jobs:
                dag.kill(sim)

            print("Number of jobs should be killed: ", num_pending)

    success = poll_until_state(current_job, "JOB_FINISHED")  # OR job killed
    if success:
        print("Completed job: %s rank=%d time=%f" % (jobname, myrank, time.time() - start))
    else:
        print(
            "Task not completed: %s rank=%d time=%f Status" % (jobname, myrank, time.time() - start), current_job.state
        )

end = time.time()
print("Done: rank=%d  time=%f" % (myrank, end - start))
