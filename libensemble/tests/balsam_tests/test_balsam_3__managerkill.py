#!/usr/bin/env python
import os             # for adding to path
import time
from mpi4py import MPI

import balsam.launcher.dag as dag


def poll_until_state(task, state, timeout_sec=120.0, delay=2.0):
    start = time.time()
    while time.time() - start < timeout_sec:
        time.sleep(delay)
        task.refresh_from_db()
        if task.state == state:
            return True
        elif task.state == 'USER_KILLED':
            return False
        raise RuntimeError("Task %s failed to reach state %s in %.1f seconds" % (task.cute_id, state, timeout_sec))


myrank = MPI.COMM_WORLD.Get_rank()
steps = 3
sleep_time = 3  # + myrank

# Create output dir
script_name = os.path.splitext(os.path.basename(__file__))[0]
sim_input_dir = 'simdir_' + script_name.split("test_", 1).pop()
dir_path = os.path.dirname(os.path.realpath(__file__))
sim_path = os.path.join(dir_path, sim_input_dir)

if myrank == 0:
    if not os.path.isdir(sim_path):
        try:
            os.mkdir(sim_path)
        except Exception as e:
            print(e)
            raise("Cannot make simulation directory %s" % sim_path)
MPI.COMM_WORLD.Barrier()  # Ensure output dir created

print("Host task rank is %d Output dir is %s" % (myrank, sim_input_dir))

start = time.time()
for sim_id in range(steps):
    taskname = 'outfile_t3_' + 'for_sim_id_' + str(sim_id) + '_ranks_' + str(myrank) + '.txt'

    current_task = dag.add_task(name=taskname,
                              workflow="libe_workflow",
                              application="helloworld",
                              application_args=str(sleep_time),
                              num_nodes=1,
                              ranks_per_node=8,
                              stage_out_url="local:" + sim_path,
                              stage_out_files=taskname + ".out")

    # Kill only from manager - pending and running tasks of given ID
    if myrank == 0:
        if sim_id == 1:
            # kill all sim_id 1 pending tasks in database
            time.sleep(0.5)

            BalsamTask = dag.BalsamTask

            # If task already finished will stage out results
            # pending_sim1_tasks = BalsamTask.objects.filter(name__contains='t3_for_sim_id_1').exclude(state='TASK_FINISHED')

            # If task already finished will NOT stage out results - once classed as USER_KILLED
            pending_sim1_tasks = BalsamTask.objects.filter(name__contains='t3_for_sim_id_1')

            num_pending = pending_sim1_tasks.count()  # will only kill if already in database

            # Iterate over the tasks and kill:
            for sim in pending_sim1_tasks:
                dag.kill(sim)

            print("Number of tasks should be killed: ", num_pending)

    success = poll_until_state(current_task, 'TASK_FINISHED')  # OR task killed
    if success:
        print("Completed task: %s rank=%d  time=%f" % (taskname, myrank, time.time()-start))
    else:
        print("Task not completed: %s rank=%d  time=%f Status" % (taskname, myrank, time.time()-start), current_task.state)

end = time.time()
print("Done: rank=%d  time=%f" % (myrank, end-start))
