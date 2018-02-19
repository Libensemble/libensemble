#!/usr/bin/env python
import sys, os             # for adding to path
import time
from mpi4py import MPI

import balsam.launcher.dag as dag

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def poll_until_state(job, state, timeout_sec=120.0, delay=5.0):
  start = time.time()
  while time.time() - start < timeout_sec:
    time.sleep(delay)
    job.refresh_from_db()
    #print(f"job state {job.state} looking for state {state} at time {time.time() - start}")
    if job.state == state:
      return True
    elif job.state == 'USER_KILLED':
      return False 
  raise RuntimeError(f"Job {job.cute_id} failed to reach state {state} in {timeout_sec} seconds")


dir_path = os.path.dirname(os.path.realpath(__file__))
myrank=MPI.COMM_WORLD.Get_rank()
steps=3
sleep_time = 5 #+ myrank

print ("libe rank is",myrank)
print ("dir_path",dir_path)


start = time.time()
for sim_id in range(steps):
  jobname = 'outfile_' + 'for_sim_id_' + str(sim_id)  + '_ranks_' + str(myrank) + '.txt'
 
  current_job = dag.add_job(name = jobname,
                            workflow = "libe_workflow",
                            application="helloworld",
                            application_args=sleep_time,
                            num_nodes=1,
                            ranks_per_node=8,
                            stage_out_url="local:" + dir_path,
                            stage_out_files=jobname + ".out")
    
  #Kill only from manager - pending and running jobs of given ID
  if myrank == 0:
 
    if sim_id == 1:
      #kill all sim_id 1 pending jobs in database

      BalsamJob = dag.BalsamJob
      
      #If job already finished will stage out results
      #pending_sim1_jobs = BalsamJob.objects.filter(name__contains='sim_id_1').exclude(state='JOB_FINISHED')
      
      #If job already finished will NOT stage out results - once classed as USER_KILLED
      pending_sim1_jobs = BalsamJob.objects.filter(name__contains='sim_id_1')

      num_pending = pending_sim1_jobs.count() #will only kill if already in database

      #Iterate over the jobs and kill:
      for sim in pending_sim1_jobs:
        dag.kill(sim)

      print("Number of jobs should be killed: ",num_pending)
      
  success = poll_until_state(current_job, 'JOB_FINISHED') #OR job killed
  if success:
    print ("Completed job: %s rank=%d  time=%f" % (jobname,myrank,time.time()-start))
  else:
    print ("Job not completed: %s rank=%d  time=%f Status" % (jobname,myrank,time.time()-start),current_job.state)

end = time.time()
print ("Done: rank=%d  time=%f" % (myrank,end-start))

#test that future job is not killed
#if myrank == 3:
  #sim_id == 1
  #jobname = 'outfile_TESTKILL_' + 'for_sim_id_' + str(sim_id)  + '_ranks_' + str(myrank) + '.txt'
 
  #current_job = dag.add_job(name = jobname,
                            #workflow = "libe_workflow",
                            #application="helloworld",
                            #application_args=sleep_time,
                            #num_nodes=1,
                            #ranks_per_node=8,
                            #stage_out_url="local:" + dir_path,
                            #stage_out_files=jobname + ".out")

  #success = poll_until_state(current_job, 'JOB_FINISHED') #OR job killed
  #if success:
    #print ("Completed job: %s rank=%d  time=%f" % (jobname,myrank,time.time()-start))
  #else:
    #print ("Job not completed: %s rank=%d  time=%f Status" % (jobname,myrank,time.time()-start),current_job.state)

