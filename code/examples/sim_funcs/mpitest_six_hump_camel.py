#!/usr/bin/env python
import sys, os             # for adding to path
from mpi4py import MPI

import balsam.launcher.dag as dag
dir_path = os.path.dirname(os.path.realpath(__file__))

myrank=MPI.COMM_WORLD.Get_rank()
print ("libe rank is",myrank)
print ("dir_path",dir_path)

jobname='helloworld_' + str(myrank)
#dag.add_job(name = jobname, workflow = "libe_workflow", application="helloworld", num_nodes=1, ranks_per_node=8)


#--url-out="local:/home/shudson/xhffi-libe/libensemble-balsam/code/examples/sim_funcs"


dag.add_job(name = jobname,
            workflow = "libe_workflow",
            application="helloworld",
            num_nodes=1,
            ranks_per_node=8,
            stage_out_url="local:" + dir_path,
            stage_out_files=jobname + ".out")


print ("done", str(myrank))
