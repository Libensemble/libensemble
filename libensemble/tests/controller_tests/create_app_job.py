#!/usr/bin/env python
import os

def del_jobs():
    """ Deletes all jobs whose names contains substring .simfunc or .genfunc"""
    #""" For now just deletes all jobs """
    import balsam.launcher.dag as dag
    from balsam.service import models
    Job = models.BalsamJob
    deletion_objs = Job.objects.all()
    deletion_objs.delete()


import balsam.launcher.dag as dag

#stage_in = '/home/shudson/libensemble/rsync-to-clusters/opal_work/opal/libensemble/libensemble/tests/controller_tests/balsam_test_bk'

stage_in = os.getcwd()
#stage_in = 'my_simjob.x'

job = dag.add_job(name = 'test_jobcontroller',
                  workflow = "libe_workflow", #add arg for this
                  application = 'test',
                  #application_args = job.app_args,           
                  num_nodes = 1,
                  ranks_per_node = 1,
                  stage_in_url="local:/" + stage_in,
                  stage_out_url = "local:/" + stage_in, #same as in
                  stage_out_files = "*.out")   
