#!/usr/bin/env python

"""
Script to clean database and create top level app/job for Balsam tests

This creates only the parent job, not the sim/gen (they are created
in libEnsemble).

Run this before each run to ensure database is in correct state

Usage: ./create_balsam_job.py <script_name>
eg: ./create_balsam_job.py test_jobexecutor.py

This is equivalent to creating an app and job using <balsam app> and <balsam job> on the commdand line

To check create app and job do:
balsam ls apps
balsam ls jobs
"""

import os
import sys

import balsam.launcher.dag as dag
from balsam.service import models


def del_jobs():
    """Deletes all jobs"""
    Job = models.BalsamJob
    deletion_objs = Job.objects.all()
    deletion_objs.delete()


def add_app(name, exepath, desc):
    """Add application to database"""
    AppDef = models.ApplicationDefinition
    app = AppDef()
    app.name = name
    app.executable = exepath
    app.description = desc
    # app.default_preprocess = '' # optional
    # app.default_postprocess = '' # optional
    app.save()


# import balsam.launcher.dag as dag
stage_in = os.getcwd()

default_script = "test_jobexecutor.py"

# if script provided use that - else default
if len(sys.argv) > 1:
    script = sys.argv[1]
else:
    script = default_script

# print("script is", script)

script_basename = os.path.splitext(script)[0]  # rm .py extension
app_name = script_basename + ".app"

# Add app if its not already there
AppDef = models.ApplicationDefinition
app_exists = AppDef.objects.filter(name__contains=app_name)
if not app_exists:
    app_path = sys.executable + " " + script
    app_desc = "Test " + script
    add_app(app_name, app_path, app_desc)

# Delete existing jobs
del_jobs()

# Add the job
job = dag.add_job(
    name="job_" + script_basename,
    workflow="libe_workflow",  # add arg for this
    application=app_name,
    # application_args=job.app_args,
    num_nodes=1,
    procs_per_node=1,
    stage_in_url="local:/" + stage_in,
    stage_out_url="local:/" + stage_in,  # same as in
    stage_out_files="*.out",
)
