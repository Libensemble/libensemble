#!/usr/bin/env python

"""
Script to clean database and create top level app/task for Balsam tests

This creates only the parent task, not the sim/gen (they are created
in libEnsemble).

Run this before each run to ensure database is in correct state

Usage: ./create_balsam_task.py <script_name>
eg: ./create_balsam_task.py test_taskexecutor.py

This is equivalent to creating an app and task using <balsam app> and <balsam task> on the commdand line

To check create app and task do:
balsam ls apps
balsam ls tasks
"""

import os
import sys

import balsam.launcher.dag as dag
from balsam.service import models


def del_tasks():
    """ Deletes all tasks """
    Task = models.BalsamTask
    deletion_objs = Task.objects.all()
    deletion_objs.delete()


def add_app(name, exepath, desc):
    """ Add application to database """
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

default_script = 'test_taskexecutor.py'

# if script provided use that - else default
if len(sys.argv) > 1:
    script = sys.argv[1]
else:
    script = default_script

# print("script is", script)

script_basename = os.path.splitext(script)[0]  # rm .py extension
app_name = script_basename + '.app'

# Add app if its not already there
AppDef = models.ApplicationDefinition
app_exists = AppDef.objects.filter(name__contains=app_name)
if not app_exists:
    app_path = sys.executable + ' ' + script
    app_desc = 'Test ' + script
    add_app(app_name, app_path, app_desc)

# Delete existing tasks
del_tasks()

# Add the task
task = dag.add_task(name='task_' + script_basename,
                  workflow="libe_workflow",  # add arg for this
                  application=app_name,
                  # application_args=task.app_args,
                  num_nodes=1,
                  ranks_per_node=1,
                  stage_in_url="local:/" + stage_in,
                  stage_out_url="local:/" + stage_in,  # same as in
                  stage_out_files="*.out")
