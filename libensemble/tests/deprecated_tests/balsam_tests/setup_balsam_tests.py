#!/usr/bin/env python

""" Script to set up apps and jobs for balsam tests """
# Note: To see use of command line interface see bash_scripts/setup_balsam_tests.sh script.
#       Currently that script does not create deps between jobs so may run simultaneously
#       This script tests setup within python (could in theory be integrated with job!)

import os
import sys
import subprocess

import balsam.launcher.dag as dag
from balsam.service import models

AppDef = models.ApplicationDefinition


# Ok so more low level - but can interface app stuff in python directly
def add_app(name, exepath, desc):
    """Add application to database"""
    app = AppDef()
    app.name = name
    app.executable = exepath  # “/full/path/to/python/interpreter /full/path/to/script.py"
    app.description = desc
    # app.default_preprocess = '' # optional
    # app.default_postprocess = '' # optional
    app.save()


# As balsam req python 3.6 lets use subprocess.run
# For any stuff requiring CLI
def run_cmd(cmd, echo=False):
    """Run a bash command"""
    if echo:
        print(f"\nRunning {cmd} ...\n")
    try:
        subprocess.run(cmd.split(), check=True)
    except Exception as e:
        print(e)
        raise (f"Error: Command {cmd} failed to run")


# Use relative paths to balsam_tests dir
work_dir = os.path.dirname(os.path.abspath(__file__))

sim_input_dir = os.path.abspath("../../examples/sim_funcs")
sim_app = "helloworld.py"

# For host code
num_nodes = 1
procs_per_node = 4

job_list = ["test_balsam_1__runjobs.py", "test_balsam_2__workerkill.py", "test_balsam_3__managerkill.py"]


# Currently think only CLI interface for this stuff??
# Check for empty database if poss
run_cmd("balsam rm apps --all", True)
run_cmd("balsam rm jobs --all", True)

# Add user apps - eg helloworld.py
sim_app_name = os.path.splitext(sim_app)[0]  # rm .py extension
sim_app_path = os.path.join(sim_input_dir, sim_app)  # Full path
sim_app_desc = "Run " + sim_app_name
run_line = sys.executable + " " + sim_app_path
add_app(sim_app_name, run_line, sim_app_desc)


# Add test jobs apps and jobs - and set to run one at a time
prev_job_name = None

for job in job_list:

    app_name = os.path.splitext(job)[0]
    app_path = os.path.join(work_dir, job)
    app_desc = "Run " + app_name
    run_line = sys.executable + " " + app_path
    add_app(app_name, run_line, app_desc)

    job_name = "job_" + app_name
    dag.add_job(
        name=job_name,
        workflow="libe_workflow",
        application=app_name,
        num_nodes=num_nodes,
        procs_per_node=procs_per_node,
        stage_out_url="local:" + work_dir,
        stage_out_files=job_name + ".out",
    )

    # Add dependency between jobs so run one at a time.
    if prev_job_name:
        BalsamJob = dag.BalsamJob
        parent = BalsamJob.objects.get(name=prev_job_name)
        child = BalsamJob.objects.get(name=job_name)
        dag.add_dependency(parent, child)

    prev_job_name = job_name


# Check how to do in API - until then use CLI
run_cmd("balsam ls apps", True)
run_cmd("balsam ls jobs", True)

print("")
run_cmd("echo -e To launch jobs run: balsam launcher --consume-all")
print("")
