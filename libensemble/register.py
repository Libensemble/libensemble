#!/usr/bin/env python

""" Script to set up apps for balsam """

#Quit possibly make this part of job/job_controller module - and when job is run - it will
#for example get the sim_calc from here. In fact I'm almost sure thats where it should go now.
#Or I guess an app object could be an attribute of job - so job relates to this app - now I'm
#almost sure about that - thats how fickle I am!

import os
import sys
import subprocess

import balsam.launcher.dag as dag
from balsam.service import models
AppDef = models.ApplicationDefinition

def add_app(name,exepath,desc):
    """ Add application to database """
    app = AppDef()
    app.name = name
    app.executable = exepath    # “/full/path/to/python/interpreter /full/path/to/script.py" 
    app.description = desc
    #app.default_preprocess = '' # optional
    #app.default_postprocess = '' # optional
    app.save()

#As balsam req python 3.6 lets use subprocess.run
#For any stuff requiring CLI
def run_cmd(cmd,echo=False):
    """ Run a bash command """
    if echo:
        print("\nRunning %s ...\n" % cmd)
    try:
        subprocess.run(cmd.split(),check=True)
    except:
        raise("Error: Command %s failed to run" % cmd)


def register_calc(full_path, calc_type='sim', desc=None):    
    calc_dir, calc_exe = os.path.split(full_path)    
    calc_name = calc_exe + '_' + calc_type
    
    if desc is None:
        desc = calc_exe + ' ' + calc_type + ' function'
        
    add_app(calc_name, full_path, desc)
    

def register_init():
    #Check for empty database if poss
    run_cmd("balsam rm apps --all", True)
    run_cmd("balsam rm jobs --all", True)


