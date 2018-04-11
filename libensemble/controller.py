#!/usr/bin/env python

""" Script to launch and control running jobs """

import os
import subprocess
import logging
from libensemble.register import Register

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

class JobControllerException(Exception): pass

class JobController:

    controller = None
    
    def __init__(self, registry=None):
        
        if registry is None:
            self.registry = Register.default_registry
        else:
            self.registry = registry
        
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")
        
        JobController.controller = self
        
        #Configured possiby by a launcher abstract class/subclasses for launcher type - based on autodetection
        #currently hardcode here
        self.mpi_launcher = 'mpirun'
        self.mfile = '-machinefile'
        self.nprocs = '-np'
        self.nnodes = ''
        self.ppn = '--ppn'
        
        #If this could share multiple launches could set default job parameters here (nodes/ranks etc...)
    
    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, app_args=None, stdout=None, test=False):
        
        #Could take optional app arg - if they want to supply here - instead of taking from registry
        #Here could be options to specify an alternative function - else registry.sim_default_app

        if calc_type == 'sim':
            if self.registry.sim_default_app is None:
                raise JobControllerException("Default sim app is not set")
            else:
                app = self.registry.sim_default_app
        elif calc_type == 'gen':
            if self.registry.gen_default_app is not None:
                raise JobControllerException("Default gen app is not set")
            else:
                app = self.registry.gen_default_app
        else:
            raise JobControllerException("Unrecognized calculation type", calc_type)

        
        if num_procs is None:
            if num_nodes is None or ranks_per_node is None:
                raise JobControllerException("Must set either num_procs or num_nodes and ranks_per_node")
        
        #if supply -np and ppn may need to calculate
        
        #May use job here from worker... whether to keep that a sep thing.
        #Also could register default job setups.
        
        #Construct run line - poss subroutine
        runline = []
        runline.append(self.mpi_launcher)
        
        if machinefile is not None:
            runline.append(self.mfile)
            runline.append(machinefile)
        if num_procs is not None:
            runline.append(self.nprocs)
            runline.append(str(num_procs))
        if num_nodes is not None:
            runline.append(self.nnodes)
            runline.append(str(num_nodes))
            
        runline.append(app.full_path)
        if app_args is not None:
            app_args_list = app_args.split()
            for iarg in app_args_list:
                runline.append(iarg)      
        
        if test:
            print('runline args are', runline)
            print('stdout to', stdout)
            #logger.info(runline)
        else:
            #pass #Actual line
            
            #I'm not sure about some of these options - this would be the format of original opal line - did not work theta
            #p = subprocess.Popen(runline, cwd='./', stdout = open('out.txt','w'), shell=False, preexec_fn=os.setsid)
            
            #This was on theta - still dont think need cwd option
            if stdout is None:
                p = subprocess.Popen(runline, cwd='./', shell=False)
            else:
                p = subprocess.Popen(runline, cwd='./', stdout = open(stdout,'w'), shell=False)

    
    def kill(self):
        pass
    
    def poll(self):
        pass
    
    
        
