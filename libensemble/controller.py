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

STATES = '''
UNKNOWN
RUNNING
FINISHED
KILLED
FAILED'''.split()
#Others may include FAILED, PAUSED...

#IMPORTANT: Need to determine if jobcontroller controls a single job - or can be re-used
#If to be re-used then needs not just an __init__ for controller but a refresh for a new job launch.
#If init controller in calling script - it will need to be latter.
#Either need reset or jobs as sub-objects.

class JobControllerException(Exception): pass

class JobController:

    controller = None

    #Currently not using sub-job so reset job attributes - as opposed to job_controller attributes
    def reset(self):
        #This may be placed in a job object (and maybe a list of jobs for controller)
        #job will have ID that can be used
        self.current_process_id = None
        self.state = 'UNKNOWN' #Create as string or integer macro?
        self.errcode = None
        self.finished = False # True means job has run - not whether was successful
        self.success = False        
    
    def __init__(self, registry=None):
        
        if registry is None:
            self.registry = Register.default_registry #Error handling req.
        else:
            self.registry = registry
        
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")
        
        #Configured possiby by a launcher abstract class/subclasses for launcher type - based on autodetection
        #currently hardcode here
        self.mpi_launcher = 'mpirun'
        self.mfile = '-machinefile'
        self.nprocs = '-np'
        self.nnodes = ''
        self.ppn = '--ppn'
        
        #Job controller settings - can be set in user function.
        self.kill_signal = SIGTERM
        self.wait_and_kill = True #If true - wait for wait_time After signal and then kill with SIGKILL
        self.wait_time = 60
        
        #Reset current job attributes
        self.reset()
        
        JobController.controller = self
        
        #If this could share multiple launches could set default job parameters here (nodes/ranks etc...)
        
    
    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, app_args=None, stdout=None, test=False):
     
        self.reset()        
        
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
            #I'm not sure about some of these options - this would be the format of original opal line - did not work theta
            #p = subprocess.Popen(runline, cwd='./', stdout = open('out.txt','w'), shell=False, preexec_fn=os.setsid)
            
            #testing - tmp - use logger.debug
            print('runline is: %s' % " ".join(runline))
            
            #This was on theta - still dont think need cwd option
            if stdout is None:
                self.current_process_id = subprocess.Popen(runline, cwd='./', shell=False)
            else:
                self.current_process_id = subprocess.Popen(runline, cwd='./', stdout = open(stdout,'w'), shell=False)
        
        #Could return self.current_process_id - or some independent job/process ID - which will always have common format.
        #tmp use actual process id
        #return self.current_process_id
        #Mayb just return if successfully launched. Process ID can be queried.
    
    def poll(self):
        
        #Check the jobs been launched (i.e. it has a process ID)
        if self.current_process_id is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job has no process ID - check jobs been launched')
        
        #Here question of checking the existing state before polling - some error handling is required
        if self.state == KILLED:
            logger.warning('Polled job has already been killed') #could poll to check....
            return self.state
        if self.state == FAILED:
            logger.warning('Polled job has already been set to failed') #could poll to check....
            return self.state
        if self.state == FINISHED:
            logger.warning('Polled job has already been set to finished') #could poll to check....
            return self.state   
        
        # Poll the job
        poll = self.current_process_id.poll()
        if poll is None:
            self.state = 'RUNNING'
        else:
            self.finished = True
            #logger.debug("Process {} Completed".format(self.current_process_id))
            
            if self.current_process_id.returncode == 0:
                self.success = True
                self.errcode = 0
                logger.debug("Process {} completed successfully".format(self.current_process_id))
                self.state = 'FINISHED'
            else:
                #Need to differentiate failure from user killed !!!!!
                #Currently FAILED MEANS BOTH
                self.errcode = self.current_process_id.returncode
                self.state = 'FAILED'
        
        return self.state
                
    
    def kill(self):
        #In here can set state to user killed!
        #- but if killed by remote job (eg. through balsam database) may be different ....
        #maybe have a function JobController.set_kill_mode() 

        #Issue signal
        if self.kill_signal == SIGTERM:
            self.current_process_id.terminate()
        elif self.kill_signal == SIGKILL:
            self.current_process_id.terminate()
        else:
            self.current_process_id.send_signal(self.kill_signal) #Prob only need this line!
            
        #Wait for job to be killed
        if self.wait_and_kill:
            try:
                self.current_process_id.wait(timeout=self.wait_time)
                #stdout,stderr = self.current_process_id.communicate(timeout=self.wait_time) #Wait for process to finish
            except subprocess.TimeoutExpired:
                logger.warning("Kill signal {} timed out - issuing SIGKILL".format(self.kill_signal))
                self.current_process_id.kill()
                self.current_process_id.wait()
        else:
            self.current_process_id.wait(timeout=self.wait_time)

        self.state = 'KILLED'
        self.finished = True
        
        #Need to test out what to do with
        #self.errcode #Can it be discovered after killing?
        #self.success #Could set to false but should be already - only set to true on success            
                
    def set_kill_mode(self, signal=None, wait_and_kill=None, wait_time=None):        
        if signal is not None:
            self.kill_signal = signal
            
        if wait_and_kill is not None:
            self.wait_and_kill = wait_and_kill
            
        if wait_time is not None: 
            self.wait_time = wait_time

        
class BalsamJobController(JobController):

    def reset(self):       
        pass
    
    def __init__(self):
        pass
    
    def launch(self):
        pass
    
    def poll(self):
        pass    
    
    def kill(self):
        pass    
