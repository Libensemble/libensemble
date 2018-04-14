#!/usr/bin/env python

""" Script to launch and control running jobs """

import os
import subprocess
import logging
from libensemble.register import Register

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
logger.setLevel(logging.DEBUG)

STATES = '''
UNKNOWN
WAITING
RUNNING
FINISHED
USER_KILLED
FAILED'''.split()

#IMPORTANT: Need to determine if jobcontroller controls a single job - or can be re-used
#If to be re-used then needs not just an __init__ for controller but a refresh for a new job launch.
#If init controller in calling script - it will need to be latter.
#Either need reset or jobs as sub-objects.

#Also - I may want to use a top-level abstrac/base class for maximum re-use
# - else inherited controller will be reimplementing all


class JobControllerException(Exception): pass

class JobController:

    controller = None
    
    #Create unit test - that checks all combos behave as expected
    #could set self.nodes etc.. or have as static function and set those in place
    #If static - only need to change values when they must be changed - or add if not provided - else they pass through
    #- if set object values here - must set all
    #- setting self for now. - but makes more sense I think to only change values and pass through others?
    
    #If NOT static - then make it an internal function I think
    #If static can test without creating job_controller object...
    
    #@staticmethod
    def _job_partition(self, num_procs, num_nodes, ranks_per_node, machinefile=None):
        #""" Takes provided nprocs/nodes/ranks and outputs working configuration of procs/nodes/ranks or error"""
        """ Takes provided nprocs/nodes/ranks and sets jobs working configuration of procs/nodes/ranks or error"""
        
        #If machinefile is provided - ignore everything else
        if machinefile is not None:        
            if num_procs is not None or num_nodes is not None or ranks_per_node is not None:
                logger.warning('Machinefile provided - overriding procs/nodes/ranks_per_node')
            self.num_procs = None
            self.num_nodes = None
            self.ranks_per_node = None
            return

        #If all set then check num_procs equals num_nodes*ranks_per_node and set values as given
        if num_procs is not None and num_nodes is not None and ranks_per_node is not None:
            if num_procs != num_nodes*ranks_per_node:
                raise JobControllerException("num_procs does not equal num_nodes*ranks_per_node")
            else:
                self.num_procs = num_procs #pass through
                self.num_nodes = num_nodes #pass through
                self.ranks_per_node = ranks_per_node #pass through
                return

        #If num_procs not set then need num_nodes and ranks_per_node and set num_procs
        if num_procs is None:
            #Note this covers case where none are set - may want to use job_controller defaults in that case - not implemented yet.
            if num_nodes is None or ranks_per_node is None:
                raise JobControllerException("Must set either num_procs or num_nodes/ranks_per_node or machinefile")
            else:
                self.num_procs = num_nodes * ranks_per_node
                self.num_nodes = num_nodes #pass through
                self.ranks_per_node = ranks_per_node #pass through
                return
        
        #If num_procs is set - fill in any other values 
        if num_procs is not None:
            self.num_procs = num_procs #pass through
            if num_nodes is None:
                if ranks_per_node is None:
                    #Currently not auto-detecting so if only num_procs - you are on 1 node
                    self.num_nodes = 1
                    self.ranks_per_node = num_procs
                else:
                    self.ranks_per_node = ranks_per_node #pass through
                    self.num_nodes = num_procs/ranks_per_node
            else:
                self.num_nodes = num_nodes #pass through
                self.ranks_per_node = num_procs/num_nodes
        
        #For static version return
        #see collections model for best return type
        #return nprocs, nodes, ranks_per_node
        return

    #Currently not using sub-job so reset job attributes - as opposed to job_controller attributes
    def reset(self):
        #This may be placed in a job object (and maybe a list of jobs for controller)
        #job will have ID that can be used
        self.process = None
        self.state = 'UNKNOWN' #Create as string or integer macro?
        self.errcode = None
        self.finished = False # True means job has run - not whether was successful
        self.success = False
        
        #job job_partition - prob. replace with set_to_default function - so can have default set at job_controller level
        self.num_procs = 1
        self.num_nodes = 1
        self.ranks_per_node = 1
        self.app_args = None
    
    def __init__(self, registry=None):
        
        if registry is None:
            self.registry = Register.default_registry #Error handling req.
        else:
            self.registry = registry
        
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")
        
        #Configured possiby by a launcher abstract class/subclasses for launcher type - based on autodetection
        #currently hardcode here - prob prefix with cmd - eg. self.cmd_nprocs
        self.mpi_launcher = 'mpirun'
        self.mfile = '-machinefile'
        self.nprocs = '-np'
        self.nnodes = ''
        self.ppn = '--ppn'
        
        #Job controller settings - can be set in user function.
        self.kill_signal = 'SIGTERM'
        self.wait_and_kill = True #If true - wait for wait_time After signal and then kill with SIGKILL
        self.wait_time = 60
        
        #Reset current job attributes
        self.reset()
        
        JobController.controller = self
        
        #If this could share multiple launches could set default job parameters here (nodes/ranks etc...)
        
    
    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, app_args=None, stdout=None, stage_out=None, test=False):
     
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

        
        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#
        #_job_partition - eg. balsam class  may want to call _job_partition without machinefile... as does not accept!
        
        #Set self.num_procs, self.num_nodes and self.ranks_per_node for this job
        self._job_partition(num_procs, num_nodes, ranks_per_node, machinefile)
        
        #Static version
        #nprocs, nodes, ranks_per_node = _job_partition(nprocs, nodes, ranks_per_node, machinefile) 
            
        if stage_out is not None:
            logger.warning('stage_out option ignored in this job_controller - runs in-place')
         
        #Construct run line - poss subroutine
        runline = []
        runline.append(self.mpi_launcher)
        
        #I will set attributes for these - eg. self.app_args - but maybe in a job object. So deferring for now.
        #Already issue of whether same job object Worker is using - already diff in e.g. timing - where launch app?
        if machinefile is not None:
            runline.append(self.mfile)
            runline.append(machinefile)
        
        #self.num_procs only if used non-static _job_partition - else just num_procs
        if self.num_procs is not None:
            runline.append(self.nprocs)
            runline.append(str(self.num_procs))
        
        #Not currently setting nodes
        #- as not always supported - but should always have the other two after calling _job_partition
        #if self.num_nodes is not None:
            #runline.append(self.nnodes)
            #runline.append(str(self.num_nodes))
        
        #Currently issues - command depends on mpich/openmpi etc...
        #if self.ranks_per_node is not None:
            #runline.append(self.ppn)
            #runline.append(str(self.ranks_per_node))        

        
        runline.append(app.full_path)
        
        if app_args is not None:
            self.app_args = app_args
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
            

            logger.debug("Launching job: {}".format(" ".join(runline)))
            
            #This was on theta - still dont think need cwd option
            if stdout is None:
                self.process = subprocess.Popen(runline, cwd='./', shell=False)
            else:
                self.process = subprocess.Popen(runline, cwd='./', stdout = open(stdout,'w'), shell=False)
        
        #Could return self.process - or some independent job/process ID - which will always have common format.
        #tmp use actual process id
        #return self.process
        #Mayb just return if successfully launched. Process ID can be queried.

    
    #Poll returns a job state - currently as a string. Includes running or various finished/end states
    #An alternative I think may be better is if it just returns whether job is finished and then
    #you can get end_state either via an attribute directly or through a function (for API reasons).
    
    def poll(self):
        
        #Check the jobs been launched (i.e. it has a process ID)
        if self.process is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job has no process ID - check jobs been launched')
        
        #Here question of checking the existing state before polling - some error handling is required
        #if self.state == 'USER_KILLED':
            #logger.warning('Polled job has already been killed') #could poll to check....
            #return self.state
        #if self.state == 'FAILED':
            #logger.warning('Polled job has already been set to failed') #could poll to check....
            #return self.state
        #if self.state == 'FINISHED':
            #logger.warning('Polled job has already been set to finished') #could poll to check....
            #return self.state   
        
        #Quicker - maybe should poll job to check (in case self.finished set in error!)
        if self.finished:
            logger.warning('Polled job has already finished. Not re-polling. Status is {}'.format(self.state))
            return self.state
        
        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#
        
        # Poll the job
        poll = self.process.poll()
        if poll is None:
            self.state = 'RUNNING'
        else:
            self.finished = True
            #logger.debug("Process {} Completed".format(self.process))
            
            if self.process.returncode == 0:
                self.success = True
                self.errcode = 0
                logger.debug("Process {} completed successfully".format(self.process))
                self.state = 'FINISHED'
            else:
                #Need to differentiate failure from user killed !!!!!
                #Currently FAILED MEANS BOTH
                self.errcode = self.process.returncode
                self.state = 'FAILED'
        
        return self.state
                
    
    def kill(self):
        #In here can set state to user killed!
        #- but if killed by remote job (eg. through balsam database) may be different ....
        #maybe have a function JobController.set_kill_mode() 

        #Issue signal
        if self.kill_signal == 'SIGTERM':
            self.process.terminate()
        elif self.kill_signal == 'SIGKILL':
            self.process.terminate()
        else:
            self.process.send_signal(self.kill_signal) #Prob only need this line!
            
        #Wait for job to be killed
        if self.wait_and_kill:
            try:
                self.process.wait(timeout=self.wait_time)
                #stdout,stderr = self.process.communicate(timeout=self.wait_time) #Wait for process to finish
            except subprocess.TimeoutExpired:
                logger.warning("Kill signal {} timed out - issuing SIGKILL".format(self.kill_signal))
                self.process.kill()
                self.process.wait()
        else:
            self.process.wait(timeout=self.wait_time)

        self.state = 'USER_KILLED'
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
    
    def read_file_in_workdir(self, filename):
        if not os.path.exists(filename):
            raise ValueError("%s not found in working directory".format(filename))
        else:
            return open(filename).read()


        
class BalsamJobController(JobController):
    
    controller = None
    
    def reset(self):       
        super().reset()
        self.jobname = None #Might set jobname in super class also
        self.balsam_state = None

  
    def __init__(self, registry=None):
        
        #Will use super - but for now dont want to set self.mpi_launcher etc...
        
        if registry is None:
            self.registry = Register.default_registry #Error handling req.
        else:
            self.registry = registry
        
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")
        
        
        ##Job controller settings - can be set in user function.
        #self.kill_signal = 'SIGTERM'
        #self.wait_and_kill = True #If true - wait for wait_time After signal and then kill with SIGKILL
        #self.wait_time = 60
        
        #Reset current job attributes
        self.reset()
        
        BalsamJobController.controller = self
        
    
    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, app_args=None, stdout=None, stage_out=None, test=False):
        
        import balsam.launcher.dag as dag
        
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
        
        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#
        
        #As above - I will set attributes (eg. self.app_args) - but maybe in a job object.
        
        #Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")  
        
        #Set self.num_procs, self.num_nodes and self.ranks_per_node for this job
        self._job_partition(num_procs, num_nodes, ranks_per_node) #Note: not included machinefile option
        
        self.app_args = app_args

        #default
        self.jobname = 'job_' + app.name
        
        #prob want app to go into self.app for consistency - either way be consistent
        #Either dont use self for job attributes or use for all
        
        #Re-do debug launch line for balsam job
        #logger.debug("Launching job: {}".format(" ".join(runline)))
        
        if stage_out is not None:
            #For now hardcode staging - for testing
            self.process = dag.add_job(name = self.jobname,
                                                  workflow = "libe_workflow", #add arg for this
                                                  application = app.name,
                                                  application_args = self.app_args,                            
                                                  num_nodes = self.num_nodes,
                                                  ranks_per_node = self.ranks_per_node,
                                                  stage_out_url = "local:" + stage_out,
                                                  stage_out_files = "*")  
        else:
            #No staging
            self.process = dag.add_job(name = self.jobname,
                                                  workflow = "libe_workflow", #add arg for thi
                                                  application = app.name,
                                                  application_args = self.app_args,           
                                                  num_nodes = self.num_nodes,
                                                  ranks_per_node = self.ranks_per_node)            


    #Todo - consideration of better way of polling and extracting information on job status
    #Balsam jobs will have more states than direct launchers.
    #I need some generality and then maybe a more specific state
    #E.g generality: Unknown, Waiting, Running, Finished.
    #Detailed - everything else. Store in self.balsam_state
    #Now I think I understand why Balsam wen't with command job.refresh_from_db() and then you query the state....
    #Could be poll just implements a poll which updates the object. And then user does jobctrl.state
    #Direct access to object attributes is still uncomfortable.....
    
    def poll(self):
        
        #Check the jobs been launched (i.e. it has a process ID)
        if self.process is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job has no process ID - check jobs been launched')
        
        #Here question of checking the existing state before polling - some error handling is required
        #if self.state == 'USER_KILLED':
            #logger.warning('Polled job has already been killed') #could poll to check....
            #return self.state
        #if self.state == 'FAILED':
            #logger.warning('Polled job has already been set to failed') #could poll to check....
            #return self.state
        #if self.state == 'FINISHED':
            #logger.warning('Polled job has already been set to finished') #could poll to check....
            #return self.state   
        
        #Quicker - maybe should poll job to check (in case self.finished set in error!)
        if self.finished:
            logger.warning('Polled job has already finished. Not re-polling. Status is {}'.format(self.state))
            return self.state  
        
        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#
        
        #Get current state of jobs from Balsam database
        job = self.process
        job.refresh_from_db()
        self.balsam_state = job.state
        logger.debug('balsam_state is {}'.format(self.balsam_state))
        
        import balsam.launcher.dag as dag #Might need this before get models - test
        from balsam.service import models

        if job.state in models.END_STATES:
            self.finished = True
            if job.state == 'JOB_FINISHED':
                self.success = True
                self.state = 'FINISHED'
            elif job.state == 'PARENT_KILLED': #I'm not using this currently
                self.state = 'USER_KILLED'
                #self.success = False #Shld already be false - init to false
                #self.errcode = #Can I get errcode??? - Else should remain as None
            elif job.state in STATES: #In my states
                self.state = job.state
                #self.success = False #All other end states are failrues currently - bit risky
            else:
                logger.warning("Job finished, but in unrecognized Balsam state {}".format(job.state))
                self.state = 'UNKNOWN'
                
        elif job.state in models.ACTIVE_STATES:
            self.state = 'RUNNING'
            
        elif job.state in models.PROCESSABLE_STATES + models.RUNNABLE_STATES: #Does this work - concatenate lists
            self.state = 'WAITING'
        else:
            raise JobControllerException('Job state returned from Balsam is not in known list of Balsam states. Job state is {}'.format(job.state))
        
        return self.state
    
    def kill(self):
        import balsam.launcher.dag as dag
        dag.kill(self.process)
        #Check if can wait for kill to complete - affect signal used etc....
    
    def set_kill_mode(self, signal=None, wait_and_kill=None, wait_time=None):
        logger.warning("set_kill_mode currently has no action with Balsam controller")
        
    def read_file_in_workdir(self, filename):
        out = self.process.read_file_in_workdir(filename)
        return out
        
    
