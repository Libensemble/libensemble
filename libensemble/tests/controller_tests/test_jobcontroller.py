#Test of job controller module for libensemble
#Test does not require running full libensemble
import os

def build_simfunc():
    import subprocess
    
    #Build simfunc
    #buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring='mpicc -o my_simjob.x my_simjob.c'
    os.chdir('simdir')
    #subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())
    os.chdir('../')

#--------------- Calling script ------------------------------------------

from libensemble.register import Register, BalsamRegister
from libensemble.controller import JobController, BalsamJobController

sim_app = 'simdir/my_simjob.x'
#gen_app = 'gendir/my_genjob.x'

if not os.path.isfile(sim_app):
    build_simfunc()

USE_BALSAM = False

#Create and add exes to registry
if USE_BALSAM:
    registry = BalsamRegister()
    jobctrl = BalsamJobController(registry = registry)    
else:
    registry = Register()
    jobctrl = JobController(registry = registry)
    
registry.register_calc(full_path=sim_app, calc_type='sim')

#Alternative to IF could be using eg. fstring to specify: e.g:
#JOB_CONTROLLER = 'Balsam'
#registry = f"{JOB_CONTROLLER}Register()"

#--------------- Worker: sim func ----------------------------------------
#Should work with Balsam or not


def polling_loop(jobctl, job, outfilename, timeout_sec=20.0,delay=2.0):
    import time
    start = time.time()
    
    while time.time() - start < timeout_sec:
        time.sleep(delay)
        print('Polling at time', time.time() - start)
        #job.poll()
        jobctl.poll(job)        
        if job.finished: break
        elif job.state == 'WAITING': print('Job waiting to launch')    
        elif job.state == 'RUNNING': print('Job still running ....') 
        
        #Check output file for error
        #if 'Error' in open(outfilename).read(): #Direct
        #read_file_in_workdir could be job function or jobctl with job supplied
        if 'Error' in job.read_file_in_workdir(outfilename): #Works if JobController creates a workdir.
            print("Found (deliberate) Error in ouput file - cancelling job")
            jobctl.kill(job)
            break
    
    if job.finished:
        if job.state == 'FINISHED':
            print('Job finished succesfully. Status:',job.state)
        elif job.state == 'FAILED':
            print('Job failed. Status:',job.state)  
        elif job.state == 'USER_KILLED':
            print('Job has been killed. Status:',job.state)
        else:
            print('Job status:', job.state)
    else:
        print("Job timed out")
        jobctl.kill(job)
        if job.finished: 
            print('Now killed')
            #double check
            #job.poll()
            jobctl.poll(job)
            print('Job state is', job.state)
    
    
# Tests

#From worker call JobController by different name to ensure getting registered app from JobController
jobctl = JobController.controller
#jobctl = BalsamJobController.controller


print('\nTest 1 - should complete succesfully with status FINISHED :\n')
#machinefilename = 'machinefile_for_rank'
cores = 4
args_for_sim = 'sleep 5'
outfilename = 't1.txt'

job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, stdout=outfilename)

#job.launch(calc_type='sim', machinefile=machinefilename, num_procs=cores, app_args=args_for_sim,
            #stdout=outfilename, test=True)
polling_loop(jobctl, job, outfilename)


print('\nTest 2 - Job should be USER_KILLED \n')
#machinefilename = 'machinefile_for_rank'
cores = 4
args_for_sim = 'sleep 5 Error'
outfilename = 't2.txt'

#From worker call JobController by different name to ensure getting registered app from JobController
#jobctl = JobController.controller
#jobctl = BalsamJobController.controller

job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, stdout=outfilename)

#jobctl.launch(calc_type='sim', machinefile=machinefilename, num_procs=cores, app_args=args_for_sim,
            #stdout=outfilename, test=True)
polling_loop(jobctl, job, outfilename)

