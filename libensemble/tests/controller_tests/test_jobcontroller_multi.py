#Test of job controller running multiple jobs for libensemble
#Could support hybrid mode - including, eg. running multi jobs per node (launched locally), or
#simply sharing burden on central system/consecutive pipes to balsam database - could enable
#use of threads if supply run-directories rather than assuming in-place runs etc....

#Test does not require running full libensemble
import os

def build_simfunc():
    import subprocess
    
    #Build simfunc
    #buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring='mpicc -o my_simjob.x simdir/my_simjob.c'
    #subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())

#--------------- Calling script ---------------------------------------------------------------

from libensemble.register import Register, BalsamRegister
from libensemble.controller import JobController, BalsamJobController

#sim_app = 'simdir/my_simjob.x'
#gen_app = 'gendir/my_genjob.x'

#temp
sim_app = './my_simjob.x'

if not os.path.isfile(sim_app):
    build_simfunc()

USE_BALSAM = False #Take as arg
#USE_BALSAM = True #Take as arg

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

#--------------- Worker: sim func -------------------------------------------------------------
#Should work with Balsam or not

#Can also use an internal iterable list of jobs in JOB_CONTROLLER - along with all_done func etc...

def polling_loop(jobctl, job_list, timeout_sec=40.0, delay=1.0):
    import time
    start = time.time()

    while time.time() - start < timeout_sec:
        
        #Test all done - (return list of not-finished jobs and test if empty)
        active_list = [job for job in job_list if not job.finished]
        if not active_list:
            break
                
        for job in job_list:
            if not job.finished:
                time.sleep(delay)
                print('Polling job %d at time %f' % (job.id, time.time() - start))
                #job.poll()
                jobctl.poll(job)
                if job.finished: continue
                elif job.state == 'WAITING': print('Job %d waiting to launch' % (job.id))   
                elif job.state == 'RUNNING': print('Job %d still running ....' % (job.id)) 

                #Check output file for error
                if job.stdout_exists():
                    if 'Error' in job.read_stdout():
                        print("Found (deliberate) Error in ouput file - cancelling job %d" % (job.id))
                        jobctl.kill(job)
                        time.sleep(delay) #Give time for kill
                        continue
                                    
                #But if I want to do something different - I want to make a file - no function for THAT!
                #But you can get all the job attributes!
                #Uncomment to test
                #path = os.path.join(job.workdir,'newfile'+str(time.time()))
                #open(path, 'a')
           
    print('Loop time', time.time() - start)
    
    for job in job_list:
        if job.finished:
            if job.state == 'FINISHED':
                print('Job %d finished succesfully. Status: %s' % (job.id, job.state))
            elif job.state == 'FAILED':
                print('Job %d failed. Status: %s' % (job.id, job.state))  
            elif job.state == 'USER_KILLED':
                print('Job %d has been killed. Status: %s' % (job.id, job.state))
            else:
                print('Job %d status: %s' % (job.id, job.state))
        else:
            print('Job %d timed out. Status: %s' % (job.id, job.state))
            jobctl.kill(job)
            if job.finished: 
                print('Job %d Now killed. Status: %s' % (job.id, job.state))
                #double check
                #job.poll()
                jobctl.poll(job)
                print('Job %d state is %s' % (job.id, job.state))
    
    
# Tests

#From worker call JobController by different name to ensure getting registered app from JobController
jobctl = JobController.controller


print('\nTest 1 - 3 jobs should complete succesfully with status FINISHED :\n')

job_list = []
cores = 4

for j in range(3):
    #outfilename = 'out_' + str(j) + '.txt' #Could allow launch to generate outfile names based on job.id
    sleeptime = 6 + j*3 #Change args
    args_for_sim = 'sleep' + ' ' + str(sleeptime)
    rundir = 'run_' + str(sleeptime)
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job_list.append(job)
            
          
polling_loop(jobctl, job_list)


