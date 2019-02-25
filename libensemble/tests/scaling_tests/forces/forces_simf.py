import os
import time
import numpy as np

from libensemble.controller import JobController
from libensemble.message_numbers import * # Have to at least put these in some data structure to import....


def run_forces(x,gen_specs,sim_specs,libE_info):
    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # output            = what will be returned to libE
    # simdir_basename   = Basename for simulation directories
    
    calc_status = 0 # Returns to worker
    
    simdir_basename = sim_specs['simdir_basename']
    cores           = sim_specs['cores']
    keys            = sim_specs['keys']
    sim_particles   = sim_specs['sim_particles']
    sim_timesteps   = sim_specs['sim_timesteps']
    time_limit      = sim_specs['sim_kill_minutes']
       
    ## Composing variable names and x values to set up simulation
    #arguments = []
    #sim_dir   = [simdir_basename]
    #for i,key in enumerate(keys):
       #variable = key+'='+str(x[i])
       #arguments.append(variable)
       #sim_dir.append('_'+variable)
    #print(os.getcwd(), sim_dir)
    
    # For one key
    seed = int(np.rint(x[0][0]))
    print(seed)
    simdir = simdir_basename + '_' + keys[0] + '_' + str(seed)
        
    # For now assume unique - will need to check for matches...
    os.mkdir(simdir)
    
    # At this point you will be in the sim directory (really worker dir) for this worker (eg. sim_1).
    # The simdir below is created for each job for this worker.
    # Any input needs to be copied into this directory. Currently there is none.
    os.chdir(simdir)
    jobctl = JobController.controller # Get JobController
    
    args = str(int(sim_particles)) + ' ' + str(sim_timesteps) + ' ' + str(seed)
    
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args, stdout='out.txt', stderr='err.txt')
    
    poll_interval = 1 # secs
    while(not job.finished):
        time.sleep(poll_interval)
        job.poll()

    if job.finished:
        if job.state == 'FINISHED':
            print("Job {} completed".format(job.name))
            calc_status = WORKER_DONE
        elif job.state == 'FAILED':
            print("Warning: Job {} failed: Error code {}".format(job.name, job.errcode))
            calc_status = JOB_FAILED
        elif job.state == 'USER_KILLED':
            print("Warning: Job {} has been killed".format(job.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Job {} in unknown state {}. Error code {}".format(job.name, job.state, job.errcode))           
    
    
    os.chdir('../')
    
    statfile = simdir_basename+'.stat'
    filepath = os.path.join(job.workdir, statfile)

    if job.file_exists_in_workdir(statfile):
        data = np.loadtxt(filepath)
        #or
        #job.read_file_in_workdir(statfile)
            
        final_energy = data[-1]    
            
    outspecs = sim_specs['out']
    output = np.zeros(1,dtype=outspecs)
    output['energy'][0] = final_energy
    
    #calc_status = WORKER_DONE
    
    return output, gen_specs, calc_status
