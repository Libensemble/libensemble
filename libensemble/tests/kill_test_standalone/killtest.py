import os
import subprocess
import sys
import time
import signal

#Does not kill, even on laptop
#kill 1
def kill_job_1(process):
    process.kill()
    process.wait()

#kill 2 - with  preexec_fn=os.setsid in subprocess
def kill_job_2(process):
    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.wait()

    
if len(sys.argv) != 4:
    raise Exception("Usage: python killtest.py <kill_type> <num_nodes> <num_procs_per_node>")

# sys.argv[0] is python exe.
kill_type = int(sys.argv[1]) # 1, 2
num_nodes = int(sys.argv[2])
num_procs_per_node = int(sys.argv[3])
num_procs = num_nodes * num_procs_per_node

print("\nKill type: {}   num_nodes: {}   procs_per_node: {}".format(kill_type,num_nodes,num_procs_per_node))

# Create common components of launch line (currently all of it)
mpicmd_launcher = 'mpirun'
mpicmd_numprocs = '-np'
mpicmd_ppn = '-ppn'
user_code = "./burn_time.x"

# As runline common to jobs currently - construct here.
runline = []                            # E.g: 2 nodes run
runline.append(mpicmd_launcher)         # mpirun
runline.append(mpicmd_numprocs)         # mpirun -np
runline.append(str(num_procs))          # mpirun -np 8
runline.append(mpicmd_ppn)              # mpirun -np 8 --ppn 
runline.append(str(num_procs_per_node)) # mpirun -np 8 --ppn 4
runline.append(user_code)               # mpirun -np 8 --ppn 4 ./burn_time.x


#print("Running killtest.py with job size {} procs".format(num_procs))
total_start_time = time.time()

for run_num in range(2):
    time.sleep(4) #Show gap where none should be running
    stdout = "out_" + str(run_num) + ".txt"
    #runline = ['mpirun', '-np', str(num_procs), user_code]
    print('\nRun num: {}   Runline: {}\n'.format(run_num," ".join(runline)))
    
    if kill_type == 1:
        process = subprocess.Popen(runline, cwd='./', stdout = open(stdout,'w'), shell=False) #with kill 1
    elif kill_type == 2:
        process = subprocess.Popen(runline, cwd='./', stdout = open(stdout,'w'), shell=False, preexec_fn=os.setsid)#kill 2
    else:
        raise Exception("kill_type not recognized")

    time_limit = 6
    start_time  = time.time()
    finished = False
    state = "Not started"
    while(not finished):
        
        time.sleep(2)
        poll = process.poll()
        if poll is None:
            state = 'RUNNING'
            print('Running....')
            
        else:
            finished = True
            if process.returncode == 0:
                state = 'PASSED'
            else:
                state = 'FAILED'
        
        if(time.time() - start_time > time_limit):
            print('killing job', run_num)
            #kill_job(process, user_code)
            
            if kill_type == 1:
                kill_job_1(process)
            elif kill_type == 2:
                kill_job_2(process)
            state = 'KILLED'
            finished = True

total_end_time = time.time()   
total_time = total_end_time - total_start_time
print("Total_time {}. State {}".format(total_time,state))

