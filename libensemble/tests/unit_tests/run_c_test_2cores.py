import subprocess, time


runline="mpirun -np 2 ./my_simjob.x sleep 3".split()
stdout="tmp_c_test_2cores.out"

process = subprocess.Popen(runline, cwd='./', stdout = open(stdout,'w'), shell=False)
finished = False
state=''
while True:
    time.sleep(1)
    poll = process.poll()
    if poll is None:
        state = 'RUNNING'
    else:
        finished = True
        break

print('returncode is:',process.returncode)

if process.returncode == 0:
    print('FINISHED')
else:
    print('FAILED')
    
