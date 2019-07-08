import subprocess

comms = ['mpi', 'local', 'tcp']
nprocs = [2, 3, 4]
test = './libensemble/tests/regression_tests/test_jobcontroller_hworld.py'

for comm in comms:
    for nproc in nprocs:
        if comm == 'mpi':
            cmd = ['mpiexec', '-n', str(nproc), 'python', test]
        else:
            cmd = ['python', test, '--comms', comm, '--nworkers', str(nproc)]
        print("----- COMMS: {} --- PROCESSES: {} -----".format(comm, nproc))
        process = subprocess.Popen(cmd)
        process.wait()
