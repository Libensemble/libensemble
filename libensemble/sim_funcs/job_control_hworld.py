from libensemble.mpi_controller import MPIJobController
from libensemble.message_numbers import UNSET_TAG, WORKER_KILL_ON_ERR, MAN_SIGNAL_FINISH, WORKER_DONE, JOB_FAILED, WORKER_KILL_ON_TIMEOUT
import numpy as np

__all__ = ['job_control_hworld']

# Alt send values through X
sim_count = 0


def polling_loop(comm, jobctl, job, timeout_sec=6.0, delay=1.0):
    import time
    start = time.time()

    calc_status = UNSET_TAG  # Sim func determines status of libensemble calc - returned to worker

    while time.time() - start < timeout_sec:
        time.sleep(delay)

        # print('Probing manager at time: ', time.time() - start)
        jobctl.manager_poll(comm)
        if jobctl.manager_signal == 'finish':
            jobctl.kill(job)
            calc_status = MAN_SIGNAL_FINISH  # Worker will pick this up and close down
            print('Job {} killed by manager on worker {}'.format(job.id, jobctl.workerID))
            break

        # print('Polling job at time', time.time() - start)
        job.poll()
        if job.finished:
            break
        elif job.state == 'RUNNING':
            print('Job {} still running on worker {} ....'.format(job.id, jobctl.workerID))

        # Check output file for error
        # print('Checking output file for error at time:', time.time() - start)
        if job.stdout_exists():
            if 'Error' in job.read_stdout():
                print("Found (deliberate) Error in ouput file - cancelling job")
                jobctl.kill(job)
                calc_status = WORKER_KILL_ON_ERR
                break

    # After exiting loop
    if job.finished:
        print('Job {} done on worker {}'.format(job.id, jobctl.workerID))
        # Fill in calc_status if not already
        if calc_status == UNSET_TAG:
            if job.state == 'FINISHED':  # Means finished succesfully
                calc_status = WORKER_DONE
            elif job.state == 'FAILED':
                calc_status = JOB_FAILED
            # elif job.state == 'USER_KILLED':
            #     calc_status = WORKER_KILL
    else:
        # assert job.state == 'RUNNING', "job.state expected to be RUNNING. Returned: " + str(job.state)
        print("Job {} timed out - killing on worker {}".format(job.id, jobctl.workerID))
        jobctl.kill(job)
        if job.finished:
            print('Job {} done on worker {}'.format(job.id, jobctl.workerID))
        calc_status = WORKER_KILL_ON_TIMEOUT

    return job, calc_status


def job_control_hworld(H, persis_info, sim_specs, libE_specs):
    """ Test of launching and polling job and exiting on job finish"""
    jobctl = MPIJobController.controller
    cores = sim_specs['cores']
    comm = libE_specs['comm']

    args_for_sim = 'sleep 3'
    # pref send this in X as a sim_in from calling script
    global sim_count
    sim_count += 1
    timeout = 6.0
    if sim_count == 1:
        args_for_sim = 'sleep 3'  # Should finish
    elif sim_count == 2:
        args_for_sim = 'sleep 3 Error'  # Worker kill on error
    elif sim_count == 3:
        args_for_sim = 'sleep 5'  # Worker kill on timeout
        timeout = 3.0
    elif sim_count == 4:
        args_for_sim = 'sleep 1 Fail'  # Manager kill - if signal received else completes
    elif sim_count == 5:
        args_for_sim = 'sleep 18'  # Manager kill - if signal received else completes
        timeout = 20.0

    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, hyperthreads=True)
    job, calc_status = polling_loop(comm, jobctl, job, timeout)

    # assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    # assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # This is temp - return something - so doing six_hump_camel_func again...
    batch = len(H['x'])
    O = np.zeros(batch, dtype=sim_specs['out'])
    for i, x in enumerate(H['x']):
        O['f'][i] = six_hump_camel_func(x)

    # This is just for testing at calling script level - status of each job
    O['cstat'] = calc_status

    # v = np.random.uniform(0, 10)
    # print('About to sleep for :' + str(v))
    # time.sleep(v)

    return O, persis_info, calc_status


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2

    return term1 + term2 + term3
