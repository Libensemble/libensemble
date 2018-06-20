UNSET_TAG = 0 #sh temp - this is a libe feature that is to be reviewed for best solution
EVAL_SIM_TAG = 1
EVAL_GEN_TAG = 2
STOP_TAG = 3
PERSIS_STOP = 4                     # manager tells a persistent worker to stop (and go back into general worker mode)
#ALLGOOD = 5
FINISHED_PERSISTENT_SIM_TAG = 11    # tells manager sim_f is done with persistent mode
FINISHED_PERSISTENT_GEN_TAG = 12    # tells manager gen_f is done with persistent mode

MAN_SIGNAL_FINISH = 20 # Kill jobs and shutdown worker
MAN_SIGNAL_KILL = 21   # Kill running job - but don't stop worker
WORKER_KILL = 22
JOB_FAILED = 23
WORKER_DONE = 24
