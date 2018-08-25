UNSET_TAG = 0 #sh temp - this is a libe feature that is to be reviewed for best solution
EVAL_SIM_TAG = 1
EVAL_GEN_TAG = 2
STOP_TAG = 3
PERSIS_STOP = 4                     # manager tells a persistent worker to stop (and go back into general worker mode)

FINISHED_PERSISTENT_SIM_TAG = 11    # tells manager sim_f is done with persistent mode
FINISHED_PERSISTENT_GEN_TAG = 12    # tells manager gen_f is done with persistent mode
ABORT_ENSEMBLE = 13 # Worker sends to manager to tell to abort (and dump history)

# CALC STATUS/SIGNAL FLAGS: In future these will be in a data structure
MAN_SIGNAL_FINISH = 20 # Kill jobs and shutdown worker
MAN_SIGNAL_KILL = 21   # Kill running job - but don't stop worker
MAN_SIGNAL_REQ_RESEND = 22 # Request worker to resend message
MAN_SIGNAL_REQ_PICKLE_DUMP = 23 # Request worker to dump pickled file of message

WORKER_KILL = 30 #Currently for worker kills that are not covered by more specific. In future will allow user description
WORKER_KILL_ON_ERR = 31
WORKER_KILL_ON_TIMEOUT = 32
JOB_FAILED = 33
WORKER_DONE = 34
CALC_EXCEPTION = 35
