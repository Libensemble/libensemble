# --- Tags

UNSET_TAG = 0  # sh temp - this is a libe feature that is to be reviewed for best solution
EVAL_SIM_TAG = 1
EVAL_GEN_TAG = 2
STOP_TAG = 3
PERSIS_STOP = 4                     # manager tells persistent worker to desist
FINISHED_PERSISTENT_SIM_TAG = 11    # tells manager sim_f done persistent mode
FINISHED_PERSISTENT_GEN_TAG = 12    # tells manager gen_f done persistent mode
# ABORT_ENSEMBLE = 13 # Worker asks manager to abort (and dump history)

calc_type_strings = {
    EVAL_SIM_TAG: 'sim',
    EVAL_GEN_TAG: 'gen',
    None: 'No type set'
}


# --- Signal flags (in message body vs tags)

# CALC STATUS/SIGNAL FLAGS: In future these will be in a data structure
MAN_SIGNAL_FINISH = 20  # Kill jobs and shutdown worker
MAN_SIGNAL_KILL = 21    # Kill running job - but don't stop worker

WORKER_KILL = 30             # Worker kills not covered by a more specific case
WORKER_KILL_ON_ERR = 31
WORKER_KILL_ON_TIMEOUT = 32
JOB_FAILED = 33
WORKER_DONE = 34
CALC_EXCEPTION = 35

calc_status_strings = {
    MAN_SIGNAL_FINISH: "Manager killed on finish",
    MAN_SIGNAL_KILL: "Manager killed job",
    WORKER_KILL_ON_ERR: " Worker killed job on Error",
    WORKER_KILL_ON_TIMEOUT: "Worker killed job on Timeout",
    WORKER_KILL: "Worker killed",
    JOB_FAILED: "Job Failed",
    WORKER_DONE: "Completed",
    CALC_EXCEPTION: "Exception occurred",
    None: "Unknown Status"
}
