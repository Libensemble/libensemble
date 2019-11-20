# --- Tags

UNSET_TAG = 0
EVAL_SIM_TAG = 1
EVAL_GEN_TAG = 2
STOP_TAG = 3
PERSIS_STOP = 4                     # manager tells persistent worker to desist
FINISHED_PERSISTENT_SIM_TAG = 11    # tells manager sim_f done persistent mode
FINISHED_PERSISTENT_GEN_TAG = 12    # tells manager gen_f done persistent mode
# last_message_number_rst_tag

calc_type_strings = {
    EVAL_SIM_TAG: 'sim',
    EVAL_GEN_TAG: 'gen',
    None: 'No type set'
}


# --- Signal flags (in message body vs tags)
# first_calc_status_rst_tag
# CALC STATUS/SIGNAL FLAGS
MAN_SIGNAL_FINISH = 20  # Kill jobs and shutdown worker
MAN_SIGNAL_KILL = 21    # Kill running job - but don't stop worker
WORKER_KILL = 30        # Worker kills not covered by a more specific case
WORKER_KILL_ON_ERR = 31
WORKER_KILL_ON_TIMEOUT = 32
JOB_FAILED = 33  # Calc had jobs that failed
WORKER_DONE = 34  # Calculation was successful
CALC_EXCEPTION = 35  # Reserved: Automatically used if gen_f or sim_f raised an exception.
# last_calc_status_rst_tag

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
# last_calc_status_string_rst_tag
