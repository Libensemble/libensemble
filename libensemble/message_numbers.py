# -------------------------------
# -- Basic User Function Tags ---
# -------------------------------

UNSET_TAG = 0

# When received by a worker, tells worker to do a sim eval;
# When received by the manager, tells manager that worker is done with sim eval.
EVAL_SIM_TAG = 1

# When received by a worker, tells worker to do a gen eval;
# When received by the manager, tells manager that worker is done with sim eval.
EVAL_GEN_TAG = 2

STOP_TAG = 3  # Manager tells worker (or persistent user_f) to stop
PERSIS_STOP = 4  # Manager tells persistent user_f to stop

# last_message_number_rst_tag

calc_type_strings = {
    EVAL_SIM_TAG: "sim",
    EVAL_GEN_TAG: "gen",
    PERSIS_STOP: "STOP with work",
    None: "No type set",
}

# --------------------------------------
# -- Calculation Status/Signal Tags ----
# --------------------------------------

# first_calc_status_rst_tag
FINISHED_PERSISTENT_SIM_TAG = 11  # tells manager sim_f done persistent mode
FINISHED_PERSISTENT_GEN_TAG = 12  # tells manager gen_f done persistent mode
MAN_SIGNAL_FINISH = 20  # Kill tasks and shutdown worker
MAN_SIGNAL_KILL = 21  # Kill running task - but don't stop worker
WORKER_KILL = 30  # Worker kills not covered by a more specific case
WORKER_KILL_ON_ERR = 31  # Worker killed due to an error in results
WORKER_KILL_ON_TIMEOUT = 32  # Worker killed on timeout
TASK_FAILED = 33  # Calc had tasks that failed
WORKER_DONE = 34  # Calculation was successful
# last_calc_status_rst_tag
CALC_EXCEPTION = 35  # Reserved: Automatically used if user_f raised an exception

MAN_KILL_SIGNALS = [MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL]

calc_status_strings = {
    UNSET_TAG: "Not set",
    FINISHED_PERSISTENT_SIM_TAG: "Persis sim finished",
    FINISHED_PERSISTENT_GEN_TAG: "Persis gen finished",
    MAN_SIGNAL_FINISH: "Manager killed on finish",
    MAN_SIGNAL_KILL: "Manager killed task",
    WORKER_KILL_ON_ERR: "Worker killed task on Error",
    WORKER_KILL_ON_TIMEOUT: "Worker killed task on Timeout",
    WORKER_KILL: "Worker killed",
    TASK_FAILED: "Task Failed",
    WORKER_DONE: "Completed",
    CALC_EXCEPTION: "Exception occurred",
    None: "Unknown Status",
}
# last_calc_status_string_rst_tag
