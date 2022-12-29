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

# ==================== User Sim-ID Warning =====================================

_USER_SIM_ID_WARNING = (
    "\n"
    + 79 * "*"
    + "\n"
    + "User generator script will be creating sim_id.\n"
    + "Take care to do this sequentially.\n"
    + "Information given back to the gen_f for existing sim_id values may be overwritten!\n"
    + "\n"
    + 79 * "*"
    + "\n\n"
)

# ==================== Ensemble directory re-use error =========================

_USER_CALC_DIR_WARNING = (
    "\n"
    + 79 * "*"
    + "\n"
    + "libEnsemble attempted to reuse {} as a parent directory for calc dirs.\n"
    + "If allowed to continue, previous results may have been overwritten!\n"
    + "Resolve this by ensuring libE_specs['ensemble_dir_path'] is unique for each run."
    + "\n"
    + 79 * "*"
    + "\n\n"
)

# ==================== Warning that persistent return data is not used ==========

_PERSIS_RETURN_WARNING = (
    "\n"
    + 79 * "*"
    + "\n"
    + "A persistent worker has returned history data on shutdown. This data is\n"
    + "not currently added to the manager's history to avoid possibly overwriting, but\n"
    + "will be added to the manager's history in a future release. If you want to\n"
    + "overwrite/append, you can set the libE_specs option ``use_persis_return_gen``\n"
    + "or ``use_persis_return_sim``"
    "\n" + 79 * "*" + "\n\n"
)

_WALLCLOCK_MSG_ALL_RETURNED = """
Termination due to wallclock_max has occurred.
All completed work has been returned.
Posting kill messages for all workers.
"""

_WALLCLOCK_MSG_ACTIVE = """
Termination due to wallclock_max has occurred.
Some issued work has not been returned.
Posting kill messages for all workers.
"""