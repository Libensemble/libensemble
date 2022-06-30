"""
Below are the fields used within libEnsemble
"""


libE_fields = [
    ("sim_id", int),  # Unique id of a generated entry in H
    ("gen_worker", int),  # Worker that generated this entry
    ("gen_started_time", float),  # Time gen_worker was initiated that produced this entry
    ("gen_ended_time", float),  # Time gen_worker requested this entry
    ("sim_worker", int),  # Worker that did (or is doing) the sim eval for this entry
    ("sim_started", bool),  # True if entry was given to sim_worker for sim eval
    ("sim_started_time", float),  # Time entry was given to sim_worker for a sim eval
    ("sim_ended", bool),  # True if entry's sim eval completed
    ("sim_ended_time", float),  # Time entry's sim eval completed
    ("gen_informed", bool),  # True if gen_worker was informed about the sim eval of this entry
    ("gen_informed_time", float),  # Time gen_worker was informed about the sim eval of this entry
    ("cancel_requested", bool),  # True if cancellation was requested for this entry
    ("kill_sent", bool),  # True if a kill signal was sent to worker for this entry
]
# end_libE_fields_rst_tag


protected_libE_fields = [
    "gen_worker",
    "gen_started_time",
    "gen_ended_time",
    "sim_worker",
    "sim_started",
    "sim_started_time",
    "sim_ended",
    "sim_ended_time",
    "gen_informed",
    "gen_informed_time",
    "kill_sent",
]

allowed_sim_spec_keys = [
    "sim_f",  #
    "in",  #
    "persis_in",
    "out",  #
    "funcx_endpoint",  #
    "user",
]

allowed_gen_spec_keys = [
    "gen_f",  #
    "in",  #
    "persis_in",
    "out",  #
    "funcx_endpoint",  #
    "user",
]

allowed_alloc_spec_keys = [
    "alloc_f",  #
    "out",  #
    "user",
]

libE_spec_calc_dir_misc = [
    "ensemble_copy_back",
    "ensemble_dir_path",
    "use_worker_dirs",
]

libE_spec_sim_dir_keys = [
    "sim_dirs_make",
    "sim_dir_copy_files",
    "sim_dir_symlink_files",
    "sim_input_dir",
]

libE_spec_gen_dir_keys = [
    "gen_dirs_make",
    "gen_dir_copy_files",
    "gen_dir_symlink_files",
    "gen_input_dir",
]

libE_spec_calc_dir_combined = libE_spec_calc_dir_misc + libE_spec_sim_dir_keys + libE_spec_gen_dir_keys

allowed_libE_spec_keys = [
    "abort_on_exception",  #
    "enforce_worker_core_bounds",  #
    "authkey",  #
    "disable_resource_manager",  #
    "dedicated_mode",  #
    "comms",  #
    "resource_info",  #
    "disable_log_files",  #
    "final_fields",  #
    "ip",  #
    "kill_canceled_sims",  #
    "mpi_comm",  #
    "num_resource_sets",  #
    "nworkers",  #
    "port",  #
    "profile",  #
    "safe_mode",  #
    "save_every_k_gens",  #
    "save_every_k_sims",  #
    "save_H_and_persis_on_abort",  #
    "scheduler_opts",  #
    "stats_fmt",  #
    "use_persis_return_gen",  #
    "use_persis_return_sim",  #
    "workerID",  #
    "worker_timeout",  #
    "zero_resource_workers",  #
    "worker_cmd",
] + libE_spec_calc_dir_combined
