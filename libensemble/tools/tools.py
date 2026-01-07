"""
The libEnsemble utilities module assists in writing consistent calling scripts
and user functions.
"""

import logging
import os
import pickle
import sys
import time

import numpy as np
import numpy.typing as npt

# Create logger
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)

# Set up format (Alt. Import LogConfig and base on that)
utils_logformat = "%(message)s"
formatter = logging.Formatter(utils_logformat)

# Log to file
# util_filename = 'util.log'
# fh = logging.FileHandler(util_filename, mode='w')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

# Log to standard error
sth = logging.StreamHandler(stream=sys.stderr)
sth.setFormatter(formatter)
logger.addHandler(sth)

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
    + "Resolve this either by ensuring libE_specs['ensemble_dir_path'] is unique for each run\n"
    + "or by setting libE_specs['reuse_output_dir'] = True.\n"
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


def _get_shortname(basename):
    script_name = os.path.splitext(os.path.basename(basename))[0]
    short_name = script_name.split("test_", 1).pop()
    return short_name


# =================== save libE output to pickle and np ========================


def save_libE_output(
    H: npt.NDArray,
    persis_info: dict,
    basename: str,
    nworkers: int,
    dest_path: str = None,
    mess: str = "Run completed",
    append_attrs: bool = True,
) -> str:
    """
    Writes out history array and persis_info to files.

    Format: <basename>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>

    To use just basename, set append_attrs=False

    .. code-block:: python

        save_libE_output(H, persis_info, __file__, nworkers)

    Parameters
    ----------

    H: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_

        History array storing rows for each point.
        :ref:`(example)<funcguides-history>`

    persis_info: :obj:`dict`

        Persistent information dictionary.
        :doc:`(example)<data_structures/persis_info>`

    basename  : :obj:`str`

        Name of user-calling script (or user chosen name) to prefix output files.
        The convention is to send __file__ from user calling script.

    nworkers: :obj:`int`

        The number of workers in this ensemble. Added to output file names.

    dest_path: :obj:`str`, optional

        The path to save the file to.

    mess: :obj:`str`

        A message to print/log when saving the file.

    append_attrs: `bool`

        Append run attributes to the base filename.

    """
    if dest_path is None:
        dest_path = os.getcwd()

    short_name = _get_shortname(basename)

    prob_str = hist_name = persis_name = ""
    if append_attrs:
        prob_str = "length=" + str(len(H)) + "_evals=" + str(sum(H["sim_ended"])) + "_workers=" + str(nworkers)
        hist_name = "_history_" + prob_str
        persis_name = "_persis_info_" + prob_str

    h_filename = os.path.join(dest_path, short_name + hist_name)
    p_filename = os.path.join(dest_path, short_name + persis_name)

    status_mess = " ".join(["------------------", mess, "-------------------"])
    logger.info(f"{status_mess}\nSaving results to file: {h_filename}")
    np.save(h_filename, H)

    with open(p_filename + ".pickle", "wb") as f:
        pickle.dump(persis_info, f)

    return h_filename + ".npy"


# ===================== per-process numpy random-streams =======================


def add_unique_random_streams(persis_info: dict, nstreams: int, seed: str = "") -> dict:
    """
    Creates nstreams random number streams for the libE manager and workers
    when nstreams is num_workers + 1. Stream i is initialized with seed i by default.
    Otherwise the streams can be initialized with a provided seed.

    The entries are appended to the provided persis_info dictionary.

    .. code-block:: python

        persis_info = add_unique_random_streams(old_persis_info, nworkers + 1)

    Parameters
    ----------

    persis_info: :obj:`dict`

        Persistent information dictionary.
        :ref:`(example)<datastruct-persis-info>`

    nstreams: :obj:`int`

        Number of independent random number streams to produce.

    seed: :obj:`int`

        (Optional) Seed for identical random number streams for each worker. If
        explicitly set to ``None``, random number streams are unique and seed
        via other pseudorandom mechanisms.

    """

    for i in range(nstreams):
        if isinstance(seed, int) or seed is None:
            random_seed = seed
        else:
            random_seed = i

        if i in persis_info:
            persis_info[i].update(
                {
                    "rand_stream": np.random.default_rng(random_seed),
                    "worker_num": i,
                }
            )
        else:
            persis_info[i] = {
                "rand_stream": np.random.default_rng(random_seed),
                "worker_num": i,
            }
    return persis_info


def check_npy_file_exists(filename: str, basename: bool = False, max_wait: int = 3) -> bool:
    """Checks a file is created in a parallel environment

    Parameters
    ----------

    filename: `str`

        Name of file.

    basename: `bool`

        Whether name provided is a basename for save_libE_output.

    max_wait: `int`

        The maximum number of seconds to wait for the file to exist.
    """

    def check_exact_file_exists():
        return os.path.exists(filename)

    def check_basename_file_exists():
        short_name = _get_shortname(filename)
        file_list = [f for f in os.listdir(".") if f.startswith(short_name) and f.endswith(".npy")]
        return bool(file_list)

    check_file_exists = check_basename_file_exists if basename else check_exact_file_exists
    sleep_interval = 0.1
    total_wait_time = 0
    file_exists = False
    while total_wait_time < max_wait:
        if check_file_exists():
            file_exists = True
            break
        time.sleep(sleep_interval)
        total_wait_time += sleep_interval
    return file_exists


# A very specific exception to using the logger.
def eprint(*args, **kwargs):
    """Prints a user message to standard error"""
    print(*args, file=sys.stderr, **kwargs)
