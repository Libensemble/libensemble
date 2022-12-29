"""
The libEnsemble utilities module assists in writing consistent calling scripts
and user functions.
"""

import logging
import os
import pickle
import sys

import numpy as np

# Create logger
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)

# Set up format (Alt. Import LogConfig and base on that)
utils_logformat = "%(message)s"
formatter = logging.Formatter(utils_logformat)

# Log to standard error
sth = logging.StreamHandler(stream=sys.stderr)
sth.setFormatter(formatter)
logger.addHandler(sth)


def save_libE_output(H, persis_info, calling_file, nworkers, mess="Run completed"):
    """
    Writes out history array and persis_info to files.

    Format: <calling_script>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>

    .. code-block:: python

        save_libE_output(H, persis_info, __file__, nworkers)

    Parameters
    ----------

    H: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_

        History array storing rows for each point.
        :ref:`(example)<funcguides-history>`

    persis_info: :obj:`dict`

        Persistent information dictionary
        :doc:`(example)<data_structures/persis_info>`

    calling_file  : :obj:`string`

        Name of user-calling script (or user chosen name) to prefix output files.
        The convention is to send __file__ from user calling script.

    nworkers: :obj:`int`

        The number of workers in this ensemble. Added to output file names.

    mess: :obj:`String`

        A message to print/log when saving the file.

    """

    script_name = os.path.splitext(os.path.basename(calling_file))[0]
    short_name = script_name.split("test_", 1).pop()
    prob_str = "length=" + str(len(H)) + "_evals=" + str(sum(H["sim_ended"])) + "_workers=" + str(nworkers)

    h_filename = short_name + "_history_" + prob_str
    p_filename = short_name + "_persis_info_" + prob_str

    status_mess = " ".join(["------------------", mess, "-------------------"])
    logger.info(f"{status_mess}\nSaving results to file: {h_filename}")
    np.save(h_filename, H)

    with open(p_filename + ".pickle", "wb") as f:
        pickle.dump(persis_info, f)


# ===================== per-process numpy random-streams =======================


def add_unique_random_streams(persis_info, nstreams, seed=""):
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

        Persistent information dictionary
        :ref:`(example)<datastruct-persis-info>`

    nstreams: :obj:`int`

        Number of independent random number streams to produce

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


# A very specific exception to using the logger.
def eprint(*args, **kwargs):
    """Prints a user message to standard error"""
    print(*args, file=sys.stderr, **kwargs)
