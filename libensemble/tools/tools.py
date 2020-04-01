"""
The libEnsemble utilities module assists in writing consistent calling scripts
and user functions.
"""

import os
import sys
import logging
import numpy as np
import pickle

# Create logger
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)

# Set up format (Alt. Import LogConfig and base on that)
utils_logformat = '%(name)s: %(message)s'
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

_USER_SIM_ID_WARNING = \
    ('\n' + 79*'*' + '\n' +
     "User generator script will be creating sim_id.\n" +
     "Take care to do this sequentially.\n" +
     "Also, any information given back for existing sim_id values will be overwritten!\n" +
     "So everything in gen_specs['out'] should be in gen_specs['in']!" +
     '\n' + 79*'*' + '\n\n')


# =================== save libE output to pickle and np ========================


def save_libE_output(H, persis_info, calling_file, nworkers, mess='Run completed'):
    """
    Writes out history array and persis_info to files.

    Format: <calling_script>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>

    .. code-block:: python

        save_libE_output(H, persis_info, __file__, nworkers)

    Parameters
    ----------

    H: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_

        History array storing rows for each point.
        :doc:`(example)<data_structures/history_array>`

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
    prob_str = '_length=' + str(len(H)) \
                          + '_evals=' + str(sum(H['returned'])) \
                          + '_workers=' + str(nworkers)

    h_filename = short_name + '_history_' + prob_str
    p_filename = short_name + '_persis_info_' + prob_str

    status_mess = ' '.join(['------------------', mess, '-------------------'])
    logger.info('{}\nSaving results to file: {}'.format(status_mess, h_filename))
    np.save(h_filename, H)

    with open(p_filename + ".pickle", "wb") as f:
        pickle.dump(persis_info, f)

# ===================== per-process numpy random-streams =======================


def add_unique_random_streams(persis_info, nstreams):
    """
    Creates nstreams random number streams for the libE manager and workers
    when nstreams is num_workers + 1. Stream i is initialized with seed i.

    The entries are appended to the existing persis_info dictionary.

    .. code-block:: python

        persis_info = add_unique_random_streams(old_persis_info, nworkers + 1)

    Parameters
    ----------

    persis_info: :obj:`dict`

        Persistent information dictionary
        :doc:`(example)<data_structures/persis_info>`

    nstreams: :obj:`int`

        Number of independent random number streams to produce

    """

    for i in range(nstreams):
        if i in persis_info:
            persis_info[i].update({
                'rand_stream': np.random.RandomState(i),
                'worker_num': i})
        else:
            persis_info[i] = {
                'rand_stream': np.random.RandomState(i),
                'worker_num': i}
    return persis_info


# A very specific exception to using the logger.
def eprint(*args, **kwargs):
    """Prints a user message to standard error"""
    print(*args, file=sys.stderr, **kwargs)
