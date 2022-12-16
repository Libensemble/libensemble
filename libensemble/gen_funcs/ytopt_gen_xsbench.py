"""
This module wraps around the ytopt generator.
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


__all__ = ['persistent_ytopt']


def persistent_ytopt(H, persis_info, gen_specs, libE_info):

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    user_specs = gen_specs['user']
    ytoptimizer = user_specs['ytoptimizer']

    tag = None
    calc_in = None
    first_call = True
    fields = [i[0] for i in gen_specs['out']]

    # Send batches until manager sends stop tag
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if first_call:
            ytopt_points = ytoptimizer.ask_initial(n_points=user_specs['num_sim_workers'])  # Returns a list
            batch_size = len(ytopt_points)
            first_call = False
        else:
            batch_size = len(calc_in)
            elapsed_secs = calc_in['sim_ended_time']-calc_in['sim_started_time']
            print(elapsed_secs)
            results = []
            for entry in calc_in:
                field_params = {}
                for field in fields:
                    field_params[field] = entry[field][0]
                results += [(field_params, entry['RUN_TIME'])]
            print('results: ', results)
            ytoptimizer.tell(results)

            ytopt_points = ytoptimizer.ask(n_points=batch_size)  # Returns a generator that we convert to a list
            ytopt_points = list(ytopt_points)[0]

        # The hand-off of information from ytopt to libE is below. This hand-off may be brittle.
        H_o = np.zeros(batch_size, dtype=gen_specs['out'])
        for i, entry in enumerate(ytopt_points):
            for key, value in entry.items():
                H_o[i][key] = value

        # This returns the requested points to the libE manager, which will
        # perform the sim_f evaluations and then give back the values.
        tag, Work, calc_in = ps.send_recv(H_o)
        print('received:', calc_in, flush=True)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
