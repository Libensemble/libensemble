"""
Generator built on Tasmanian example:
https://github.com/ORNL/TASMANIAN/blob/master/InterfacePython/example_sparse_grids_04.py
"""

import numpy as np
import Tasmanian
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg


def sparse_grid(H, persis_info, gen_specs, libE_info):
    U = gen_specs['user']

    iNumInputs = U['NumInputs']
    iNumOutputs = U['NumOutputs']
    precisions = U['precisions']
    aPointOfInterest = U['x0']

    tag = None

    persis_info['aResult'] = {}

    for prec in precisions:
        # Generate Tasmanian grid
        grid = Tasmanian.makeGlobalGrid(iNumInputs, iNumOutputs, prec,
                                        "iptotal", "clenshaw-curtis")
        aPoints = grid.getNeededPoints()

        # Return the points of that need to be evaluated to the manager
        H0 = np.zeros(len(aPoints), dtype=gen_specs['out'])
        H0['x'] = aPoints

        # Receive values from manager
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H0)
        if tag in [STOP_TAG, PERSIS_STOP]:
            break
        aModelValues = calc_in['f']

        # Update surrogate on grid
        t = aModelValues.reshape((aModelValues.shape[0], iNumOutputs))
        t = t.flatten()
        t = np.atleast_2d(t).T
        grid.loadNeededPoints(t)

        # Evaluate grid
        aResult = grid.evaluate(aPointOfInterest)

        persis_info['aResult'][prec] = aResult

    tag = FINISHED_PERSISTENT_GEN_TAG
    return H0, persis_info, tag
