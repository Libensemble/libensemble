"""
Generator built on Tasmanian example:
https://github.com/ORNL/TASMANIAN/blob/master/InterfacePython/example_sparse_grids_04.py
"""

import numpy as np
import Tasmanian
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP
from libensemble.gen_funcs.support import sendrecv_mgr_worker_msg

def sparse_grid(H, persis_info, gen_specs, libE_info):
    U = gen_specs['user']

    iNumInputs = U['NumInputs']
    iNumOutputs = U['NumOutputs']
    precisions = U['precisions']
    aPointOfInterest = U['x0']

    aReferenceSolution = np.exp(-aPointOfInterest[0]**2) * np.cos(aPointOfInterest[1])

    tag = None
    for prec in precisions:
        grid = Tasmanian.makeGlobalGrid(iNumInputs, iNumOutputs, prec,
                                        "iptotal", "clenshaw-curtis")
        aPoints = grid.getNeededPoints()

        H0 = np.zeros(len(aPoints), dtype=gen_specs['out'])
        H0['x'] = aPoints

        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H0)

        if tag in [STOP_TAG, PERSIS_STOP]:
            break

        import ipdb; ipdb.set_trace()
        aModelValues = calc_in['f'] 
        t = aModelValues.reshape((aModelValues.shape[0], iNumOutputs))
        # aModelValues = np.exp(-aPoints[:,0]**2) * np.cos(aPoints[:,1])
        grid.loadNeededPoints(t)


        # when using multiple points at once, evaluateBatch() is more efficient
        aResult = grid.evaluate(aPointOfInterest)
        fError = np.abs(aResult[0] - aReferenceSolution)

    return H0, persis_info, tag
    
