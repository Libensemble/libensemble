"""
A persistent generator using the uncertainty quantification capabilities in
`Tasmanian <https://tasmanian.ornl.gov/>`_.
"""

import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


def sparse_grid_batched(H, persis_info, gen_specs, libE_info):
    """
    Implements batched construction for a Tasmanian sparse grid,
    using the loop described in Tasmanian Example 09:
    `sparse grid example <https://github.com/ORNL/TASMANIAN/blob/master/InterfacePython/example_sparse_grids_09.py>`_

    """
    U = gen_specs['user']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    grid = U['tasmanian_init']()  # initialize the grid
    allowed_refinements = ['setAnisotropicRefinement', 'setSurplusRefinement', 'none']
    assert 'refinement' in U and U['refinement'] in allowed_refinements, \
        "Must provide a gen_specs['user']['refinement'] in: {}".format(allowed_refinements)

    while grid.getNumNeeded() > 0:
        aPoints = grid.getNeededPoints()

        H0 = np.zeros(len(aPoints), dtype=gen_specs['out'])
        H0['x'] = aPoints

        # Receive values from manager
        tag, Work, calc_in = ps.send_recv(H0)
        if tag in [STOP_TAG, PERSIS_STOP]:
            break
        aModelValues = calc_in['f']

        # Update surrogate on grid
        t = aModelValues.reshape((aModelValues.shape[0], grid.getNumOutputs()))
        t = t.flatten()
        t = np.atleast_2d(t).T
        grid.loadNeededPoints(t)

        if 'tasmanian_checkpoint_file' in U:
            grid.write(U['tasmanian_checkpoint_file'])

        # set refinement, using user['refinement'] to pick the refinement strategy
        if U['refinement'] == 'setAnisotropicRefinement':
            assert 'sType' in U
            assert 'iMinGrowth' in U
            assert 'iOutput' in U
            grid.setAnisotropicRefinement(U['sType'], U['iMinGrowth'], U['iOutput'])
        elif U['refinement'] == 'setSurplusRefinement':
            assert 'fTolerance' in U
            assert 'iOutput' in U
            assert 'sCriteria' in U
            grid.setSurplusRefinement(U['fTolerance'], U['iOutput'], U['sCriteria'])

    return H0, persis_info, FINISHED_PERSISTENT_GEN_TAG
