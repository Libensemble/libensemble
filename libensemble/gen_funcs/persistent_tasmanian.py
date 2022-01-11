"""
A persistent generator using the uncertainty quantification capabilities in
`Tasmanian <https://tasmanian.ornl.gov/>`_.
"""

import numpy as np
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


def lex_le(x, y, tol=1E-12):
    """
    Returns True if x <= y lexicographically up to some tolerance.
    """
    cmp = np.fabs(x - y) > tol
    ind = np.argmax(cmp)
    if not cmp[ind]:
        return True
    return x[ind] <= y[ind]


def get_2D_insert_indices(x, y, x_ord=np.empty(0, dtype='int'), y_ord=np.empty(0, dtype='int'), tol=1E-12):
    """
    Finds the row indices in a 2D numpy array `x` for which `y` can be inserted into. If `x_ord` (resp. `y_ord`) is empty,
    then `x` (resp. `y`) must be lexicographically sorted. Otherwise, `x[x_ord]` (resp. `y[y_ord]`) must be lexicographically
    sorted. Complexity is O(x.shape[0] + y.shape[0]).
    """
    if x.size == 0:
        return np.zeros(y.shape[0], dtype='int')
    else:
        if x_ord.size == 0:
            x_ord = np.arange(x.shape[0], dtype='int')
        if y_ord.size == 0:
            y_ord = np.arange(y.shape[0], dtype='int')
        x_ptr = 0
        y_ptr = 0
        out_ord = np.empty(0, dtype='int')
        while(y_ptr < y.shape[0]):
            # The case where y[k] <= max of x[k:end, :]
            xk = x[x_ord[x_ptr], :]
            yk = y[y_ord[y_ptr], :]
            if lex_le(yk, xk, tol=tol):
                out_ord = np.append(out_ord, x_ord[x_ptr])
                y_ptr += 1
            else:
                x_ptr += 1
                # The edge case where y[k] is the largest of all elements of x.
                if x_ptr >= x_ord.shape[0]:
                    for i in range(y_ptr, y_ord.shape[0], 1):
                        out_ord = np.append(out_ord, x_ord.shape[0])
                        y_ptr += 1
                    break
        return out_ord


def get_2D_duplicate_indices(x, y, x_ord=[], y_ord=[], tol=1E-12):
    """
    Finds the row indices of a 2D numpy array `x` which overlap with `y`. If `x_ord` (resp. `y_ord`) is empty, then `x` (resp.
    `y`) must be lexicographically sorted. Otherwise, `x[x_ord]` (resp. `y[y_ord]`) must be lexicographically sorted.Complexity
    is O(x.shape[0] + y.shape[0]).
    """
    if x.size == 0:
        return np.empty(0, dtype='int')
    else:
        if x_ord.size == 0:
            x_ord = np.arange(x.shape[0], dtype='int')
        if y_ord.size == 0:
            y_ord = np.arange(y.shape[0], dtype='int')
        x_ptr = 0
        y_ptr = 0
        out_ord = np.empty(0, dtype='int')
        while(y_ptr < y.shape[0]):
            # The case where y[k] <= max of x[k:end, :]
            xk = x[x_ord[x_ptr], :]
            yk = y[y_ord[y_ptr], :]
            if all(np.fabs(yk - xk) <= tol):
                out_ord = np.append(out_ord, x_ord[x_ptr])
                x_ptr += 1
            y_ptr += 1
        return out_ord


def sparse_grid_batched(H, persis_info, gen_specs, libE_info):
    """
    Implements batched construction for a Tasmanian sparse grid,
    using the loop described in Tasmanian Example 09:
    `sparse grid example <https://github.com/ORNL/TASMANIAN/blob/master/InterfacePython/example_sparse_grids_09.py>`_

    """
    U = gen_specs['user']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    grid = U['tasmanian_init']()  # initialize the grid
    allowed_refinements = ['anisotropic', 'setAnisotropicRefinement', 'surplus', 'setSurplusRefinement', 'none']
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
        if U['refinement'] in ['anisotropic', 'setAnisotropicRefinement']:
            assert 'sType' in U
            assert 'iMinGrowth' in U
            assert 'iOutput' in U
            grid.setAnisotropicRefinement(U['sType'], U['iMinGrowth'], U['iOutput'])
        elif U['refinement'] in ['surplus', 'setSurplusRefinement']:
            assert 'fTolerance' in U
            assert 'iOutput' in U
            assert 'sCriteria' in U
            grid.setSurplusRefinement(U['fTolerance'], U['iOutput'], U['sCriteria'])

    return H0, persis_info, FINISHED_PERSISTENT_GEN_TAG


def sparse_grid_async(H, persis_info, gen_specs, libE_info, num_tol=1E-12):
    """
    Implements asynchronous construction for a Tasmanian sparse grid,
    using the logic in the dynamic Tasmanian model construction function:
    `sparse grid example <https://github.com/ORNL/TASMANIAN/blob/master/Addons/tsgConstructSurrogate.hpp>`_

    """
    U = gen_specs['user']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    grid = U['tasmanian_init']()  # initialize the grid
    allowed_refinements = ['anisotropic', 'surplus']
    assert 'refinement' in U and U['refinement'] in allowed_refinements, \
        "Must provide a gen_specs['user']['refinement'] in: {}".format(allowed_refinements)

    # Choose the refinement function based on U['refinement'].
    if U['refinement'] == 'anisotropic':
        assert 'sType' in U
        assert 'liAnisotropicWeightsOrOutput' in U
        get_refined_points = lambda g : g.getCandidateConstructionPoints(U['sType'], U['liAnisotropicWeightsOrOutput'])
    if U['refinement'] == 'surplus':
        assert 'fTolerance' in U
        assert 'sRefinementType' in U
        assert 'iOutput' in U
        get_refined_points = lambda g : g.getCandidateConstructionPointsSurplus(U['fTolerance'], U['sRefinementType'], U['iOutput'])

    # Asynchronous helper variables.
    num_dims = grid.getNumDimensions()
    num_vals = grid.getNumOutputs()
    # The following two arrays MUST remain sorted according to the ordering of the point array in a pair.
    completed_points = np.empty(0, dtype='float')
    completed_values = np.empty(0, dtype='float')
    running_points = np.empty(0, dtype='float')

    # First run.
    needed_points = grid.getNeededPoints()
    running_points = needed_points[np.lexsort(np.rot90(needed_points)),:]
    H0 = np.zeros(running_points.shape[0], dtype=gen_specs['out'])
    H0['x'] = running_points
    H0['priority'] = np.arange(H['x'].shape[0], dtype='int')
    tag, Work, calc_in = ps.send_recv(H0)

    # Subsequent runs.
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Update running arrays.
        received_ord = np.lexsort(np.rot90(calc_in['x']))
        insert_ind = get_2D_insert_indices(completed_points, calc_in['x'], y_ord=received_ord)
        if completed_points.size == 0:
            completed_points = calc_in['x']
        else:
            completed_points = np.insert(completed_points, insert_ind, calc_in['x'])
        if completed_values.size == 0:
            completed_values = calc_in['f']
        else:
            completed_values = np.insert(completed_values, insert_ind, calc_in['f'])
        delete_ind = get_2D_duplicate_indices(running_points, calc_in['x'], y_ord=received_ord)
        running_points = np.delete(running_points, delete_ind)

        # Allow more work when a sufficient of past work has been completed.
        if grid.getNumLoaded() or completed_points.size[0] > 0.2 * grid.getNumLoaded():
            grid.loadConstructedPoint(calc_in['x'], calc_in['f'])
            if 'tasmanian_checkpoint_file' in U:
                grid.write(U['tasmanian_checkpoint_file'])
            refined_points = get_refined_points(grid)
            # Shut down all nodes if the grid does not give out more work.
            if refined_points.size == 0:
                tag = STOP_TAG
                break
            refined_ord = np.lexsort(np.rot90(refined_points))
            delete_ind = get_2D_duplicate_indices(refined_points, running_points, x_ord=refined_ord)
            H0['x'] = np.delete(refined_points, delete_ind)
            H0['priority'] = np.arange(H['x'].shape[0], dtype='int')
            completed_points = np.empty(0, dtype='float')
            completed_values = np.empty(0, dtype='float')
            tag, Work, calc_in = ps.send_recv(H0)

        # Otherwise, wait for more completed work from the workers.
        else:
            tag, Work, calc_in = ps.recv()

    return H0, persis_info, FINISHED_PERSISTENT_GEN_TAG

