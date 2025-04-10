"""
A persistent generator using the uncertainty quantification capabilities in
`Tasmanian <https://github.com/ORNL/Tasmanian>`_.
"""

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as allocf
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools import parse_args
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "sparse_grid_batched",
    "sparse_grid_async",
]


def lex_le(x, y, tol=1e-12):
    """
    Returns True if x <= y lexicographically up to some tolerance.
    """
    cmp = np.fabs(x - y) > tol
    ind = np.argmax(cmp)
    if not cmp[ind]:
        return True
    return x[ind] <= y[ind]


def get_2D_insert_indices(x, y, x_ord=np.empty(0, dtype="int"), y_ord=np.empty(0, dtype="int"), tol=1e-12):
    """
    Finds the row indices in a 2D numpy array `x` for which the sorted values of `y` can be inserted
    into. If `x_ord` (resp. `y_ord`) is empty, then `x` (resp. `y`) must be lexicographically
    sorted. Otherwise, `x[x_ord]` (resp. `y[y_ord]`) must be lexicographically sorted. Complexity is
    O(x.shape[0] + y.shape[0]).
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    if x.size == 0:
        return np.zeros(y.shape[0], dtype="int")
    else:
        if x_ord.size == 0:
            x_ord = np.arange(x.shape[0], dtype="int")
        if y_ord.size == 0:
            y_ord = np.arange(y.shape[0], dtype="int")
        x_ptr = 0
        y_ptr = 0
        out_ord = np.empty(0, dtype="int")
        while y_ptr < y.shape[0]:
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


def get_2D_duplicate_indices(x, y, x_ord=np.empty(0, dtype="int"), y_ord=np.empty(0, dtype="int"), tol=1e-12):
    """
    Finds the row indices of a 2D numpy array `x` which overlap with `y`. If `x_ord` (resp. `y_ord`)
    is empty, then `x` (resp. `y`) must be lexicographically sorted. Otherwise, `x[x_ord]` (resp.
    `y[y_ord]`) must be lexicographically sorted.Complexity is O(x.shape[0] + y.shape[0]).
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    if x.size == 0:
        return np.empty(0, dtype="int")
    else:
        if x_ord.size == 0:
            x_ord = np.arange(x.shape[0], dtype="int")
        if y_ord.size == 0:
            y_ord = np.arange(y.shape[0], dtype="int")
        x_ptr = 0
        y_ptr = 0
        out_ord = np.empty(0, dtype="int")
        while y_ptr < y.shape[0] and x_ptr < x.shape[0]:
            # The case where y[k] <= max of x[k:end, :]
            xk = x[x_ord[x_ptr], :]
            yk = y[y_ord[y_ptr], :]
            if all(np.fabs(yk - xk) <= tol):
                out_ord = np.append(out_ord, x_ord[x_ptr])
                x_ptr += 1
            elif lex_le(xk, yk, tol=tol):
                x_ptr += 1
            else:
                y_ptr += 1
        return out_ord


def get_state(queued_pts, queued_ids, id_offset, new_points=np.array([]), completed_points=np.array([]), tol=1e-12):
    """
    Creates the data to be sent and updates the state arrays and scalars if new information
    (new_points or completed_points) arrives. Ensures that the output state arrays remain sorted if
    the input state arrays are already sorted.
    """
    if new_points.size > 0:
        new_points_ord = np.lexsort(np.rot90(new_points))
        new_points_ids = id_offset + np.arange(new_points.shape[0])
        id_offset += new_points.shape[0]
        insert_idx = get_2D_insert_indices(queued_pts, new_points, y_ord=new_points_ord, tol=tol)
        queued_pts = np.insert(queued_pts, insert_idx, new_points[new_points_ord], axis=0)
        queued_ids = np.insert(queued_ids, insert_idx, new_points_ids[new_points_ord], axis=0)

    if completed_points.size > 0:
        completed_ord = np.lexsort(np.rot90(completed_points))
        delete_ind = get_2D_duplicate_indices(queued_pts, completed_points, y_ord=completed_ord, tol=tol)
        queued_pts = np.delete(queued_pts, delete_ind, axis=0)
        queued_ids = np.delete(queued_ids, delete_ind, axis=0)

    return queued_pts, queued_ids, id_offset


def get_H0(gen_specs, refined_pts, refined_ord, queued_pts, queued_ids, tol=1e-12):
    """
    For runs following the first one, get the history array H0 based on the ordering in `refined_pts`
    """

    def approx_eq(x, y):
        return np.argmax(np.fabs(x - y)) <= tol

    num_ids = queued_ids.shape[0]
    H0 = np.zeros(num_ids, dtype=gen_specs["out"])
    refined_priority = np.flip(np.arange(refined_pts.shape[0], dtype="int"))
    rptr = 0
    for qptr in range(num_ids):
        while not approx_eq(refined_pts[refined_ord[rptr]], queued_pts[qptr]):
            rptr += 1
        assert rptr <= refined_pts.shape[0]
        H0["x"][qptr] = queued_pts[qptr]
        H0["sim_id"][qptr] = queued_ids[qptr]
        H0["priority"][qptr] = refined_priority[refined_ord[rptr]]
    return H0


# ========================
# Main generator functions
# ========================


def sparse_grid_batched(H, persis_info, gen_specs, libE_info):
    """
    Implements batched construction for a Tasmanian sparse grid,
    using the loop described in Tasmanian Example 09:
    `sparse grid example <https://github.com/ORNL/TASMANIAN/blob/master/InterfacePython/example_sparse_grids_09.py>`_

    """
    U = gen_specs["user"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    grid = U["tasmanian_init"]()  # initialize the grid
    allowed_refinements = [
        "setAnisotropicRefinement",
        "getAnisotropicRefinement",
        "setSurplusRefinement",
        "getSurplusRefinement",
        "none",
    ]
    assert (
        "refinement" in U and U["refinement"] in allowed_refinements
    ), f"Must provide a gen_specs['user']['refinement'] in: {allowed_refinements}"

    while grid.getNumNeeded() > 0:
        aPoints = grid.getNeededPoints()

        H0 = np.zeros(len(aPoints), dtype=gen_specs["out"])
        H0["x"] = aPoints

        # Receive values from manager
        tag, Work, calc_in = ps.send_recv(H0)
        if tag in [STOP_TAG, PERSIS_STOP]:
            break
        aModelValues = calc_in["f"]

        # Update surrogate on grid
        t = aModelValues.reshape((aModelValues.shape[0], grid.getNumOutputs()))
        t = t.flatten()
        t = np.atleast_2d(t).T
        grid.loadNeededPoints(t)

        if "tasmanian_checkpoint_file" in U:
            grid.write(U["tasmanian_checkpoint_file"])

        # set refinement, using user["refinement"] to pick the refinement strategy
        if U["refinement"] in ["setAnisotropicRefinement", "getAnisotropicRefinement"]:
            assert "sType" in U
            assert "iMinGrowth" in U
            assert "iOutput" in U
            grid.setAnisotropicRefinement(U["sType"], U["iMinGrowth"], U["iOutput"])
        elif U["refinement"] in ["setSurplusRefinement", "getSurplusRefinement"]:
            assert "fTolerance" in U
            assert "iOutput" in U
            assert "sCriteria" in U
            grid.setSurplusRefinement(U["fTolerance"], U["iOutput"], U["sCriteria"])

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def sparse_grid_async(H, persis_info, gen_specs, libE_info):
    """
    Implements asynchronous construction for a Tasmanian sparse grid,
    using the logic in the dynamic Tasmanian model construction function:
    `sparse grid dynamic example <https://github.com/ORNL/TASMANIAN/blob/master/Addons/tsgConstructSurrogate.hpp>`_

    """
    U = gen_specs["user"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    grid = U["tasmanian_init"]()  # initialize the grid
    allowed_refinements = ["getCandidateConstructionPoints", "getCandidateConstructionPointsSurplus"]
    assert (
        "refinement" in U and U["refinement"] in allowed_refinements
    ), f"Must provide a gen_specs['user']['refinement'] in: {allowed_refinements}"
    tol = U["_match_tolerance"] if "_match_tolerance" in U else 1.0e-12

    # Choose the refinement function based on U["refinement"].
    if U["refinement"] == "getCandidateConstructionPoints":
        assert "sType" in U
        assert "liAnisotropicWeightsOrOutput" in U
    if U["refinement"] == "getCandidateConstructionPointsSurplus":
        assert "fTolerance" in U
        assert "sRefinementType" in U

    def get_refined_points(g, U):
        if U["refinement"] == "getCandidateConstructionPoints":
            return g.getCandidateConstructionPoints(U["sType"], U["liAnisotropicWeightsOrOutput"])
        else:
            assert U["refinement"] == "getCandidateConstructionPointsSurplus"
            return g.getCandidateConstructionPointsSurplus(U["fTolerance"], U["sRefinementType"])
        # else:
        #     raise ValueError("Unknown refinement string")

    # Asynchronous helper and state variables.
    num_dims = grid.getNumDimensions()
    num_completed = 0
    offset = 0
    queued_pts = np.empty((0, num_dims), dtype="float")
    queued_ids = np.empty(0, dtype="int")

    # First run.
    grid.beginConstruction()
    init_pts = get_refined_points(grid, U)
    queued_pts, queued_ids, offset = get_state(queued_pts, queued_ids, offset, new_points=init_pts, tol=tol)
    H0 = np.zeros(init_pts.shape[0], dtype=gen_specs["out"])
    H0["x"] = init_pts
    H0["sim_id"] = np.arange(init_pts.shape[0], dtype="int")
    H0["priority"] = np.flip(H0["sim_id"])
    tag, Work, calc_in = ps.send_recv(H0)

    # Subsequent runs.
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Parse the points returned by the allocator.
        num_completed += calc_in["x"].shape[0]
        queued_pts, queued_ids, offset = get_state(
            queued_pts, queued_ids, offset, completed_points=calc_in["x"], tol=tol
        )

        # Compute the next batch of points (if they exist).
        new_pts = np.empty((0, num_dims), dtype="float")
        refined_pts = np.empty((0, num_dims), dtype="float")
        refined_ord = np.empty(0, dtype="int")
        if grid.getNumLoaded() < 1000 or num_completed > 0.2 * grid.getNumLoaded():
            # A copy is needed because the data in the calc_in arrays are not contiguous.
            grid.loadConstructedPoint(np.copy(calc_in["x"]), np.copy(calc_in["f"]))
            if "tasmanian_checkpoint_file" in U:
                grid.write(U["tasmanian_checkpoint_file"])
            refined_pts = get_refined_points(grid, U)
            # If the refined points are empty, then there is a stopping condition internal to the
            # Tasmanian sparse grid that is being triggered by the loaded points.
            if refined_pts.size == 0:
                break
            refined_ord = np.lexsort(np.rot90(refined_pts))
            delete_ind = get_2D_duplicate_indices(refined_pts, queued_pts, x_ord=refined_ord, tol=tol)
            new_pts = np.delete(refined_pts, delete_ind, axis=0)

        if new_pts.shape[0] > 0:
            # Update the state variables with the refined points and update the queue in the allocator.
            num_completed = 0
            queued_pts, queued_ids, offset = get_state(queued_pts, queued_ids, offset, new_points=new_pts, tol=tol)
            H0 = get_H0(gen_specs, refined_pts, refined_ord, queued_pts, queued_ids, tol=tol)
            tag, Work, calc_in = ps.send_recv(H0)
        else:
            tag, Work, calc_in = ps.recv()

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def get_sparse_grid_specs(user_specs, sim_f, num_dims, num_outputs=1, mode="batched"):
    """
    Helper function that generates the simulator, generator, and allocator specs as well as the
    persis_info dictionary to ensure that they are compatible with the custom generators in this
    script. The outputs should be used in the main libE() call.

    INPUTS:
        user_specs  (dict)   : a dictionary of user specs that is needed in the generator specs;
                               expects the key "tasmanian_init" whose value is a 0-argument lambda
                               that initializes an appropriate Tasmanian sparse grid object.

        sim_f       (func)   : a lambda function that takes in generator outputs (simulator inputs)
                               and returns simulator outputs.

        num_dims    (int)    : number of model inputs.

        num_outputs (int)    : number of model outputs.

        mode        (string) : can either be "batched" or "async".

    OUTPUTS:
        sim_specs   (dict) : a dictionary of simulation specs and also one of the inputs of libE().

        gen_specs   (dict) : a dictionary of generator specs and also one of the inputs of libE().

        alloc_specs (dict) : a dictionary of allocation specs and also one of the inputs of libE().

        persis_info (dict) : a dictionary containing common information that is passed to all
                             workers and also one of the inputs of libE().

    """

    assert "tasmanian_init" in user_specs
    assert mode in ["batched", "async"]

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
    }
    gen_out = [
        ("x", float, (num_dims,)),
        ("sim_id", int),
        ("priority", int),
    ]
    gen_specs = {
        "persis_in": [t[0] for t in gen_out] + ["f"],
        "out": gen_out,
        "user": user_specs,
    }
    alloc_specs = {
        "alloc_f": allocf,
        "user": {},
    }

    if mode == "batched":
        gen_specs["gen_f"] = sparse_grid_batched
        sim_specs["out"] = [("f", float, (num_outputs,))]
    if mode == "async":
        gen_specs["gen_f"] = sparse_grid_async
        sim_specs["out"] = [("x", float, (num_dims,)), ("f", float, (num_outputs,))]
        alloc_specs["user"]["active_recv_gen"] = True
        alloc_specs["user"]["async_return"] = True

    nworkers, _, _, _ = parse_args()
    persis_info = {}
    for i in range(nworkers + 1):
        persis_info[i] = {"worker_num": i}

    return sim_specs, gen_specs, alloc_specs, persis_info
