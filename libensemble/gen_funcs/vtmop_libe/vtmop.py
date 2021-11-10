"""
Wrapper for the VTMOP genfuncs drivers, for solving multiobjective
optimization problems with arbitrary number of objectives.
"""
import numpy as np
import ctypes
from sys import float_info
from libensemble.gen_funcs.vtmop_libe import vtmop_libe_mod


def vtmop_gen(H, persis_info, gen_specs, _):
    """
     This generator function solves multiobjective optimization problems
     with d design variables (subject to simple bound constraints) and
     p objectives using VTMOP.

     This generator produces batches of simulation inputs to evaluate, using
     the Fortran package VTMOP. In order to facilitate an interface between
     Fortran and Python, wrappers have been generated for a subset of VTMOP's
     functionality using fmodpy.

     The first time that this generator is called on your machine, the
     necessary VTMOP libraries will be automatically built using gfortran.
     If a different compiler is desired, edit lines 16 and 20 of
     gen_funcs/vtmop_libe/vtmop_libe_python_wrapper.py to specify your
     compiler and the appropriate compiler flags, respectively.

     After the Fortran and C binaries have been built, VTMOP's Python wrapper
     will be automatically imported every time the generator is called without
     need to recompile. Note that this wrapper uses a .so file, which is only
     supported by Mac and Linux/Unix systems (will not work on Windows).

     All inputs and arguments to VTMOP are specified through the
     gen_specs['user'] dictionary.

     Required args:

         gen_specs['user']['d'] (int): is the dimension of the design space.

         gen_specs['user']['p'] (int): is the number of objectives.

         gen_specs['user']['lb'] (np.ndarray): is a 1d array of length D,
             containing the lower bound constraints on the design variables.

         gen_specs['user']['ub'] (np.ndarray): is a 1d array of length D,
             containing the upper bound constraints on the design variables.
             Note that gen_specs['user']['ub'] must be elementwise greater than
             gen_specs['user']['lb'] (up to the design tolerance, specified
             below).

         gen_specs['user']['new_run'] (bool): should be True if you are
             starting a new run, and False if you would like to reload an old
             run using the checkpoint file 'vtmop.chkpt' in your current
             working directory.

         gen_specs['user']['isnb'] (int): is the preferred initial search batch
             size and is used as the size of the Latin hypercube design in the
             zeroeth iteration (initial search). ISNB should be much greater
             than SNB and a multiple of the number of simulation workers.
             Empirically, a good starting place is
             gen_specs['user']['isnb'] ~ 1000. If a large initial database
             is given, it is possible to set gen_specs['user']['isnb'] = 0.

         gen_specs['user']['snb'] (int): is the preferred search batch size
             for all other search steps and is used as the size of the Latin
             hypercube design in the search phase. Empirically, a good starting
             place is gen_specs['user']['snb'] ~ 2 * d.

         gen_specs['user']['onb'] (int): is the preferred batch size for all
             optimization phases. VTMOP will pad out batches of candidates in
             order to produce a multiple of gen_specs['user']['onb'] candidates
             in each iteration. In almost every case, gen_specs['user']['onb']
             should be equal to the number of available simulation workers.

     Optional args:

         gen_specs['user']['lopt_budget'] (int): specifies the budget for GPS
             optimizer iteration when solving the surrogate problems. The
             default value is LOPT_BUDGET is 2500 surrogate evaluations.

         gen_specs['user']['decay'] (float): specifies the decay rate for the
             local trust region (LTR) radius. This value affects how many
             times an isolated point can be the center of a LTR before it i
             discarded. By default, DECAY = 0.5.

         gen_specs['user']['des_tol'] (float): the tolerance for the design
             space. A design point that is within DES_TOL of another evaluated
             design point will not be reevaluated. The default value for
             DES_TOL is the square-root of the working precision EPS (below).
             Note that any value that is smaller than the working precsion EPS
             will be ignored and EPS will be used.

         gen_specs['user']['eps'] (float): specifies the working precision of the
             machine. The default value for EPS is SQRT(EPSILON), where EPSILON
             is the unit roundoff. Note that if the value supplied is smaller
             than the default value then the default value will be used.

         gen_specs['user']['epsw'] (float): is a small positive number, which is
             used as the fudge factor for zero-valued weights. A zero-valued
             weight does not guarantee Pareto optimality. Therefore, all zero
             weights are set to EPSW. The appropriate value of EPSW is problem
             dependent. By default, EPSW is the fourth root of EPSILON (the
             unit roundoff). Note that any value that is smaller than
             SQRT(EPSILON) is ignored and SQRT(EPSILON) will be used.

         gen_specs['user']['obj_tol'] (float): is the tolerance for the
             objective space. An objective point that is within OBJ_TOL of
             being dominated by another objective point will be treated as such.
             The default value of OBJ_TOL is the square-root of EPS. This
             value should be strictly greater than the value of EPS. Note that
             any value that is smaller than the working precsion EPS will be
             ignored and EPS will be used.

         gen_specs['user']['obj_bounds'] (np.ndarray) is an optional 2d array
             of dimensions p by 2, whose first column is a list of lower
             bounds and whose second column is a list of upper bounds on the
             range of interesting objective values. When present, this value
             is used to prune the list of potential LTR centers, so
             that only objective values in the specified range are used when
             looking for gaps in the Pareto. In particular, an objective value
             F(x) will be considered for a LTR center if and only if
             OBJ_BOUNDS(I,1) .LE. F_I(x) .LE. OBJ_BOUNDS(I,2) for all
             I = 1, ..., P. By default, there are no bounds on the interesting
             range. Note that this value is intentionally not saved during
             checkpointing, and must be reset by the user when recovery mode is
             active, whenever a non-default value is desired. This is the only
             input, for which changing the value after loading from a previous
             checkpoint is not ill-advised.

         gen_specs['user']['min_radf'] (float): is the smallest value for the
             fraction r defining the trust region box dimensions r * (UB - LB),
             before an isolated point is abandoned. By default,
             MIN_RADF = 0.1 * TRUST_RADF, and is also set to this default
             value if it is less than DES_TOL. After MIN_RADF and TRUST_RADF
             are set, MIN_RADF < TRUST_RADF must hold.

         gen_specs['user']['trust_radf'] (float): defines the initial trust
             region centered at an isolated point X as
             [X - TRUST_RADF * (UB - LB), X + TRUST_RADF * (UB - LB)]
             intersected with [LB, UB].  By default, TRUST_RADF = 0.2, and is
             also set to this value if the value given is outside the interval
             (DES_TOL, 1 - DES_TOL).

         gen_specs['user']['pmode'] (bool): specifies whether or not iteration
             tasks should be performed in parallel. By default, PMODE = False.
             If changed to True, VTMOP uses OpenMP parallelism to perform
             iteration tasks, and uses the environment variable OMP_NUM_THREADS
             to determine the number of parallel threads per generation.

    Function values/simulation database; VTMOP can also be initialized with
    a precomputed database of n pre-evaluated simulations. To do so, initialize
    a history array H0 as follows:

         H0['x'] (np.ndarray): is a 2d n-by-d history array containing a list
             of precomputed design points.

         H0['f'] (np.ndarray): is a 2d n-by-p history array containing the list
             of corresponding precomputed objective values.

         H0['sim_id'] (np.ndarray): Set H0['sim_id'] = range(n).

         H0[['given', 'returned']] = True

     A checkpoint file (``vtmop.chkpt``) will be generated in the calling
     directory to save the status of VTMOP, and can be used to recover in
     the event of a crash, or can be used to reload any run after pausing.

    """

    # First get the problem dimensions and data
    d = np.int32(gen_specs["user"]["d"])  # design dimension
    p = np.int32(gen_specs["user"]["p"])  # objective dimension
    ub = np.asarray(gen_specs["user"]["ub"], dtype=ctypes.c_double, order="f")
    lb = np.asarray(gen_specs["user"]["lb"], dtype=ctypes.c_double, order="f")
    new_run = bool(gen_specs["user"]["new_run"])  # Are we starting a new run?
    isnb = np.int32(gen_specs["user"]["isnb"])  # first iter batch size
    snb = np.int32(gen_specs["user"]["snb"])  # search batch size
    onb = np.int32(gen_specs["user"]["onb"])  # preferred opt batch size
    # Check for any optional inputs
    lopt_budget = np.int32(2500)
    if "lopt_budget" in gen_specs["user"].keys():
        lopt_budget = gen_specs["user"]["lopt_budget"]
    decay = ctypes.c_double(0.5)
    if "decay" in gen_specs["user"].keys():
        decay = ctypes.c_double(gen_specs["user"]["decay"])
    eps = ctypes.c_double(np.sqrt(float_info.epsilon))
    if "eps" in gen_specs["user"].keys():
        eps = ctypes.c_double(gen_specs["user"]["eps"])
    des_tol = ctypes.c_double(np.sqrt(eps))
    if "des_tol" in gen_specs["user"].keys():
        des_tol = ctypes.c_double(gen_specs["user"]["des_tol"])
    epsw = ctypes.c_double(np.sqrt(np.sqrt(float_info.epsilon)))
    if "epsw" in gen_specs["user"].keys():
        epsw = ctypes.c_double(gen_specs["user"]["epsw"])
    obj_tol = ctypes.c_double(np.sqrt(eps))
    if "obj_tol" in gen_specs["user"].keys():
        obj_tol = ctypes.c_double(gen_specs["user"]["obj_tol"])
    obj_bounds = np.zeros((p, 2), dtype=ctypes.c_double, order="f")
    obj_bounds[:, 0] = -float_info.max
    obj_bounds[:, 1] = float_info.max
    if "obj_bounds" in gen_specs["user"].keys():
        obj_bounds = np.asarray(gen_specs["user"]["obj_bounds"], dtype=ctypes.c_double)
    trust_radf = ctypes.c_double(0.2)
    trust_radf_float = 0.2
    if "trust_radf" in gen_specs["user"].keys():
        trust_radf = ctypes.c_double(gen_specs["user"]["trust_radf"])
        trust_radf_float = float(gen_specs["user"]["trust_radf"])
    min_radf = ctypes.c_double(0.1 * trust_radf_float)
    if "min_radf" in gen_specs["user"].keys():
        min_radf = ctypes.c_double(gen_specs["user"]["min_radf"])
    pmode = bool(False)
    if "pmode" in gen_specs["user"].keys():
        pmode = bool(gen_specs["user"]["pmode"])

    # Copy the history arrays
    des_pts = np.asarray(H["x"].T, dtype=ctypes.c_double, order="f")
    obj_pts = np.asarray(H["f"].T, dtype=ctypes.c_double, order="f")

    # Is this the beginning of a new run? Reinitialize VTMOP
    if new_run:
        ierror = vtmop_libe_mod.vtmop_libe_init(d, p, lb, ub, lopt_budget, decay, des_tol, eps, epsw, obj_tol, min_radf,
                                                trust_radf, obj_bounds, pmode)
        if ierror != 0:
            return
        else:
            gen_specs["user"]["new_run"] = bool(False)

        # Get the first batch right away
        lbatch, batchx, ierr = vtmop_libe_mod.vtmop_libe_generate(d, p, lb, ub, des_pts, obj_pts, isnb, snb, onb,
                                                                  obj_bounds)
        # If lbatch is 0, immediately generate the second batch
        if lbatch == 0:
            lbatch, batchx, ierr = vtmop_libe_mod.vtmop_libe_generate(d, p, lb, ub, des_pts, obj_pts, isnb, snb, onb,
                                                                      obj_bounds)
    # Is this a resuming run? Just generate a batch of points
    else:
        # Generate a batch
        lbatch, batchx, ierr = vtmop_libe_mod.vtmop_libe_generate(d, p, lb, ub, des_pts, obj_pts, isnb, snb, onb,
                                                                  obj_bounds)

    # Read record
    Out = np.zeros(lbatch, dtype=gen_specs["out"])
    for i in range(0, lbatch):
        Out["x"][i, :] = batchx[:, i]

    return Out, persis_info
