"""This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.
"""

import os
import ctypes
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
#
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "vtmop_libe.so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ["-fPIC", "-shared", "-O3", "-std=legacy"]
_ordered_dependencies = [
    "blas.f",
    "shared_modules.f90",
    "qnstop.f90",
    "lapack.f",
    "linear_shepard.f90",
    "slatec.f",
    "delsparse.f90",
    "vtmop.f90",
    "vtmop_libe.f90",
    "vtmop_libe_c_wrapper.f90", ]
#
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the existing object. If that fails, recompile and then try.
try:
    clib = ctypes.CDLL(_path_to_lib)
except Exception:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = " ".join([_fort_compiler] + _compile_options + ["-o", _shared_object_name] + _ordered_dependencies)
    if _verbose:
        print("Running system command with arguments")
        print("  ", _command)
    # Run the compilation command.
    import subprocess

    subprocess.run(_command, shell=True, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


class vtmop_libe_mod:
    """"""

    # ----------------------------------------------
    # Wrapper for the Fortran subroutine VTMOP_LIBE_INIT

    def vtmop_libe_init(self, d, p, lb, ub, lopt_budget, decay, des_tol, eps, epsw, obj_tol, min_radf, trust_radf,
                        obj_bounds, pmode):
        """! This subroutine initializes a VTMOP object for tracking the adaptive
        ! weighting scheme described in
        !
        ! Deshpande, Shubhangi, Layne T. Watson, and Robert A. Canfield.
        ! "Multiobjective optimization using an adaptive weighting scheme."
        ! Optimization Methods and Software 31.1 (2016): 110-133.
        !
        ! This is a wrapper for the subroutine VTMOP_INIT in VTMOP_MOD.
        !
        !
        ! On input:
        !
        ! D is the dimension of the design space.
        !
        ! P is the dimension of the objective space.
        !
        ! LB(1:D) is the real vector of lower bound constraints for the
        !    D design variables.
        !
        ! UB(1:D) is the real vector of upper bound constraints for the
        !    D design variables.
        !
        !
        ! On output:
        !
        ! IERR is an integer error flag.
        !
        ! Hundreds digit:
        !  000 : Normal output. Successful initialization of VTMOP object.
        !
        !  1xx : Errors detected.
        !   Tens digit:
        !     11x : The input parameters contained illegal dimensions or values.
        !       Ones digit:
        !         110 : D (design dimension) must be a positive integer.
        !         111 : P (objective dimension) must be at least two.
        !         112 : The lead dimension of LB(:) must match D.
        !         113 : The lead dimension of UB(:) must match D.
        !         114 : LB(:) must be elementwise strictly less than UB(:) - DES_TOL.
        !     12x : The optional dummy arguments contained illegal values.
        !       Ones digit:
        !         123 : LOPT_BUDGET must be positive.
        !         124 : DECAY must be in the range (EPS, 1-EPS).
        !         125 : TRUST_RADF must be larger than or equal to MIN_RADF.
        !         129 : If OBJ_BOUNDS is given, then it must have dimensions P by 2
        !               and all OBJ_BOUNDS(:,1) < OBJ_BOUNDS(:,2).
        !     13x : A memory allocation error has occurred.
        !       Ones digit:
        !         130 : A memory allocation error occurred.
        !         131 : A memory deallocation error occurred.
        !
        !  9xx : A checkpointing error has occurred.
        !    901 : WARNING: the VTMOP object was successfully recovered from the
        !          checkpoint but does not match the input data.
        !
        !  The following error codes are returned by VTMOP_CHKPT_NEW. Further details
        !  can be found in the header for VTMOP_CHKPT_NEW.
        !    91x : The VTMOP passed to the checkpoint was invalid.
        !    92x : Error creating the checkpoint file.
        !
        !  The following error codes are returned by VTMOP_CHKPT_RECOVER, when
        !  VTMOP_INIT is called in recovery mode. Further details can be found in
        !  the header for VTMOP_CHKPT_RECOVER.
        !    95x : Error reading data from the checkpoint file.
        !    96x : A memory management error occurred during recovery.
        !
        !
        ! Optional input arguments.
        !
        ! LOPT_BUDGET is an integer input, which specifies the budget for the
        !    local optimization subroutine. The default value for LOPT_BUDGET is
        !    2500 surrogate evaluations. Note that this value is not saved during
        !    checkpointing, and must be reset by the user when recovery mode is
        !    active, whenever a non-default value is desired.
        !
        ! DECAY is a real input specifying the decay rate for the local
        !    trust region (LTR) radius. This value affects how many times an
        !    isolated point can be the center of a LTR before it is discarded.
        !    By default, DECAY = 0.5.
        !
        ! DES_TOL is the tolerance for the design space. A design point that
        !    is within DES_TOL of an evaluated design point will not be reevaluated.
        !    The default value for DES_TOL is the square-root of the working precision
        !    EPS. Note that any value that is smaller than the working precsion EPS
        !    will be ignored and EPS will be used.
        !
        ! EPS is a real input, which specifies the working precision of the
        !    machine. The default value for EPS is SQRT(EPSILON), where EPSILON
        !    is the unit roundoff. Note that if the value supplied is smaller than
        !    the default value then the default value will be used.
        !
        ! EPSW is a small positive number, which is used as the fudge factor for
        !    zero-valued weights. A zero-valued weight does not guarantee Pareto
        !    optimality. Therefore, all zero weights are set to EPSW. The appropriate
        !    value of EPSW is problem dependent. By default, EPSW is the fourth root
        !    of EPSILON (the unit roundoff). Note that any value that is smaller
        !    than SQRT(EPSILON) is ignored and SQRT(EPSILON) will be used.
        !
        ! OBJ_TOL is the tolerance for the objective space. An objective point
        !    that is within OBJ_TOL of being dominated by another objective point
        !    will be treated as such. The default value of OBJ_TOL is the
        !    square-root of EPS. This value should be strictly greater than the
        !    value of EPS. Note that any value that is smaller than the
        !    working precsion EPS will be ignored and EPS will be used.
        !
        ! OBJ_BOUNDS(1:P,1:2) is an optional real P by 2 array, whose first column
        !    is a list of lower bounds and whose second column is a list of upper
        !    bounds on the range of interesting objective values. When present,
        !    this value is used to prune the list of potential LTR centers, so
        !    that only objective values in the specified range are used when
        !    looking for gaps in the Pareto. In particular, an objective value
        !    F(x) will be considered for a LTR center if and only if
        !    OBJ_BOUNDS(I,1) .LE. F_I(x) .LE. OBJ_BOUNDS(I,2) for all I = 1, ..., P.
        !    By default, there are no bounds on the interesting range. Note that
        !    this value is intentionally not saved during checkpointing, and must
        !    be reset by the user when recovery mode is active, whenever a
        !    non-default value is desired. This is the only input, for which
        !    changing the value after loading from a previous checkpoint is not
        !    ill-advised.
        !
        ! MIN_RADF is the smallest value for the fraction r defining the trust region
        !    box dimensions r * (UB - LB), before an isolated point is abandoned.
        !    By default, MIN_RADF = 0.1 * TRUST_RADF, and is also set to this default
        !    value if it is less than DES_TOL. After MIN_RADF and TRUST_RADF are set,
        !    MIN_RADF < TRUST_RADF must hold.
        !
        ! TRUST_RADF defines the initial trust region centered at an isolated
        !    point X as [X - TRUST_RADF * (UB - LB), X + TRUST_RADF * (UB - LB)]
        !    intersected with [LB, UB].  By default, TRUST_RADF = 0.2, and is also set
        !    to this value if the value given is outside the interval
        !    (DES_TOL, 1 - DES_TOL).
        !
        ! PMODE is a logical input that specifies whether or not iteration tasks
        !    should be performed in parallel. By default, PMODE = .FALSE. Note
        !    that this value is not saved during checkpointing, and must be reset
        !    by the user when recovery mode is active, whenever a non-default value
        !    is desired.
        !
        ! In recovery mode the inputs D, P, LB, and UB are still referenced
        !    (for sanity checks). Also, the procedure arguments are still
        !    needed to recover the procedure settings. No other optional
        !    arguments are referenced.
        !"""

        # Setting up "d"
        if type(d) is not ctypes.c_int:
            d = ctypes.c_int(d)

        # Setting up "p"
        if type(p) is not ctypes.c_int:
            p = ctypes.c_int(p)

        # Setting up "lb"
        if ((not issubclass(type(lb), numpy.ndarray)) or (not numpy.asarray(lb).flags.f_contiguous)
                or (not (lb.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'lb' was not an f_contiguous NumPy array " +
                          "of type 'ctypes.c_double' (or equivalent). Automatically " +
                          "converting (probably creating a full copy).")
            lb = numpy.asarray(lb, dtype=ctypes.c_double, order="F")
        lb_dim_1 = ctypes.c_int(lb.shape[0])

        # Setting up "ub"
        if ((not issubclass(type(ub), numpy.ndarray)) or (not numpy.asarray(ub).flags.f_contiguous)
                or (not (ub.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'ub' was not an f_contiguous NumPy array " +
                          "of type 'ctypes.c_double' (or equivalent). Automatically " +
                          "converting (probably creating a full copy).")
            ub = numpy.asarray(ub, dtype=ctypes.c_double, order="F")
        ub_dim_1 = ctypes.c_int(ub.shape[0])

        # Setting up "ierr"
        ierr = ctypes.c_int()

        # Setting up "lopt_budget"
        if type(lopt_budget) is not ctypes.c_int:
            lopt_budget = ctypes.c_int(lopt_budget)

        # Setting up "decay"
        if type(decay) is not ctypes.c_double:
            decay = ctypes.c_double(decay)

        # Setting up "des_tol"
        if type(des_tol) is not ctypes.c_double:
            des_tol = ctypes.c_double(des_tol)

        # Setting up "eps"
        if type(eps) is not ctypes.c_double:
            eps = ctypes.c_double(eps)

        # Setting up "epsw"
        if type(epsw) is not ctypes.c_double:
            epsw = ctypes.c_double(epsw)

        # Setting up "obj_tol"
        if type(obj_tol) is not ctypes.c_double:
            obj_tol = ctypes.c_double(obj_tol)

        # Setting up "min_radf"
        if type(min_radf) is not ctypes.c_double:
            min_radf = ctypes.c_double(min_radf)

        # Setting up "trust_radf"
        if type(trust_radf) is not ctypes.c_double:
            trust_radf = ctypes.c_double(trust_radf)

        # Setting up "obj_bounds"
        if ((not issubclass(type(obj_bounds), numpy.ndarray)) or (not numpy.asarray(obj_bounds).flags.f_contiguous)
                or (not (obj_bounds.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'obj_bounds' was not an f_contiguous " +
                          "NumPy array of type 'ctypes.c_double' (or equivalent). " +
                          "Automatically converting (probably creating a full copy).")
            obj_bounds = numpy.asarray(obj_bounds, dtype=ctypes.c_double, order="F")
        obj_bounds_dim_1 = ctypes.c_int(obj_bounds.shape[0])
        obj_bounds_dim_2 = ctypes.c_int(obj_bounds.shape[1])

        # Setting up "pmode"
        if type(pmode) is not ctypes.c_bool:
            pmode = ctypes.c_bool(pmode)

        # Call C-accessible Fortran wrapper.
        clib.c_vtmop_libe_init(
            ctypes.byref(d),
            ctypes.byref(p),
            ctypes.byref(lb_dim_1),
            ctypes.c_void_p(lb.ctypes.data),
            ctypes.byref(ub_dim_1),
            ctypes.c_void_p(ub.ctypes.data),
            ctypes.byref(ierr),
            ctypes.byref(lopt_budget),
            ctypes.byref(decay),
            ctypes.byref(des_tol),
            ctypes.byref(eps),
            ctypes.byref(epsw),
            ctypes.byref(obj_tol),
            ctypes.byref(min_radf),
            ctypes.byref(trust_radf),
            ctypes.byref(obj_bounds_dim_1),
            ctypes.byref(obj_bounds_dim_2),
            ctypes.c_void_p(obj_bounds.ctypes.data),
            ctypes.byref(pmode),
        )

        # Return final results, 'INTENT(OUT)' arguments only.
        return ierr.value

    # ----------------------------------------------
    # Wrapper for the Fortran subroutine VTMOP_LIBE_GENERATE

    def vtmop_libe_generate(self, d, p, lb, ub, des_pts, obj_pts, isnb, snb, onb, obj_bounds):
        """! The VTMOP_LIBE_GENERATE subroutine produces static batches of candidate
        ! points using VTMOP_MOD. Each call to VTMOP_LIBE_GENERATE performs a half
        ! iteration of the VTMOP_SOLVE algorithm (from VTMOP_LIB), using a Latin
        ! hypercube design of size SNB.
        !
        ! VTMOP_INITIALIZE must still be used to initialize the VTMOP object, and
        ! VTMOP_FINALIZE must be used to terminate and post process the results.
        !
        !
        ! On input:
        !
        ! D is the dimension of the design space.
        !
        ! P is the dimension of the objective space.
        !
        ! LB(1:D) is the real vector of lower bound constraints for the
        !    D design variables.
        !
        ! UB(1:D) is the real vector of upper bound constraints for the
        !    D design variables.
        !
        ! DES_PTS(D,N) is the current list of design points.
        !
        ! OBJ_PTS(P,N) is the current list of objective points.
        !
        ! ISNB is the preferred initial search batch size and is used as the
        !    size of the Latin hypercube design in the zeroeth iteration
        !    (initial search). ISNB should be much greater than SNB.
        !
        ! SNB is the preferred search batch size and is used as the
        !    size of the Latin hypercube design in the search phase.
        !
        ! ONB is the preferred optimization batch size and is used to
        !    pad out the batch of candidate designs in the optimization phase.
        !
        !
        ! On output:
        !
        ! LBATCH returns the size of the next batch.
        !
        ! BATCHX(D,LBATCH) is the next batch of design points to evaluate.
        !
        ! IERR is an integer error flag. Error codes carried from worker
        !    subroutines. In general:
        !
        !  000 : successful iteration.
        !  003 : stopping criterion 3 achieved.
        !
        !  1xx : illegal input error.
        !   Tens digit:
        !    11x : The VTMOP object contained illegal dimensions or values, and
        !          may not have been properly initialized.
        !       Ones digit:
        !         110 : The problem dimensions D and P are not legal.
        !         111 : The internal arrays have not been allocated.
        !         112 : The internal arrays have been initialized, but to invalid
        !               dimensions.
        !         113 : The lower and upper bound constraints contain illegal values,
        !               subject to the design space tolerance.
        !    12x : Illegal values in other inputs.
        !         120 : The lead dimension of DES_PTS does not match the data in VTMOP.
        !         121 : The lead dimension of OBJ_PTS does not match the data in VTMOP.
        !         122 : The second dimensions of DES_PTS and OBJ_PTS do not match.
        !         123 : The preferred batch size NB must be nonnegative.
        !
        !    2xx, 3xx : Error in VTMOP_LTR or VTMOP_OPT, respectively.
        !
        !    5xx, 6xx, 7xx, 8xx : Error in DELAUNAYGRAPH, FIT_SURROGATES, LOCAL_OPT,
        !                         or LH_DESIGN, respectively. See documentation for
        !                         those procedures for more details.
        !
        !    9xx : Checkpointing error. See checkpointing subroutine documentation.
        !"""

        # Setting up "d"
        if type(d) is not ctypes.c_int:
            d = ctypes.c_int(d)

        # Setting up "p"
        if type(p) is not ctypes.c_int:
            p = ctypes.c_int(p)

        # Setting up "lb"
        if ((not issubclass(type(lb), numpy.ndarray)) or (not numpy.asarray(lb).flags.f_contiguous)
                or (not (lb.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'lb' was not an f_contiguous NumPy array " +
                          "of type 'ctypes.c_double' (or equivalent). Automatically " +
                          "converting (probably creating a full copy).")
            lb = numpy.asarray(lb, dtype=ctypes.c_double, order="F")
        lb_dim_1 = ctypes.c_int(lb.shape[0])

        # Setting up "ub"
        if ((not issubclass(type(ub), numpy.ndarray)) or (not numpy.asarray(ub).flags.f_contiguous)
                or (not (ub.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'ub' was not an f_contiguous NumPy array " +
                          "of type 'ctypes.c_double' (or equivalent). Automatically " +
                          "converting (probably creating a full copy).")
            ub = numpy.asarray(ub, dtype=ctypes.c_double, order="F")
        ub_dim_1 = ctypes.c_int(ub.shape[0])

        # Setting up "des_pts"
        if ((not issubclass(type(des_pts), numpy.ndarray)) or (not numpy.asarray(des_pts).flags.f_contiguous)
                or (not (des_pts.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'des_pts' was not an f_contiguous NumPy " +
                          "array of type 'ctypes.c_double' (or equivalent). Automatically " +
                          "converting (probably creating a full copy).")
            des_pts = numpy.asarray(des_pts, dtype=ctypes.c_double, order="F")
        des_pts_dim_1 = ctypes.c_int(des_pts.shape[0])
        des_pts_dim_2 = ctypes.c_int(des_pts.shape[1])

        # Setting up "obj_pts"
        if ((not issubclass(type(obj_pts), numpy.ndarray)) or (not numpy.asarray(obj_pts).flags.f_contiguous)
                or (not (obj_pts.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'obj_pts' was not an f_contiguous NumPy " +
                          "array of type 'ctypes.c_double' (or equivalent). Automatically " +
                          "converting (probably creating a full copy).")
            obj_pts = numpy.asarray(obj_pts, dtype=ctypes.c_double, order="F")
        obj_pts_dim_1 = ctypes.c_int(obj_pts.shape[0])
        obj_pts_dim_2 = ctypes.c_int(obj_pts.shape[1])

        # Setting up "isnb"
        if type(isnb) is not ctypes.c_int:
            isnb = ctypes.c_int(isnb)

        # Setting up "snb"
        if type(snb) is not ctypes.c_int:
            snb = ctypes.c_int(snb)

        # Setting up "onb"
        if type(onb) is not ctypes.c_int:
            onb = ctypes.c_int(onb)

        # Setting up "obj_bounds"
        if ((not issubclass(type(obj_bounds), numpy.ndarray)) or (not numpy.asarray(obj_bounds).flags.f_contiguous)
                or (not (obj_bounds.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings

            warnings.warn("The provided argument 'obj_bounds' was not an f_contiguous " +
                          "NumPy array of type 'ctypes.c_double' (or equivalent). " +
                          "Automatically converting (probably creating a full copy).")
            obj_bounds = numpy.asarray(obj_bounds, dtype=ctypes.c_double, order="F")
        obj_bounds_dim_1 = ctypes.c_int(obj_bounds.shape[0])
        obj_bounds_dim_2 = ctypes.c_int(obj_bounds.shape[1])

        # Setting up "lbatch"
        lbatch = ctypes.c_int()

        # Setting up "batchx"
        batchx = ctypes.c_void_p()
        batchx_dim_1 = ctypes.c_int()
        batchx_dim_2 = ctypes.c_int()

        # Setting up "ierr"
        ierr = ctypes.c_int()

        # Call C-accessible Fortran wrapper.
        clib.c_vtmop_libe_generate(
            ctypes.byref(d),
            ctypes.byref(p),
            ctypes.byref(lb_dim_1),
            ctypes.c_void_p(lb.ctypes.data),
            ctypes.byref(ub_dim_1),
            ctypes.c_void_p(ub.ctypes.data),
            ctypes.byref(des_pts_dim_1),
            ctypes.byref(des_pts_dim_2),
            ctypes.c_void_p(des_pts.ctypes.data),
            ctypes.byref(obj_pts_dim_1),
            ctypes.byref(obj_pts_dim_2),
            ctypes.c_void_p(obj_pts.ctypes.data),
            ctypes.byref(isnb),
            ctypes.byref(snb),
            ctypes.byref(onb),
            ctypes.byref(obj_bounds_dim_1),
            ctypes.byref(obj_bounds_dim_2),
            ctypes.c_void_p(obj_bounds.ctypes.data),
            ctypes.byref(lbatch),
            ctypes.byref(batchx_dim_1),
            ctypes.byref(batchx_dim_2),
            ctypes.byref(batchx),
            ctypes.byref(ierr),
        )

        # Post-processing "batchx"
        batchx_size = (batchx_dim_1.value) * (batchx_dim_2.value)
        if batchx_size > 0:
            batchx = numpy.array(
                ctypes.cast(batchx, ctypes.POINTER(ctypes.c_double * batchx_size)).contents, copy=False)
            batchx = batchx.reshape(batchx_dim_2.value, batchx_dim_1.value).T
        elif batchx_size == 0:
            batchx = numpy.zeros((batchx_dim_2.value, batchx_dim_1.value), dtype=ctypes.c_double, order="F")
        else:
            batchx = None

        # Return final results, 'INTENT(OUT)' arguments only.
        return lbatch.value, batchx, ierr.value


vtmop_libe_mod = vtmop_libe_mod()
