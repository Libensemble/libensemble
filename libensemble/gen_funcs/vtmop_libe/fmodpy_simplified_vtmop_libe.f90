MODULE VTMOP_LIBE_MOD


SUBROUTINE VTMOP_LIBE_INIT ( D , P , LB , UB , IERR , LOPT_BUDGET , DECAY , DES_TOL , EPS , EPSW , OBJ_TOL , MIN_RADF , TRUST_RADF , OBJ_BOUNDS , PMODE )
! This subroutine initializes a VTMOP object for tracking the adaptive
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
!
USE VTMOP_MOD
USE ISO_FORTRAN_ENV
IMPLICIT NONE
! Input parameters.
INTEGER , INTENT ( IN ) : : D
INTEGER , INTENT ( IN ) : : P
REAL ( KIND = R8 ) , INTENT ( IN ) : : LB ( : )
REAL ( KIND = R8 ) , INTENT ( IN ) : : UB ( : )
! Output parameters.
INTEGER , INTENT ( OUT ) : : IERR
! Optional parameters from VTMOP (mandatory for this interface).
INTEGER , INTENT ( IN ) : : LOPT_BUDGET
LOGICAL , INTENT ( IN ) : : PMODE
REAL ( KIND = R8 ) , INTENT ( IN ) : : DECAY
REAL ( KIND = R8 ) , INTENT ( IN ) : : DES_TOL
REAL ( KIND = R8 ) , INTENT ( IN ) : : EPS
REAL ( KIND = R8 ) , INTENT ( IN ) : : EPSW
REAL ( KIND = R8 ) , INTENT ( IN ) : : OBJ_TOL
REAL ( KIND = R8 ) , INTENT ( IN ) : : MIN_RADF
REAL ( KIND = R8 ) , INTENT ( IN ) : : TRUST_RADF
REAL ( KIND = R8 ) , INTENT ( IN ) : : OBJ_BOUNDS ( : , : )

! Local variables.
TYPE ( VTMOP_TYPE ) : : VTMOP

! Initialize the VTMOP status object.

END SUBROUTINE VTMOP_LIBE_INIT

SUBROUTINE VTMOP_LIBE_GENERATE ( D , P , LB , UB , DES_PTS , OBJ_PTS , ISNB , SNB , ONB , OBJ_BOUNDS , LBATCH , BATCHX , IERR )
! The VTMOP_LIBE_GENERATE subroutine produces static batches of candidate
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
!
USE VTMOP_MOD
USE ISO_FORTRAN_ENV
IMPLICIT NONE
! Input parameters.
INTEGER , INTENT ( IN ) : : D
INTEGER , INTENT ( IN ) : : P
REAL ( KIND = R8 ) , INTENT ( IN ) : : LB ( : )
REAL ( KIND = R8 ) , INTENT ( IN ) : : UB ( : )
REAL ( KIND = R8 ) , INTENT ( IN ) : : DES_PTS ( : , : )
REAL ( KIND = R8 ) , INTENT ( IN ) : : OBJ_PTS ( : , : )
INTEGER , INTENT ( IN ) : : ISNB
INTEGER , INTENT ( IN ) : : SNB
INTEGER , INTENT ( IN ) : : ONB
REAL ( KIND = R8 ) , INTENT ( IN ) : : OBJ_BOUNDS ( : , : )
! Output parameters.
INTEGER , INTENT ( OUT ) : : LBATCH
REAL ( KIND = R8 ) , ALLOCATABLE , INTENT ( OUT ) : : BATCHX ( : , : )
INTEGER , INTENT ( OUT ) : : IERR
! Local variables.
TYPE ( VTMOP_TYPE ) : : VTMOP
INTEGER : : I , J , K
INTEGER : : NW
REAL ( KIND = R8 ) : : LTR_LB ( D )
REAL ( KIND = R8 ) : : LTR_UB ( D )
REAL ( KIND = R8 ) , ALLOCATABLE : : TMP ( : , : )
REAL ( KIND = R8 ) , ALLOCATABLE : : WEIGHTS ( : , : )
! BLAS function for computing Euclidean distance.
REAL ( KIND = R8 ) , EXTERNAL : : DNRM2

! Recover the VTMOP status object.

! Check for illegal input parameter values.
! Check for illegal dimensions to DES_PTS and OBJ_PTS.

! Perform a half-iteration and request the next batch, based on the state of
! the VTMOP object.

! There are two cases.
! Execute the search phase.

! Choose the most isolated point(s), and construct the LTR(s).

! Get the batch size, using a special rule for the first iteration
! and respecting the preferred batch size SNB.
! In first iteration, generate a much larger LH_DESIGN.
! In later iterations, generate a smaller LH_DESIGN.
! Request a static exploration of the LTR.
! Allocate the output array.
! Filter out any redundant points.
! Check for any repeated design points.
! If the point is not repeated, add it to the output array.
! Check the size of the output array.
! No resizing is required, free the temporary array (typical case).
! Resize the output array (rare case).
! Free the temporary array.
! Execute the optimization phase.

! Recover the last LTR.

! Add additional weights at random in order to match ONB.
! Generate twice as many points as needed, in case several points
! produce redundant solutions.
! Generate a uniform sample from the probability simplex
! Make a copy of VTMOP%WEIGHTS.
! Reallocate the VTMOP%WEIGHTS array.
! Copy in the full list of weights.
! Free the temporary arrays.

! Optimize the surrogates in the LTR.

! If less than NW + ONB - (NW mod ONB) points were returned, return all the
! candidates.
! Otherwise, return the first NW + ONB - (NW mod ONB) candidates.

! Free the temporary array.
! Get the size of BATCHX.

END SUBROUTINE VTMOP_LIBE_GENERATE

END MODULE VTMOP_LIBE_MOD