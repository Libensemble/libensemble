! Module and subroutines implementing an adaptive weighting scheme for
! generating uniformly spaced points on the Pareto front for multiobjective
! optimization problems.
!
! Created : 5/20 by T. H. Chang (THC)
! Updates : 4/21 (THC)
MODULE VTMOP_MOD
! Use the R8 data type from HOMPACK90 for approximately 64-bit precision on
! all known machines.
USE REAL_PRECISION, ONLY : R8

! The default scope for VTMOP_MOD is public.
PUBLIC
! The following module arrays and pointers are private.
PRIVATE :: VTMOP_MOD_WEIGHTS, VTMOP_MOD_SURROGATES
! The following LSHEP (ACM TOMS Algorithm 905) parameters are private.
PRIVATE :: LSHEP_D, LSHEP_P, LSHEP_N_PTS, LSHEP_N_TABOO, LSHEP_A, &
           LSHEP_DES_TOL, LSHEP_FVALS, LSHEP_RW, LSHEP_SCALE,     &
           LSHEP_SHIFT, LSHEP_TABOO, LSHEP_XVALS
! The following auxiliary subroutines are private.
PRIVATE :: SCALAR_FUNC, SURROGATE_FUNC

! The derived data VTMOP_TYPE type carries metadata about the multiobjective
! optimization problem between iterations of the algorithm.
TYPE VTMOP_TYPE
   ! Contents of the VTMOP_TYPE data object.
   INTEGER :: D, P ! Problem dimensions.
   INTEGER :: ITERATE ! Total number of iterations elapsed.
   INTEGER :: LCLIST ! Length of CLIST(:,:) array.
   INTEGER :: LOPT_BUDGET ! Budget for the local optimizer.
   LOGICAL :: CHKPT ! Checkpointing mode.
   LOGICAL :: PMODE ! Parallel execution mode.
   REAL(KIND=R8) :: DECAY ! Rate of decay for the local trust region (LTR).
   REAL(KIND=R8) :: DES_TOL ! Design point tolerance.
   REAL(KIND=R8) :: EPS ! Working precision for the problem.
   REAL(KIND=R8) :: EPSW ! Fudge factor for zero weights.
   REAL(KIND=R8) :: MIN_RADF ! Minimum LTR radius as a fraction of UB-LB.
   REAL(KIND=R8) :: OBJ_TOL ! Objective point tolerance.
   REAL(KIND=R8) :: TRUST_RADF ! Initial LTR radius as a fraction of UB-LB.
   REAL(KIND=R8), ALLOCATABLE :: CLIST(:,:) ! Previously used LTR centers.
   REAL(KIND=R8), ALLOCATABLE :: LB(:), UB(:) ! Lower/upper design bounds.
   REAL(KIND=R8), ALLOCATABLE :: OBJ_BOUNDS(:,:) ! Lower/upper obj bounds.
   REAL(KIND=R8), ALLOCATABLE :: WEIGHTS(:,:) ! Adaptive weights.

   ! Pointers to procedures that are called by the subroutine VTMOP_OPT.
   ! Subroutine to evaluate surrogate models.
   PROCEDURE(VTMOP_MOD_SEVAL_INT), NOPASS, POINTER :: EVAL_SURROGATES
   ! Subroutine to fit surrogate models.
   PROCEDURE(VTMOP_MOD_SFIT_INT), NOPASS, POINTER :: FIT_SURROGATES
   ! Subroutine to perform local optimization over surrogate models.
   PROCEDURE(VTMOP_MOD_LOCAL_INT), NOPASS, POINTER :: LOCAL_OPT
END TYPE VTMOP_TYPE

! Module variables.
! Module variables for checkpointing.
CHARACTER(LEN=20) :: VTMOP_CHKPTFILE = "vtmop.chkpt" ! Checkpoint file name.
CHARACTER(LEN=20) :: VTMOP_DATAFILE = "vtmop.dat" ! Database file name.
! LSHEP (ACM TOMS Algorithm 905) surrogate model parameters.
INTEGER :: LSHEP_D, LSHEP_N_PTS, LSHEP_N_TABOO, LSHEP_P
REAL(KIND=R8) :: LSHEP_DES_TOL
! Module variables for checkpointing.
INTEGER :: VTMOP_CHKPTUNIT = 11 ! Iteration data checkpoint unit.
INTEGER :: VTMOP_DATAUNIT = 12 ! Function evaluation database unit.
! Module variables containing problem metadata.
INTEGER :: VTMOP_MOD_BB_BUDGET ! The blackbox function evaluation budget.
INTEGER :: VTMOP_MOD_D ! Number of design variables.
INTEGER :: VTMOP_MOD_DBN ! Size of the databases.
INTEGER :: VTMOP_MOD_P ! Number of objectives.
! Module variables for checkpointing.
LOGICAL :: VTMOP_MOD_CHKPT ! Global copy of the checkpointing status.
! Module variables containing problem metadata.
REAL(KIND=R8) :: VTMOP_MOD_DES_TOL ! Design space tolerance.

! Dynamic module arrays.
! Arrays for the LSHEP (ACM TOMS Algorithm 905) surrogate models.
REAL(KIND=R8), ALLOCATABLE :: LSHEP_A(:,:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_FVALS(:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_RW(:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_TABOO(:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_SCALE(:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_SHIFT(:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_XVALS(:,:)
! Module arrays for maintaining the database and weights.
REAL(KIND=R8), ALLOCATABLE :: VTMOP_MOD_DBF(:,:) ! Database of objective values.
REAL(KIND=R8), ALLOCATABLE :: VTMOP_MOD_WEIGHTS(:) ! Scalarization weights.
REAL(KIND=R8), ALLOCATABLE :: VTMOP_MOD_DBX(:,:) ! Database of design points.

! Pointers to module procedures for the objective function and its surrogate.
PROCEDURE(VTMOP_MOD_OBJ_INT), POINTER :: VTMOP_MOD_OBJ_FUNC
PROCEDURE(VTMOP_MOD_SEVAL_INT), POINTER :: VTMOP_MOD_SURROGATES

! The THREADPRIVATE list specifies a list of public variables that will be
! initialized PRIVATE to each thread and persist outside of PARALLEL blocks.
!$OMP THREADPRIVATE(VTMOP_MOD_WEIGHTS)

! Interfaces for external procedures.
INTERFACE
   ! QSORTC_DVEC interface.
   SUBROUTINE QSORTC_DVEC(A, IDX)
      USE REAL_PRECISION
      ! Input/output parameters.
      REAL(KIND=R8), INTENT(INOUT) :: A(:, :)
      INTEGER, INTENT(OUT) :: IDX(SIZE(A, 2))
   END SUBROUTINE QSORTC_DVEC

   ! DELAUNAYGRAPH interface.
   SUBROUTINE DELAUNAYGRAPH( D, N, PTS, GRAPH, IERR, EPS, IBUDGET, PMODE )
      USE REAL_PRECISION
      ! Input arguments.
      INTEGER, INTENT(IN) :: D, N
      REAL(KIND=R8), INTENT(INOUT) :: PTS(:,:)
      ! Output arguments.
      LOGICAL, INTENT(OUT) :: GRAPH(:,:)
      INTEGER, INTENT(OUT) :: IERR
      ! Optional arguments.
      REAL(KIND=R8), OPTIONAL, INTENT(IN) :: EPS
      INTEGER, OPTIONAL, INTENT(IN) :: IBUDGET
      LOGICAL, OPTIONAL, INTENT(IN) :: PMODE
   END SUBROUTINE DELAUNAYGRAPH

   ! Scalarized objective function interface.
   FUNCTION VTMOP_MOD_SCALAR_INT(C, IERR) RESULT(F)
      USE REAL_PRECISION, ONLY : R8
      REAL(KIND=R8), INTENT(IN) :: C(:)
      INTEGER, INTENT(OUT) :: IERR
      REAL(KIND=R8) :: F
   END FUNCTION VTMOP_MOD_SCALAR_INT

   ! Local optimization subroutine interface.
   SUBROUTINE VTMOP_MOD_LOCAL_INT(D, X, LB, UB, OBJ_FUNC, BUDGET, TOL, IERR)
      USE REAL_PRECISION, ONLY : R8
      INTEGER, INTENT(IN) :: D
      REAL(KIND=R8), INTENT(INOUT) :: X(:)
      REAL(KIND=R8), INTENT(IN) :: LB(:)
      REAL(KIND=R8), INTENT(IN) :: UB(:)
      INTERFACE
         FUNCTION OBJ_FUNC(C, IERR) RESULT(F)
            USE REAL_PRECISION, ONLY : R8
            REAL(KIND=R8), INTENT(IN) :: C(:)
            INTEGER, INTENT(OUT) :: IERR
            REAL(KIND=R8) :: F
         END FUNCTION OBJ_FUNC
      END INTERFACE
      INTEGER, INTENT(IN) :: BUDGET
      REAL(KIND=R8), INTENT(IN) :: TOL
      INTEGER, INTENT(OUT) :: IERR
   END SUBROUTINE VTMOP_MOD_LOCAL_INT

   ! Objective subroutine interface.
   SUBROUTINE VTMOP_MOD_OBJ_INT(C, V, IERR)
      USE REAL_PRECISION, ONLY : R8
      REAL(KIND=R8), INTENT(IN) :: C(:)
      REAL(KIND=R8), INTENT(OUT) :: V(:)
      INTEGER, INTENT(OUT) :: IERR
   END SUBROUTINE VTMOP_MOD_OBJ_INT

   ! Surrogate function evaluation interface.
   SUBROUTINE VTMOP_MOD_SEVAL_INT(C, V, IERR)
      USE REAL_PRECISION, ONLY : R8
      ! Parameters.
      REAL(KIND=R8), INTENT(IN) :: C(:)
      REAL(KIND=R8), INTENT(OUT) :: V(:)
      INTEGER, INTENT(OUT) :: IERR
   END SUBROUTINE VTMOP_MOD_SEVAL_INT

   ! Surrogate function fitting interface.
   SUBROUTINE VTMOP_MOD_SFIT_INT(D, P, N, X_VALS, Y_VALS, FIRST, PARALLEL, &
                                 DES_TOL, SCALE_FACT, SHIFT_FACT, IERR)
      USE REAL_PRECISION, ONLY : R8
      ! Parameters.
      INTEGER, INTENT(IN) :: D
      INTEGER, INTENT(IN) :: P
      INTEGER, INTENT(IN) :: N
      REAL(KIND=R8), INTENT(IN) :: X_VALS(:,:)
      REAL(KIND=R8), INTENT(IN) :: Y_VALS(:,:)
      LOGICAL, INTENT(IN) :: FIRST
      LOGICAL, INTENT(IN) :: PARALLEL
      REAL(KIND=R8), INTENT(IN) :: DES_TOL
      REAL(KIND=R8), INTENT(IN) :: SCALE_FACT(:)
      REAL(KIND=R8), INTENT(IN) :: SHIFT_FACT(:)
      INTEGER, INTENT(OUT) :: IERR
   END SUBROUTINE VTMOP_MOD_SFIT_INT
END INTERFACE

CONTAINS

! The following public subroutines are referenced by the driver VTMOP_SOLVE
! program and could be used individually by an advanced user for a
! return-to-caller interface.

SUBROUTINE VTMOP_INIT( VTMOP, D, P, LB, UB, IERR, LOPT_BUDGET, DECAY,     &
                       DES_TOL, EPS, EPSW, OBJ_TOL, MIN_RADF, TRUST_RADF, &
                       OBJ_BOUNDS, LOCAL_OPT, FIT_SURROGATES,             &
                       EVAL_SURROGATES, PMODE, ICHKPT )
! This subroutine initializes a VTMOP object for tracking the adaptive
! weighting scheme described in
!
! Deshpande, Shubhangi, Layne T. Watson, and Robert A. Canfield.
! "Multiobjective optimization using an adaptive weighting scheme."
! Optimization Methods and Software 31.1 (2016): 110-133.
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
! VTMOP is an object of derived data type VTMOP_TYPE, which carries meta data
!    about the multiobjective problem.
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
!         128 : If either FIT_SURROGATES or EVAL_SURROGATES are present,
!               then both must be present.
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
! LOCAL_OPT is a SUBROUTINE, whose interface matches that of
!    VTMOP_MOD_LOCAL_INT. LOCAL_OPT is used to optimize the surrogate model.
!    The default value for LOCAL_OPT is GPS, a lightweight Fortran
!    implementation of the polling algorithm generalized pattern search, as
!    used by NOMAD (ACM TOMS Alg. 909). Note that this value is not saved
!    during checkpointing, and must be reset by the user when recovery mode
!    is active, whenever a non-default value is desired.
!
! FIT_SURROGATES is a module subroutine that fits P surrogate models, by
!    setting variables in its module. The interface for FIT_SURROGATES
!    must match that of VTMOP_MOD_SFIT_INT. By default, FIT_SURROGATES is
!    LSHEP_FIT. Note that this value is not saved during checkpointing, and
!    must be reset by the user when recovery mode is active, whenever a
!    non-default value is desired.
!
! EVAL_SURROGATES is a module subroutine that evaluates the P surrogate
!    models fit by FIT_SURROGATES. The interface for EVAL_SURROGATES must
!    match that of VTMOP_MOD_SEVAL_INT. By default, EVAL_SURROGATES is
!    LSHEP_EVAL. Note that this value is not saved during checkpointing, and
!    must be reset by the user when recovery mode is active, whenever a
!    non-default value is desired.
!
! PMODE is a logical input that specifies whether or not iteration tasks
!    should be performed in parallel. By default, PMODE = .FALSE. Note
!    that this value is not saved during checkpointing, and must be reset
!    by the user when recovery mode is active, whenever a non-default value
!    is desired.
!
! ICHKPT is an integer that specifies the checkpointing status. The
!    checkpoint file and checkpoint unit are "vtmop.chkpt" and 10 by
!    default, but can be adjusted by setting the module variables
!    VTMOP_CHKPTFILE and VTMOP_CHKPTUNIT. Possible values are:
!
!    ICHKPT = 0 : No checkpointing (default setting).
!    ICHKPT < 0 : Recover from the last checkpoint.
!    ICHKPT > 0 : Begin a new checkpoint file.
!
! In recovery mode the inputs D, P, LB, and UB are still referenced
!    (for sanity checks). Also, the procedure arguments are still
!    needed to recover the procedure settings. No other optional
!    arguments are referenced.
!
IMPLICIT NONE
! Input parameters.
INTEGER, INTENT(IN) :: D ! Dimension of design space.
INTEGER, INTENT(IN) :: P ! Dimension of objective space.
REAL(KIND=R8), INTENT(IN) :: LB(:) ! Lower bound constraints.
REAL(KIND=R8), INTENT(IN) :: UB(:) ! Upper bound constraints.
! Output parameters.
TYPE(VTMOP_TYPE), INTENT(OUT) :: VTMOP ! Data struct containing problem info.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Optional parameters.
INTEGER, OPTIONAL, INTENT(IN) :: LOPT_BUDGET ! Local optimizer budget.
LOGICAL, OPTIONAL, INTENT(IN) :: PMODE ! Parallel execution mode.
INTEGER, OPTIONAL, INTENT(IN) :: ICHKPT ! Checkpointing mode.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: DECAY ! Decay rate for LTR.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: DES_TOL ! Design space tolerance.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: EPS ! Working precision.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: EPSW ! Fudge factor for zero weights.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: OBJ_TOL ! Objective space tolerance.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: MIN_RADF ! Minimum LTR radius fraction.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: TRUST_RADF ! Initial LTR radius fraction.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: OBJ_BOUNDS(:,:) ! Upper/lower obj bounds.
! Optional procedure arguments.
! Locally convergent optimization procedure for solving surrogate problem.
PROCEDURE(VTMOP_MOD_LOCAL_INT), OPTIONAL :: LOCAL_OPT
! Procedure for fitting the surrogate models.
PROCEDURE(VTMOP_MOD_SFIT_INT), OPTIONAL :: FIT_SURROGATES
! Procedure for evaluating the surrogate models.
PROCEDURE(VTMOP_MOD_SEVAL_INT), OPTIONAL :: EVAL_SURROGATES

! Local copy of optional variable.
INTEGER :: ICHKPTL
! External BLAS function for computing Euclidean distance.
REAL(KIND=R8), EXTERNAL :: DNRM2

! Check for illegal input dimensions and values.
IF (D < 1) THEN ! Illegal design space dimension.
   IERR = 110; RETURN; END IF
IF (P < 2) THEN ! Illegal objective space dimension.
   IERR = 111; RETURN; END IF
IF (SIZE(LB,1) .NE. D) THEN ! Lower bounds dimension must match D.
   IERR = 112; RETURN; END IF
IF (SIZE(UB,1) .NE. D) THEN ! Upper bounds dimension must match D.
   IERR = 113; RETURN; END IF
! Get optional inputs.
ICHKPTL = 0
IF (PRESENT(ICHKPT)) ICHKPTL = ICHKPT

! If in checkpoint recovery mode, read the VTMOP object in.
IF (ICHKPTL < 0) THEN
   ! Load problem status from last checkpoint.
   CALL VTMOP_CHKPT_RECOVER(VTMOP, IERR)
   IF (IERR .NE. 0) RETURN
   ! Perform the final sanity check.
   IF ( (VTMOP%D .NE. D) .OR. (VTMOP%P .NE. P) .OR. &
        ANY(VTMOP%LB(:) .NE. LB(:)) .OR. ANY(VTMOP%UB(:) .NE. UB(:)) ) THEN
      IERR = 901; END IF
   ! Set the checkpointing flag.
   VTMOP%CHKPT = .TRUE.

! Otherwise, perform a normal initialization.
ELSE
   ! Initialize the VTMOP structure to maintain the status of the problem.
   VTMOP%D = D
   VTMOP%P = P
   VTMOP%LCLIST = 20
   ALLOCATE(VTMOP%LB(D), VTMOP%UB(D), VTMOP%CLIST(D+1,VTMOP%LCLIST), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 130; RETURN; END IF
   VTMOP%ITERATE = 0
   VTMOP%LB(:) = LB(:)
   VTMOP%UB(:) = UB(:)

   ! Check for optional inputs.
   ! Initialize the working precision.
   VTMOP%EPS = SQRT(EPSILON(0.0_R8))
   IF(PRESENT(EPS)) THEN
      ! The default value of EPS cannot be decreased. Ignore such inputs.
      IF(EPS > VTMOP%EPS) VTMOP%EPS = EPS
   END IF
   ! Initialize the fudge factor for zero weights.
   VTMOP%EPSW = EPSILON(0.0_R8) ** 0.25_R8
   IF(PRESENT(EPSW)) THEN
      ! Ignore EPSW if it is less than the square-root of the machine EPSILON.
      IF(EPSW .GE. SQRT(EPSILON(0.0_R8))) VTMOP%EPSW = EPSW
   END IF
   ! Initialize the design space and objective space tolerances.
   VTMOP%DES_TOL = SQRT(VTMOP%EPS)
   IF (PRESENT(DES_TOL)) THEN
      IF(DES_TOL > VTMOP%EPS) THEN
         VTMOP%DES_TOL = DES_TOL
      ELSE
         VTMOP%DES_TOL = VTMOP%EPS
      END IF
   END IF
   VTMOP%OBJ_TOL = SQRT(VTMOP%EPS)
   IF (PRESENT(OBJ_TOL)) THEN
      IF(OBJ_TOL > VTMOP%EPS) THEN
         VTMOP%OBJ_TOL = OBJ_TOL
      ELSE
         VTMOP%OBJ_TOL = VTMOP%EPS
      END IF
   END IF
   ! Initialize the decay rate.
   VTMOP%DECAY = 0.5_R8
   IF(PRESENT(DECAY)) THEN
      ! The decay rate must be between EPS and 1-EPS.
      IF(DECAY > 1.0_R8 - VTMOP%EPS .OR. DECAY < VTMOP%EPS) THEN
         IERR = 124; RETURN; END IF
      VTMOP%DECAY = DECAY
   END IF
   ! Initialize the LTR radius fraction and minimum LTR radius fraction.
   VTMOP%TRUST_RADF = 0.2_R8
   IF(PRESENT(TRUST_RADF)) THEN
      ! The LTR radius fraction must be greater than the design space tolerance.
      IF(TRUST_RADF .GE. VTMOP%DES_TOL .AND. TRUST_RADF < 1.0_R8) THEN
         VTMOP%TRUST_RADF = TRUST_RADF; END IF
   END IF
   ! The minimum LTR radius fraction must be greater than the design
   ! space tolerance. By default, tolerate decay down to 10% of the
   ! initial LTR radius.
   VTMOP%MIN_RADF = MAX(0.1_R8 * VTMOP%TRUST_RADF, VTMOP%DES_TOL)
   IF(PRESENT(MIN_RADF)) THEN
      IF(MIN_RADF .GE. VTMOP%DES_TOL) VTMOP%MIN_RADF = MIN_RADF
   END IF
   ! Check that the size of the LTR radius is appropriate.
   IF(VTMOP%MIN_RADF > VTMOP%TRUST_RADF) THEN
      IERR = 125; RETURN; END IF
   ! Lower bounds must be elementwise strictly less than upper bounds.
   ! Use the design space tolerance to check.
   IF(ANY(LB(:) .GE. UB(:) - VTMOP%DES_TOL)) THEN
      IERR = 114; RETURN; END IF
   ! Initialize the checkpointing mode.
   VTMOP%CHKPT = .FALSE.
   ! If ICHKPT > 0, initialize the checkpoint file and activate checkpointing.
   ! Check whether checkpointing is enabled.
   IF (ICHKPTL .NE. 0) THEN
      ! Initialize the checkpoint file and save the initialized VTMOP.
      CALL VTMOP_CHKPT_NEW(VTMOP, IERR)
      IF (IERR .NE. 0) RETURN
      ! Activate checkpointing mode.
      VTMOP%CHKPT = .TRUE.
   END IF
END IF

! The remaining optional inputs are not saved when checkpointing is active.
! Therefore, if non-default values are desired, these values must be re-set
! by the user, even in recovery mode.

! Initialize the local optimization budget.
VTMOP%LOPT_BUDGET = 2500
IF(PRESENT(LOPT_BUDGET)) THEN
   ! The budget must be at least 1, return an error for nonpositive values.
   IF(LOPT_BUDGET < 1) THEN
      IERR = 123; RETURN; END IF
   VTMOP%LOPT_BUDGET = LOPT_BUDGET
END IF
! Initialize the objective bounds.
ALLOCATE(VTMOP%OBJ_BOUNDS(P,2), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 130; RETURN; END IF
VTMOP%OBJ_BOUNDS(:,1) = -HUGE(0.0_R8)
VTMOP%OBJ_BOUNDS(:,2) = HUGE(0.0_R8)
IF(PRESENT(OBJ_BOUNDS)) THEN
   ! When present, OBJ_BOUNDS must have dimension P by 2 and
   ! OBJ_BOUNDS(:,1) < OBJ_BOUNDS(:,2).
   IF(SIZE(OBJ_BOUNDS,1) .NE. P .OR. SIZE(OBJ_BOUNDS,2) .NE. 2) THEN
      IERR = 129; RETURN
   ELSE
      IF(ANY(OBJ_BOUNDS(:,1) .GE. OBJ_BOUNDS(:,2))) THEN
         IERR = 129; RETURN; END IF
      VTMOP%OBJ_BOUNDS(:,:) = OBJ_BOUNDS(:,:)
   END IF
END IF
! Set the parallel execution mode.
VTMOP%PMODE = .FALSE.
IF(PRESENT(PMODE)) VTMOP%PMODE = PMODE
! Set the procedure arguments.
VTMOP%LOCAL_OPT => GPS ! Default optimizer is GPS.
IF(PRESENT(LOCAL_OPT)) THEN
   VTMOP%LOCAL_OPT => LOCAL_OPT; END IF
VTMOP%FIT_SURROGATES => LSHEP_FIT ! Default fit is LSHEP_FIT.
IF(PRESENT(FIT_SURROGATES)) THEN
   IF (.NOT. PRESENT(EVAL_SURROGATES)) THEN
      IERR = 128; RETURN; END IF
   VTMOP%FIT_SURROGATES => FIT_SURROGATES
END IF
VTMOP%EVAL_SURROGATES => LSHEP_EVAL ! Default evaluation is LSHEP_EVAL.
IF(PRESENT(EVAL_SURROGATES)) THEN
   IF (.NOT. PRESENT(FIT_SURROGATES)) THEN
      IERR = 128; RETURN; END IF
   VTMOP%EVAL_SURROGATES => EVAL_SURROGATES
END IF

RETURN
END SUBROUTINE VTMOP_INIT

SUBROUTINE VTMOP_LTR( VTMOP, DES_PTS, OBJ_PTS, LTR_LB, LTR_UB, IERR )
! This subroutine identifies the most isolated point, builds a local
! trust region (LTR), and chooses the adaptive weights, as described in
!
! Deshpande, Shubhangi, Layne T. Watson, and Robert A. Canfield.
! "Multiobjective optimization using an adaptive weighting scheme."
! Optimization Methods and Software 31.1 (2016): 110-133.
!
!
! On input:
!
! VTMOP is an object of derived data type VTMOP_TYPE, which carries meta data
!    about the multiobjective problem. VTMOP is created using VTMOP_INIT.
!
! DES_PTS(1:D,1:N) is a real matrix of all design points in the feasible
!    design space [LB, UB], stored by column. The second dimension of
!    DES_PTS(:,:) (N) is assumed based on the shape and must be at least D+1
!    to build an accurate surrogate model. In the special case of the zeroth
!    iteration, DES_PTS need not be allocated.
!
! OBJ_PTS(1:P,1:N) is a real matrix of objective values corresponding
!    to the design points in DES_PTS(:,:), stored by column: for cost
!    function F, OBJ_PTS(:,I) = F(DES_PTS(:,I)). In the special case of the
!    zeroth iteration, OBJ_PTS need not be allocated.
!
!
! On output:
!
! LTR_LB(1:D) is a real array of lower bounds for the LTR.
!
! LTR_UB(1:D) is a real array of upper bounds for the LTR.
!
! IERR is an integer error flag.
!
! Hundreds digit:
!  0xx : Normal output.
!    Ones digit:
!      000 : Successfully constructed a new LTR and selected adaptive weights.
!      003 : Maximal accuracy has already been achieved, no isolated points
!            can be further refined for the problem tolerance.
!
!  2xx : Error detected in input, or during VTMOP_INIT (initialization) code.
!   Tens digit:
!     21x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         210 : The VTMOP object appears to be uninitialized.
!         211 : The VTMOP object is initialized, but its dimensions either
!               do not agree or contain illegal values. This is likely the
!               result of an undetected segmentation fault.
!         212 : The lead dimension of LTR_LB(:) must match the design
!               dimension D, stored in VTMOP.
!         213 : The lead dimension of LTR_UB(:) must match the design
!               dimension D, stored in VTMOP.
!         214 : The lead dimension of DES_PTS(:,:) must match D.
!         215 : The lead dimension of OBJ_PTS(:,:) must match P.
!         216 : The second dimensions of DES_PTS and OBJ_PTS must match.
!     22x : A memory error occurred while managing the dynamic memory in VTMOP.
!         220 : A memory allocation error while copying the history.
!         221 : A memory deallocation error occurred while freeing temp arrays.
!         222 : A memory allocation error while allocating the adaptive weights.
!     23x : A memory error occurred while managing the local arrays.
!       Ones digit:
!         230 : A memory allocation error occurred.
!         231 : A memory deallocation error occurred.
!     240 : Detected a set of unused adaptive weights. This subroutine
!           may have been called out of sequence.
!     250 : No objective values found (so far) within the range
!           [OBJ_BOUNDS(:,1), OBJ_BOUNDS(:,2)]. Consider relaxing these
!           bounds.
!
!  5xx : Error thrown by DELAUNAYSPARSE.
!    Tens and ones digits carry the exact error code from DELAUNAYSPARSE,
!    passed by the subroutine DELAUNAYGRAPH. The occurrence of these error
!    codes could indicate poor objective scaling. When receiving these
!    error codes, try either (1) rescaling all objective functions so that
!    their range of possible outputs is roughly equal, or (2) increasing the
!    value of OBJ_TOL.
!
!  9xx : A checkpointing error was thrown.
!    91x : The VTMOP passed to the checkpoint was invalid.
!    93x : Error writing iteration information to the checkpoint file.
!
USE IEEE_ARITHMETIC
IMPLICIT NONE
! Input parameters.
TYPE(VTMOP_TYPE), INTENT(INOUT) :: VTMOP ! Data struct containing problem info.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Table of precomputed design pts.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Table of objective values.
! Output parameters.
REAL(KIND=R8), INTENT(OUT) :: LTR_LB(:) ! LTR lower bound constraints.
REAL(KIND=R8), INTENT(OUT) :: LTR_UB(:) ! LTR upper bound constraints.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Problem dimensions.
INTEGER :: D ! Design space dimension.
INTEGER :: M ! Cardinality of the current Pareto set.
INTEGER :: N ! Number of design/objective points in the database.
INTEGER :: P ! Number of objectives.
! Local variables.
LOGICAL :: ACCEPT ! Acceptance condition for new center point.
LOGICAL :: FOUND ! Indicate whether the box center was found.
INTEGER :: I, J, K ! Loop indexing / temp variables.
INTEGER :: MAXIND ! The index of the most isolated point.
REAL(KIND=R8) :: BOX(VTMOP%D+1) ! Center and radius of next LTR.
REAL(KIND=R8) :: MINVAL_P ! Minimum value taken by the Pth objective.
! Local dynamic arrays.
LOGICAL, ALLOCATABLE :: DELGRAPH(:,:) ! Delaunay graph.
INTEGER, ALLOCATABLE :: INDICES(:) ! For tracking indices when sorting.
REAL(KIND=R8), ALLOCATABLE :: CLIST_TMP(:,:) ! Temp array for expanding CLIST.
REAL(KIND=R8), ALLOCATABLE :: DISCREP(:) ! List of star discrepancies.
REAL(KIND=R8), ALLOCATABLE :: EFFICIENT_SET(:,:) ! Efficient point set.
REAL(KIND=R8), ALLOCATABLE :: HOMOGENEOUS_PF(:,:) ! Homogeneous Pareto front.
REAL(KIND=R8), ALLOCATABLE :: PARETO_SET(:,:) ! Pareto front.
! External BLAS procedures.
REAL(KIND=R8), EXTERNAL :: DNRM2 ! Euclidean distance (BLAS).

! Retrieve problem dimensions from input data.
D = VTMOP%D; P = VTMOP%P; N = SIZE(DES_PTS,2)
! Check for illegal problem dimensions.
IF ( (.NOT. ALLOCATED(VTMOP%LB)) .OR. (.NOT. ALLOCATED(VTMOP%UB)) &
     .OR. (.NOT. ALLOCATED(VTMOP%CLIST)) ) THEN
   IERR = 210; RETURN; END IF
IF ((D < 1) .OR. (P < 2) .OR. (SIZE(VTMOP%LB,1) .NE. D) .OR. &
    (SIZE(VTMOP%UB,1) .NE. D)) THEN
   IERR = 211; RETURN; END IF
IF (SIZE(LTR_LB,1) .NE. D) THEN ! Lower bounds dimension does not match D.
   IERR = 212; RETURN; END IF
IF (SIZE(LTR_UB,1) .NE. D) THEN ! Upper bounds dimension does not match D.
   IERR = 213; RETURN; END IF
! If the adaptive weights array is already allocated, then VTMOP_OPT has not
! been called since the last call to VTMOP_LTR.
IF (ALLOCATED(VTMOP%WEIGHTS)) THEN
   IERR = 240; RETURN; END IF

! In the zeroth iteration, use static weights on the entire design space.
IF (VTMOP%ITERATE .EQ. 0) THEN
   ! Set the LTR bounds.
   LTR_LB(:) = VTMOP%LB(:)
   LTR_UB(:) = VTMOP%UB(:)
   ! Allocate the initial weights.
   ALLOCATE(VTMOP%WEIGHTS(P,P+1), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 222; RETURN; END IF
   ! Set the individual weights.
   DO I = 1, P
      ! Set zero-valued weights to VTMOP%EPS to avoid pathological cases.
      VTMOP%WEIGHTS(:,I) = VTMOP%EPSW
      VTMOP%WEIGHTS(I,I) = 1.0_R8 - VTMOP%EPSW*REAL(P-1,KIND=R8)
   END DO
   ! Set the adaptive weights.
   VTMOP%WEIGHTS(:,P+1) = 1.0_R8 / REAL(P, KIND=R8)

! Otherwise, for the Kth iteration (K > 0), perform a general iteration.
ELSE
   ! Check that the dimensions of DES_PTS and OBJ_PTS match.
   IF (SIZE(DES_PTS,1) .NE. D) THEN
      IERR = 214; RETURN; END IF
   IF (SIZE(OBJ_PTS,1) .NE. P) THEN
      IERR = 215; RETURN; END IF
   IF (SIZE(OBJ_PTS,2) .NE. N) THEN
      IERR = 216; RETURN; END IF

   ! If CLIST is at capacity, then reallocate CLIST.
   IF (VTMOP%ITERATE > VTMOP%LCLIST) THEN
      ! Allocate the temporary array to copy CLIST.
      ALLOCATE(CLIST_TMP(D+1,VTMOP%LCLIST), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 220; RETURN; END IF
      CLIST_TMP = VTMOP%CLIST
      ! Reallocate CLIST to twice its current size.
      DEALLOCATE(VTMOP%CLIST, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 221; RETURN; END IF
      ALLOCATE(VTMOP%CLIST(D+1,VTMOP%LCLIST*2), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 220; RETURN; END IF
      ! Restore values back into CLIST and free the temporary array.
      VTMOP%CLIST(:,1:VTMOP%LCLIST) = CLIST_TMP(:,:)
      DEALLOCATE(CLIST_TMP, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 221; RETURN; END IF
      ! Update the size of VTMOP%LCLIST.
      VTMOP%LCLIST = VTMOP%LCLIST * 2
   END IF

   ! Allocate the dynamic arrays for storing both copies of the Pareto front.
   ALLOCATE(INDICES(N), PARETO_SET(P,N), EFFICIENT_SET(D,N), &
            HOMOGENEOUS_PF(P-1,N), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 230; RETURN; END IF
   ! Make copies of the objective and their indices for sorting.
   FORALL (I=1:N) INDICES(I) = I
   PARETO_SET(:,:) = OBJ_PTS(:,:)
   ! Ignore NANs and values outside of the OBJ_BOUNDS.
   I = 1
   DO WHILE(I .LE. N)
      IF (ANY(IEEE_IS_NAN(PARETO_SET(:,I))) .OR.            &
          ANY(PARETO_SET(:,I) < VTMOP%OBJ_BOUNDS(:,1)) .OR. &
          ANY(PARETO_SET(:,I) > VTMOP%OBJ_BOUNDS(:,2))) THEN
         PARETO_SET(:,I:N-1) = PARETO_SET(:,I+1:N)
         EFFICIENT_SET(:,I:N-1) = EFFICIENT_SET(:,I+1:N)
         INDICES(I:N-1) = INDICES(I+1:N)
         N = N - 1
      ELSE
         I = I + 1
      END IF
   END DO
   ! Check whether there are still any viable points remaining.
   IF (N .EQ. 0) THEN
      IERR = 250; RETURN; END IF
   ! Sort objective and design points.
   CALL QSORTC_DVEC(PARETO_SET(:,1:N), INDICES(1:N))
   EFFICIENT_SET(:,1:N) = DES_PTS(:,INDICES(1:N))
   ! Free the indices; they are not needed anymore.
   DEALLOCATE(INDICES, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 231; RETURN; END IF
   ! Compute the minimum value obtained by the Pth objective. This value is
   ! used to shift the points in the Pth dimension, so that they are above
   ! the hyperplane X(P) = 1. This prevents division by zero.
   MINVAL_P = MINVAL(PARETO_SET(P,:)) - 1.0_R8
   ! The first entry in PARETO_SET/EFFICIENT_SET is always Pareto optimal.
   M = 1 ! Count the cardinality of the Pareto front.
   HOMOGENEOUS_PF(:,M) = PARETO_SET(1:P-1,1) / (PARETO_SET(P,1) - MINVAL_P)
   ! Get the current Pareto front (in both the objective space and in the
   ! homogeneous coordinates).
   OUTER : DO I = 2, N
      ! Compute the homogeneous coordinates for the Ith point.
      HOMOGENEOUS_PF(:,M+1) = PARETO_SET(1:P-1,I) / &
                                 (PARETO_SET(P,I) - MINVAL_P)
      ! Check against all points in the current solution set.
      INNER : DO J = 1, M
         ! Check whether the Jth point dominates the Ith point.
         IF (ALL(PARETO_SET(:,J) .LE. PARETO_SET(:,I) + VTMOP%OBJ_TOL)) THEN
            CYCLE OUTER ! Skip the Ith point.
         END IF
         ! Check whether the Ith point and Jth point are equal in the
         ! homogeneous coordinate system (up to the working precision).
         IF (DNRM2(P-1, HOMOGENEOUS_PF(:,M+1) - HOMOGENEOUS_PF(:,J), 1) &
                < VTMOP%OBJ_TOL) THEN
            CYCLE OUTER ! Only store the first occurrence of a duplicate.
         END IF
      END DO INNER
      ! Increment the counter and update both Pareto front arrays.
      M = M + 1
      PARETO_SET(:,M) = PARETO_SET(:,I)
      EFFICIENT_SET(:,M) = EFFICIENT_SET(:,I)
   END DO OUTER

   ! Allocate the remaining dynamic arrays.
   ALLOCATE(DELGRAPH(M,M), DISCREP(M), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 230; RETURN; END IF

   ! Use DELAUNAYGRAPH to compute the Delaunay graph (serially or in parallel).
   CALL DELAUNAYGRAPH(P-1, M, HOMOGENEOUS_PF(:,1:M), DELGRAPH, IERR, &
                       EPS=VTMOP%EPS, PMODE=VTMOP%PMODE)
   IF (IERR < 10) THEN ! Normal execution.
      IERR = 0
   ELSE ! An irrecoverable error occurred.
      IERR = IERR + 500
      RETURN
   END IF

   ! Identify the most isolated point using the star discrepancy.
   DO I = 1, M
      ! Compute the star discrepancy at index I using DELGRAPH(:,I).
      DISCREP(I) = 0.0_R8
      K = 0
      DO J = 1, M
         IF(DELGRAPH(J,I)) THEN
            DISCREP(I) = DISCREP(I) + &
                         DNRM2(P, PARETO_SET(:,J)-PARETO_SET(:,I), 1)
            K = K + 1
         END IF
      END DO
      IF (K > 0) THEN
         ! In general, point at index I has at least one Delaunay neighbor.
         DISCREP(I) = DISCREP(I) / REAL(K, KIND=R8)
      ELSE
         ! In rare cases, the point at index I has no neighbors. This
         ! can only occur when the Pareto front consists of a single point.
         DISCREP(I) = 1.0_R8
      END IF
   END DO

   ! Loop until an acceptable center is found.
   ACCEPT = .FALSE.
   DO WHILE(.NOT. ACCEPT)
      ! Identify the largest discrepancy, except negative/zero values.
      MAXIND = MAXLOC(DISCREP, DIM=1, MASK=(DISCREP > VTMOP%EPS))
      ! If no values with positive discrepancy remain, terminate.
      IF (MAXIND .EQ. 0) EXIT
      ! Otherwise, set BOX(1:D) to the corresponding design point.
      BOX(1:D) = EFFICIENT_SET(:,MAXIND)
      ! Check whether the current design point has been used before.
      FOUND = .FALSE.
      DO I = VTMOP%ITERATE-1, 1, -1
         IF ( DNRM2(D, VTMOP%CLIST(1:D,I) - BOX(1:D), 1) < VTMOP%DES_TOL ) THEN
            FOUND = .TRUE.
            EXIT
         END IF
      END DO
      ! A previous entry in VTMOP%CLIST(:,:) matches.
      IF (FOUND) THEN
         ! Copy the item and decay the LTR radius.
         BOX = VTMOP%CLIST(:,I)
         BOX(D+1) = BOX(D+1) * VTMOP%DECAY
         ! The minimum tolerance has not yet been exceeded.
         IF (BOX(D+1) > VTMOP%MIN_RADF) THEN
            ! Append BOX(:) to CLIST(:,:).
            VTMOP%CLIST(:,VTMOP%ITERATE) = BOX(:)
            ACCEPT = .TRUE.
         ! The minimum tolerance has been exceeded, set the
         ! discrepancy to a negative number.
         ELSE
            DISCREP(MAXIND) = -1.0_R8
         END IF
      ! No match found.
      ELSE
         ! Append BOX(:) to CLIST(:,:).
         BOX(D+1) = VTMOP%TRUST_RADF
         VTMOP%CLIST(:,VTMOP%ITERATE) = BOX(:)
         ACCEPT = .TRUE.
      END IF
   END DO

   ! If no point was accepted, terminate. It must be that the Pareto front
   ! has been approximated to the maximum tolerance.
   IF (.NOT. ACCEPT) THEN
      IERR = 3;
      RETURN
   END IF

   ! Build the LTR. It is the intersection over the current box and the
   ! bound constraints.
   DO I = 1, D
      LTR_LB(I) = MAX(BOX(I)-BOX(D+1)*(VTMOP%UB(I)-VTMOP%LB(I)), VTMOP%LB(I))
      LTR_UB(I) = MIN(BOX(I)+BOX(D+1)*(VTMOP%UB(I)-VTMOP%LB(I)), VTMOP%UB(I))
   END DO

   ! Count the number of Delaunay neighbors of PARETO_SET(:,MAXIND) using
   ! DELGRAPH(:,MAXIND).
   K = 0
   DO I = 1, M
      IF(DELGRAPH(I, MAXIND)) K = K + 1
   END DO
   ! Construct the adaptive weights. First, consider the special case where
   ! there are insufficiently many Delaunay neighbors.
   IF (K < 1) THEN
      ! Allocate the adaptive weights.
      ALLOCATE(VTMOP%WEIGHTS(P,P+1), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 222; RETURN; END IF
      ! Set the individual weights.
      DO I = 1, P
         ! Set zero-valued weights to VTMOP%EPSW to avoid pathological cases.
         VTMOP%WEIGHTS(:,I) = VTMOP%EPSW
         VTMOP%WEIGHTS(I,I) = 1.0_R8 - VTMOP%EPSW*REAL(P-1,KIND=R8)
      END DO
      ! Set the adaptive weights.
      VTMOP%WEIGHTS(:,P+1) = 1.0_R8 / REAL(P, KIND=R8)

   ! In the general case, use the Delaunay neighborhood.
   ELSE
      ! Allocate the adaptive weights.
      ALLOCATE(VTMOP%WEIGHTS(P,P+K), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 222; RETURN; END IF
      ! Set the individual weights.
      DO I = 1, P
         ! Set zero-valued weights to VTMOP%EPSW to avoid pathological cases.
         VTMOP%WEIGHTS(:,I) = VTMOP%EPSW
         VTMOP%WEIGHTS(I,I) = 1.0_R8 - VTMOP%EPSW*REAL(P-1,KIND=R8)
      END DO
      ! Set the adaptive weights.
      K = 1
      DO I = 1, M
         ! Check that I and MAXIND are Delaunay neighbors.
         IF (DELGRAPH(I,MAXIND)) THEN
            ! Set the adaptive weights.
            VTMOP%WEIGHTS(:,P+K) = ABS(PARETO_SET(:,I) - PARETO_SET(:,MAXIND))
            ! Invert when greater than or equal to zero.
            DO J = 1, P
               IF (VTMOP%WEIGHTS(J,P+K) < VTMOP%EPS) THEN
                  VTMOP%WEIGHTS(J,P+K) = VTMOP%EPSW
               ELSE
                  VTMOP%WEIGHTS(J,P+K) = 1.0_R8 / VTMOP%WEIGHTS(J,P+K)
               END IF
            END DO
            ! Normalize to make VTMOP%WEIGHTS(:,P+K) convex.
            VTMOP%WEIGHTS(:,P+K) = VTMOP%WEIGHTS(:,P+K) / &
                                   SUM(VTMOP%WEIGHTS(:,P+K))
            K = K + 1
         END IF
      END DO
   END IF
   ! Free heap memory.
   DEALLOCATE(DELGRAPH, DISCREP, PARETO_SET, HOMOGENEOUS_PF, STAT=IERR)
   IF (IERR .NE. 0) IERR = 231
END IF

! If CHKPT is set, then save to the checkpoint file.
IF (VTMOP%CHKPT) THEN
   ! Save the updated VTMOP object to the checkpoint file.
   CALL VTMOP_CHKPT(VTMOP, IERR)
   IF (IERR .NE. 0) RETURN
END IF
RETURN
END SUBROUTINE VTMOP_LTR

SUBROUTINE VTMOP_OPT( VTMOP, LTR_LB, LTR_UB, DES_PTS, OBJ_PTS, CAND_PTS, IERR )
! This subroutine fits and optimizes P surrogate models within the current
! local trust region (LTR), over the adaptive weights in the VTMOP object,
! as described in
!
! Deshpande, Shubhangi, Layne T. Watson, and Robert A. Canfield.
! "Multiobjective optimization using an adaptive weighting scheme."
! Optimization Methods and Software 31.1 (2016): 110-133.
!
!
! On input:
!
! VTMOP is an object of derived data type VTMOP_TYPE, which carries meta data
!    about the multiobjective problem. VTMOP is created using VTMOP_INIT.
!
! LTR_LB(1:D) is the real vector of lower bounds for the LTR.
!
! LTR_UB(1:D) is the real vector of upper bounds for the LTR.
!
! DES_PTS(1:D,1:N) is a real matrix of all design points in the feasible
!    design space [LB, UB], stored by column. The second dimension of
!    DES_PTS(:,:) (N) is assumed based on the shape and must be at least
!    D+1 to build an accurate surrogate model.
!
! OBJ_PTS(1:P,1:N) is a real matrix of objective values corresponding
!    to the design points in DES_PTS(:,:), stored by column: for
!    cost function F, OBJ_PTS(:,I) = F(DES_PTS(:,I)).
!
! CAND_PTS(:,:) is an ALLOCATABLE real array, which need not be allocated
!    on input. If allocated, any contents of CAND_PTS are lost, and CAND_PTS
!    are reallocated on output.
!
!
! On output:
!
! CAND_PTS(1:D,1:M) is a list of candidate design points to be evaluated
!    before the next iteration of the algorithm. CAND_PTS contains
!    no redundant design points.
!
! IERR is an integer error flag. IERR=0 signifies a successful iteration.
!
! Hundreds digit:
!  000 : Normal output. Successful iteration, and list CAND_PTS obtained.
!
!  3xx : Errors detected.
!   Tens digit:
!     31x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         310 : Either the LB(:) or UB(:) array is not allocated.
!         311 : Either P or D contains an illegal value, or does not
!               agree with input sizes.
!         312 : The lead dimension of LB(:) must match D.
!         313 : The lead dimension of UB(:) must match D.
!         314 : LB(:) must be elementwise strictly less than UB(:) - TOL.
!         315 : The lead dimension of DES_PTS(:,:) must match D.
!         316 : The lead dimension of OBJ_PTS(:,:) must match P.
!         317 : The second dimensions of DES_PTS and OBJ_PTS must match.
!     32x : An irregularity was detected in the supplied VTMOP data type.
!       Ones digit:
!         320 : The adaptive weights are not allocated. Check that
!               the last subroutine called was VTMOP_LTR.
!         321 : The adaptive weights array is allocated, but its lead
!               dimension does not match the number of objectives P. This
!               is most likely the result of an undetected segmentation
!               fault.
!         322 : There are too few adaptive weights. This is most likely the
!               result of an undetected segmentation fault.
!         323 : One of the adaptive weights contains a negative value. This
!               is most likely the result of an undetected segmentation
!               fault.
!     33x : A memory error has occurred.
!       Ones digit:
!         330 : A memory allocation error occurred in the local memory.
!         331 : A memory deallocation error occurred in the local memory.
!         332 : A memory deallocation error occurred while freeing the
!               adaptive weights for the next iteration.
!     340 : Too few points were supplied in DES_PTS and OBJ_PTS to construct
!           an accurate surrogate. At least D+1 points are required.
!
!  6xx : Error thrown by FIT_SURROGATES.
!    Tens and ones digits carry the error code from FIT_SURROGATES subroutine.
!    Note, this assumes that FIT_SURROGATES returns an error code less than
!    100.
!
!  7xx : Error thrown by LOCAL_OPT.
!    Tens and ones digits carry the error code from LOCAL_OPT subroutine.
!    Note, this assumes that LOCAL_OPT returns an error code less than 100.
!
!  9xx : A checkpointing error was thrown.
!    91x : The VTMOP passed to the checkpoint was invalid.
!    93x : Error writing iteration information to the checkpoint file.
!
IMPLICIT NONE
! Input parameters.
TYPE(VTMOP_TYPE), INTENT(INOUT) :: VTMOP ! Data struct containing problem info.
REAL(KIND=R8), INTENT(IN) :: LTR_LB(:) ! Lower bound constraints.
REAL(KIND=R8), INTENT(IN) :: LTR_UB(:) ! Upper bound constraints.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Table of design points.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Table of objective values.
! Output parameters.
REAL(KIND=R8), INTENT(OUT), ALLOCATABLE :: CAND_PTS(:,:) ! Efficient set.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Problem dimensions.
INTEGER :: D ! Design space dimension.
INTEGER :: L ! Number of adaptive weights for the current iteration.
INTEGER :: M ! Number of candidate design points to be returned.
INTEGER :: N ! Number of design/objective points in the database.
INTEGER :: P ! Number of objectives.
! Local variables.
INTEGER :: I, J ! Loop indexing variables.
REAL(KIND=R8) :: MIN_X(VTMOP%D,SIZE(VTMOP%WEIGHTS,2)) ! Potential candidate pts.
! External BLAS function for computing Euclidean distance.
REAL(KIND=R8), EXTERNAL :: DNRM2

! Get problem dimensions from input data.
D = VTMOP%D; P = VTMOP%P; N = SIZE(DES_PTS,2); L = SIZE(VTMOP%WEIGHTS,2)
! Check for illegal problem dimensions.
IF ((.NOT. ALLOCATED(VTMOP%LB)) .OR. (.NOT. ALLOCATED(VTMOP%UB))) THEN
   IERR = 310; RETURN; END IF
IF ((D < 1) .OR. (P < 2) .OR. (SIZE(VTMOP%LB,1) .NE. D) .OR. &
    (SIZE(VTMOP%UB,1) .NE. D)) THEN
   IERR = 311; RETURN; END IF
IF (SIZE(LTR_LB,1) .NE. D) THEN ! Lower bound dimension does not match D.
   IERR = 312; RETURN; END IF
IF (SIZE(LTR_UB,1) .NE. D) THEN ! Upper bound dimension does not match D.
   IERR = 313; RETURN; END IF
IF (ANY(LTR_LB .GE. LTR_UB - VTMOP%DES_TOL) .OR. &
    ANY(LTR_LB < VTMOP%LB - VTMOP%EPS) .OR.      &
    ANY(LTR_UB > VTMOP%UB + VTMOP%EPS)) THEN
   IERR = 314; RETURN; END IF
! Check that the dimensions of DES_PTS and OBJ_PTS match.
IF (SIZE(DES_PTS,1) .NE. D) THEN
   IERR = 315; RETURN; END IF
IF (SIZE(OBJ_PTS,1) .NE. P) THEN
   IERR = 316; RETURN; END IF
IF (SIZE(OBJ_PTS,2) .NE. N) THEN
   IERR = 317; RETURN; END IF
! Check the adaptive VTMOP%WEIGHTS array.
IF(.NOT. ALLOCATED(VTMOP%WEIGHTS)) THEN
   IERR = 320; RETURN; END IF
IF(SIZE(VTMOP%WEIGHTS,1) .NE. P) THEN
   IERR = 321; RETURN; END IF
IF(L < P+1) THEN
   IERR = 322; RETURN; END IF
IF(ANY(VTMOP%WEIGHTS < -VTMOP%EPS)) THEN
   IERR = 323; RETURN; END IF
! Too few points to fit a surrogate model.
IF (N < D+1) THEN
   IERR = 340; RETURN; END IF

! Fit P surrogate models.
IF (VTMOP%ITERATE .EQ. 0) THEN
   CALL VTMOP%FIT_SURROGATES(D, P, N, DES_PTS, OBJ_PTS, .TRUE., VTMOP%PMODE, &
                             VTMOP%DES_TOL, VTMOP%UB(:)-VTMOP%LB(:),         &
                             VTMOP%LB(:), IERR)
   IF (IERR .NE. 0) THEN
      IERR = IERR + 600; RETURN; END IF
ELSE
   CALL VTMOP%FIT_SURROGATES(D, P, N, DES_PTS, OBJ_PTS, .FALSE., VTMOP%PMODE, &
                             VTMOP%DES_TOL, VTMOP%UB(:)-VTMOP%LB(:),          &
                             VTMOP%LB(:), IERR)
   IF (IERR .NE. 0) THEN
      IERR = IERR + 600; RETURN; END IF
END IF

! Initialize the module variables containing the problem dimensions.
VTMOP_MOD_D = D
VTMOP_MOD_P = P
! Set the module surrogate functions.
VTMOP_MOD_SURROGATES => VTMOP%EVAL_SURROGATES

! Optimize the surrogate models for all adaptive weightings.
!$OMP PARALLEL &
!
! The PRIVATE list specifies uninitialized variables, of which each
! thread has a private copy.
!$OMP& PRIVATE(I), &
!
! The REDUCTION clause specifies a PRIVATE variable that will retain
! some value (i.e., max, min, sum, etc.) upon output.
!$OMP& REDUCTION(MAX:IERR), &
!
! Any variables not explicitly listed above receive the SHARED scope
! by default and are visible across all threads.
!$OMP& DEFAULT(SHARED), &
!
! Only use this level of parallelism if PMODE is .TRUE.
!$OMP& IF(VTMOP%PMODE)
! Initialize the error flag.
IERR = 0
! Allocate the adaptive weights array.
IF(ALLOCATED(VTMOP_MOD_WEIGHTS)) THEN
   DEALLOCATE(VTMOP_MOD_WEIGHTS, STAT=IERR)
   IF (IERR .NE. 0) IERR = 331
END IF
ALLOCATE(VTMOP_MOD_WEIGHTS(P), STAT=IERR)
IF (IERR .NE. 0) IERR = 330
! Optimize L adaptive weightings of the P surrogate models.
!$OMP DO SCHEDULE(STATIC)
DO I = 1, L
   ! If an error occurs, skip to the end.
   IF (IERR .NE. 0) CYCLE
   ! Set the module weights.
   VTMOP_MOD_WEIGHTS(:) = VTMOP%WEIGHTS(:,I)
   ! Call the local optimizer.
   MIN_X(:,I) = (LTR_LB(:) + LTR_UB(:)) / 2.0_R8
   CALL VTMOP%LOCAL_OPT( D, MIN_X(:,I), LTR_LB(:), LTR_UB(:),              &
                         SURROGATE_FUNC, VTMOP%LOPT_BUDGET, VTMOP%DES_TOL, &
                         IERR )
END DO
!$OMP END DO
!$OMP END PARALLEL
IF (IERR .NE. 0) THEN
   IERR = IERR + 700; RETURN; END IF

! Post processing loop to filter out redundant evaluation points.
M = L ! Track the number of distinct candidate points to return.
I = 1 ! Track the current iterate.
FILTER_LOOP : DO WHILE(.TRUE.)
   ! If the end of the reduced list is reached, exit the loop.
   IF (I > M) EXIT FILTER_LOOP
   ! If MIN_X(:,I) is already in DES_PTS(:,:), don't add to CAND_PTS(:,:).
   DO J = 1, SIZE(DES_PTS(:,:), 2)
      IF (DNRM2(D, DES_PTS(:,J)-MIN_X(:,I), 1) < VTMOP%DES_TOL) THEN
         ! Perform a deletion by overwriting the current entry with the last.
         MIN_X(:,I) = MIN_X(:,M)
         M = M - 1
         ! Skip forward.
         CYCLE FILTER_LOOP
      END IF
   END DO
   ! If MIN_X(:,J) = MIN_X(:,I), for J <= I, don't add to CAND_PTS(:,:).
   DO J = 1, I - 1
      IF (DNRM2(D, MIN_X(:,J)-MIN_X(:,I), 1) < VTMOP%DES_TOL) THEN
         ! Perform a deletion by overwriting the current entry with the last.
         MIN_X(:,I) = MIN_X(:,M)
         M = M - 1
         ! Skip forward.
         CYCLE FILTER_LOOP
      END IF
   END DO
   ! If all the tests passed, advance the iteration index.
   I = I + 1
END DO FILTER_LOOP

! Increment the iteration counter.
VTMOP%ITERATE = VTMOP%ITERATE + 1
! Free CAND_PTS if they are already allocated.
IF(ALLOCATED(CAND_PTS)) THEN
   DEALLOCATE(CAND_PTS, STAT=IERR)
   IF(IERR .NE. 0) THEN
      IERR = 331; RETURN; END IF
END IF
! Allocate and fill CAND_PTS(:,:) to return.
ALLOCATE(CAND_PTS(D,M), STAT=IERR)
IF(IERR .NE. 0) THEN
   IERR = 330; RETURN; END IF
CAND_PTS(:,:) = MIN_X(:,1:M)
! Free the WEIGHTS array for next iteration.
DEALLOCATE(VTMOP%WEIGHTS, STAT=IERR)
IF(IERR .NE. 0) IERR = 332
! Free the VTMOP_MOD_WEIGHTS array.
DEALLOCATE(VTMOP_MOD_WEIGHTS, STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 332; RETURN; END IF

! If CHKPT is set, save to the checkpoint file.
! Check whether checkpointing is enabled.
IF (VTMOP%CHKPT) THEN
   ! Save the updated VTMOP object to the checkpoint file.
   CALL VTMOP_CHKPT(VTMOP, IERR)
   IF (IERR .NE. 0) RETURN
END IF
RETURN
END SUBROUTINE VTMOP_OPT

SUBROUTINE VTMOP_FINALIZE( VTMOP, DES_PTS, OBJ_PTS, M, PARETO_F, EFFICIENT_X, &
                           IERR )
! This subroutine finalizes a multiobjective optimization problem, by
! computing the entire weakly Pareto set, and freeing all dynamic memory
! allocated to the VTMOP object.
!
!
! On input:
!
! VTMOP is an object of derived data type VTMOP_TYPE, which carries meta data
!    about the multiobjective problem. VTMOP is created using VTMOP_INIT.
!
! DES_PTS(1:D,1:N) is a real matrix of all design points in
!    the feasible design space [LB, UB], stored in column major order.
!    The second dimension of DES_PTS(:,:) (N) is assumed based on the shape
!    and must be at least D+1 to build an accurate surrogate model.
!
! OBJ_PTS(1:P,1:N) is a real matrix of objective values corresponding
!    to the design points in DES_PTS(:,:), stored in column major order.
!    I.e., for cost function F, OBJ_PTS(:,I) = F(DES_PTS(:,I)).
!
! PARETO_F(:,:) is an ALLOCATABLE real array. PARETO_F need not be allocated
!    on input. If allocated, any contents of PARETO_F are lost, as PARETO_F
!    is reallocated on output.
!
! EFFICIENT_X(:,:) is an ALLOCATABLE real array. EFFICIENT_X need not be
!    allocated on input. If allocated, any contents of EFFICIENT_X are
!    lost, as EFFICIENT_X is reallocated on output.
!
!
! On output:
!
! M is the cardinality of the weakly Pareto set.
!
! EFFICIENT_X(1:D,1:M) contains the entire weakly efficient set, stored in
!    column major ordering, with corresponding objective values in PARETO_F.
!
! PARETO_F(1:P,1:M) contains the entire weakly nondominated set. Note,
!    PARETO_F may contain duplicate values since the entire weakly nondominated
!    set is returned.
!
! IERR is an integer error flag.
!
! Hundreds digit:
!  000 : Normal output. Successful iteration, and list CAND_PTS obtained.
!
!  4xx : Errors detected.
!   Tens digit:
!     41x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         410 : The VTMOP object contains illegal problem dimensions. This
!               is most likely the result of an undetected segmentation
!               fault.
!         411 : The lead dimension of DES_PTS(:,:) must match D.
!         412 : The lead dimension of OBJ_PTS(:,:) must match P.
!         413 : The second dimensions of DES_PTS and OBJ_PTS must match.
!     42x : A memory error occurred while managing the output arrays.
!       Ones digit:
!         420 : A memory allocation error occurred while allocating an
!               output array PARETO_F and EFFICIENT_X.
!         421 : A memory deallocation error occurred while freeing an
!               output array PARETO_F or EFFICIENT_X, which was already
!               allocated on input.
!     43x : A memory error has occurred while managing internal memory.
!       Ones digit:
!         430 : A memory allocation error occurred in the local memory.
!         431 : A memory deallocation error occurred in the local memory.
!     441 : A memory error has occurred while freeing the VTMOP object or
!           module memory. The output arrays EFFICIENT_X and PARETO_F
!           should be unaffected.
!
USE IEEE_ARITHMETIC
IMPLICIT NONE
! Input parameters.
TYPE(VTMOP_TYPE), INTENT(INOUT) :: VTMOP ! Data struct containing problem info.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Table of precomputed design pts.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Table of objective values.
! Output parameters.
INTEGER, INTENT(OUT) :: M ! Cardinality of the weakly Pareto set.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: PARETO_F(:,:)
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: EFFICIENT_X(:,:)
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Problem dimensions.
INTEGER :: D ! Design space dimension.
INTEGER :: N ! Cardinality of the weakly Pareto set.
INTEGER :: P ! Number of objectives.
! Local variables.
INTEGER :: I, J ! Loop indexing variables.
! Local dynamic arrays.
INTEGER, ALLOCATABLE :: INDICES(:) ! For tracking indices when sorting.
REAL(KIND=R8), ALLOCATABLE :: PARETO_SET(:,:) ! Pareto front.
REAL(KIND=R8), ALLOCATABLE :: EFFICIENT_SET(:,:) ! Efficient set.
! External BLAS procedures.
REAL(KIND=R8), EXTERNAL :: DNRM2 ! Euclidean distance (BLAS).

! Get problem dimensions from input data.
D = VTMOP%D; P = VTMOP%P; N = SIZE(DES_PTS,2)
! Check for illegal problem dimensions.
IF ((D < 1) .OR. (P < 2)) THEN
   IERR = 410; RETURN; END IF
IF (SIZE(DES_PTS,1) .NE. D) THEN
   IERR = 411; RETURN; END IF
IF (SIZE(OBJ_PTS,1) .NE. P) THEN
   IERR = 412; RETURN; END IF
IF (SIZE(OBJ_PTS,2) .NE. N) THEN
   IERR = 413; RETURN; END IF

! Allocate the dynamic arrays for storing both copies of the Pareto front.
ALLOCATE(INDICES(N), PARETO_SET(P,N), EFFICIENT_SET(D,N), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 430; RETURN; END IF

! Make copies of the objective and their indices for sorting.
FORALL (I=1:N) INDICES(I) = I
PARETO_SET(:,:) = OBJ_PTS(:,:)
! Replace NANs with large numbers so that they will be ignored.
DO I = 1, N
   IF (ANY(IEEE_IS_NAN(PARETO_SET(:,I)))) PARETO_SET(:,I) = HUGE(0.0_R8)
END DO
! Sort objective and design points.
CALL QSORTC_DVEC(PARETO_SET, INDICES)
EFFICIENT_SET(:,:) = DES_PTS(:,INDICES)
! Free the indices; they are not needed anymore.
DEALLOCATE(INDICES, STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 431; RETURN; END IF
! The first entry in PARETO_SET/EFFICIENT_SET is always Pareto optimal.
M = 1 ! Count the cardinality of the Pareto front.
! Get the weakly Pareto and efficient sets.
OUTER : DO I = 2, N
   ! Check against all points in the current solution set.
   INNER : DO J = 1, M
      ! Check whether the points at indices I and J are equal.
      IF (DNRM2(P, PARETO_SET(:,I) - PARETO_SET(:,J), 1) < VTMOP%OBJ_TOL) THEN
         CYCLE INNER ! Consider all weakly Pareto points.
      END IF
      ! Check whether the Jth point dominates the Ith point.
      IF (ALL(PARETO_SET(:,J) .LE. PARETO_SET(:,I) + VTMOP%OBJ_TOL)) THEN
         CYCLE OUTER ! Skip the Ith point.
      END IF
   END DO INNER
   ! Increment the counter and update both Pareto front arrays.
   M = M + 1
   PARETO_SET(:,M) = PARETO_SET(:,I)
   EFFICIENT_SET(:,M) = EFFICIENT_SET(:,I)
END DO OUTER

! Reallocate the output arrays.
IF (ALLOCATED(PARETO_F)) THEN
   DEALLOCATE(PARETO_F, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 421; RETURN; END IF
END IF
IF (ALLOCATED(EFFICIENT_X)) THEN
   DEALLOCATE(EFFICIENT_X, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 421; RETURN; END IF
END IF
ALLOCATE(EFFICIENT_X(D,M), PARETO_F(P,M), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 420; RETURN; END IF
! Populate the output arrays.
EFFICIENT_X(:,:) = EFFICIENT_SET(:,1:M)
PARETO_F(:,:) = PARETO_SET(:,1:M)
! Free the local memory.
DEALLOCATE(EFFICIENT_SET, PARETO_SET, STAT=IERR)
IF (IERR .NE. 0) THEN
    IERR = 431; RETURN; END IF
! Free the VTMOP data structure's memory.
IF (ALLOCATED(VTMOP%WEIGHTS)) THEN
   DEALLOCATE(VTMOP%WEIGHTS, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(VTMOP%LB)) THEN
   DEALLOCATE(VTMOP%LB, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(VTMOP%UB)) THEN
   DEALLOCATE(VTMOP%UB, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(VTMOP%CLIST)) THEN
   DEALLOCATE(VTMOP%CLIST, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(VTMOP%OBJ_BOUNDS)) THEN
   DEALLOCATE(VTMOP%OBJ_BOUNDS, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
RETURN
END SUBROUTINE VTMOP_FINALIZE

! The following subroutines are for performing checkpointing.

SUBROUTINE VTMOP_CHKPT_NEW(VTMOP, IERR)
! Create a new checkpoint file for a given instance of VTMOP.
!
!
! On input:
!
! VTMOP is an object of derived data type VTMOP_TYPE, which carries metadata
!    about the multiobjective problem.
!
!
! On output:
!
! IERR is an integer error flag.
!
!  000 : Normal output. Successful initialization of a new VTMOP checkpoint.
!
!  9xx : Errors detected.
!     91x : The input parameters contained illegal dimensions or values.
!         910 : VTMOP does not appear to have been properly allocated.
!         911 : VTMOP has been allocated, but appears to contain corrupted
!               or inconsistent data.
!     92x : A file I/O error was detected.
!         920 : An error occurred while opening the checkpoint file.
!         921 : An error occurred while writing data to the checkpoint file.
!         922 : An error occurred while closing the checkpoint file.
!
IMPLICIT NONE
! Input parameters.
TYPE(VTMOP_TYPE), INTENT(IN) :: VTMOP ! Data structure containing problem info.
! Output parameters.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Check for uninitialized values.
IF ( (.NOT. ALLOCATED(VTMOP%LB)) .OR. (.NOT. ALLOCATED(VTMOP%UB)) ) THEN
   IERR = 910; RETURN; END IF
! Check for illegal/mismatched values.
IF ( VTMOP%D < 1 .OR. VTMOP%P < 2) THEN
   IERR = 911; RETURN; END IF
IF (SIZE(VTMOP%LB, 1) .NE. VTMOP%D .OR. SIZE(VTMOP%UB) .NE. VTMOP%D) THEN
   IERR = 911; RETURN; END IF
! Open the checkpoint file, using unformatted write.
OPEN(VTMOP_CHKPTUNIT, FILE=VTMOP_CHKPTFILE, FORM="unformatted", &
     ACTION="write", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 920; RETURN; END IF
! Write unformatted VTMOP metadata to the checkpoint file.
WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%D, VTMOP%P
IF (IERR .NE. 0) THEN
   IERR = 921; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%DECAY, VTMOP%DES_TOL, VTMOP%EPS,     &
                                    VTMOP%EPSW, VTMOP%OBJ_TOL, VTMOP%MIN_RADF, &
                                    VTMOP%TRUST_RADF
IF (IERR .NE. 0) THEN
   IERR = 921; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%LB(1:VTMOP%D), VTMOP%UB(1:VTMOP%D)
IF (IERR .NE. 0) THEN
   IERR = 921; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Close the checkpoint file.
CLOSE(VTMOP_CHKPTUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 922; RETURN; END IF
RETURN
END SUBROUTINE VTMOP_CHKPT_NEW

SUBROUTINE VTMOP_CHKPT(VTMOP, IERR)
! Save VTMOP's iteration data to an existing checkpoint file.
!
!
! On input:
!
! VTMOP is an object of derived data type VTMOP_TYPE, which carries meta data
!    about the multiobjective problem.
!
!
! On output:
!
! IERR is an integer error flag.
!
!  000 : Normal output. Iteration data saved to the checkpoint file.
!
!  9xx : Errors detected.
!     91x : The input parameters contained illegal dimensions or values.
!         910 : VTMOP does not appear to have been properly allocated.
!         911 : VTMOP has been allocated, but appears to contain corrupted
!               or inconsistent data.
!     93x : A file I/O error was detected.
!         930 : The checkpoint file could not be opened, check whether
!               CHKPTFILE has been properly initialized.
!         931 : An error occurred while writing data to the checkpoint file.
!         932 : An error occurred while closing the checkpoint file.
!
IMPLICIT NONE
! Input parameters.
TYPE(VTMOP_TYPE), INTENT(IN) :: VTMOP ! Data structure containing problem info.
! Output parameters.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Check for uninitialized values.
IF ( (.NOT. ALLOCATED(VTMOP%LB)) .OR. (.NOT. ALLOCATED(VTMOP%UB)) ) THEN
   IERR = 910; RETURN; END IF
! Check for illegal/mismatched values.
IF ( VTMOP%D < 1 .OR. VTMOP%P < 2) THEN
   IERR = 911; RETURN; END IF
IF (SIZE(VTMOP%LB, 1) .NE. VTMOP%D .OR. SIZE(VTMOP%UB) .NE. VTMOP%D) THEN
   IERR = 911; RETURN; END IF
! Open the checkpoint file, using unformatted append.
OPEN(VTMOP_CHKPTUNIT, FILE=VTMOP_CHKPTFILE, FORM="unformatted", ACTION="write", &
     POSITION="append", STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 930; RETURN; END IF
! The status of VTMOP%WEIGHTS tells the phase of the VTMOP algorithm.
IF (ALLOCATED(VTMOP%WEIGHTS)) THEN
   IF (VTMOP%ITERATE > 0) THEN ! Don't write CLIST for the zeroth iteration.
      ! Write unformatted VTMOP iteration data to the checkpoint file.
      WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%CLIST(1:VTMOP%D+1,VTMOP%ITERATE)
      IF (IERR .NE. 0) THEN
         IERR = 931; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
   END IF
   ! Write unformatted adaptive weights to the checkpoint file.
   WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) SIZE(VTMOP%WEIGHTS, 2)
   IF (IERR .NE. 0) THEN
      IERR = 931; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
   WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%WEIGHTS(:,:)
   IF (IERR .NE. 0) THEN
      IERR = 931; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
ELSE
   ! Write the iteration counter.
   WRITE(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%ITERATE
   IF (IERR .NE. 0) THEN
      IERR = 931; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
END IF
! Close the checkpoint file.
CLOSE(VTMOP_CHKPTUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 932; RETURN; END IF
RETURN
END SUBROUTINE VTMOP_CHKPT

SUBROUTINE VTMOP_CHKPT_RECOVER(VTMOP, IERR)
! Recover VTMOP's progress from a checkpoint file.
!
!
! On output:
!
! The status of VTMOP is as specified in CHKPTFILE.
!
! IERR is an integer error flag.
!
!  000 : Normal output. Iteration data saved to the checkpoint file.
!
!  9xx : Errors detected.
!     94x : A file I/O error was detected.
!         940 : The checkpoint file could not be opened, check whether
!               CHKPTFILE has been properly initialized.
!         941 : An error occurred while writing data to the checkpoint file.
!         942 : An error occurred while closing the checkpoint file.
!     95x : A memory allocation error occurred.
!         950 : A memory allocation error occurred.
!         951 : A memory deallocation error occurred.
!     960 : Failed the sanity check. Either the checkpoint feature was
!           implemented improperly, or VTMOP_CHKPTFILE was corrupted.
!
IMPLICIT NONE
! Output parameters.
TYPE(VTMOP_TYPE), INTENT(OUT) :: VTMOP ! Data structure containing problem info.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Temporary arrays.
INTEGER :: NW
REAL(KIND=R8), ALLOCATABLE :: TMP(:,:)
! Open the checkpoint file, using unformatted write.
OPEN(VTMOP_CHKPTUNIT, FILE=VTMOP_CHKPTFILE, FORM="unformatted", ACTION="read", &
     STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 940; RETURN; END IF
! Read in the problem dimensions.
READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%D, VTMOP%P
IF (IERR .NE. 0) THEN
   IERR = 941; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Read in the problem parameters.
READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%DECAY, VTMOP%DES_TOL, VTMOP%EPS,     &
                                   VTMOP%EPSW, VTMOP%OBJ_TOL, VTMOP%MIN_RADF, &
                                   VTMOP%TRUST_RADF
IF (IERR .NE. 0) THEN
   IERR = 941; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Allocate the bound constraints.
ALLOCATE(VTMOP%LB(VTMOP%D), VTMOP%UB(VTMOP%D), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 950; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Read in the bound constraints.
READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%LB(1:VTMOP%D), VTMOP%UB(1:VTMOP%D)
IF (IERR .NE. 0) THEN
   IERR = 941; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Initialize the iteration data.
VTMOP%ITERATE = 0
VTMOP%LCLIST = 20
ALLOCATE(VTMOP%CLIST(VTMOP%D+1,VTMOP%LCLIST), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 950; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF

! Recreate the 0th iteration.

! Read the size of the next batch of adaptive weights.
READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) NW
IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
   IERR = 0; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Allocate VTMOP%WEIGHTS accordingly.
ALLOCATE(VTMOP%WEIGHTS(VTMOP%P, NW), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 950; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Read in the next batch of adaptive weights.
READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%WEIGHTS(:,:)
IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
   IERR = 0; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Next checkpoint tells that the search phase has completed.
READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) NW
IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
   IERR = 0; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Update the iteration counter, and do a sanity check.
VTMOP%ITERATE = VTMOP%ITERATE + 1
IF (NW .NE. VTMOP%ITERATE) THEN
   IERR = 960; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
! Free the adaptive weights for the next iteration.
DEALLOCATE(VTMOP%WEIGHTS, STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 951; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF

! Read data into VTMOP%CLIST(:,:) until the end of file.
READ_LOOP : DO WHILE (.TRUE.)
   ! Check if VTMOP%CLIST(:,:) needs to be resized.
   IF (VTMOP%ITERATE .EQ. VTMOP%LCLIST) THEN
      ! Allocate the temporary array.
      ALLOCATE(TMP(VTMOP%D+1,VTMOP%ITERATE), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 950; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
      ! Make a temporary copy.
      TMP(:,:) = VTMOP%CLIST(:,:)
      ! Update the size of VTMOP%LCLIST.
      VTMOP%LCLIST = VTMOP%LCLIST * 2
      ! Reallocate VTMOP%CLIST to twice its current size.
      DEALLOCATE(VTMOP%CLIST, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 951; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
      ALLOCATE(VTMOP%CLIST(VTMOP%D+1, VTMOP%LCLIST), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 950; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
      ! Restore values back into CLIST and free the temporary array.
      VTMOP%CLIST(:,1:VTMOP%ITERATE) = TMP(:,:)
      DEALLOCATE(TMP, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 951; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
   END IF
   ! Now, read in the next point in the center list from the file.
   READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%CLIST(1:VTMOP%D+1, VTMOP%ITERATE)
   IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Read the size of the next batch of adaptive weights.
   READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) NW
   IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Allocate VTMOP%WEIGHTS accordingly.
   ALLOCATE(VTMOP%WEIGHTS(VTMOP%P, NW), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 950; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
   ! Read in the next batch of adaptive weights.
   READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) VTMOP%WEIGHTS(:,:)
   IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Next checkpoint tells that the search phase has completed.
   READ(VTMOP_CHKPTUNIT, IOSTAT=IERR) NW
   IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Update the iteration counter, and do a sanity check.
   VTMOP%ITERATE = VTMOP%ITERATE + 1
   IF (NW .NE. VTMOP%ITERATE) THEN
      IERR = 960; EXIT READ_LOOP; END IF
   ! Free the adaptive weights for the next iteration.
   DEALLOCATE(VTMOP%WEIGHTS, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 951; CLOSE(VTMOP_CHKPTUNIT); RETURN; END IF
END DO READ_LOOP
! Close the checkpoint file.
CLOSE(VTMOP_CHKPTUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 942; RETURN; END IF
RETURN
END SUBROUTINE VTMOP_CHKPT_RECOVER

SUBROUTINE VTMOP_NEW_DATA(D, P, IERR)
! Create a new checkpoint data file to store VTMOP's function evaluation data.
!
!
! On input:
!
! D is the dimension of the design space.
!
! P is the dimension of the objective space.
!
!
! On output:
!
! IERR is an integer error flag.
!
!  000 : Normal output. Successful initialization of a new data checkpoint
!        file.
!
!  97x : Errors detected.
!     97x : A data I/O error was detected.
!         970 : An error occurred while opening the data file.
!         971 : An error occurred while writing data to the data file.
!         972 : An error occurred while closing the data file.
!
IMPLICIT NONE
! Input parameters.
INTEGER, INTENT(IN) :: D, P ! Problem dimensions.
! Output parameters.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Create a new data file, using unformatted write.
OPEN(VTMOP_DATAUNIT, FILE=VTMOP_DATAFILE, FORM="unformatted", ACTION="write", &
     IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 970; RETURN; END IF
! Write the problem dimensions.
WRITE(VTMOP_DATAUNIT, IOSTAT=IERR) D, P
IF (IERR .NE. 0) THEN
   IERR = 971; CLOSE(VTMOP_DATAUNIT); RETURN; END IF
! Close the data file.
CLOSE(VTMOP_DATAUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 972; RETURN; END IF
RETURN
END SUBROUTINE VTMOP_NEW_DATA

SUBROUTINE VTMOP_SAVE_DATA(DES_PT, OBJ_PT, IERR)
! Save VTMOP's function evaluation data to an existing checkpoint file.
!
!
! On input:
!
! DES_PT(:) is a design point to save.
!
! OBJ_PT(:) is a corresponding objective value to save.
!
!
! On output:
!
! IERR is an integer error flag.
!
!  000 : Normal output. Successful initialization of a new VTMOP checkpoint.
!
!  9xx : Errors detected.
!     98x : A data I/O error was detected.
!         980 : An error occurred while opening the data file.
!         981 : An error occurred while writing data to the data file.
!         982 : An error occurred while closing the data file.
!
IMPLICIT NONE
! Input parameters.
REAL(KIND=R8), INTENT(IN) :: DES_PT(:), OBJ_PT(:)
! Output parameters.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Open an existing data file, using unformatted append.
OPEN(VTMOP_DATAUNIT, FILE=VTMOP_DATAFILE, FORM="unformatted", ACTION="write", &
     POSITION="append", STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 980; RETURN; END IF
! Write the design point and objective value.
WRITE(VTMOP_DATAUNIT, IOSTAT=IERR) DES_PT(:), OBJ_PT(:)
IF (IERR .NE. 0) THEN
   IERR = 981; CLOSE(VTMOP_DATAUNIT); RETURN; END IF
! Close the checkpoint file.
CLOSE(VTMOP_DATAUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 982; RETURN; END IF
RETURN
END SUBROUTINE VTMOP_SAVE_DATA

SUBROUTINE VTMOP_RECOVER_DATA(DBN, DBX, DBF, IERR, DB_SIZE)
! Recover VTMOP's function evaluation database from an existing checkpoint
! file.
!
!
! On input:
!
! DBX(:,:) and DBF(:,:) are unallocated, allocatable arrays.
!
! On output:
!
! DBN is the integer counter specifying the final length of DB{X|F}
!
! DBX(:,:) is allocated to the problem dimensions and contains the
!    recovered database of design points.
!
! DBF(:,:) is allocated to the problem dimensions and contains the
!    recovered database of objective points.
!
! IERR is an integer error flag.
!
!  000 : Normal output. Successful initialization of a new VTMOP checkpoint.
!
!  99x : Errors detected.
!     99x : A data I/O error was detected.
!         990 : An error occurred while opening the data file.
!         991 : An error occurred while reading data from the data file.
!         992 : An error occurred while closing the data file.
!         993 : There was an issue allocating the local memory.
!         994 : The number of recovered data points exceeds the budget.
!
!
! Optional arguments:
!
! DB_SIZE, when present, specifies the amount of memory to allocate for
!    DBX and DBF.  By default, DBX and DBF are allocated to a length of
!    1000.
!
IMPLICIT NONE
! Output parameters.
INTEGER, INTENT(OUT) :: DBN ! The size of the final database.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: DBX(:,:) ! Recovered design points.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: DBF(:,:) ! Recovered objective pts.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
INTEGER, OPTIONAL, INTENT(IN) :: DB_SIZE ! Size of the database.
! Local variables.
INTEGER :: D, P ! Problem dimensions.
INTEGER :: DB_SIZEL ! Local copy of database size parameters.
REAL(KIND=R8), ALLOCATABLE :: DES_PT(:), OBJ_PT(:) ! Temp arrays.
! Read in the optional inputs.
DB_SIZEL = 1000
IF (PRESENT(DB_SIZE)) THEN
   IF (DB_SIZE > 0) DB_SIZEL = DB_SIZE
END IF
! Create a new data file, using unformatted write.
OPEN(VTMOP_DATAUNIT, FILE=VTMOP_DATAFILE, FORM="unformatted", ACTION="read", &
     STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 990; RETURN; END IF
! Read in the problem dimensions.
READ(VTMOP_DATAUNIT, IOSTAT=IERR) D, P
IF (IERR .NE. 0) THEN
   IERR = 991; CLOSE(VTMOP_DATAUNIT); RETURN; END IF
! Allocate the output array memory.
IF (ALLOCATED(DBX)) THEN
   DEALLOCATE(DBX, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 993; RETURN; END IF
END IF
IF (ALLOCATED(DBF)) THEN
   DEALLOCATE(DBF, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 993; RETURN; END IF
END IF
ALLOCATE(DBX(D,DB_SIZEL), DBF(P,DB_SIZEL), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 993; RETURN; END IF
! Allocate the local memory.
ALLOCATE(DES_PT(D), OBJ_PT(P), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 993; RETURN; END IF
! Read until all data points are recovered.
DBN = 0
DO WHILE(.TRUE.)
   READ(VTMOP_DATAUNIT, IOSTAT=IERR) DES_PT, OBJ_PT
   IF (IERR .NE. 0) THEN ! A read error occurred. This must be EOF.
      IERR = 0; EXIT; END IF
   ! If the database is at capacity, return an error.
   IF (DBN .GE. DB_SIZEL) THEN
      IERR = 994; RETURN; END IF
   ! Add DES_PT and OBJ_PT to the module database.
   DBN = DBN + 1
   DBX(:,DBN) = DES_PT(:)
   DBF(:,DBN) = OBJ_PT(:)
END DO
! Close the data file.
CLOSE(VTMOP_DATAUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 992; RETURN; END IF
RETURN
END SUBROUTINE VTMOP_RECOVER_DATA

! The following module procedures are used internally.

FUNCTION SURROGATE_FUNC(C, IERR) RESULT(F)
! This module procedure uses the private module array VTMOP_MOD_WEIGHTS
! to scalarize the output of VTMOP_MOD_SURROGATES, matches the interface
! of VTMOP_MOD_SCALAR_INT, and can be passed as input to a generic single
! objective optimization procedure.
!
!
! On input:
!
! C(:) is a design point to be evaluated.
!
!
! On output:
!
! IERR is an integer error flag. Any nonzero value indicates an
!    illegal/missing value, as reported by VTMOP_MOD_SURROGATES.
!
! F is a scalar output, as determined by VTMOP_MOD_WEIGHTS(:).
!
IMPLICIT NONE
! Parameters.
REAL(KIND=R8), INTENT(IN) :: C(:)
INTEGER, INTENT(OUT) :: IERR
REAL(KIND=R8) :: F
! Local variables.
REAL(KIND=R8) :: V(VTMOP_MOD_P)
! BLAS function for computing inner products.
REAL(KIND=R8), EXTERNAL :: DDOT
! Evaluate the surrogates.
CALL VTMOP_MOD_SURROGATES(C, V, IERR)
IF (IERR .NE. 0) RETURN
! Compute the weighted sum.
F = DDOT(VTMOP_MOD_P, VTMOP_MOD_WEIGHTS, 1, V, 1)
RETURN
END FUNCTION SURROGATE_FUNC

! The following procedures define the default surrogate, using
! the module LINEAR_SHEPARD from SHEPPACK (ACM TOMS Alg. 905).

SUBROUTINE LSHEP_FIT(D, P, N, X_VALS, Y_VALS, FIRST, PARALLEL, DES_TOL, &
                     SCALE_FACT, SHIFT_FACT, IERR)
! This subroutine fits all P surrogate using the module LINEAR_SHEPARD
! from SHEPPACK.
!
! Thacker, William I., J. Zhang, L. T. Watson, J. B. Birch, M. A. Iyer, and
! M. W. Berry. Algorithm 905: SHEPPACK: Modified Shepard algorithm for
! interpolation of scattered multivariate data. ACM Trans. Math. Softw. (TOMS)
! 37.3 (2010): 34.
!
!
! On input:
!
! D is the dimension of the design space.
!
! P is the dimension of the objective space.
!
! N is the current size of the internal database.
!
! X_VALS(1:D,1:N) is a real vector containing the current database
!    of evaluated design points.
!
! Y_VALS(1:P,1:N) is a real vector containing the current database
!    of corresponding objective values.
!
! FIRST is a logical type that specifies whether this is the first iteration
!    of the algorithm. In the first iteration, since data is sparse, the
!    radius of influence for each design point is doubled.
!
! PARALLEL is a logical type that specifies whether the P LSHEP models should
!    be fit in parallel, using OpenMP.
!
! DES_TOL is a real type that specifies the design space tolerance.
!
! SCALE_FACT(1:D) is a real type that specifies the rescale factor for each
!    dimension of the input data. In particular, each point in X_VALS is
!    transformed by (X_VALS(:,I) - SHIFT_FACT(:)) / SCALE_FACT(:).
!
! SHIFT_FACT(1:D) is a real type that specifies the shift factor for each
!    dimension of the input data. In particular, each point in X_VALS is
!    transformed by (X_VALS(:,I) - SHIFT_FACT(:)) / SCALE_FACT(:).
!
!
! On output:
!
! IERR is an integer error flag.
!
!  00 : Normal output. Successfully fit the P LSHEP models.
!
!  1x : An illegal input was supplied.
!      11 : The problem dimensions D, P, or N contain illegal values.
!      12 : The sizes of the databases X_VALS(:,:) and Y_VALS(:,:) do not
!           match the problem dimensions
!  2x : Error reported by the LSHEP fit subroutine.
!      21 : The number of non-NaN values in the database was not enough
!           to fit the LSHEP models. Try increasing the search budget,
!           or consider whether the cost function is defined within
!           the given bound constraints.
!  3x : A memory allocation error occurred.
!      30 : A memory allocation error occurred.
!      31 : A memory deallocation error occurred.
!
!
! The following module variables are also altered on output.
!
! The P local linear fits are stored in the private module array
! LSHEP_A(:,:,1:P) and the radii of influence in the private module array
! LSHEP_RW(:,1:P). Also, this subroutine makes copies of the current dataset
! in LSHEP_XVALS(:,:) and LSHEP_FVALS(:,:), along with problem dimensions
! in LSHEP_D, LSHEP_P, and LSHEP_N_PTS. LSHEP_DES_TOL is set to the value
! DES_TOL. LSHEP_SCALE and LSHEP_SHIFT are set to SCALE_FACT and SHIFT_FACT,
! respectively.
!
! Finally, a taboo list LSHEP_TABOO(:,:) is created with length LSHEP_N_TABOO.
! If there are any missing values (marked as NaN values) in Y_VALS, then
! these entries appear in LSHEP_TABOO(:,:).
!
!
USE LINEAR_SHEPARD_MOD
USE IEEE_ARITHMETIC
IMPLICIT NONE
! Parameters.
INTEGER, INTENT(IN) :: D ! The dimension of the design space.
INTEGER, INTENT(IN) :: P ! The dimension of the objective space.
INTEGER, INTENT(IN) :: N ! The number of points in X_VALS/Y_VALS.
REAL(KIND=R8), INTENT(IN) :: X_VALS(:,:) ! N points in the design space.
REAL(KIND=R8), INTENT(IN) :: Y_VALS(:,:) ! Corresponding objective values.
LOGICAL, INTENT(IN) :: FIRST ! FIRST is .TRUE. in the 0th iteration.
LOGICAL, INTENT(IN) :: PARALLEL ! Fit the P surrogate models in parallel.
REAL(KIND=R8), INTENT(IN) :: DES_TOL ! Design space tolerance.
REAL(KIND=R8), INTENT(IN) :: SCALE_FACT(:) ! Scale factor for data points.
REAL(KIND=R8), INTENT(IN) :: SHIFT_FACT(:) ! Shift factor for data points.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Local variables.
INTEGER :: I ! Loop indexing variable.
REAL(KIND=R8) :: TMP_FVALS(P,N) ! Temporary list of F values.
REAL(KIND=R8) :: TMP_TABOO(D,N) ! Temporary taboo list.
REAL(KIND=R8) :: TMP_XVALS(D,N) ! Temporary list of x values.
! Check for illegal input dimensions.
IF (D < 1 .OR. P < 2 .OR. N < D+2) THEN
   IERR = 11; RETURN; END IF
! Check that the database dimensions match the input dimensions.
IF ( SIZE(X_VALS, 1) .NE. D .OR.    &
     SIZE(X_VALS, 2) .NE. N .OR.    &
     SIZE(Y_VALS, 1) .NE. P .OR.    &
     SIZE(Y_VALS, 2) .NE. N .OR.    &
     SIZE(SCALE_FACT,1) .NE. D .OR. &
     SIZE(SHIFT_FACT,1) .NE. D ) THEN
   IERR = 12; RETURN; END IF
! Set the LSHEP problem dimensions.
LSHEP_D = D
LSHEP_P = P
LSHEP_DES_TOL = DES_TOL
! Get the SCALE and SHIFT factors.
IF (ALLOCATED(LSHEP_SCALE)) THEN
   DEALLOCATE(LSHEP_SCALE, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 31; RETURN; END IF
END IF
IF (ALLOCATED(LSHEP_SHIFT)) THEN
   DEALLOCATE(LSHEP_SHIFT, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 31; RETURN; END IF
END IF
ALLOCATE(LSHEP_SCALE(D), LSHEP_SHIFT(D), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 30; RETURN; END IF
LSHEP_SCALE(:) = SCALE_FACT(:)
LSHEP_SHIFT(:) = SHIFT_FACT(:)
! Populate the temporary arrays with correct values.
LSHEP_N_PTS = 0
LSHEP_N_TABOO = 0
DO I = 1, N
   ! Check for NAN values.
   IF (ANY(IEEE_IS_NAN(Y_VALS(:,I)))) THEN
      ! Build the taboo list.
      LSHEP_N_TABOO = LSHEP_N_TABOO + 1
      TMP_TABOO(:,LSHEP_N_TABOO) = (X_VALS(:,I)-LSHEP_SHIFT(:))/LSHEP_SCALE(:)
   ELSE
      ! Build the LSHEP surrogate data set.
      LSHEP_N_PTS = LSHEP_N_PTS + 1
      TMP_XVALS(:,LSHEP_N_PTS) = (X_VALS(:,I)-LSHEP_SHIFT(:))/LSHEP_SCALE(:)
      TMP_FVALS(:,LSHEP_N_PTS) = Y_VALS(:,I)
   END IF
END DO
! Check that a reasonable number of non-NaN points were found.
IF (LSHEP_N_PTS > D+1) THEN
   ! Free the module arrays for resizing.
   IF (ALLOCATED(LSHEP_A)) THEN
      DEALLOCATE(LSHEP_A, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 31; RETURN; END IF
   END IF
   IF (ALLOCATED(LSHEP_RW)) THEN
      DEALLOCATE(LSHEP_RW, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 31; RETURN; END IF
   END IF
   IF (ALLOCATED(LSHEP_XVALS)) THEN
      DEALLOCATE(LSHEP_XVALS, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 31; RETURN; END IF
   END IF
   IF (ALLOCATED(LSHEP_FVALS)) THEN
      DEALLOCATE(LSHEP_FVALS, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 31; RETURN; END IF
   END IF
   IF (ALLOCATED(LSHEP_TABOO)) THEN
      DEALLOCATE(LSHEP_TABOO, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 31; RETURN; END IF
   END IF
   ! Reallocate LSHEP_A, LSHEP_RW, LSHEP_FVALS, and LSHEP_XVALS.
   ALLOCATE( LSHEP_A(D, LSHEP_N_PTS, P), LSHEP_RW(LSHEP_N_PTS, P),   &
             LSHEP_XVALS(D,LSHEP_N_PTS), LSHEP_FVALS(LSHEP_N_PTS,P), &
             STAT=IERR )
   IF (IERR .NE. 0) THEN
      IERR = 30; RETURN; END IF
! If not enough non-NAN points were found, LSHEP_FIT cannot proceed.
ELSE
   IERR = 21; RETURN; END IF
! Populate the LSHEP arrays.
LSHEP_XVALS(:,:) = TMP_XVALS(:,1:LSHEP_N_PTS)
LSHEP_FVALS(:,:) = TRANSPOSE(TMP_FVALS(:,1:LSHEP_N_PTS))
! If LSHEP_N_TABOO > 0, allocate the taboo list.
IF (LSHEP_N_TABOO > 0) THEN
   ALLOCATE(LSHEP_TABOO(D,LSHEP_N_TABOO), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 30; RETURN; END IF
   LSHEP_TABOO(:,:) = TMP_TABOO(:,1:LSHEP_N_TABOO)
END IF
! This is the beginning of an iteration task parallelism block. This block
! fits P surrogate models in parallel and stores the weights/radii in A and RW.
!$OMP PARALLEL DO SCHEDULE(STATIC) &
!
! The PRIVATE list specifies uninitialized variables, of which each
! thread has a private copy.
!$OMP& PRIVATE(I, IERR), &
!
! Any variables not explicitly listed above receive the SHARED scope
! by default and are visible across all threads.
!$OMP& DEFAULT(SHARED), &
!
! Only execute if in parallel mode.
!$OMP& IF(PARALLEL)
DO I = 1, LSHEP_P
   ! Perform the LSHEP fit for each of the P surrogate models.
   ! There is no need to consider IERR, since the only serious error
   ! case has already been handled.
   CALL LSHEP( LSHEP_D, LSHEP_N_PTS, LSHEP_XVALS, LSHEP_FVALS(:,I), &
               LSHEP_A(:,:,I), LSHEP_RW(:,I), IERR )
END DO
!$OMP END PARALLEL DO
! Reset the error flag to zero to indicate a successful output.
IERR = 0
! Double the radii of influence in the 0th iteration.
IF (FIRST) LSHEP_RW(:,:) = LSHEP_RW(:,:) * 2.0_R8
RETURN
END SUBROUTINE LSHEP_FIT

SUBROUTINE LSHEP_EVAL(C, V, IERR)
! This subroutine evaluates the P surrogate using the module LINEAR_SHEPARD
! from SHEPPACK.
!
! Thacker, William I., J. Zhang, L. T. Watson, J. B. Birch, M. A. Iyer, and
! M. W. Berry. Algorithm 905: SHEPPACK: Modified Shepard algorithm for
! interpolation of scattered multivariate data. ACM Trans. Math. Softw. (TOMS)
! 37.3 (2010): 34.
!
! This subroutine uses the private module variables and arrays LSHEP_D,
! LSHEP_P, LSHEP_N_PTS, LSHEP_XVALS, LSHEP_FVALS, LSHEP_A, LSHEP_DES_TOL,
! and LSHEP_RW, as set by the subroutine LSHEP_FIT.
!
! Also, this subroutine checks the taboo list
! LSHEP_TABOO(1:LSHEP_D,1:LSHEP_N_TABOO)
! for missing values.
!
!
! On input:
!
! C(:) is a real point in the design space.
!
!
! On output:
!
! V(:) is the objective value predicted at C(:), according to the LSHEP
!    surrogate functions.
!
! IERR is an integer error flag. Any nonzero value indicates an
!    illegal/missing value, as recorded in the taboo list LSHEP_TABOO.
!
USE LINEAR_SHEPARD_MOD
IMPLICIT NONE
! Parameters.
REAL(KIND=R8), INTENT(IN) :: C(:) ! Input point (from the design space).
REAL(KIND=R8), INTENT(OUT) :: V(:) ! Output point (from the objective space).
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Local variables.
INTEGER I ! Loop indexing variable.
REAL(KIND=R8) :: CL(SIZE(C,1)) ! Rescaled input point.
! BLAS function for Euclidean distance.
REAL(KIND=R8), EXTERNAL :: DNRM2
! Rescale the input point and store in CL(:).
CL(:) = (C(:)-LSHEP_SHIFT(:)) / LSHEP_SCALE(:)
! First check the taboo list.
DO I = 1, LSHEP_N_TABOO
   IF (DNRM2(LSHEP_D, CL(:) - LSHEP_TABOO(:,I), 1) < LSHEP_DES_TOL) THEN
      IERR = -1; RETURN; END IF
END DO
! Evaluate the surrogate models.
DO I = 1, LSHEP_P
   V(I) = LSHEPVAL( CL, LSHEP_D, LSHEP_N_PTS, LSHEP_XVALS, LSHEP_FVALS(:,I), &
                    LSHEP_A(:,:,I), LSHEP_RW(:,I), IERR )
   IF (IERR .GE. 10) RETURN
END DO
! Reset error flag to success code.
IERR = 0
RETURN
END SUBROUTINE LSHEP_EVAL

! The following is the optimization procedure. It is a lightweight native
! Fortran implementation of the polling algorithm generalized pattern search
! (GPS) as in the NOMAD software package (ACM TOMS Alg. 909).

SUBROUTINE GPS(D, X, LB, UB, OBJ_FUNC, BUDGET, TOL, IERR)
! This is a lightweight implementation of MADS designed for usage with
! computationally cheap surrogate functions, based on the algorithm GPS
! described in
!
! Le Digabel, Sébastien. Algorithm 909: NOMAD: Nonlinear Optimization with
! the MADS Algorithm. ACM Trans. Math. Softw. (TOMS) 37.4 (2011): 15.
!
! All features not relevant for local optimization of computationally cheap
! surrogates, such as quadratic surrogate models for poll ordering, the global
! search phase, the variable neighborhood search, and the taboo list are
! omitted.
!
!
! On input:
!
! D is the dimension of the design space.
!
! X(1:D) contains a design point from which to start the local optimization
!    procedure.
!
! LB(1:D) contains the lower bound constraints for the design space or the
!    current local trust region (LTR).
!
! UB(1:D) contains the upper bound constraints for the design space or the
!    current LTR.
!
! OBJ_FUNC is a subroutine, whose interface matches VTMOP_MOD_SCALAR_INT.
!    OBJ_FUNC returns a scalarization of the surrogate models. Any
!    missing/illegal values requested should trigger OBJ_FUNC to return
!    a nonzero error flag.
!
! BUDGET is the iteration budget for iterations of the algorithm GPS.
!
! TOL is the tolerance for the design space. Once the mesh fineness reaches
!    TOL the algorithm terminates, regardless of the value of BUDGET.
!
!
! On output:
!
! X(:) is a local minimizer of the scalarized surrogate functions, described
!    by OBJ_FUNC.
!
! IERR is an integer error flag.
!
!  00 : Normal output. Successfully converged on a local minimizer, or
!       BUDGET iterations of GPS.
!
!  1x : An illegal input was supplied.
!      10 : The problem dimensions D, P, or N contain illegal values.
!      11 : The dimension of the bound constraint arrays LB(:) or UB(:) do
!           not match the design dimension.
!      12 : LB(:) must be elementwise strictly less than UB(:) - TOL.
!      13 : X(:) must be a point in the specified bound constraints [LB, UB].
!
IMPLICIT NONE
! Parameter list.
INTEGER, INTENT(IN) :: D ! The dimension of the design space.
REAL(KIND=R8), INTENT(INOUT) :: X(:) ! The starting point for GPS.
REAL(KIND=R8), INTENT(IN) :: LB(:) ! The lower bound constraints.
REAL(KIND=R8), INTENT(IN) :: UB(:) ! The upper bound constraints.
PROCEDURE(VTMOP_MOD_SCALAR_INT) :: OBJ_FUNC ! The scalarized objective function.
INTEGER, INTENT(IN) :: BUDGET ! The iteration budget.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
REAL(KIND=R8), INTENT(IN) :: TOL ! The design space tolerance.
! Local variables.
INTEGER :: I, J ! Loop index variables.
INTEGER :: MIN_POLL ! Minimum poll index.
REAL(KIND=R8) :: X_VAL ! Current X value.
REAL(KIND=R8) :: POLLS(D,2*D) ! List of poll directions.
REAL(KIND=R8) :: POLL_VALS(2*D) ! List of poll values.
REAL(KIND=R8) :: MESH_GPS(D,2*D) ! The GPS mesh.
REAL(KIND=R8) :: MESH_SIZE ! The current mesh size.
REAL(KIND=R8) :: RESCALE(D) ! Rescale factors for bounding box.
! Check for bad inputs.
IF (SIZE(X, 1) .NE. D) THEN
   IERR = 10; RETURN; END IF
IF ( (SIZE(LB, 1) .NE. D) .OR. (SIZE(UB,1) .NE. D) ) THEN
   IERR = 11; RETURN; END IF
IF ( ANY(LB(:) .GE. UB(:) - TOL) ) THEN
   IERR = 12; RETURN; END IF
IF ( ANY(X(:) .GE. UB(:) + TOL) .OR. ANY(X(:) .LE. LB(:) - TOL) ) THEN
   IERR = 13; RETURN; END IF
! Initialize the mesh fineness and rescale factors.
RESCALE(:) = (UB(:) - LB(:)) / 2.0_R8
MESH_SIZE = 1.0_R8
! Generate the GPS mesh.
DO I = 1, D
   MESH_GPS(:,2*I-1) = 0.0_R8
   MESH_GPS(I,2*I-1) = 1.0_R8
   MESH_GPS(:,2*I) = 0.0_R8
   MESH_GPS(I,2*I) = -1.0_R8
END DO
! Initialize the center value.
X_VAL = OBJ_FUNC(X, IERR)
! Avoid missing values.
IF (IERR .NE. 0) THEN
   X_VAL = HUGE(0.0_R8)
   IERR = 0
END IF
! Loop until the iteration budget is exhausted.
DO I = 1, BUDGET ! Stopping condition 1: budget exhausted.
   DO J = 1, 2*D
      ! Get the next poll point.
      POLLS(:,J) = X(:) + (MESH_GPS(:,J) * RESCALE(:) * MESH_SIZE)
      ! Now predict the objective value for the poll.
      IF ( ANY(POLLS(:,J) > UB(:)) .OR. &
           ANY(POLLS(:,J) < LB(:)) ) THEN
         ! Use an extreme barrier approach for bound violations.
         POLL_VALS(J) = HUGE(0.0_R8)
      ELSE
         ! For legal values, query the objective surrogate.
         POLL_VALS(J) = OBJ_FUNC(POLLS(:,J), IERR)
         ! Avoid missing values.
         IF (IERR .NE. 0) THEN
            X_VAL = HUGE(0.0_R8)
            IERR = 0
         END IF
      END IF
   END DO
   ! Check all poll directions for the best result.
   MIN_POLL = MINLOC(POLL_VALS, 1)
   IF (POLL_VALS(MIN_POLL) < X_VAL) THEN
      ! If the result is an improvement, then move to the new poll location.
      X(:) = POLLS(:,MIN_POLL)
      X_VAL = POLL_VALS(MIN_POLL)
   ELSE
      ! Otherwise, decay the mesh size.
      MESH_SIZE = MESH_SIZE * 0.5_R8
      ! If the mesh size has reached its limit, then exit.
      IF (MESH_SIZE < TOL) EXIT ! Stop cond 2: mesh tolerance reached.
   END IF
END DO
RETURN
END SUBROUTINE GPS

! The following are possible global search options. LH_DESIGN returns a
! batch of design points to evaluate.

SUBROUTINE LH_DESIGN(D, LB, UB, N_PTS, CAND_PTS, IERR, TOL, XI)
! This wrapper generates a Latin hypercube design of experiment using the
! QNSTOP subroutine LATINDESIGN, as implemented in
!
! Amos, B. D., D. R. Easterling, L. T. Watson, W. I. Thacker, B. S. Castle,
! and M. W. Trosset. Algorithm XXX: QNSTOP -- Quasi-Newton Algorithm for
! Stochastic Optimization. Tech Report. Virginia Polytechnic Institute and
! State University (2014).
!
! This design can be used to perform an exploration of the entire design
! space, or the current local trust region (LTR).
!
!
! On input:
!
! D is the dimension of the design space.
!
! LB(1:D) contains the lower bound constraints for the design space or the
!    current LTR.
!
! UB(1:D) contains the upper bound constraints for the design space or the
!    current LTR.
!
! N_PTS is an integer specifying the size of the requested design.
!
!
! On output:
!
! CAND_PTS(:,:) is a real allocatable array. On output, CAND_PTS(:,:)
!    is allocated to size D by N_PTS, and contains a Latin hypercube design.
!
! IERR is an integer error flag, which relays error messages from
!    VTdirect.
!
!  00 : Normal output. VTdirect has successfully run for MAXITERS iterations,
!       or until some other termination condition, for each of the weightings
!       in the ADAPTIVE_WEIGHTS(:,:) array.
!
!  1x : Input data error.
!     10 : D does not match the size of LB(:) or UB(:).
!     11 : LB(:) must be componentwise less than UB(:) - TOL.
!     12 : When present, the size of XI(:) must match with D.
!     13 : When present, XI(:) must be within the bound constraints [LB, UB].
!  20 : A memory allocation error has occurred.
!
!
! Optional input arguments:
!
! TOL is the tolerance for the design space. This value is only used for
!    sanity checks. By default, TOL is the square-root of the machine
!    epsilon.
!
! XI(1:D) is an initial point in the design space, to include in the Latin
!    hypercube design. When supplied, N_PTS additional points are generated,
!    with the understanding that XI has already been evaluated.
!
USE QNSTOPS_MOD, ONLY : LATINDESIGN
IMPLICIT NONE
! Parameter list.
INTEGER, INTENT(IN) :: D ! The dimension of the design space.
REAL(KIND=R8), INTENT(IN) :: LB(:), UB(:) ! Lower and upper bound constraints.
INTEGER, INTENT(IN) :: N_PTS ! Number of points in the design.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: CAND_PTS(:,:) ! The output design.
INTEGER, INTENT(OUT) :: IERR ! Error flag.
! Optional Parameters.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: TOL ! Design space tolerance.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: XI(:) ! An initial design point.
! Local variables.
REAL(KIND=R8) :: DES_PTS(D,N_PTS+1) ! Set of new candidate points.
REAL(KIND=R8) :: TOL_L ! Local copy of the tolerance.
REAL(KIND=R8) :: XIL(D) ! Local copy of initial point.
! Check for bad input dimensions.
IF (SIZE(LB,1) .NE. D .OR. SIZE(UB,1) .NE. D) THEN
   IERR = 10; RETURN; END IF
! Get optional inputs.
TOL_L = SQRT(EPSILON(0.0_R8))
IF (PRESENT(TOL)) THEN
   IF (TOL > TOL_L) TOL_L = TOL
END IF
! Check that the bounds are appropriate.
IF (ANY(UB(:) - LB(:) < TOL_L)) THEN
   IERR = 11; RETURN; END IF
! Different rules depending on whether an initial point is supplied.
IF (PRESENT(XI)) THEN
   ! Check that XI is of the correct dimension.
   IF (SIZE(XI,1) .NE. D) THEN
      IERR = 12; RETURN; END IF
   ! Check that XI is within the bounds.
   IF (ANY(XI(:) > UB(:) + TOL_L) .OR. ANY(XI(:) < LB(:) - TOL_L)) THEN
      IERR = 13; RETURN; END IF
   ! Create a design with N_PTS+1 points, so that XI(:) is included in the
   ! design but N_PTS new points are generated.
   IF (N_PTS > 0) THEN
      CALL LATINDESIGN(D, N_PTS+1, LB, UB, XI, DES_PTS)
   ELSE
      DES_PTS(1:D,1) = XI(:)
   END IF
ELSE
   ! Otherwise, randomly generate XIL.
   CALL RANDOM_NUMBER(XIL(:))
   ! Rescale to bounds.
   XIL(:) = XIL(:) * (UB(:) - LB(:)) + LB(:)
   ! Create a design with N_PTS points.
   IF (N_PTS > 1) THEN
      CALL LATINDESIGN(D, N_PTS, LB, UB, XIL, DES_PTS(1:D,1:N_PTS))
   ELSE
      DES_PTS(1:D,1) = XIL(:)
   END IF
END IF
! Allocate the output array.
ALLOCATE(CAND_PTS(D,N_PTS), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 20; RETURN; END IF
! Copy the points into the output array.
CAND_PTS(:,:) = DES_PTS(:,1:N_PTS)
RETURN
END SUBROUTINE LH_DESIGN

END MODULE VTMOP_MOD

! The subroutine DELAUNAYGRAPH is an external procedure, as it may be of
! interest independently.

SUBROUTINE DELAUNAYGRAPH(D, N, PTS, GRAPH, IERR, EPS, IBUDGET, PMODE)
! This subroutine produces the Delaunay graph for a set of N points in R^D
! in polynomial time using DELAUNAYSPARSE. If DELAUNAYSPARSE reports that PTS
! is embedded in a lower-dimensional linear manifold (error code 31), then
! PTS is projected onto the span of its left singular vectors with nonzero
! singular values (i.e., principle component analysis).
!
!
! On input:
!
! D is the dimension of the space for PTS.
!
! N is the number of data points in PTS.
!
! PTS(1:D,1:N) is a real matrix with N columns, each containing the
!    coordinates of a single data point in R^D.
!
!
! On output:
!
! PTS has been rescaled and shifted. All the data points in PTS are now
!    contained in the unit hyperball in R^D.
!
! GRAPH(1:N,1:N) is a symmetric matrix of type LOGICAL containing the
!    Delaunay graph structure. If GRAPH(I,J) is .TRUE., then the vertices
!    PTS(:,I) and PTS(:,J) are Delaunay neighbors in some Delaunay
!    triangulation. Otherwise, if GRAPH(I,J) is .FALSE., then PTS(:,I) and
!    PTS(:,J) are not neighbors in some Delaunay triangulation.
!
! IERR is an integer error flag. For the most part, IERR relays
!    error codes from DELAUNAYSPARSE{S|P}.
!    The error codes are:
!
! 00 : Successfully interpolated all midpoints and constructed the Delaunay
!      graph structure.
! 01 : Too few points were provided to compute a complete triangulation.
!      The Delaunay graph was still computed, under the assumption that
!      all points are Delaunay neighbors.
! 02 : The input set PTS is embedded in a lower dimensional linear manifold.
!      The Delaunay graph was still computed for a dimension reduced copy of
!      PTS.
!
! 10 : The dimension D must be positive.
! 12 : The supplied LOGICAL matrix GRAPH(:,:) is not of dimension N x N.
! 13 : The first dimension of PTS does not agree with the dimension D.
! 14 : The second dimension of PTS does not agree with the number of points N.
!
! 26 : The budget supplied in IBUDGET does not contain a positive
!      integer.
!
! 30 : Two or more points in the data set PTS are too close together with
!      respect to the working precision (EPS), which would result in a
!      numerically degenerate simplex.
!
! 40 : An error caused DELAUNAYSPARSE to terminate before the entire
!      Delaunay graph could be computed.
!
! 50 : A memory allocation error occurred while allocating the internal
!      work array for DELAUNAYSPARSE.
! 51 : A memory allocation error occurred while allocating the work arrays
!      for DGESVD, for performing dimension reduction.
! 52 : A memory deallocation error occurred while freeing work arrays for
!      DGESVD.
!
! 60 : The budget was exceeded before the algorithm converged for at least
!      one of the simplices. If the dimension is high, try increasing IBUDGET.
!      This error can also be caused by a working precision EPS that is too
!      small for the conditioning of the problem.
!
! 61 : A value that was judged appropriate later caused LAPACK to encounter a
!      singularity. Try increasing the value of EPS.
!
! 70 : Allocation error for the extrapolation work arrays.
! 71 : The SLATEC subroutine DWNNLS failed to converge during the projection
!      of an extrapolation point onto the convex hull.
! 72 : The SLATEC subroutine DWNNLS has reported a usage error.
! 73 : At least one of the midpoints was ruled significantly outside the
!      convex hull of PTS by DELAUNAYSPARSE. This is a numerical precision
!      issue that should never occur. Consider increasing EPS.
!
!      The errors 72, 80--83, and 90 should never occur, and likely indicate
!      a compiler bug or hardware failure.
! 80 : The LAPACK subroutine DGEQP3 has reported an illegal value.
! 81 : The LAPACK subroutine DGETRF has reported an illegal value.
! 82 : The LAPACK subroutine DGETRS has reported an illegal value.
! 83 : The LAPACK subroutine DORMQR has reported an illegal value.
!
! 90 : The LAPACK subroutine DGESVD has reported an illegal value.
! 91 : The LAPACK subroutine DGESVD has failed to converge.
!
!
! Optional arguments:
!
! EPS contains the working precision for the problem on input. By default,
!    EPS is assigned \sqrt{\mu} where \mu denotes the unit roundoff for the
!    machine. In general, any values that differ by less than EPS are judged
!    as equal, and any weights that are greater than -EPS are judged as
!    nonnegative.  EPS cannot take a value less than the default value of
!    \sqrt{\mu}. If any value less than \sqrt{\mu} is supplied, the default
!    value will be used instead automatically.
!
! IBUDGET contains the integer budget for performing flips while
!    iterating toward the simplex containing each interpolation point in Q.
!    This prevents DelaunayFan from falling into an infinite loop when
!    supplied with degenerate or near degenerate data.  By default,
!    IBUDGET=50000. However, for extremely high-dimensional problems and
!    pathological data sets, the default value may be insufficient.
!
! PMODE is a logical value specifying whether to compute the Delaunay graph
!    serially (using DELAUNAYSPARSES) or in parallel (using DELAUNAYSPARSEP).
!
USE DELSPARSE_MOD
IMPLICIT NONE

! Input arguments.
INTEGER, INTENT(IN) :: D, N
REAL(KIND=R8), INTENT(INOUT) :: PTS(:,:)
! Output arguments.
LOGICAL, INTENT(OUT) :: GRAPH(:,:)
INTEGER, INTENT(OUT) :: IERR
! Optional arguments.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: EPS
INTEGER, OPTIONAL, INTENT(IN) :: IBUDGET
LOGICAL, OPTIONAL, INTENT(IN) :: PMODE

! Local variables.
INTEGER :: IBUDGETL ! Local copy of IBUDGET.
INTEGER :: I, J, K ! Loop iteration variables.
INTEGER :: SIMPS(D+1,N*(N-1)/2) ! Matrix of simplices.
INTEGER :: IERR_LIST(N*(N-1)/2) ! Array of error codes for DELAUNAYSPARSE.
LOGICAL :: PMODEL ! Local copy of PMODE.
REAL(KIND=R8) :: EPSL ! Local copy of EPS.
REAL(KIND=R8) :: WEIGHTS(D+1,N*(N-1)/2) ! Matrix of interpolation weights.
REAL(KIND=R8) :: Q(D,N*(N-1)/2) ! Matrix of interpolation points.

! Work arrays for DGESVD, only referenced if dimension reduction is required.
INTEGER :: LWORK ! Length of the work array.
REAL(KIND=R8), ALLOCATABLE :: LSV(:,:) ! Left singular vectors.
REAL(KIND=R8), ALLOCATABLE :: RED_PTS(:,:) ! Dimension reduced PTS.
REAL(KIND=R8), ALLOCATABLE :: RED_Q(:,:) ! Dimension reduced Q.
REAL(KIND=R8), ALLOCATABLE :: S(:) ! Singular values.
REAL(KIND=R8), ALLOCATABLE :: WORK(:) ! Work array.
REAL(KIND=R8) :: U(1), VT(1) ! Optional outputs, not referenced by DGESVD.

! Check whether GRAPH is a legal size.
IF (SIZE(GRAPH, 1) .NE. N .OR. SIZE(GRAPH, 2) .NE. N) THEN
   IERR = 12
   RETURN
END IF
! Compute the machine precision.
EPSL = SQRT(EPSILON(1.0_R8))
! Check for the optional value EPS, and ensure that EPS is large enough.
IF (PRESENT(EPS)) THEN
   IF(EPSL < EPS) EPSL = EPS
END IF
! Set the budget.
IBUDGETL = 50000
! Check for the optional input IBUDGET, and ensure that it is valid.
IF (PRESENT(IBUDGET)) THEN
   IF (IBUDGET < 1) THEN; IERR = 26; RETURN; END IF
   IBUDGETL = IBUDGET
END IF
! Get optional input PMODE.
PMODEL = .FALSE.
IF (PRESENT(PMODE)) PMODEL = PMODE

! Initialize the interpolation points.
K = 0
DO I = 1, N
   DO J = I+1, N
      ! Interpolate the midpoint between PTS(:,I) and PTS(:,J).
      Q(:,K+J-I) = (PTS(:,I) + PTS(:,J)) / 2.0_R8
   END DO
   ! Track the offset K.
   K = K + N - I
END DO

! Check whether iteration parallelism is turned on.
IF(PMODEL) THEN
   ! Compute the Delaunay simplices containing the midpoints in parallel.
   CALL DELAUNAYSPARSEP( D, N, PTS, N*(N-1)/2, Q, SIMPS, WEIGHTS, IERR_LIST, &
                         EPS=EPSL, IBUDGET=IBUDGETL, EXACT=.FALSE., PMODE=1 )
ELSE
   ! Compute the Delaunay simplices containing the midpoints serially.
   CALL DELAUNAYSPARSES( D, N, PTS, N*(N-1)/2, Q, SIMPS, WEIGHTS, IERR_LIST, &
                         EPS=EPSL, IBUDGET=IBUDGETL, EXACT=.FALSE. )
END IF

! Check for errors.
IERR = 0
DO I = 1, N*(N-1)/2
   IF(IERR_LIST(I) > 2) THEN
      IERR = IERR_LIST(I) ! Pass the error code on.
   ELSE IF(IERR_LIST(I) .EQ. 2) THEN
      IERR = 73; END IF ! Error code 2 -> 73.
END DO

! Handle relevant error codes for DELAUNAYGRAPH.
IF (IERR .EQ. 11) THEN
   ! N is too small to triangulate PTS, so all PTS are considered neighbors.
   GRAPH(:,:) = .TRUE.
   FORALL ( I = 1 : N ) GRAPH(I,I) = .FALSE.
   IERR = 1
ELSE IF (IERR .EQ. 31) THEN
   ! PTS lie in a lower dimensional linear manifold. Reduce the dimension
   ! and retry the triangulation.

   ! Allocate the left singular vectors and singular values.
   ALLOCATE(LSV(D,N), S(D), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 51; RETURN; END IF
   ! Query the optimal size for WORK and store in LWORK.
   LWORK = -1
   CALL DGESVD('O', 'N', D, N, LSV, D, S, VT, 1, VT, 1, U, LWORK, IERR)
   IF (IERR < 0) THEN
      IERR = 90; RETURN
   ELSE IF (IERR > 0) THEN
      IERR = 91; RETURN; END IF
   LWORK = INT(U(1))
   ! Allocate the input/output matrix.
   ALLOCATE(WORK(LWORK), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 51; RETURN; END IF
   ! Copy the input array.
   LSV(:,:) = PTS(:,:)
   ! Compute the SVD and overwrite LSV with the left singular vectors.
   ! The dummy arguments U and VT are not referenced for these settings.
   CALL DGESVD('O', 'N', D, N, LSV, D, S, U, 1, VT, 1, WORK, LWORK, IERR)
   IF (IERR < 0) THEN
      IERR = 90; RETURN
   ELSE IF (IERR > 0) THEN
      IERR = 91; RETURN; END IF
   ! Perform the dimension reduction by eliminating directions with small
   ! singular values.
   DO I = 1, D; IF (S(I) < EPSL) EXIT; END DO
   I = I - 1
   ! Free the memory that was used for the SVD.
   DEALLOCATE(S, WORK, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 52; RETURN; END IF
   ! Allocate and initialize the reduced data and interpolation point sets.
   ALLOCATE(RED_PTS(I,N), RED_Q(I,N*(N-1)/2), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 51; RETURN; END IF
   RED_PTS(:,:) = 0.0_R8
   RED_Q(:,:) = 0.0_R8
   ! Compute the projection RED_PTS(1:I,1:N) = LSV(1:D,1:I)^T PTS(1:D,1:N).
   CALL DGEMM('T', 'N', I, N, D, 1.0_R8, LSV, D, PTS, D, 0.0_R8, RED_PTS, I)
   ! Compute the projection RED_Q(1:I,1:N) = LSV(1:D,1:I)^T Q(1:D,:)
   CALL DGEMM('T', 'N', I, N*(N-1)/2, D, 1.0_R8, LSV, D, Q, D, 0.0_R8, RED_Q, I)
   ! Check whether iteration tasks are parallelized.
   IF (PMODEL) THEN
      ! Compute the reduced dimensional solution, using DELAUNAYSPARSEP.
      CALL DELAUNAYSPARSEP( I, N, RED_PTS, N*(N-1)/2, RED_Q, SIMPS(1:I+1,:), &
                            WEIGHTS(1:I+1,:), IERR_LIST, EPS=EPSL,           &
                            IBUDGET=IBUDGETL, EXACT=.FALSE., PMODE=1 )
   ELSE
      ! Compute the reduced dimensional solution, using DELAUNAYSPARSES.
      CALL DELAUNAYSPARSES( I, N, RED_PTS, N*(N-1)/2, RED_Q, SIMPS(1:I+1,:), &
                            WEIGHTS(1:I+1,:), IERR_LIST, EPS=EPSL,           &
                            IBUDGET=IBUDGETL, EXACT=.FALSE. )
   END IF
   SIMPS(I+2:D+1,:) = 0
   ! Check for errors.
   IERR = 2
   DO I = 1, N
      IF(IERR_LIST(I) > 2) THEN
         IERR = IERR_LIST(I) ! Pass the error code on.
      ELSE IF(IERR_LIST(I) .EQ. 2) THEN
         IERR = 73; END IF ! Error code 2 -> 73.
   END DO

   ! Free the temporary data matrices used for dimension reduction.
   DEALLOCATE(RED_PTS, RED_Q, LSV, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 52; RETURN; END IF
END IF

! If an error code persists, quit.
IF (IERR .GE. 10) RETURN

! Initialize the Delaunay graph.
GRAPH(:,:) = .FALSE.
! Construct the Delaunay graph from the resulting simplices.
K = 0
DO I = 1, N
   DO J = I+1, N
      ! Check SIMPS(:,K+J-I) for any edge between PTS(:,I) and PTS(:,J).
      IF (ANY(SIMPS(:,K+J-I) .EQ. I) .AND. ANY(SIMPS(:,K+J-I) .EQ. J)) THEN
         GRAPH(I,J) = .TRUE.; GRAPH(J,I) = .TRUE.;  END IF

      ! If SIMPS(:,K+J-I) does not contain the edge ( PTS(:,I), PTS(:,J) ),
      ! then SIMPS(:,K+J-I) serves as a certificate of separation in some
      ! Delaunay triangulation of PTS.
   END DO
   ! Track the offset K.
   K = K + N - I
END DO

RETURN
END SUBROUTINE DELAUNAYGRAPH

! The subroutine QSORTC_DVEC is an external procedure, as it may be of
! interest independently.

SUBROUTINE QSORTC_DVEC(A, IDX)
! This is a QuickSort routine adapted from QNSTOP adapted from Orderpack 2.0.
!
! Also, this implementation incorporates ideas from "A Practical Introduction
! to Data Structures and Algorithm Analysis", by Clifford Shaffer.
!
! It sorts real vectors into ascending numerical order by their sums
! and keeps an index of the value's original array position.
!
! Author: Will Thacker, Winthrop University, July 2013.
! Update: Tyler Chang, Argonne National Lab, Jan 2021 (sort by column sum).
!
! QSORTC_DVEC sorts the real 2D array A by its column sum and keeps the index
! of the value's original array position along with the value (integer array
! IDX).
!
! On input:
!
! A(:,:) is the array to be sorted by column sum.
!
! On output:
!
! A(:,:) is sorted by column sum.
!
! IDX(1:SIZEOF(A,2)) contains the original positions of the sorted values.
!    I.e., sorted(i) = orginal_unsorted(IDX(i)).
!
USE REAL_PRECISION, ONLY : R8
IMPLICIT NONE

! Input/output parameters.
REAL(KIND=R8), INTENT(INOUT) :: A(:,:)
INTEGER, INTENT(OUT) :: IDX(SIZE(A,2))

! Local variables
INTEGER :: I   ! Loop iteration variable.

! Initialize the array of original positions.
FORALL (I=1:SIZE(A,2)) IDX(I)=I

CALL QSORTC_DVEC_HELPER(A, IDX, 1, SIZE(A,2))
RETURN

CONTAINS

RECURSIVE SUBROUTINE QSORTC_DVEC_HELPER(A, IDX, ISTART, ISTOP)
! This internal recursive subroutine performs the recursive quicksort
! algorithm.  It is needed because the boundaries of the part of the
! array being sorted change and the initial call to the sort routine
! does not need to specify boundaries since, generally, the user will
! want to sort the entire array passed.
!
! On input:
!
! A(:,:) contains a subset of columns to be sorted.
!
! IDX(i) contains the initial position of the value A(:,i) before sorting.
!
! ISTART is the starting position of the subarray to be sorted.
!
! ISTOP is the ending position of the subarray to be sorted.
!
! On output:
!
! A(:,ISTART:ISTOP) will be sorted by column sum.
!
! IDX(i) contains the original position for the value at A(:,i).
!

! Input/output parameters
REAL(KIND=R8), INTENT(INOUT) :: A(:,:)
INTEGER, INTENT(INOUT) :: IDX(SIZE(A,2))
INTEGER, INTENT(IN) :: ISTART, ISTOP

!  Local variables
INTEGER :: ILEFT ! A position on the left to be swapped with value at IRIGHT.
INTEGER :: IMID ! The middle position used to select the pivot.
INTEGER :: IRIGHT ! A position on the right to be swapped with value at ILEFT.
INTEGER :: ITEMP  ! Used for swapping within IDX.
REAL(KIND=R8) :: ATEMP(SIZE(A,1)) ! Used for swapping.
REAL(KIND=R8) :: PIVOT(SIZE(A,1)) ! Holds the temporary pivot.

! INSMAX is used to stop recursively dividing the array and to instead
! use a sort that is more efficient for small arrays than quicksort.
!
! The best cutoff point is system dependent.
INTEGER, PARAMETER:: INSMAX=24

! Check to see if we have enough values to make quicksort useful.
! Otherwise let the insertion sort handle it.
IF ((ISTOP - ISTART) < INSMAX) THEN
  CALL INSERTION_DVEC(A, IDX, ISTART, ISTOP)
ELSE

   ! Use the median of the first, middle and last items for the pivot
   ! and place the median (pivot) at the end of the list.
   ! Putting it at the end of the list allows for a guard value to keep
   ! the loop from falling off the right end of the array (no need to
   ! check for at the end of the subarray EACH time through the loop).
   IMID = (ISTART + ISTOP)/2

   IF (SUM(A(:,ISTOP)) < SUM(A(:,ISTART))) THEN
      ATEMP(:) = A(:,ISTART)
      A(:,ISTART) = A(:,ISTOP)
      A(:,ISTOP) = ATEMP(:)

      ITEMP = IDX(ISTART)
      IDX(ISTART) = IDX(ISTOP)
      IDX(ISTOP) = ITEMP
   END IF

   IF (SUM(A(:,IMID)) < SUM(A(:,ISTOP))) THEN
      ATEMP(:) = A(:,ISTOP)
      A(:,ISTOP) = A(:,IMID)
      A(:,IMID) = ATEMP(:)

      ITEMP = IDX(ISTOP)
      IDX(ISTOP) = IDX(IMID)
      IDX(IMID) = ITEMP

      IF (SUM(A(:,ISTOP)) < SUM(A(:,ISTART))) THEN
         ATEMP(:) = A(:,ISTOP)
         A(:,ISTOP) = A(:,ISTART)
         A(:,ISTART) = ATEMP(:)

         ITEMP = IDX(ISTOP)
         IDX(ISTOP) = IDX(ISTART)
         IDX(ISTART) = ITEMP
      END IF
   END IF

   ! Now, the first position has a value that is less or equal to the
   ! partition. So, we know it belongs in the left side of the partition
   ! and we can skip it. Also, the pivot is at the end.  So, there is
   ! no need to compare the pivot with itself.
   PIVOT(:) = A(:,ISTOP)
   ILEFT = ISTART + 1
   IRIGHT = ISTOP - 1

   DO WHILE (ILEFT < IRIGHT)
      ! Find a value in the left side that is bigger than the pivot value.
      ! Pivot is at the right end so ILEFT will not fall off the end
      ! of the subarray.
      DO WHILE (SUM(A(:,ILEFT)) < SUM(PIVOT(:)))
         ILEFT = ILEFT + 1
      END DO

      DO WHILE (IRIGHT .NE. ILEFT)
         IF (SUM(A(:,IRIGHT)) .LT. SUM(PIVOT(:))) EXIT
            IRIGHT = IRIGHT - 1
      END DO

      ! Now we have a value bigger than pivot value on the left side that can
      ! be swapped with a value smaller than the pivot on the right side.
      !
      ! This gives us all values less than pivot on the left side of the
      ! array and all values greater than the pivot on the right side.
      ATEMP(:) = A(:,IRIGHT)
      A(:,IRIGHT) = A(:,ILEFT)
      A(:,ILEFT) = ATEMP(:)

      ITEMP = IDX(IRIGHT)
      IDX(IRIGHT) = IDX(ILEFT)
      IDX(ILEFT) = ITEMP

   END DO
   !
   ! The last swap was in error (since the while condition is not checked
   ! until after the swap is done) so we swap again to fix it.
   !
   ! This is done (once) rather than having an if (done many times) in the
   ! loop to prevent the swapping.
   ATEMP(:) = A(:,IRIGHT)
   A(:,IRIGHT) = A(:,ILEFT)
   A(:,ILEFT) = ATEMP(:)

   ITEMP = IDX(IRIGHT)
   IDX(IRIGHT) = IDX(ILEFT)
   IDX(ILEFT) = ITEMP

   ! Put the pivot value in its correct spot (between the 2 partitions)
   ! When the WHILE condition finishes, ILEFT is greater than IRIGHT.
   ! So, ILEFT has the position of the first value in the right side.
   ! This is where we can put the pivot (and where it will finally rest,
   ! so no need to look at it again).  Also, place the first value of
   ! the right side (being displaced by the pivot) at the end of the
   ! subarray (since it is bigger than the pivot).
   ATEMP(:) = A(:,ISTOP)
   A(:,ISTOP) = A(:,ILEFT)
   A(:,ILEFT) = ATEMP(:)

   ITEMP = IDX(ISTOP)
   IDX(ISTOP) = IDX(ILEFT)
   IDX(ILEFT) = ITEMP

   CALL QSORTC_DVEC_HELPER(A, IDX, ISTART, ILEFT-1)
   CALL QSORTC_DVEC_HELPER(A, IDX, ILEFT+1, ISTOP)
END IF
RETURN
END SUBROUTINE QSORTC_DVEC_HELPER

SUBROUTINE INSERTION_DVEC(A, IDX, ISTART, ISTOP)
! This subroutine performs an insertion sort used for sorting
! small subarrays efficiently.
!
! This subroutine sorts a subarray of A by column sum (between positions
! ISTART and ISTOP) keeping a record of the original position (array IDX).
!
! On input:
!
! A(:,:) contains a subset of columns to be sorted.
!
! IDX(i) contains the initial position of the value A(:,i) before sorting.
!
! ISTART is the starting position of the subarray to be sorted.
!
! ISTOP is the ending position of the subarray to be sorted.
!
! On output:
!
! A(:,ISTART:ISTOP) will be sorted by column sum.
!
! IDX(i) contains the original position for the value at A(:,i).
!

! Input/output parameters.
REAL(KIND=R8), INTENT(INOUT) :: A(:,:)
INTEGER, INTENT(INOUT) :: IDX(SIZE(A,2))
INTEGER, INTENT(IN) :: ISTART, ISTOP

! Local variables.
REAL(KIND=R8) :: AMIN(SIZE(A,1))  ! Temporary minimum.
REAL(KIND=R8) :: ATEMP(SIZE(A,1))  ! The value to be inserted.
INTEGER :: I    ! Index variable.
INTEGER :: IABOVE ! Index to find insertion point.
INTEGER :: IMIN ! Temporary minimum position.
INTEGER :: ITEMP ! Temporary for swapping.

IF (ISTOP .EQ. ISTART) THEN
   RETURN
END IF

! Find the smallest and put it at the top as a "guard" so there is
! no need for the DO WHILE to check if it is going past the top.
AMIN(:) = A(:,ISTART)
IMIN = ISTART

DO I=ISTOP,ISTART+1,-1
   IF (SUM(A(:,I)) < SUM(AMIN(:))) THEN
      AMIN(:) = A(:,I)
      IMIN = I
   END IF
END DO

A(:,IMIN) = A(:,ISTART)
A(:,ISTART) = AMIN(:)

ITEMP = IDX(ISTART)
IDX(ISTART) = IDX(IMIN)
IDX(IMIN) = ITEMP

! Insertion sort the rest of the array.
DO I=ISTART+2,ISTOP
   ATEMP(:) = A(:,I)
   ITEMP = IDX(I)
   IABOVE = I - 1
   IF (SUM(ATEMP(:)) < SUM(A(:,IABOVE))) THEN
      A(:,I) = A(:,IABOVE)
      IDX(I) = IDX(IABOVE)
      IABOVE = IABOVE - 1

      ! Stop moving items down when the position for "insertion" is found.
      !
      ! Do not have to check for "falling off" the beginning of the
      ! array since the smallest value is a guard value in the first position.

      DO WHILE (SUM(ATEMP(:)) < SUM(A(:,IABOVE)))
         A(:,IABOVE+1) = A(:,IABOVE)
         IDX(IABOVE+1) = IDX(IABOVE)
         IABOVE = IABOVE - 1
      END DO
   END IF
   A(:,IABOVE+1) = ATEMP(:)
   IDX(IABOVE+1) = ITEMP
END DO
RETURN
END SUBROUTINE INSERTION_DVEC

END SUBROUTINE QSORTC_DVEC
