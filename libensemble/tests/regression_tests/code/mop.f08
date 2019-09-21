
! Lightweight copy of MOP_MOD, with everything not used by genfunc deleted.

! Module and subroutines implementing an adaptive weighting scheme for
! generating uniformly spaced points on the Pareto front for a
! multiobjective optimization problem (MOP).
!
! Last Update : September, 2019 by Tyler Chang
MODULE MOP_MOD_LIGHT

USE REAL_PRECISION, ONLY : R8

! The default scope for MOP_MOD is private.
PRIVATE
! The following KIND parameters are public.
PUBLIC :: R8
! The following datatypes are public.
PUBLIC :: MOP_TYPE
! The following MOP_MOD subroutines are public.
PUBLIC :: MOP_INIT, MOP_LTR, MOP_OPT, MOP_FINALIZE, MOP_GENERATE
! The following checkpointing hyperparameters are public.
PUBLIC :: MOP_CHKPTFILE, MOP_CHKPTUNIT, MOP_DATAFILE, MOP_DATAUNIT
! The MOP_MOD module procedure dummy arguments are public.
PUBLIC :: GPSMADS, GPSSEARCH, LSHEP_FIT, LSHEP_EVAL
! The MOP_MOD interface list is public.
PUBLIC :: MOP_MOD_LOCAL_INT, MOP_MOD_SCALAR_INT, MOP_MOD_SEVAL_INT, &
          MOP_MOD_SFIT_INT
! The following checkpoint recoveries are needed for the command line interface.
PUBLIC :: MOP_CHKPT_RECOVER, MOP_RECOVER_DATA

! Interfaces for external procedures.
INTERFACE 

   ! Scalarized function interface.
   FUNCTION MOP_MOD_SCALAR_INT(C, IERR) RESULT(F)
      USE REAL_PRECISION, ONLY : R8
      REAL(KIND=R8), INTENT(IN) :: C(:)
      INTEGER, INTENT(OUT) :: IERR
      REAL(KIND=R8) :: F
   END FUNCTION MOP_MOD_SCALAR_INT

   ! Local optimization subroutine interface.
   SUBROUTINE MOP_MOD_LOCAL_INT(D, X, LB, UB, OBJ_FUNC, BUDGET, IERR, TOLL)
      USE REAL_PRECISION, ONLY : R8
      INTEGER, INTENT(IN) :: D
      REAL(KIND=R8), INTENT(INOUT) :: X(:)
      REAL(KIND=R8), INTENT(IN) :: LB(:)
      REAL(KIND=R8), INTENT(IN) :: UB(:)
      PROCEDURE(MOP_MOD_SCALAR_INT) :: OBJ_FUNC
      INTEGER, INTENT(IN) :: BUDGET
      INTEGER, INTENT(OUT) :: IERR
      REAL(KIND=R8), INTENT(IN) :: TOLL
   END SUBROUTINE MOP_MOD_LOCAL_INT
   
   ! Surrogate function fitting interface.
   SUBROUTINE MOP_MOD_SFIT_INT(D, P, N, X_VALS, Y_VALS, IERR)
      USE REAL_PRECISION, ONLY : R8
      ! Parameters.
      INTEGER, INTENT(IN) :: D
      INTEGER, INTENT(IN) :: P
      INTEGER, INTENT(IN) :: N
      REAL(KIND=R8), INTENT(IN) :: X_VALS(:,:)
      REAL(KIND=R8), INTENT(IN) :: Y_VALS(:,:)
      INTEGER, INTENT(OUT) :: IERR
   END SUBROUTINE MOP_MOD_SFIT_INT

   ! Surrogate function evaluation interface.
   SUBROUTINE MOP_MOD_SEVAL_INT(C, V, IERR)
      USE REAL_PRECISION, ONLY : R8
      ! Parameters.
      REAL(KIND=R8), INTENT(IN) :: C(:)
      REAL(KIND=R8), INTENT(OUT) :: V(:)
      INTEGER, INTENT(OUT) :: IERR
   END SUBROUTINE MOP_MOD_SEVAL_INT
END INTERFACE

! Derived data type for multiobjective optimization problems.
TYPE MOP_TYPE
   ! The SEQUENCE keyword forces the compiler to store MOP_TYPE in contiguous
   ! memory and is needed for compatibility with IBM compilers.
   SEQUENCE
   ! External users cannot directly access the internal contents of a MOP_TYPE.
   ! A MOP_TYPE object is meant to be used as a black box for passing
   ! iteration data between calls to MOP_INIT, MOP_SOLVE, MOP_LTR, and
   ! MOP_FINALIZE.
   PRIVATE

   ! Contents of the MOP_TYPE data object.
   INTEGER :: D, P ! Problem dimensions.
   INTEGER :: LCLIST ! Length of CLIST(:,:) array.
   INTEGER :: ITERATE ! Total number of iterations elapsed.
   INTEGER :: LOPT_BUDGET ! Budget for the local optimizer.
   REAL(KIND=R8) :: DECAY ! Rate of decay for trust region.
   REAL(KIND=R8) :: EPS ! Working precision for the problem.
   REAL(KIND=R8) :: DES_TOLL ! Design point tollerance.
   REAL(KIND=R8) :: OBJ_TOLL ! Objective point tollerance.
   REAL(KIND=R8) :: MIN_RAD ! Trust region tollerance.
   REAL(KIND=R8) :: TRUST_RAD ! Initial trust region radius.
   REAL(KIND=R8), ALLOCATABLE :: CLIST(:,:) ! Previously used LTR centers.
   REAL(KIND=R8), ALLOCATABLE :: LB(:), UB(:) ! Bound constraints.
   REAL(KIND=R8), ALLOCATABLE :: WEIGHTS(:,:) ! Adaptive weights.

   ! Pointers to procedures that are called by the MOP_OPT subroutine.

   ! Subroutine to perform local optimization over surrogate models.
   PROCEDURE(MOP_MOD_LOCAL_INT), NOPASS, POINTER :: LOCAL_OPT
   ! Subroutine to fit surrogate models.
   PROCEDURE(MOP_MOD_SFIT_INT), NOPASS, POINTER :: FIT_SURROGATES
   ! Subroutine to evaluate surrogate models.
   PROCEDURE(MOP_MOD_SEVAL_INT), NOPASS, POINTER :: EVAL_SURROGATES
END TYPE MOP_TYPE

! Public checkpointing variables.
INTEGER :: MOP_CHKPTUNIT = 10 ! Iteration information unit.
INTEGER :: MOP_DATAUNIT = 11 ! Database unit.
CHARACTER(LEN=20) :: MOP_CHKPTFILE = "mop.chkpt" ! Iteration information file.
CHARACTER(LEN=20) :: MOP_DATAFILE = "mop.dat" ! Database file.

! Module variables.
INTEGER :: MOP_MOD_ICHKPT ! Global copy of the checkpointing status.

! Dynamic module arrays.
REAL(KIND=R8), ALLOCATABLE :: MOP_MOD_WEIGHTS(:) ! Scalarization weights.

! Pointer to module procedures for the objective function and its surrogate.
PROCEDURE(MOP_MOD_SEVAL_INT), POINTER :: MOP_MOD_SURROGATES

! LSHEP surrogate model hyperparameters.
INTEGER :: LSHEP_N, LSHEP_D, LSHEP_P
REAL(KIND=R8), ALLOCATABLE :: LSHEP_A(:,:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_RW(:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_XVALS(:,:)
REAL(KIND=R8), ALLOCATABLE :: LSHEP_FVALS(:,:)

CONTAINS

! The following public subroutines are referenced by the driver
! program, and could be used individually by an advanced user, for a
! return-to-caller interface.

SUBROUTINE MOP_INIT( MOP, D, P, LB, UB, IERR, LOPT_BUDGET, DECAY, EPS, &
                     DES_TOLL, OBJ_TOLL, MIN_RAD, TRUST_RAD,           &
                     LOCAL_OPT, FIT_SURROGATES, EVAL_SURROGATES, ICHKPT )
! This subroutine initializes a MOP object for tracking the adaptive
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
! LB(1:D) is the real valued vector of lower bound constraints for the
!    D design variables.
!
! UB(1:D) is the real valued vector of upper bound constraints for the
!    D design variables.
!
!
! On output:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem.
!
! IERR is an integer valued error flag.
!
! Hundreds digit:
!  000 : Normal output. Successful initialization of MOP object.
!
!  1xx : Errors detected.
!   Tens digit:
!     11x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         110 : D (design dimension) must be a positive integer.
!         111 : P (objective dimension) must be at least two.
!         112 : The lead dimension of LB(:) must match D.
!         113 : The lead dimension of UB(:) must match D.
!         114 : LB(:) must be elementwise strictly less than UB(:).
!     12x : The optional dummy arguments contained illegal values.
!       Ones digit:
!         123 : LOPT_BUDGET must be positive.
!         124 : DECAY must be in the range (EPS, 1-EPS).
!         125 : TRUST_RAD must be larger than MIN_RAD.
!     13x : A memory allocation error has occured.
!       Ones digit:
!         130 : A memory allocation error occured.
!         131 : A memory deallocation error occured.
!
!  9xx : A checkpointing error has occurred.
!    901 : WARNING the MOP object was successfully recovered from the
!          checkpoint but does not match the input data.
!    91x : The MOP passed to the checkpoint was invalid.
!    92x : Error creating the checkpoint file.
!    93x : Error writing iteration information to the checkpoint file.
!    95x : Error reading data from the checkpoint file.
!    96x : A memory management error occured during recovery.
!    970 : A sanity check failed during recovery, indicating that the
!          checkpoint file may have been corrupted.
!
!
! Optional input arguments.
!
! LOPT_BUDGET is an integer valued input, which specifies the budget for the
!    local optimization subroutine. The default value for LOPT_BUDGET is
!    5000 surrogate evaluations.
!
! EPS is real valued input, which specifies the working precision of the
!    machine. The default value for EPS is SQRT(EPSILON), where EPSILON
!    is the unit roundoff. Note, any value supplied that is smaller than
!    the default value will be ignored.
!
! DES_TOLL is the tollerance for the design space. A design points that
!    is within DES_TOLL of another design point will not be evaluated.
!    The default value for DES_TOLL is the square-root of EPS.
!
! OBJ_TOLL is the tollerance for the objective space. An objective point
!    that is within OBJ_TOLL of being dominated by another objective point
!    will be treated as such. The default value of OBJ_TOLL is the
!    square-root of EPS.
!
! MIN_RAD is the smallest trust region radius that is allowed, before
!    an isolated point is abandoned. MIN_RAD must be significantly larger
!    than EPS. By default, MIN_RAD = 10% TRUST_RAD. If MIN_RAD is set
!    to less than DES_TOLL, then MIN_RAD is ignored.
!    
! TRUST_RAD is the initial trust region radius. By default, TRUST_RAD is
!    0.25 * MAXVAL(UB(:) - LB(:)). TRUST_RAD must be positive and cannot
!    be less than MIN_RAD.
!
! LOCAL_OPT is a SUBROUTINE, whose interface matches MOP_MOD_LOCAL_INT.
!    LOCAL_OPT is used to optimize the surrogate model. The default value
!    for LOCAL_OPT = GPSMADS, a lightweight Fortran implementation of the
!    GPS MADS algorithm from NOMAD (ACM TOMS Alg. 909).
!
! FIT_SURROGATES is a module subroutine that fits P surrogate models, by
!    setting variables in its module. The interface for FIT_SURROGATES
!    must match MOP_MOD_SFIT_INT. By default, FIT_SURROGATES = LSHEP_FIT.
!
! EVAL_SURROGATES is a module subroutine that evaluates the P surrogate
!    models fit by FIT_SURROGATES. The interface for EVAL_SURROGATES must
!    match MOP_MOD_SEVAL_INT. By default, EVAL_SURROGATES = LSHEP_EVAL.
!
! ICHKPT is an integer that specifies the checkpointing status. The
!    checkpoint file and checkpoint unit are "mop.chkpt" and 10 by
!    default, but can be adjusted by setting the module variables
!    MOP_CHKPTFILE and MOP_CHKPTUNIT. Possible values are:
!
!    ICHKPT = 0 : No checkpointing (default setting).
!    ICHKPT < 0 : Recover from the last checkpoint.
!    ICHKPT > 0 : Begin a new checkpoint file.
!
! In recovery mode, then the inputs D, P, LB, and UB are still referenced
!    (for sanity checks). Also, the optional procedure arguments are still
!    needed to recover algorithmic modifications. No other optional
!    arguments are referenced.
!
IMPLICIT NONE
! Input parameters.
INTEGER, INTENT(IN) :: D ! Dimension of design space.
INTEGER, INTENT(IN) :: P ! Dimension of objective space.
REAL(KIND=R8), INTENT(IN) :: LB(:) ! Lower bound constraints.
REAL(KIND=R8), INTENT(IN) :: UB(:) ! Upper bound constraints.
! Output parameters.
TYPE(MOP_TYPE), INTENT(OUT) :: MOP ! Data structure containing problem info.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Optional parameters.
INTEGER, OPTIONAL, INTENT(IN) :: LOPT_BUDGET ! Local optimizer budget.
INTEGER, OPTIONAL, INTENT(IN) :: ICHKPT ! Checkpointing mode.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: DECAY ! Decay rate for LTR.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: EPS ! Working precision.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: DES_TOLL ! Design space tollerance.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: OBJ_TOLL ! Objective space tollerance.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: MIN_RAD ! Minimum LTR radius.
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: TRUST_RAD ! Default LTR radius.
! Optional procedure arguments.
PROCEDURE(MOP_MOD_LOCAL_INT), OPTIONAL :: LOCAL_OPT
PROCEDURE(MOP_MOD_SFIT_INT), OPTIONAL :: FIT_SURROGATES
PROCEDURE(MOP_MOD_SEVAL_INT), OPTIONAL :: EVAL_SURROGATES
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

! If in checkpoint recovery mode, read the MOP object in.
IF (PRESENT(ICHKPT)) THEN
   ! Checkpoint recovery mode.
   IF (ICHKPT < 0) THEN
      ! Load problem status form last checkpoint.
      CALL MOP_CHKPT_RECOVER(MOP, IERR)
      IF (IERR .NE. 0) RETURN
      ! Perform the final sanity check.
      IF ( (MOP%D .NE. D) .OR. (MOP%P .NE. P) .OR. &
           ANY(MOP%LB(:) .NE. LB(:)) .OR. ANY(MOP%UB(:) .NE. UB(:)) ) THEN
         IERR = 901; END IF
      ! Set the procedure arguments, which cannot be recovered from the
      ! checkpoint file.
      MOP%LOCAL_OPT => GPSMADS ! Default optimizer is GPSMADS.
      IF(PRESENT(LOCAL_OPT)) THEN
         MOP%LOCAL_OPT => LOCAL_OPT
      END IF
      MOP%FIT_SURROGATES => LSHEP_FIT ! Default fit is LSHEP_FIT.
      IF(PRESENT(FIT_SURROGATES)) THEN
         IF (.NOT. PRESENT(EVAL_SURROGATES)) THEN
            IERR = 123; RETURN; END IF
         MOP%FIT_SURROGATES => FIT_SURROGATES
      END IF
      MOP%EVAL_SURROGATES => LSHEP_EVAL ! Default evaluation is LSHEP_EVAL.
      IF(PRESENT(EVAL_SURROGATES)) THEN
         IF (.NOT. PRESENT(FIT_SURROGATES)) THEN
            IERR = 123; RETURN; END IF
         MOP%EVAL_SURROGATES => EVAL_SURROGATES
      END IF
      ! The MOP has been initialized, return the initialized MOP object.
      RETURN
   END IF
END IF

! Initialize the MOP structure to maintain the status of the problem.
MOP%D = D
MOP%P = P
MOP%LCLIST = 20
ALLOCATE(MOP%LB(D), MOP%UB(D), MOP%CLIST(D+1,MOP%LCLIST), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 130; RETURN; END IF
MOP%ITERATE = 0
MOP%LB(:) = LB(:)
MOP%UB(:) = UB(:)

! Check for optional inputs.
! Initialize the local optimization budget.
MOP%LOPT_BUDGET = 5000
IF(PRESENT(LOPT_BUDGET)) THEN
   ! The budget must be at least 1, return an error for nonpositive values.
   IF(LOPT_BUDGET < 1) THEN
      IERR = 123; RETURN; END IF
   MOP%LOPT_BUDGET = LOPT_BUDGET
END IF
! Initialize the working precision.
MOP%EPS = SQRT(EPSILON(0.0_R8))
IF(PRESENT(EPS)) THEN
   ! The default value of EPS cannot be decreased. Simply ignore such inputs.
   IF(EPS > MOP%EPS) MOP%EPS = EPS
END IF
! Initialize the design space and objective space tollerances.
MOP%DES_TOLL = SQRT(MOP%EPS)
IF (PRESENT(DES_TOLL)) THEN
   IF(DES_TOLL > MOP%EPS) THEN
      MOP%DES_TOLL = DES_TOLL
   ELSE
      MOP%DES_TOLL = MOP%EPS
   END IF
END IF
MOP%OBJ_TOLL = SQRT(MOP%EPS)
IF (PRESENT(OBJ_TOLL)) THEN
   IF(OBJ_TOLL > MOP%EPS) THEN
      MOP%OBJ_TOLL = OBJ_TOLL
   ELSE
      MOP%OBJ_TOLL = MOP%EPS
   END IF
END IF
! Initialize the decay rate.
MOP%DECAY = 0.5_R8
IF(PRESENT(DECAY)) THEN
   ! The decay rate must be between 0 and 1, up to the working precision EPS.
   IF(DECAY > 1.0_R8 - MOP%EPS .OR. DECAY < MOP%EPS) THEN 
      IERR = 124; RETURN; END IF
   MOP%DECAY = DECAY
END IF
! Initialize trust region radius and tollerance.
MOP%TRUST_RAD = MAXVAL(UB(:) - LB(:)) / 4.0_R8
IF(PRESENT(TRUST_RAD)) THEN
   ! The trust region radius must be greater than the design space tollerance.
   IF(TRUST_RAD .GE. MOP%DES_TOLL) THEN
      MOP%TRUST_RAD = TRUST_RAD
   ELSE
      MOP%TRUST_RAD = MOP%DES_TOLL
   END IF
END IF
! The tollerance for the trust region radius must be greater than the
! design space tollerance. By default, tollerate decay down to 10% of the
! initial trust region radius.
MOP%MIN_RAD = MAX(0.1_R8 * MOP%TRUST_RAD, MOP%DES_TOLL)
IF(PRESENT(MIN_RAD)) THEN
   MOP%MIN_RAD = MAX(MIN_RAD, MOP%DES_TOLL); END IF
! Check that the size of the trust region radius is appropriate.
IF(MOP%MIN_RAD > MOP%TRUST_RAD) THEN
   IERR = 125; RETURN; END IF
! Lower bounds must be elementwise strictly less than upper bounds.
! Use the design space tollerance to check.
IF(ANY(LB(:) .GE. UB(:) - MOP%DES_TOLL)) THEN
   IERR = 114; RETURN; END IF

! Set the procedure arguments.
MOP%LOCAL_OPT => GPSMADS ! Default optimizer is GPSMADS.
IF(PRESENT(LOCAL_OPT)) THEN
   MOP%LOCAL_OPT => LOCAL_OPT; END IF
MOP%FIT_SURROGATES => LSHEP_FIT ! Default fit is LSHEP_FIT.
IF(PRESENT(FIT_SURROGATES)) THEN
   IF (.NOT. PRESENT(EVAL_SURROGATES)) THEN
      IERR = 129; RETURN; END IF
   MOP%FIT_SURROGATES => FIT_SURROGATES
END IF
MOP%EVAL_SURROGATES => LSHEP_EVAL ! Default evaluation is LSHEP_EVAL.
IF(PRESENT(EVAL_SURROGATES)) THEN
   IF (.NOT. PRESENT(FIT_SURROGATES)) THEN
      IERR = 129; RETURN; END IF
   MOP%EVAL_SURROGATES => EVAL_SURROGATES
END IF

! If ICHKPT > 0, initialize the checkpoint file.
IF (PRESENT(ICHKPT)) THEN
   ! Check whether checkpointing is enabled.
   IF (ICHKPT > 0) THEN
      ! Initialize the checkpoint file and save the initialized MOP.
      CALL MOP_CHKPT_NEW(MOP, IERR)
      IF (IERR .NE. 0) RETURN
   END IF
END IF
RETURN
END SUBROUTINE MOP_INIT

SUBROUTINE MOP_LTR( MOP, DES_PTS, OBJ_PTS, LTR_LB, LTR_UB, IERR, ICHKPT)
! This subroutine identifies the most isolated point, builds a local
! trust region, and chooses the adaptive weights, as described in
! 
! Deshpande, Shubhangi, Layne T. Watson, and Robert A. Canfield.
! "Multiobjective optimization using an adaptive weighting scheme."
! Optimization Methods and Software 31.1 (2016): 110-133.
! 
! 
! On input:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem. MOP is created using MOP_INIT.
!
! DES_PTS(1:P,1:N) is a real valued matrix of all design points in
!    the feasible design space [LB, UB], stored in column major order.
!    The second dimension of DES_PTS(:,:) (N) is assumed based on the shape
!    and must be at least D+1 to build an accurate surrogate model.
!
! OBJ_PTS(1:P,1:N) is a real valued matrix of objective values corresponding
!    to the design points in DES_PTS(:,:), stored in column major order.
!    I.e., for cost function F, OBJ_PTS(:,I) = F(DES_PTS(:,I)).
!
!
! On output:
!
! LTR_LB(1:D) is a real valued array of lower bound constraints for the
!    local trust region.
!
! LTR_UB(1:D) is a real valued array of upper bound constraints for the
!    local trust region.
!
! IERR is an integer valued error flag.
!
! Hundreds digit:
!  0xx : Normal output.
!    Ones digit:
!      000 : Successfully constructed a new LTR and selected adaptive weights.
!      003 : Maximal accuracy has already been achieved, no isolated points
!            can be further refined for the problem tollerance.
!
!  2xx : Error detected in input, or during MOP_INIT (initialization) code.
!   Tens digit:
!     21x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         210 : The MOP object appears to be uninitialized.
!         211 : The MOP object is initialized, but its dimensions either
!               do not agree or contain illegal values. This is likely the
!               result of an undetected segmentation fault.
!         212 : The lead dimension of LTR_LB(:) must match the design
!               dimension D, stored in MOP.
!         213 : The lead dimension of LTR_UB(:) must match the design
!               dimension D, stored in MOP.
!         214 : The lead dimension of DES_PTS(:,:) must match D.
!         215 : The lead dimension of OBJ_PTS(:,:) must match P.
!         216 : The second dimensions of DES_PTS and OBJ_PTS must match.
!     22x : A memory error occured while managing the dynamic memory in MOP.
!         220 : A memory allocation error while copying the history.
!         221 : A memory deallocation error occured while freeing temp arrays.
!         222 : A memory allocation error while allocating the adaptive weights.
!     23x : A memory error occured while managing the local arrays.
!       Ones digit:
!         230 : A memory allocation error occured.
!         231 : A memory deallocation error occured.
!     240 : Detected a set of unused adaptive weights. This subroutine
!           may have been called out of sequence.
!
!  5xx : Error thrown by DELAUNAYSPARSE.
!    Tens and ones digits carry the exact error code from DELAUNAYSPARSE,
!    passed by the DELAUNAYGRAPH subroutine.
!
!  9xx : A checkpointing error was thrown.
!    91x : The MOP passed to the checkpoint was invalid.
!    93x : Error writing iteration information to the checkpoint file.
!
!
! Optional input arguments.
!
! ICHKPT is an integer that specifies the checkpointing status. The
!    checkpoint file and checkpoint unit are "mop.chkpt" and 10 by
!    default, but can be adjusted by setting the module variables
!    MOP_CHKPTFILE and MOP_CHKPTUNIT. Possible values are:
!
!    0 (default) : No checkpointing.
!    Any other number : Save algorithm iteration data to the checkpoint file
!
USE DELAUNAYGRAPH_MOD, ONLY : DELAUNAYGRAPH
IMPLICIT NONE
! Input parameters.
TYPE(MOP_TYPE), INTENT(INOUT) :: MOP ! Data structure containing problem info.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Table of precomputed design pts.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Table of objective values.
! Output parameters.
REAL(KIND=R8), INTENT(OUT) :: LTR_LB(:) ! LTR lower bound constraints.
REAL(KIND=R8), INTENT(OUT) :: LTR_UB(:) ! LTR upper bound constraints.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Optional parameters.
INTEGER, OPTIONAL, INTENT(IN) :: ICHKPT ! Checkpointing mode.
! Local variables.
LOGICAL :: ACCEPT ! Acceptance condition for new center point.
INTEGER :: D, P, M, N ! Problem dimensions.
INTEGER :: I, J, K ! Loop indexing / temp variables.
INTEGER :: LWORK ! Length of LAPACK work array.
INTEGER :: MAXIND ! The index of the most isolated point.
REAL(KIND=R8) :: BOX(MOP%D+1) ! Center and radius of next trust region.
REAL(KIND=R8) :: MINVAL_P ! Minimum value taken by the Pth objective.
REAL(KIND=R8) :: TMP(MOP%P-1) ! Temporary array of length P-1.
! Local dynamic arrays.
LOGICAL, ALLOCATABLE :: DELGRAPH(:,:) ! Delaunay graph.
REAL(KIND=R8), ALLOCATABLE :: CLIST_TMP(:,:) ! Temp array for expanding CLIST.
REAL(KIND=R8), ALLOCATABLE :: DISCREP(:) ! List of star discrepancies.
REAL(KIND=R8), ALLOCATABLE :: HOMOGENEOUS_PF(:,:) ! Homogeneous Pareto front.
REAL(KIND=R8), ALLOCATABLE :: PARETO_SET(:,:) ! Pareto front.
REAL(KIND=R8), ALLOCATABLE :: EFFICIENT_SET(:,:) ! Efficient point set.
! External BLAS procedures.
REAL(KIND=R8), EXTERNAL :: DNRM2 ! Euclidean distance (BLAS).

! Retrieve problem dimensions from input data.
D = MOP%D; P = MOP%P; N = SIZE(DES_PTS,2)
! Check for illegal problem dimensions.
IF ( (.NOT. ALLOCATED(MOP%LB)) .OR. (.NOT. ALLOCATED(MOP%UB)) &
     .OR. (.NOT. ALLOCATED(MOP%CLIST)) ) THEN
   IERR = 210; RETURN; END IF
IF ((D < 1) .OR. (P < 2) .OR. (SIZE(MOP%LB,1) .NE. D) .OR. &
    (SIZE(MOP%UB,1) .NE. D)) THEN
   IERR = 211; RETURN; END IF
IF (SIZE(LTR_LB,1) .NE. D) THEN ! Lower bounds dimension does not match D.
   IERR = 212; RETURN; END IF
IF (SIZE(LTR_UB,1) .NE. D) THEN ! Upper bounds dimension does not match D.
   IERR = 213; RETURN; END IF
! If the adaptive weights array is already allocated, then MOP_OPT has not
! been called and the previous iteration must not be complete.
IF (ALLOCATED(MOP%WEIGHTS)) THEN
   IERR = 240; RETURN; END IF

! In the zeroeth iteration, use static weights on the entire design space.
IF (MOP%ITERATE .EQ. 0) THEN
   ! Set the LTR bounds.
   LTR_LB(:) = MOP%LB(:)
   LTR_UB(:) = MOP%UB(:)
   ! Allocate the initial weights.
   ALLOCATE(MOP%WEIGHTS(P,P+1), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 232; RETURN; END IF
   ! Set the individual weights.
   DO I = 1, P
      ! Set zero-valued weights to MOP%EPS to avoid pathological cases.
      MOP%WEIGHTS(:,I) = MOP%EPS
      MOP%WEIGHTS(I,I) = 1.0_R8 - MOP%EPS*REAL(P-1,KIND=R8)
   END DO
   ! Set the adaptive weights.
   MOP%WEIGHTS(:,P+1) = 1.0_R8 / REAL(P, KIND=R8)

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
   IF (MOP%ITERATE > MOP%LCLIST) THEN
      ! Allocate the temporary array to copy CLIST.
      ALLOCATE(CLIST_TMP(D+1,MOP%LCLIST), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 220; RETURN; END IF
      CLIST_TMP = MOP%CLIST
      ! Reallocate CLIST to twice its current size.
      DEALLOCATE(MOP%CLIST, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 221; RETURN; END IF
      ALLOCATE(MOP%CLIST(D+1,MOP%LCLIST*2), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 220; RETURN; END IF
      ! Restore values back into CLIST and free the temporary array.
      MOP%CLIST(:,1:MOP%LCLIST) = CLIST_TMP(:,:)
      DEALLOCATE(CLIST_TMP, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 221; RETURN; END IF
      ! Update the size of MOP%LCLIST.
      MOP%LCLIST = MOP%LCLIST * 2
   END IF

   ! Allocate the dynamic arrays for storing both copies of the Pareto front.
   ALLOCATE(PARETO_SET(P,N), EFFICIENT_SET(D,N), HOMOGENEOUS_PF(P-1,N), &
            STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 230; RETURN; END IF
   ! Compute the minimum value obtained by the Pth objective.
   MINVAL_P = MINVAL(OBJ_PTS(P,:)) - 1.0_R8
   ! Get the current Pareto front (in both the objective space and in the
   ! homogeneous coordinates).
   M = 0 ! Count the cardinality of the Pareto front.
   OUTER : DO I = 1, N
      INNER : DO J = 1, N
         ! Skip index I.
         IF (I .EQ. J) CYCLE INNER
         ! Compute the homogeneous coordinates.
         TMP(:) = (OBJ_PTS(1:P-1,I) / (OBJ_PTS(P,I) - MINVAL_P)) &
                  - (OBJ_PTS(1:P-1,J) / (OBJ_PTS(P,J) - MINVAL_P))
         ! Check whether OBJ_PTS(:,I) and OBJ_PTS(:,J) are equal in the
         ! homogeneous coordinate system (up to the working precision).
         IF (DNRM2(P-1, TMP, 1) < MOP%OBJ_TOLL) THEN
            IF ( I > J ) THEN
               CYCLE OUTER ! Only store the first occurence of a duplicate.
            ELSE
               CYCLE INNER ! This is the first occurence of a duplicate.
            END IF
         END IF
         ! Check whether OBJ_PTS(:,J) dominates OBJ_PTS(:,I).
         IF (ALL(OBJ_PTS(:,J) .LE. OBJ_PTS(:,I) + MOP%OBJ_TOLL)) CYCLE OUTER
      END DO INNER
      ! Increment the counter and update both Pareto front arrays.
      M = M + 1
      PARETO_SET(:,M) = OBJ_PTS(:,I)
      EFFICIENT_SET(:,M) = DES_PTS(:,I)
      HOMOGENEOUS_PF(:,M) = OBJ_PTS(1:P-1,I) / (OBJ_PTS(P,I) - MINVAL_P)
   END DO OUTER

   ! Allocate the remaining dynamic arrays.
   ALLOCATE(DELGRAPH(M,M), DISCREP(M), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 230; RETURN; END IF

   ! Generate the Delaunay graph.
   CALL DELAUNAYGRAPH(P-1, M, HOMOGENEOUS_PF(:,1:M), DELGRAPH, IERR, &
                      EPS=MOP%EPS)
   IF (IERR .LT. 10) THEN ! Normal execution.
      IERR = 0
   ELSE ! An irrecoverable error occured.
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
               & DNRM2(P, PARETO_SET(:,J)-PARETO_SET(:,I), 1)
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
      ! Indentify the largest discrepancy, excepting negative/zero values.
      MAXIND = MAXLOC(DISCREP, DIM=1, MASK=(DISCREP > MOP%EPS))
      ! If no values with positive discrepancy remain, terminate.
      IF (MAXIND .EQ. 0) EXIT
      ! Otherwise, set BOX(1:D) to the corresponding design point.
      BOX(1:D) = EFFICIENT_SET(:,MAXIND)
      ! Check whether the current design point has been used before.
      DO I = MOP%ITERATE-1, 1, -1
         IF ( DNRM2(D, MOP%CLIST(1:D,I) - BOX(1:D), 1) < MOP%DES_TOLL ) EXIT
      END DO
      ! A previous entry in MOP%CLIST(:,:) matches.
      IF (I > 0) THEN
         ! Copy the item and decay the trust region.
         BOX = MOP%CLIST(:,I)
         BOX(D+1) = BOX(D+1) * MOP%DECAY
         ! The minimum tolerance has not yet been exceeded.
         IF (BOX(D+1) > MOP%MIN_RAD) THEN
            ! Append BOX(:) to CLIST(:,:).
            MOP%CLIST(:,MOP%ITERATE) = BOX(:)
            ACCEPT = .TRUE.
         ! The minimum tolerance has been exceeded, set the
         ! discrepancy to a negative number.
         ELSE
            DISCREP(MAXIND) = -1.0_R8
         END IF
      ! No match found.
      ELSE
         ! Append BOX(:) to CLIST(:,:).
         BOX(D+1) = MOP%TRUST_RAD
         MOP%CLIST(:,MOP%ITERATE) = BOX(:)
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
      LTR_UB(I) = MIN(BOX(I) + BOX(D+1), MOP%UB(I))
      LTR_LB(I) = MAX(BOX(I) - BOX(D+1), MOP%LB(I))
   END DO

   ! Count the number of Delaunay neighbors of PARETO_SET(:,MAXIND) using
   ! DELGRAPH(:,MAXIND).
   K = 0
   DO I = 1, M
      IF(DELGRAPH(I, MAXIND)) K = K + 1
   END DO
   ! Construct the adaptive weights. First, consider the special case where
   ! there is only one point on the Pareto front.
   IF (K .EQ. 0) THEN
      ! Allocate the adaptive weights.
      ALLOCATE(MOP%WEIGHTS(P,P+1), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 232; RETURN; END IF
      ! Set the individual weights.
      DO I = 1, P
         ! Set zero-valued weights to MOP%EPS to avoid pathological cases.
         MOP%WEIGHTS(:,I) = MOP%EPS
         MOP%WEIGHTS(I,I) = 1.0_R8 - MOP%EPS*REAL(P-1,KIND=R8)
      END DO
      ! Set the adaptive weights.
      MOP%WEIGHTS(:,P+1) = 1.0_R8 / REAL(P, KIND=R8)

   ! In the general case, the Delaunay neighborhood is nonempty.
   ELSE
      ! Allocate the adaptive weights.
      ALLOCATE(MOP%WEIGHTS(P,P+K), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 232; RETURN; END IF
      ! Set the individual weights.
      DO I = 1, P
         ! Set zero-valued weights to MOP%EPS to avoid pathological cases.
         MOP%WEIGHTS(:,I) = MOP%EPS
         MOP%WEIGHTS(I,I) = 1.0_R8 - MOP%EPS*REAL(P-1,KIND=R8)
      END DO
      ! Set the adaptive weights.
      K = 1
      DO I = 1, M
         ! Check that I and MAXIND are Delaunay neighbors.
         IF (DELGRAPH(I,MAXIND)) THEN
            ! Set the adaptive weights.
            MOP%WEIGHTS(:,P+K) = ABS(PARETO_SET(:,I) - PARETO_SET(:,MAXIND))
            ! Invert when greater than or equal to zero.
            DO J = 1, P
               IF (MOP%WEIGHTS(J,P+K) < MOP%EPS) THEN
                  MOP%WEIGHTS(J,P+K) = MOP%EPS
               ELSE
                  MOP%WEIGHTS(J,P+K) = 1.0_R8 / MOP%WEIGHTS(J,P+K)
               END IF
            END DO
            ! Normalize to make MOP%WEIGHTS(:,P+K) convex.
            MOP%WEIGHTS(:,P+K) = MOP%WEIGHTS(:,P+K) / SUM(MOP%WEIGHTS(:,P+K))
            K = K + 1
         END IF
      END DO
   END IF
   ! Free heap memory.
   DEALLOCATE(DELGRAPH, DISCREP, PARETO_SET, HOMOGENEOUS_PF, STAT=IERR)
   IF (IERR .NE. 0) IERR = 231
END IF

! If ICHKPT is present, then save to the checkpoint file.
IF (PRESENT(ICHKPT)) THEN
   ! Check whether checkpointing is enabled.
   IF (ICHKPT .NE. 0) THEN
      ! Save the updated MOP object to the checkpoint file.
      CALL MOP_CHKPT(MOP, IERR)
      IF (IERR .NE. 0) RETURN
   END IF
END IF
RETURN
END SUBROUTINE MOP_LTR

SUBROUTINE MOP_OPT( MOP, LTR_LB, LTR_UB, DES_PTS, OBJ_PTS, CAND_PTS, IERR, &
                    ICHKPT )
! This subroutine fits and optimizes P surrogate models within the LTR,
! over the adaptive weights in the MOP object, as described in
! 
! Deshpande, Shubhangi, Layne T. Watson, and Robert A. Canfield.
! "Multiobjective optimization using an adaptive weighting scheme."
! Optimization Methods and Software 31.1 (2016): 110-133.
! 
! 
! On input:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem. MOP is created using MOP_INIT.
!
! LTR_LB(1:D) is the real valued vector of lower bound constraints for the
!    local trust region.
!
! LTR_UB(1:D) is the real valued vector of upper bound constraints for the
!    local trust region.
!
! DES_PTS(1:P,1:N) is a real valued matrix of all design points in
!    the feasible design space [LB, UB], stored in column major order.
!    The second dimension of DES_PTS(:,:) (N) is assumed based on the shape
!    and must be at least D+1 to build an accurate surrogate model.
!
! OBJ_PTS(1:P,1:N) is a real valued matrix of objective values corresponding
!    to the design points in DES_PTS(:,:), stored in column major order.
!    I.e., for cost function F, OBJ_PTS(:,I) = F(DES_PTS(:,I)).
!
! CAND_PTS(:,:) is an ALLOCATABLE real valued array of rank 2. CAND_PTS need
!    not be allocated on input. If allocated, any contents of CAND_PTS is
!    lost, and CAND_PTS is reallocated on output.
!
!
! On output:
!
! CAND_PTS(1:D,1:M) is a list of candidate design points to be evaluated
!    before the next iteration of the algorithm. M <= L, and CAND_PTS
!    contains no redundant design points.
!
! IERR is an integer valued error flag. IERR=0 signifies a successful iteration.
!
! Hundreds digit:
!  000 : Normal output. Successful iteration, and list CAND_PTS obtained.
!
!  3xx : Errors detected.
!   Tens digit:
!     31x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         310 : D (design dimension) must be a positive integer.
!         311 : P (objective dimension) must be at least two.
!         312 : The lead dimension of LB(:) must match D.
!         313 : The lead dimension of UB(:) must match D.
!         314 : LB(:) must be elementwise strictly less than UB(:).
!         315 : The lead dimension of DES_PTS(:,:) must match D.
!         316 : The lead dimension of OBJ_PTS(:,:) must match P.
!         317 : The second dimensions of DES_PTS and OBJ_PTS must match.
!     32x : An irregularity was detected in the supplied MOP data type.
!       Ones digit:
!         320 : The adaptive weights are not allocated. Check that
!               the last subroutine called was MOP_LTR.
!         321 : The adaptive weights array is allocated, but its lead
!               dimension does not match the number of objectives P. This
!               is most likely the result of an undetected segmentation
!               fault.
!         322 : There are too few adaptive weights. This is most likely the
!               result of an undetected segmentation fault.
!         323 : One of the adaptive weights contains a negative value. This
!               is most likely the result of an undetected segmentation
!               fault.
!     33x : A memory error has occured.
!       Ones digit:
!         330 : A memory allocation error occured in the local memory.
!         331 : A memory deallocation error occured in the local memory.
!         332 : A memory deallocation error occured while freeing the
!               adaptive weights for the next iteration.
!     340 : Too few points were supplied in DES_PTS and OBJ_PTS to construct
!           an accurate surrogate. At least D+1 points are required.
!
!  6xx : Error thrown by FIT_SURROGATES.
!    Tens and ones digits carry the error code from FIT_SURROGATES subroutine.
!
!  7xx : Error thrown by LOCAL_OPT.
!    Tens and ones digits carry the error code from LOCAL_OPT subroutine.
!
!  9xx : A checkpointing error was thrown.
!    91x : The MOP passed to the checkpoint was invalid.
!    93x : Error writing iteration information to the checkpoint file.
!
!
! Optional input arguments.
!
! ICHKPT is an integer that specifies the checkpointing status. The
!    checkpoint file and checkpoint unit are "mop.chkpt" and 10 by
!    default, but can be adjusted by setting the module variables
!    MOP_CHKPTFILE and MOP_CHKPTUNIT. Possible values are:
!
!    0 (default) : No checkpointing.
!    Any other number : Save algorithm iteration data to the checkpoint file
!
IMPLICIT NONE
! Input parameters.
TYPE(MOP_TYPE), INTENT(INOUT) :: MOP ! Data structure containing problem info.
REAL(KIND=R8), INTENT(IN) :: LTR_LB(:) ! Lower bound constraints.
REAL(KIND=R8), INTENT(IN) :: LTR_UB(:) ! Upper bound constraints.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Table of design points.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Table of objective values.
! Output parameters.
REAL(KIND=R8), INTENT(OUT), ALLOCATABLE :: CAND_PTS(:,:) ! Efficient set.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Optional parameters.
INTEGER, OPTIONAL, INTENT(IN) :: ICHKPT ! Checkpointing mode.
! Local variables.
INTEGER :: D, P, L, M, N ! Problem dimensions.
INTEGER :: I, J ! Loop indexing variables.
REAL(KIND=R8) :: MIN_X(MOP%D,SIZE(MOP%WEIGHTS,2)) ! Potential candidate points.
! External BLAS function for computing Euclidean distance.
REAL(KIND=R8), EXTERNAL :: DNRM2

! Get problem dimensions from input data.
D = MOP%D; P = MOP%P; N = SIZE(DES_PTS,2); L = SIZE(MOP%WEIGHTS,2)
! Check for illegal problem dimensions.
IF ((.NOT. ALLOCATED(MOP%LB)) .OR. (.NOT. ALLOCATED(MOP%UB))) THEN
   IERR = 310; RETURN; END IF
IF ((D < 1) .OR. (P < 2) .OR. (SIZE(MOP%LB,1) .NE. D) .OR. &
    (SIZE(MOP%UB,1) .NE. D)) THEN
   IERR = 311; RETURN; END IF
IF (SIZE(LTR_LB,1) .NE. D) THEN ! Lower bound dimension does not match D.
   IERR = 312; RETURN; END IF
IF (SIZE(LTR_UB,1) .NE. D) THEN ! Upper bound dimension does not match D.
   IERR = 313; RETURN; END IF
IF (ANY(LTR_LB .GE. LTR_UB - MOP%DES_TOLL) .OR. &
    ANY(LTR_LB .LT. MOP%LB - MOP%EPS) .OR.      &
    ANY(LTR_UB .GT. MOP%UB + MOP%EPS)) THEN
   IERR = 314; RETURN; END IF
! Check that the dimensions of DES_PTS and OBJ_PTS match.
IF (SIZE(DES_PTS,1) .NE. D) THEN
   IERR = 315; RETURN; END IF
IF (SIZE(OBJ_PTS,1) .NE. P) THEN
   IERR = 316; RETURN; END IF
IF (SIZE(OBJ_PTS,2) .NE. N) THEN
   IERR = 317; RETURN; END IF
! Check the adaptive MOP%WEIGHTS array.
IF(.NOT. ALLOCATED(MOP%WEIGHTS)) THEN
   IERR = 320; RETURN; END IF
IF(SIZE(MOP%WEIGHTS,1) .NE. P) THEN
   IERR = 321; RETURN; END IF
IF(L < P+1) THEN
   IERR = 322; RETURN; END IF
IF(ANY(MOP%WEIGHTS < -MOP%EPS)) THEN
   IERR = 323; RETURN; END IF
! Too few points to fit a surrogate model.
IF (N < D+1) THEN
   IERR = 340; RETURN; END IF

! Fit P surrogate models.
CALL MOP%FIT_SURROGATES(D, P, N, DES_PTS, OBJ_PTS, IERR)
IF (IERR .NE. 0) THEN
   IERR = IERR + 600; RETURN; END IF

! Optimize the surrogate models for all P+1 weightings.
M = 0 ! Track the number of distinct candidate points to return.
LOPT_LOOP : DO I = 1, L
   ! Set the module weights.
   MOP_MOD_WEIGHTS(:) = MOP%WEIGHTS(:,I)
   ! Call the local optimizer.
   MIN_X(:,M+1) = (LTR_LB(:) + LTR_UB(:)) / 2.0_R8
   CALL MOP%LOCAL_OPT(D, MIN_X(:,M+1), LTR_LB(:), LTR_UB(:), &
                      SURROGATE_FUNC, MOP%LOPT_BUDGET, IERR, MOP%DES_TOLL)
   IF (IERR .NE. 0) THEN
      IERR = IERR + 700; RETURN; END IF
   ! If MIN_X(:,M+1) is already in DES_PTS(:,:), don't add to CAND_PTS(:,:).
   DO J = 1, SIZE(DES_PTS(:,:), 2)
      IF (DNRM2(D, DES_PTS(:,J)-MIN_X(:,M+1), 1) < MOP%DES_TOLL) CYCLE LOPT_LOOP
   END DO
   ! If MIN_X(:,J) = MIN_X(:,M+1), for J \leq M, don't add to CAND_PTS(:,:).
   DO J = 1, M
      IF (DNRM2(D, MIN_X(:,J)-MIN_X(:,M+1), 1) < MOP%DES_TOLL) CYCLE LOPT_LOOP
   END DO
   ! Increment the counter.
   M = M + 1
END DO LOPT_LOOP

! Increment the iteration counter.
MOP%ITERATE = MOP%ITERATE + 1
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
DEALLOCATE(MOP%WEIGHTS, STAT=IERR)
IF(IERR .NE. 0) IERR = 332

! If ICHKPT is present, save to the checkpoint file.
IF (PRESENT(ICHKPT)) THEN
   ! Check whether checkpointing is enabled.
   IF (ICHKPT .NE. 0) THEN
      ! Save the updated MOP object to the checkpoint file.
      CALL MOP_CHKPT(MOP, IERR)
      IF (IERR .NE. 0) RETURN
   END IF
END IF
RETURN
END SUBROUTINE MOP_OPT

SUBROUTINE MOP_FINALIZE( MOP, DES_PTS, OBJ_PTS, M, PARETO_F, EFFICIENT_X, IERR )
! This subroutine finalizes a multiobjective optimization problem, by
! computing the entire weakly Pareto set, and freeing all dynamic memory
! allocated to the MOP object.
! 
! 
! On input:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem. MOP is created using MOP_INIT.
!
! DES_PTS(1:D,1:N) is a real valued matrix of all design points in
!    the feasible design space [LB, UB], stored in column major order.
!    The second dimension of DES_PTS(:,:) (N) is assumed based on the shape
!    and must be at least D+1 to build an accurate surrogate model.
!
! OBJ_PTS(1:P,1:N) is a real valued matrix of objective values corresponding
!    to the design points in DES_PTS(:,:), stored in column major order.
!    I.e., for cost function F, OBJ_PTS(:,I) = F(DES_PTS(:,I)).
!
! PARETO_F(:,:) is an ALLOCATABLE real valued array of rank 2. PARETO_F need
!    not be allocated on input. If allocated, any contents of PARETO_F are
!    lost, as PARETO_F is reallocated on output.
!
! EFFICIENT_X(:,:) is an ALLOCATABLE real valued array of rank 2. EFFICIENT_X
!    need not be allocated on input. If allocated, any contents of EFFICIENT_X
!    are lost, as EFFICIENT_X is reallocated on output.
!
!
! On output:
!
! M is the cardinality of the weakly Pareto set.
!
! EFFICIENT_X(1:D,1:M) contains the entire weakly efficient set, stored in
!    column major ordering, with corresponding objective values in PARETO_F.
!
! PARETO_F(1:P,1:M) contains the entire weakly Pareto set. Note, PARETO_F may
!    contain duplicate values since the entire weakly Pareto set is returned.
!
! IERR is an integer valued error flag.
!
! Hundreds digit:
!  000 : Normal output. Successful iteration, and list CAND_PTS obtained.
!
!  4xx : Errors detected.
!   Tens digit:
!     41x : The input parameters contained illegal dimensions or values.
!       Ones digit:
!         410 : The MOP object contains illegal problem dimensions. This
!               is most likely the result of an undetected segmentation
!               fault.
!         411 : The lead dimension of DES_PTS(:,:) must match D.
!         412 : The lead dimension of OBJ_PTS(:,:) must match P.
!         413 : The second dimensions of DES_PTS and OBJ_PTS must match.
!     42x : A memory error occured while managing the output arrays.
!       Ones digit:
!         420 : A memory allocation error occured while allocating an
!               output array PARETO_F and EFFICIENT_X.
!         421 : A memory deallocation error occured while freeing an
!               output array PARETO_F or EFFICIENT_X, which was already
!               allocated on input.
!     43x : A memory error has occured while managing internal memory.
!       Ones digit:
!         130 : A memory allocation error occured in the local memory.
!         131 : A memory deallocation error occured in the local memory.
!     441 : A memory error has occured while freeing the MOP object or
!           module memory. The output arrays EFFICIENT_X and PARETO_F
!           should be unaffected.
!
IMPLICIT NONE
! Input parameters.
TYPE(MOP_TYPE), INTENT(INOUT) :: MOP ! Data structure containing problem info.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Table of precomputed design pts.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Table of objective values.
! Output parameters.
INTEGER, INTENT(OUT) :: M ! Cardinality of the weakly Pareto set.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: PARETO_F(:,:)
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: EFFICIENT_X(:,:)
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Local variables.
INTEGER :: D, P, N ! Problem dimensions.
INTEGER :: I, J ! Loop indexing variables.
! Local dynamic arrays.
REAL(KIND=R8), ALLOCATABLE :: PARETO_SET(:,:) ! Pareto front.
REAL(KIND=R8), ALLOCATABLE :: EFFICIENT_SET(:,:) ! Efficient set.
! External BLAS procedures.
REAL(KIND=R8), EXTERNAL :: DNRM2 ! Euclidean distance (BLAS).

! Get problem dimensions from input data.
D = MOP%D; P = MOP%P; N = SIZE(DES_PTS,2)
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
ALLOCATE(PARETO_SET(P,N), EFFICIENT_SET(D,N), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 430; RETURN; END IF

! Get the weakly Pareto and efficient sets.
M = 0 ! Count the cardinality of the Pareto set.
OUTER : DO I = 1, N
   INNER : DO J = 1, N
      ! Skip this point.
      IF (I .EQ. J) CYCLE
      ! Check whether OBJ_PTS(:,I) and OBJ_PTS(:,J) are equal.
      IF (DNRM2(P, OBJ_PTS(:,I) - OBJ_PTS(:,J), 1) < MOP%OBJ_TOLL) THEN
         CYCLE INNER ! Consider all weakly Pareto points.
      END IF
      ! Check whether OBJ_PTS(:,J) dominates OBJ_PTS(:,I).
      IF (ALL(OBJ_PTS(:,J) .LE. OBJ_PTS(:,I) + MOP%OBJ_TOLL)) CYCLE OUTER
   END DO INNER
   ! Increment the counter and update both Pareto and efficient set.
   M = M + 1
   PARETO_SET(:,M) = OBJ_PTS(:,I)
   EFFICIENT_SET(:,M) = DES_PTS(:,I)
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
! Free the MOP data structure's memory and the MOP_MOD_WEIGHTS array, which
! does not contain any useful information on output.
IF (ALLOCATED(MOP_MOD_WEIGHTS)) THEN
   DEALLOCATE(MOP_MOD_WEIGHTS, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(MOP%WEIGHTS)) THEN
   DEALLOCATE(MOP%WEIGHTS, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(MOP%LB)) THEN
   DEALLOCATE(MOP%LB, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(MOP%UB)) THEN
   DEALLOCATE(MOP%UB, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
IF (ALLOCATED(MOP%CLIST)) THEN
   DEALLOCATE(MOP%CLIST, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 441; RETURN; END IF
END IF
RETURN
END SUBROUTINE MOP_FINALIZE

SUBROUTINE MOP_GENERATE(MOP, DES_PTS, OBJ_PTS, LBATCH, BATCHX, IERR, &
                        NB, ICHKPT)
! Generator function for generating static batches of candidate points
! using MOP_MOD.
!
!
! On input:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem. MOP is created using MOP_INIT.
!
! DES_PTS(D,N) is the current list of design points.
!
! OBJ_PTS(P,N) is the current list of objective points.
!
!
! On output:
!
! LBATCH returns the size of the next batch. If NB is present, then when
!    possible, LBATCH is a multiple of the input NB.
!
! BATCHX(D,LBATCH) is the next batch of design points to evaluate.
!
! IERR is an integer-valued error flag. Error codes carried from worker
!    subroutines. In general:
!
!  000 : successful iteration.
!  003 : stopping criterion 3 achieved.
!
!  1xx : illegal input error.
!   Tens digit:
!    11x : The MOP object contained illegal dimensions or values, and
!          may not have been properly initialized.
!       Ones digit:
!         110 : The problem dimensions D and P are not legal.
!         111 : The internal arrays have not been allocated.
!         112 : The internal arrays have been initialized, but to invalid
!               dimensions.
!         113 : The lower and upper bound constraints contain illegal values,
!               subject to the design space tollerance.
!    12x : Illegal values in other inputs.
!         120 : The lead dimension of DES_PTS does not match the data in MOP.
!         121 : The lead dimension of OBJ_PTS does not match the data in MOP.
!         122 : The second dimensions of DES_PTS and OBJ_PTS do not match.
!         123 : The preferred batch size NB must be greater than zero.
!
!    2xx, 3xx : Error in MOP_LTR or MOP_OPT, respectively.
!
!    5xx, 6xx, 7xx, 8xx : Error in one of MOP_OPT's dependencies.
!
!    9xx : Checkpointing error.
!
!
! Optional input arguments.
!
! NB is the preferred batch size. When possible, LBATCH will be a multiple
!    of NB.
!
! ICHKPT is an integer that specifies the checkpointing status. The
!    checkpoint file and checkpoint unit are "mop.chkpt" and 10 by
!    default, but can be adjusted by setting the module variables
!    MOP_CHKPTFILE and MOP_CHKPTUNIT. Possible values are:
!
!    0 (default) : No checkpointing.
!    Any other number : Save algorithm iteration data to the checkpoint file
!
IMPLICIT NONE
! Parameter list.
TYPE(MOP_TYPE), INTENT(INOUT) :: MOP ! MOP object.
REAL(KIND=R8), INTENT(IN) :: DES_PTS(:,:) ! Design point database.
REAL(KIND=R8), INTENT(IN) :: OBJ_PTS(:,:) ! Objective point database.
INTEGER, INTENT(OUT) :: LBATCH ! Requested batch size.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: BATCHX(:,:) ! Batch of requested pts.
INTEGER, INTENT(OUT) :: IERR ! Integer valued error flag.
! Optional inputs.
INTEGER, OPTIONAL, INTENT(IN) :: NB ! Preferred batch size.
INTEGER, OPTIONAL, INTENT(IN) :: ICHKPT ! Checkpointing mode.
! Local variables.
INTEGER :: D, P ! Problem dimensions.
INTEGER :: I ! Loop indexing variable.
INTEGER :: ICHKPTL ! Local copy of ICHKPT.
REAL(KIND=R8) :: LTR_LB(SIZE(MOP%LB,1)) ! Local trust region lower bound.
REAL(KIND=R8) :: LTR_UB(SIZE(MOP%UB,1)) ! Local trust region upper bound.

! Get problem dimensions.
D = MOP%D; P = MOP%P
! Check for illegal input parameter values.
IF ( D < 1 .OR. P < 2 ) THEN
   IERR = 110; RETURN; END IF
IF ( .NOT. ( ALLOCATED(MOP%LB) .AND. ALLOCATED(MOP%UB) .AND. &
             ALLOCATED(MOP%CLIST) ) ) THEN
   IERR = 111; RETURN; END IF
IF ( SIZE(MOP%LB,1) .NE. D .OR. SIZE(MOP%UB,1) .NE. D .OR. &
     SIZE(MOP%CLIST,1) .NE. D+1)  THEN
   IERR = 112; RETURN; END IF
IF ( ANY(MOP%LB(:) .GE. MOP%UB(:) - MOP%DES_TOLL) ) THEN
   IERR = 113; RETURN; END IF
! Load optional input argument.
ICHKPTL = 0
IF (PRESENT(ICHKPT)) THEN
   ICHKPTL = ICHKPT; END IF
IF (PRESENT(NB)) THEN
   IF (NB < 1) THEN
      IERR = 123; RETURN; END IF
END IF

! Perform a half-iteration and request the next batch, based on the state of
! the MOP object.

! There are two cases.
IF (.NOT. ALLOCATED(MOP%WEIGHTS)) THEN
   ! Execute the search phase.

   ! Choose the most isolated point(s), and construct the LTR(s).
   CALL MOP_LTR( MOP, DES_PTS, OBJ_PTS, LTR_LB, LTR_UB, IERR, ICHKPT=ICHKPTL )
   IF (IERR .NE. 0) RETURN

   ! Get the batch size, using a special rule for the first iteration
   ! and respecting the preferred batch size NB.
   IF (MOP%ITERATE .EQ. 0) THEN
      ! In first iteration, allow for 2 iterations of GPSSEARCH.
      LBATCH = (2*D+1)**2
      ! Correct for NB, when present.
      IF (PRESENT(NB)) THEN
         LBATCH = LBATCH + MOD(NB-MOD(LBATCH,NB),NB); END IF
   ELSE
      ! In later iterations, allow for just 1 iteration of GPSSEARCH.
      LBATCH = 2*D+1
      ! Correct for NB, when present.
      IF (PRESENT(NB)) THEN
         LBATCH = LBATCH + MOD(NB-MOD(LBATCH,NB),NB); END IF
   END IF
   ! Request a static exploration of the LTR.
   CALL GPSSEARCH(D, P, LTR_LB, LTR_UB, LBATCH, .FALSE., BATCHX, IERR, &
                  MOP%DES_TOLL)
   IF (IERR .NE. 0) THEN
      IERR = IERR + 800; RETURN; END IF

ELSE
   ! Execute the optimization phase.

   ! Recover the last LTR.
   IF (MOP%ITERATE .EQ. 0) THEN
      LTR_LB(:) = MOP%LB(:)
      LTR_UB(:) = MOP%UB(:)
   ELSE
      DO I = 1, D
         LTR_UB(I) = MIN( MOP%CLIST(I,MOP%ITERATE) +  &
                          MOP%CLIST(D+1,MOP%ITERATE), MOP%UB(I) )
         LTR_LB(I) = MAX( MOP%CLIST(I,MOP%ITERATE) -  &
                          MOP%CLIST(D+1,MOP%ITERATE), MOP%LB(I) )
      END DO
   END IF

   ! Optimize the surrogates in the LTR.
   CALL MOP_OPT( MOP, LTR_LB, LTR_UB, DES_PTS, OBJ_PTS, BATCHX, IERR, &
                 ICHKPT=ICHKPTL )
   IF (IERR .NE. 0) RETURN

   ! Get the size of BATCHX.
   LBATCH = SIZE(BATCHX, 2)
END IF
RETURN
END SUBROUTINE MOP_GENERATE

! The following module procedures are used for checkpointing. The
! checkpointing feature is primarily for restoring a crashed or prematurely
! terminated instance of a MOP_TYPE object. Another method should be used
! for checkpointing objective function data.

SUBROUTINE MOP_CHKPT_NEW(MOP, IERR)
! Create a new checkpoint file for a given MOP.
!
!
! On input:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem.
!
!
! On output:
!
! IERR is an integer valued error flag.
!
!  000 : Normal output. Successful initialization of a new MOP checkpoint.
!
!  9xx : Errors detected.
!     91x : The input parameters contained illegal dimensions or values.
!         910 : MOP does not appear to have been properly initialized.
!         911 : MOP has been initialized, but appears to contain corrupted
!               or inconsistent data.
!     92x : A file I/O error was detected.
!         920 : An error occured while opening the checkpoint file.
!         921 : An error occured while writing data to the checkpoint file.
!         922 : An error occured while closing the checkpoint file.
!
IMPLICIT NONE
! Input parameters.
TYPE(MOP_TYPE), INTENT(IN) :: MOP ! Data structure containing problem info.
! Output parameters.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.

! Check for uninitialized values.
IF ( (.NOT. ALLOCATED(MOP%LB)) .OR. (.NOT. ALLOCATED(MOP%UB)) ) THEN
   IERR = 910; RETURN; END IF
! Check for illegal/mismatched values.
IF ( MOP%D < 1 .OR. MOP%P < 2) THEN
   IERR = 911; RETURN; END IF
IF (SIZE(MOP%LB, 1) .NE. MOP%D .OR. SIZE(MOP%UB) .NE. MOP%D) THEN
   IERR = 911; RETURN; END IF

! Open the checkpoint file, using unformatted write.
OPEN(MOP_CHKPTUNIT, FILE=MOP_CHKPTFILE, FORM="unformatted", &
     ACTION="write", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 920; RETURN; END IF
! Write unformatted MOP metadata to the checkpoint file.
WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%D, MOP%P
IF (IERR .NE. 0) THEN
   IERR = 921; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%LOPT_BUDGET, MOP%DECAY, & 
                     MOP%EPS, MOP%DES_TOLL, MOP%OBJ_TOLL,     &
                     MOP%MIN_RAD, MOP%TRUST_RAD
IF (IERR .NE. 0) THEN
   IERR = 921; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%LB(1:MOP%D), MOP%UB(1:MOP%D)
IF (IERR .NE. 0) THEN
   IERR = 921; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Close the checkpoint file.
CLOSE(MOP_CHKPTUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 922; RETURN; END IF

RETURN
END SUBROUTINE MOP_CHKPT_NEW

SUBROUTINE MOP_CHKPT(MOP, IERR)
! Save MOP iteration data to an existing checkpoint file.
!
!
! On input:
!
! MOP is a derived data type of TYPE(MOP_TYPE), which carries meta data
!    about the multiobjective problem.
!
!
! On output:
!
! IERR is an integer valued error flag.
!
!  000 : Normal output. Iteration data saved to the checkpoint file.
!
!  9xx : Errors detected.
!     91x : The input parameters contained illegal dimensions or values.
!         910 : MOP does not appear to have been properly initialized.
!         911 : MOP has been initialized, but appears to contain corrupted
!               or inconsistent data.
!     93x : A file I/O error was detected.
!         930 : The checkpoint file could not be opened, check whether
!               CHKPTFILE has been properly initialized.
!         931 : An error occured while writing data to the checkpoint file.
!         932 : An error occured while closing the checkpoint file.
!
IMPLICIT NONE
! Input parameters.
TYPE(MOP_TYPE), INTENT(IN) :: MOP ! Data structure containing problem info.
! Output parameters.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.

! Check for uninitialized values.
IF ( (.NOT. ALLOCATED(MOP%LB)) .OR. (.NOT. ALLOCATED(MOP%UB)) ) THEN
   IERR = 910; RETURN; END IF
! Check for illegal/mismatched values.
IF ( MOP%D < 1 .OR. MOP%P < 2) THEN
   IERR = 911; RETURN; END IF
IF (SIZE(MOP%LB, 1) .NE. MOP%D .OR. SIZE(MOP%UB) .NE. MOP%D) THEN
   IERR = 911; RETURN; END IF

! Open the checkpoint file, using unformatted append.
OPEN(MOP_CHKPTUNIT, FILE=MOP_CHKPTFILE, FORM="unformatted", ACTION="write", &
     POSITION="append", STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 930; RETURN; END IF
! The status of MOP%WEIGHTS tells the phase of the MOP algorithm.
IF (ALLOCATED(MOP%WEIGHTS)) THEN
   IF (MOP%ITERATE > 0) THEN ! Don't write CLIST for the zeroeth iteration.
      ! Write unformatted MOP iteration data to the checkpoint file.
      WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%CLIST(1:MOP%D+1,MOP%ITERATE)
      IF (IERR .NE. 0) THEN
         IERR = 931; RETURN; CLOSE(MOP_CHKPTUNIT); END IF
   END IF
   ! Write unformatted adaptive weights to the checkpoint file.
   WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) SIZE(MOP%WEIGHTS, 2)
   IF (IERR .NE. 0) THEN
      IERR = 931; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
   WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%WEIGHTS(:,:)
   IF (IERR .NE. 0) THEN
      IERR = 931; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
ELSE
   ! Write the iteration counter.
   WRITE(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%ITERATE
   IF (IERR .NE. 0) THEN
      IERR = 931; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
END IF
! Close the checkpoint file.
CLOSE(MOP_CHKPTUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 932; RETURN; END IF

RETURN
END SUBROUTINE MOP_CHKPT

SUBROUTINE MOP_CHKPT_RECOVER(MOP, IERR)
! Recover MOP progress from a checkpoint file.
!
!
! On output:
!
! The status of MOP is as specified in CHKPTFILE.
!
! IERR is an integer valued error flag.
!
!  000 : Normal output. Iteration data saved to the checkpoint file.
!
!  9xx : Errors detected.
!     94x : A file I/O error was detected.
!         940 : The checkpoint file could not be opened, check whether
!               CHKPTFILE has been properly initialized.
!         941 : An error occured while writing data to the checkpoint file.
!         942 : An error occured while closing the checkpoint file.
!     95x : A memory allocation error ocured.
!         960 : A memory allocation error occured.
!         961 : A memory deallocation error occured.
!     960 : Failed the sanity check. Either the checkpoint feature was
!           implemented improperly, or MOP_CHKPTFILE was corrupted.
!
IMPLICIT NONE
! Output parameters.
TYPE(MOP_TYPE), INTENT(OUT) :: MOP ! Data structure containing problem info.
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Temporary arrays.
INTEGER :: NW
REAL(KIND=R8), ALLOCATABLE :: TMP(:,:)

! Open the checkpoint file, using unformatted write.
OPEN(MOP_CHKPTUNIT, FILE=MOP_CHKPTFILE, FORM="unformatted", ACTION="read", &
     STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 940; RETURN; END IF
! Read in the problem dimensions.
READ(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%D, MOP%P 
IF (IERR .NE. 0) THEN
   IERR = 941; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Read in the problem hyperparameters.
READ(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%LOPT_BUDGET, MOP%DECAY, & 
                     MOP%EPS, MOP%DES_TOLL, MOP%OBJ_TOLL,    &
                     MOP%MIN_RAD, MOP%TRUST_RAD
IF (IERR .NE. 0) THEN
   IERR = 941; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Allocate the bound constraints.
ALLOCATE(MOP%LB(MOP%D), MOP%UB(MOP%D), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 950; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Read in the bound constraints.
READ(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%LB(1:MOP%D), MOP%UB(1:MOP%D) 
IF (IERR .NE. 0) THEN
   IERR = 941; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Initialize the iteration data.
MOP%ITERATE = 0
MOP%LCLIST = 20
ALLOCATE(MOP%CLIST(MOP%D+1,MOP%LCLIST), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 950; CLOSE(MOP_CHKPTUNIT); RETURN; END IF

! Recreate the 0th iteration.

! Read the size of the next batch of adaptive weights.
READ(MOP_CHKPTUNIT, IOSTAT=IERR) NW
IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
   IERR = 0; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Allocate MOP%WEIGHTS accordingly.
ALLOCATE(MOP%WEIGHTS(MOP%P, NW), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 250; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Read in the next batch of adaptive weights.
READ(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%WEIGHTS(:,:)
IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
   IERR = 0; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Next checkpoint tells that the search phase has completed.
READ(MOP_CHKPTUNIT, IOSTAT=IERR) NW
IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
   IERR = 0; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Update the iteration counter, and do a sanity check.
MOP%ITERATE = MOP%ITERATE + 1
IF (NW .NE. MOP%ITERATE) THEN
   IERR = 260; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
! Free the adaptive weights for the next iteration.
DEALLOCATE(MOP%WEIGHTS, STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 251; CLOSE(MOP_CHKPTUNIT); RETURN; END IF

! Read data into MOP%CLIST(:,:) until the end of file.
READ_LOOP : DO WHILE (.TRUE.)
   ! Check if MOP%CLIST(:,:) needs to be resized.
   IF (MOP%ITERATE .EQ. MOP%LCLIST) THEN
      ! Allocate the temporary array.
      ALLOCATE(TMP(MOP%D+1,MOP%ITERATE), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 950; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
      ! Make a temporary copy.
      TMP(:,:) = MOP%CLIST(:,:)
      ! Update the size of MOP%LCLIST.
      MOP%LCLIST = MOP%LCLIST * 2
      ! Reallocate MOP%CLIST to twice its current size.
      DEALLOCATE(MOP%CLIST, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 951; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
      ALLOCATE(MOP%CLIST(MOP%D+1, MOP%LCLIST), STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 950; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
      ! Restore values back into CLIST and free the temporary array.
      MOP%CLIST(:,1:MOP%ITERATE) = TMP(:,:)
      DEALLOCATE(TMP, STAT=IERR)
      IF (IERR .NE. 0) THEN
         IERR = 251; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
   END IF
   ! Now, read in the next point in the center list from the file.
   READ(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%CLIST(1:MOP%D+1, MOP%ITERATE)
   IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Read the size of the next batch of adaptive weights.
   READ(MOP_CHKPTUNIT, IOSTAT=IERR) NW
   IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Allocate MOP%WEIGHTS accordingly.
   ALLOCATE(MOP%WEIGHTS(MOP%P, NW), STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 250; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
   ! Read in the next batch of adaptive weights.
   READ(MOP_CHKPTUNIT, IOSTAT=IERR) MOP%WEIGHTS(:,:)
   IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Next checkpoint tells that the search phase has completed.
   READ(MOP_CHKPTUNIT, IOSTAT=IERR) NW
   IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF.
      IERR = 0; EXIT READ_LOOP; END IF
   ! Update the iteration counter, and do a sanity check.
   MOP%ITERATE = MOP%ITERATE + 1
   IF (NW .NE. MOP%ITERATE) THEN
      IERR = 960; EXIT READ_LOOP; END IF
   ! Free the adaptive weights for the next iteration.
   DEALLOCATE(MOP%WEIGHTS, STAT=IERR)
   IF (IERR .NE. 0) THEN
      IERR = 951; CLOSE(MOP_CHKPTUNIT); RETURN; END IF
END DO READ_LOOP
! Close the checkpoint file.
CLOSE(MOP_CHKPTUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 942; RETURN; END IF
RETURN
END SUBROUTINE MOP_CHKPT_RECOVER

SUBROUTINE MOP_RECOVER_DATA(NB, DES_PTS, OBJ_PTS, IERR)
! Save MOP iteration data to an existing checkpoint file.
!
!
! On input:
!
!
! On output:
!
! Suggested batch size is NB.
!
! DES_PTS and OBJ_PTS are recovered.
!
! IERR is an integer valued error flag.
!
!  000 : Normal output. Successful initialization of a new MOP checkpoint.
!
!  99x : Errors detected.
!     99x : A data I/O error was detected.
!         990 : An error occured while opening the data file.
!         991 : An error occured while reading data from the data file.
!         992 : An error occured while closing the data file.
!         993 : The problem dimensions do not agree with MOP_MOD's internal
!               database.
!         994 : There was an issue allocating the local memory.
!
IMPLICIT NONE
! Output parameters.
INTEGER, INTENT(OUT) :: NB ! Preferred block size.
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: DES_PTS(:,:), OBJ_PTS(:,:)
INTEGER, INTENT(OUT) :: IERR ! Error flag arrays.
! Local variables.
INTEGER :: D, P, N ! Database dimensions.
INTEGER :: I ! Loop indexing variable.
REAL(KIND=R8), ALLOCATABLE :: DES_PT(:), OBJ_PT(:)

! Create a new data file, using unformatted write.
OPEN(MOP_DATAUNIT, FILE=MOP_DATAFILE, FORM="unformatted", ACTION="read", &
     STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 990; RETURN; END IF
! Read in the problem dimensions.
READ(MOP_DATAUNIT, IOSTAT=IERR) D, P, N, NB
IF (IERR .NE. 0) THEN
   IERR = 991; CLOSE(MOP_DATAUNIT); RETURN; END IF
! Allocate the local memory.
ALLOCATE(DES_PTS(D,N), OBJ_PTS(P,N), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 994; RETURN; END IF
! Read until all data points are recovered.
DO I = 1, N
   READ(MOP_DATAUNIT, IOSTAT=IERR) DES_PTS(:,I), OBJ_PTS(:,I)
   IF (IERR .NE. 0) THEN ! A read error occured. This must be EOF. 
      IERR = 0; EXIT; END IF
END DO
! Close the data file.
CLOSE(MOP_DATAUNIT, IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 992; RETURN; END IF
RETURN
END SUBROUTINE MOP_RECOVER_DATA

! The following module procedures can be passed to external procedures.

FUNCTION SURROGATE_FUNC(C, IERR) RESULT(F)
! Module procedure that uses the private module array MOP_MOD_WEIGHTS(:) to
! scalarize the output of MOP_MOD_SURROGATES. Matches the interface of
! MOP_MOD_SCALAR_INT, and can be passed as input to a generic single
! objective optimization procedure.
!
IMPLICIT NONE
! Parameters.
REAL(KIND=R8), INTENT(IN) :: C(:)
INTEGER, INTENT(OUT) :: IERR
REAL(KIND=R8) :: F
! Local variables.
REAL(KIND=R8) :: V(LSHEP_P)
! BLAS function for computing inner products.
REAL(KIND=R8), EXTERNAL :: DDOT
! Evaluate the surrogates.
CALL MOP_MOD_SURROGATES(C, V, IERR)
IF (IERR .NE. 0) RETURN
! Compute the weighted sum.
F = DDOT(LSHEP_P, MOP_MOD_WEIGHTS, 1, V, 1)
RETURN
END FUNCTION SURROGATE_FUNC

! The following procedures define the default surrogate function, using
! the LINEAR_SHEPARD module from SHEPPACK (ACM TOMS Alg. 905).

SUBROUTINE LSHEP_FIT(D, P, N, X_VALS, Y_VALS, IERR)
! Fit all P surrogate models using the LINEAR_SHEPARD module from SHEPPACK.
!
! Thacker, William I., J. Zhang, L. T. Watson, J. B. Birch, M. A. Iyer, and
! M. W. Berry. Algorithm 905: SHEPPACK: Modified Shepard algorithm for
! interpolation of scattered multivariate data. ACM Trans. Math. Softw. (TOMS)
! 37.3 (2010): 34.
!
! Store the P local linear fits in the private module array LSHEP_A(:,:,1:P)
! and the radii of influence in the private module array LSHEP_RW(:,1:P).
! Also make copies of the current dataset in LSHEP_XVALS(:,:) and
! LSHEP_FVALS(:,:), along with problem dimensions in LSHEP_D, LSHEP_P, and
! LSHEP_N.
! 
USE LINEAR_SHEPARD_MOD
IMPLICIT NONE
! Parameters.
INTEGER, INTENT(IN) :: D
INTEGER, INTENT(IN) :: P
INTEGER, INTENT(IN) :: N
REAL(KIND=R8), INTENT(IN) :: X_VALS(:,:)
REAL(KIND=R8), INTENT(IN) :: Y_VALS(:,:)
INTEGER, INTENT(OUT) :: IERR
! Local variables.
INTEGER :: I ! Loop indexing variable.
! Set LSHEP problem dimensions.
LSHEP_D = D
LSHEP_P = P
LSHEP_N = N
! Reallocate both LSHEP_A and LSHEP_RW to appropriate sizes.
IF (ALLOCATED(LSHEP_A)) DEALLOCATE(LSHEP_A)
IF (ALLOCATED(LSHEP_RW)) DEALLOCATE(LSHEP_RW)
IF (ALLOCATED(LSHEP_XVALS)) DEALLOCATE(LSHEP_XVALS)
IF (ALLOCATED(LSHEP_FVALS)) DEALLOCATE(LSHEP_FVALS)
IF (ALLOCATED(MOP_MOD_WEIGHTS)) DEALLOCATE(MOP_MOD_WEIGHTS)
ALLOCATE(LSHEP_A(LSHEP_D,LSHEP_N,LSHEP_P), &
       & LSHEP_RW(LSHEP_N,LSHEP_P),        &
       & LSHEP_XVALS(LSHEP_D, LSHEP_N),    &
       & LSHEP_FVALS(LSHEP_N, LSHEP_P),    &
       & MOP_MOD_WEIGHTS(LSHEP_P),         &
       & STAT=IERR)
IF (IERR .NE. 0) RETURN
! Populate data values.
LSHEP_XVALS(:,:) = X_VALS(:,:)
LSHEP_FVALS(:,:) = TRANSPOSE(Y_VALS(:,:))
! Fit P surrogate models and store weights/radii in A and RW.
DO I = 1, LSHEP_P
   CALL LSHEP( LSHEP_D, LSHEP_N, LSHEP_XVALS, LSHEP_FVALS(:,I), &
      & LSHEP_A(:,:,I), LSHEP_RW(:,I), IERR)
   IF (IERR .EQ. 1 .OR. IERR .EQ. 3) THEN; RETURN; END IF
END DO
! Return success error flag.
IERR = 0
RETURN
END SUBROUTINE LSHEP_FIT

SUBROUTINE LSHEP_EVAL(C, V, IERR)
! Evaluate the P surrogate models using the LINEAR_SHEPARD module from SHEPPACK.
!
! Thacker, William I., J. Zhang, L. T. Watson, J. B. Birch, M. A. Iyer, and
! M. W. Berry. Algorithm 905: SHEPPACK: Modified Shepard algorithm for
! interpolation of scattered multivariate data. ACM Trans. Math. Softw. (TOMS)
! 37.3 (2010): 34.
!
! Uses the private module variables and arrays LSHEP_D, LSHEP_P, LSHEP_N, 
! LSHEP_XVALS, LSHEP_FVALS, LSHEP_A, and LSHEP_RW, as set by the LSHEP_FIT
! subroutine.
!
USE LINEAR_SHEPARD_MOD
IMPLICIT NONE
! Parameters.
REAL(KIND=R8), INTENT(IN) :: C(:)
REAL(kIND=R8), INTENT(OUT) :: V(:)
INTEGER, INTENT(OUT) :: IERR
! Local variables.
INTEGER I
! Evaluate the surrogates.
DO I = 1, LSHEP_P
   V(I) = LSHEPVAL( C(:), LSHEP_D, LSHEP_N, &
      & LSHEP_XVALS(:,:), LSHEP_FVALS(:,I), &
      & LSHEP_A(:,:,I), LSHEP_RW(:,I), IERR )
   IF (IERR .GE. 10) RETURN
END DO
! Reset error flag to success code.
IERR = 0
RETURN
END SUBROUTINE LSHEP_EVAL

! The following is the default local search. A lightweight native Fortran
! implementation of the GPSMADS algorithm from the NOMAD software package
! (ACM TOMS Alg. 909).

SUBROUTINE GPSMADS(D, X, LB, UB, OBJ_FUNC, BUDGET, IERR, TOLL)
! Light weight Fortran implementation of the GPS MADS algorithm described in
!
! Le Digabel, Sébastien. Algorithm 909: NOMAD: Nonlinear Optimization with
! the MADS Algorithm. ACM Trans. Math. Softw. (TOMS) 37.4 (2011): 15.
!
! All features not relevant for this work, such as surrogate pre-ordering,
! search phases, variable neighborhood search, and taboo lists are ommitted.
!
IMPLICIT NONE
! Parameter list.
INTEGER, INTENT(IN) :: D
REAL(KIND=R8), INTENT(INOUT) :: X(:)
REAL(KIND=R8), INTENT(IN) :: LB(:)
REAL(KIND=R8), INTENT(IN) :: UB(:)
PROCEDURE(MOP_MOD_SCALAR_INT) :: OBJ_FUNC
INTEGER, INTENT(IN) :: BUDGET
INTEGER, INTENT(OUT) :: IERR
REAL(KIND=R8), INTENT(IN) :: TOLL
! Local variables.
INTEGER :: I, J ! Loop index variables.
INTEGER :: MIN_POLL ! Minimum poll index.
REAL(KIND=R8) :: X_VAL ! Current X value.
REAL(KIND=R8) :: EPS ! Working precision.
REAL(KIND=R8) :: POLLS(D,2*D) ! List of poll directions.
REAL(KIND=R8) :: POLL_VALS(2*D) ! List of poll values.
REAL(KIND=R8) :: MESH_GPS(D,2*D) ! The GPS mesh.
REAL(KIND=R8) :: MESH_SIZE ! The current mesh size.
REAL(KIND=R8) :: RESCALE(D) ! Rescale factors for bounding box.
! Get the working precision.
EPS = EPSILON(0.0_R8)
! Check for bad inputs.
IF (SIZE(X, 1) .NE. D) THEN
   IERR = 10; RETURN; END IF
IF ( (SIZE(LB, 1) .NE. D) .OR. (SIZE(UB,1) .NE. D) ) THEN
   IERR = 11; RETURN; END IF
IF ( ANY(LB(:) .GE. UB(:) - TOLL) ) THEN
   IERR = 12; RETURN; END IF
IF ( ANY(X(:) .GE. UB(:) + EPS) .OR. ANY(X(:) .LE. LB(:) - EPS) ) THEN
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
         ! Use the extreme barrier approach for bound violations.
         POLL_VALS(J) = HUGE(0.0_R8)
      ELSE
         ! For legal values, query the objective surrogate.
         POLL_VALS(J) = OBJ_FUNC(POLLS(:,J), IERR)
         ! Use an extreme barrier for missing values.
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
      IF (MESH_SIZE < TOLL) EXIT ! Stop cond 2: mesh tollerance.
   END IF
END DO
RETURN
END SUBROUTINE GPSMADS

! The following are possible global search options. GPSSEARCH can
! produce evenly spaced points, and has better properties for load
! balancing.

SUBROUTINE GPSSEARCH(D, P, LB, UB, BB_BUDGET, FIRST, CAND_PTS, IERR, TOLL)
! Light weight deterministic design space exploration code, inspired by the
! GPS MADS algorithm described in
!
! Le Digabel, Sébastien. Algorithm 909: NOMAD: Nonlinear Optimization with
! the MADS Algorithm. ACM Trans. Math. Softw. (TOMS) 37.4 (2011): 15.
!
! GPS_SEARCH carries out an initial poll phase from the center of the
! design space, then calls for a new poll from every point in its database.
! This process iterates until either the budget is exhausted, or no new
! poll points are generated. In the latter case, the mesh fineness is decayed,
! and the process continues.
!
IMPLICIT NONE
! Parameter list.
INTEGER, INTENT(IN) :: D, P
REAL(KIND=R8), INTENT(IN) :: LB(:), UB(:)
INTEGER, INTENT(IN) :: BB_BUDGET
LOGICAL, INTENT(IN) :: FIRST
REAL(KIND=R8), ALLOCATABLE, INTENT(OUT) :: CAND_PTS(:,:)
INTEGER, INTENT(OUT) :: IERR
REAL(KIND=R8), INTENT(IN) :: TOLL
! Local variables.
INTEGER :: I, J, K, L ! Loop index variables.
INTEGER :: N ! Current length of CAND_PTS(:,:) list.
REAL(KIND=R8) :: POLLS(D) ! Next poll direction.
REAL(KIND=R8) :: MESH_GPS(D,2*D) ! The GPS mesh.
REAL(KIND=R8) :: MESH_SIZE ! The current mesh size.
REAL(KIND=R8) :: RESCALE(D) ! Rescale factors for bounding box.
REAL(KIND=R8) :: EPS ! Working precision.
! External BLAS function for computing Euclidean distance.
REAL(KIND=R8), EXTERNAL :: DNRM2
! Get the working precision.
EPS = SQRT(EPSILON(0.0_R8))
! Check for bad inputs.
IF ( (SIZE(LB, 1) .NE. D) .OR. (SIZE(UB,1) .NE. D) ) THEN
   IERR = 10; RETURN; END IF
IF ( ANY(LB(:) .GE. UB(:) - EPS) ) THEN
   IERR = 11; RETURN; END IF
IF (BB_BUDGET < 2*D+1) THEN
   IERR = 12; RETURN; END IF
! This is a static exploration, so allocate the output array.
ALLOCATE(CAND_PTS(D,BB_BUDGET), STAT=IERR)
IF (IERR .NE. 0) THEN
   IERR = 20; RETURN; END IF
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
! Set the first point in the mesh to the center point.
N = 1
CAND_PTS(:,N) = (UB(:) + LB(:)) / 2.0_R8
! Loop until the iteration budget is exhausted.
OUTER : DO I = 1, BB_BUDGET
   DO J = 1, N
      INNER : DO K = 1, 2*D
         ! Get the next poll point.
         POLLS(:) = CAND_PTS(:,J) + MESH_GPS(:,K) * RESCALE(:) * MESH_SIZE
         ! Strictly enforce the bound constraints.
         DO L = 1, D
            POLLS(L) = MAX(POLLS(L), LB(L))
            POLLS(L) = MIN(POLLS(L), UB(L))
         END DO
         ! Filter out any redundant poll points.
         DO L = 1, N
            IF ( DNRM2(D,POLLS(:)-CAND_PTS(:,L),1) < TOLL ) CYCLE INNER
         END DO
         ! Add to the search mesh.
         N = N + 1
         CAND_PTS(:,N) = POLLS(:)
         ! Loop until the budget is exhausted.
         IF (N .EQ. BB_BUDGET) EXIT OUTER
      END DO INNER
   END DO
   ! Decay the mesh size.
   MESH_SIZE = MESH_SIZE * 0.5_R8
   ! Rare case where the budget is so large, the mesh tollerance is reached.
   IF (MESH_SIZE < TOLL) EXIT
END DO OUTER
! Rare case where the budget is so large, the mesh tollerance is reached.
IF (N < BB_BUDGET) THEN
   IERR = 30; RETURN; END IF
! Set the error flag to "success" and return.
IERR = 0
RETURN
END SUBROUTINE GPSSEARCH

END MODULE MOP_MOD_LIGHT
