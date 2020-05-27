PROGRAM VTMOP_INITIALIZER
! Implement a generator function to be used by libEnsemble.
USE ISO_FORTRAN_ENV
USE VTMOP_MOD
IMPLICIT NONE
! Variables.
INTEGER :: D, P ! Problem dimensions.
INTEGER :: LBATCH ! Length of the batch.
INTEGER :: NB ! Preferred batch size.
INTEGER :: IERR ! Error flag.
REAL(KIND=R8) :: TRUST_RAD ! Trust region radius.
REAL(KIND=R8), ALLOCATABLE :: BATCHX(:,:) ! Candidate design points to evaluate.
REAL(KIND=R8), ALLOCATABLE :: LB(:), UB(:) ! Bound constraints.
REAL(KIND=R8), ALLOCATABLE :: DES(:,:), OBJ(:,:)
TYPE(VTMOP_TYPE) :: VTMOP

! Open an existing I/O file using unformatted read and get input dimensions.
OPEN(12, FILE="vtmop.io", FORM="unformatted", ACTION="read", &
     STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT,"(A)") "An error occurred while opening the VTMOP input"
   GO TO 100; END IF
READ(12, IOSTAT=IERR) D, P, NB
ALLOCATE(LB(D), UB(D), STAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT,"(A)") "An error occurred while reading the VTMOP metadata"
   GO TO 100; END IF
ALLOCATE(DES(D,1), OBJ(P,1), STAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT,"(A)") "An error occurred while allocated dummy arrays"
   GO TO 100; END IF
READ(12, IOSTAT=IERR) LB(:), UB(:)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT,"(A)") "An error occurred while reading the constrainst"
   GO TO 100; END IF
READ(12, IOSTAT=IERR) TRUST_RAD
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT,"(A)") "An error occurred while reading the LTR radius"
   GO TO 100; END IF
CLOSE(12)

! Initialize the VTMOP status object.
CALL VTMOP_INIT( VTMOP, D, P, LB, UB, IERR, TRUST_RADF=TRUST_RAD, ICHKPT=1 )
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A,I4)") &
        "An error occurred while initializing. Error code: ", IERR
    GO TO 100
END IF

! Call the generator function and save to the checkpoint.
CALL VTMOP_GENERATE( VTMOP, DES, OBJ, LBATCH, BATCHX, IERR, NB=NB )
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A,I4)") &
        "An error occurred while generating initial batch. Error code: ", &
        IERR
    GO TO 100
END IF

! Open I/O file using unformatted write, and record the output.
OPEN(12, FILE="vtmop.io", FORM="unformatted", ACTION="write", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while opening the output file for the initializer"
   GO TO 100; END IF
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while writing the output file for the initializer"
   GO TO 100; END IF
WRITE(12, IOSTAT=IERR) BATCHX(:,:)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while writing the output file for the initializer"
   GO TO 100; END IF
CLOSE(12)

! End of program marker.
100 CONTINUE

END PROGRAM VTMOP_INITIALIZER
