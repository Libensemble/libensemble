PROGRAM GENERATOR
! Implement a generator function to be used by libEnsemble.
USE ISO_FORTRAN_ENV
USE VTMOP_MOD
IMPLICIT NONE
! Variables.
INTEGER :: D, P ! Problem dimensions.
INTEGER :: LBATCH ! Length of the batch.
INTEGER :: N ! Number of points in database.
INTEGER :: SNB, ONB, NB ! Preferred batch size.
INTEGER :: IERR ! Error flag.
REAL(KIND=R8), ALLOCATABLE :: BATCHX(:,:) ! Candidate design points to evaluate.
REAL(KIND=R8), ALLOCATABLE :: LB(:), UB(:) ! Bound constraints.
TYPE(VTMOP_TYPE) :: VTMOP

! Open an existing I/O file using unformatted read and get input dimensions.
OPEN(12, FILE="vtmop.io", FORM="unformatted", ACTION="read", &
     STATUS="old", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while opening the input file for the generator"
   GO TO 100; END IF
READ(12, IOSTAT=IERR) D, P, N, SNB, ONB
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while reading the input metadata for the generator"
   GO TO 100; END IF
ALLOCATE(LB(D), UB(D), STAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while allocating the bound arrays for the generator"
   GO TO 100; END IF
READ(12, IOSTAT=IERR) LB(:), UB(:)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while reading the constraints for the generator"
   GO TO 100; END IF
CLOSE(12)

! Recover the VTMOP status object.
CALL VTMOP_INIT( VTMOP, D, P, LB, UB, IERR, ICHKPT=-1 )
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A,I4)") &
        "An error occurred while recovering VTMOP object. Error code: ", IERR
   GO TO 100
END IF

! Recover the cost function data.
CALL VTMOP_RECOVER_DATA(VTMOP_MOD_DBN, VTMOP_MOD_DBX, VTMOP_MOD_DBF, IERR, &
                        DB_SIZE=N)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A,I4)") &
        "An error occurred while recovering data. Error code: ", IERR
   GO TO 100
END IF

! Check whether this is an optimization or search phase.
IF (ALLOCATED(VTMOP%WEIGHTS)) THEN
   NB = ONB
ELSE
   NB = SNB
END IF

! Call the generator function and save to the checkpoint.
CALL VTMOP_GENERATE( VTMOP, VTMOP_MOD_DBX(:,1:VTMOP_MOD_DBN),          &
                     VTMOP_MOD_DBF(:,1:VTMOP_MOD_DBN), LBATCH, BATCHX, &
                     IERR, NB=NB )
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A,I4)") &
        "An error occurred while generating candidates. Error code: ", IERR
   GO TO 100
END IF

! Open I/O file using unformatted write, and record the output.
OPEN(12, FILE="vtmop.io", FORM="unformatted", ACTION="write", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while opening the output file for the generator"
   GO TO 100; END IF
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while writing the output file for the generator"
   GO TO 100; END IF
WRITE(12, IOSTAT=IERR) BATCHX(:,:)
IF (IERR .NE. 0) THEN
   WRITE(ERROR_UNIT, "(A)") &
        "An error occurred while writing the output file for the generator"
   GO TO 100; END IF
CLOSE(12)

! End of program marker.
100 CONTINUE

END PROGRAM GENERATOR
