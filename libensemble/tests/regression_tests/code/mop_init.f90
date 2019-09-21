PROGRAM INIT
! Import MOP_MOD.
USE MOP_MOD_LIGHT
IMPLICIT NONE

! Local variables used for defining the problem.
TYPE(MOP_TYPE) :: MOP
INTEGER :: D, P
INTEGER :: IERR, INB, LBATCH
REAL(KIND=R8), ALLOCATABLE :: LB(:), UB(:), DES_PTS(:,:), OBJ_PTS(:,:), &
                              BATCHX(:,:)

! Read the data file, using unformatted write.
OPEN(11, FILE="mop.dat", FORM="unformatted", ACTION="read", STATUS="old", &
     IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   PRINT *, 'mop.dat does not exist!'; GO TO 999; END IF
READ(11, IOSTAT=IERR) D, P, INB
IF (IERR .NE. 0) THEN
   PRINT *, 'reading error'; GO TO 999; END IF
ALLOCATE(LB(D), UB(D), STAT=IERR)
IF (IERR .NE. 0) THEN
   PRINT *, 'reading error'; GO TO 999; END IF
READ(11, IOSTAT=IERR) LB(:)
IF (IERR .NE. 0) THEN
   PRINT *, 'reading error'; GO TO 999; END IF
READ(11, IOSTAT=IERR) UB(:)
IF (IERR .NE. 0) THEN
   PRINT *, 'reading error'; GO TO 999; END IF
CLOSE(11)

! Initialize the MOP object.
CALL MOP_INIT(MOP, D, P, LB, UB, IERR, ICHKPT=1)
IF (IERR .GE. 10) THEN
   PRINT *, 'initialization error: ', IERR; GO TO 999; END IF

! Call MOP generator for zeroeth iteration.
CALL MOP_GENERATE(MOP, DES_PTS, OBJ_PTS, LBATCH, BATCHX, IERR, &
                        NB=INB, ICHKPT=1)
IF (IERR .GE. 10) THEN
   PRINT *, 'generator error: ', IERR; GO TO 999; END IF

! Create a new data file, using unformatted write.
OPEN(12, FILE="mop.out", FORM="unformatted", ACTION="write", IOSTAT=IERR)
IF (IERR .NE. 0) THEN
   PRINT *, 'writing error'; GO TO 999; END IF
WRITE(12, IOSTAT=IERR) LBATCH
IF (IERR .NE. 0) THEN
   PRINT *, 'writing error'; GO TO 999; END IF
WRITE(12, IOSTAT=IERR) BATCHX
IF (IERR .NE. 0) THEN
   PRINT *, 'writing error'; GO TO 999; END IF
CLOSE(12)

! Exit statement.
999 CONTINUE

END PROGRAM INIT
