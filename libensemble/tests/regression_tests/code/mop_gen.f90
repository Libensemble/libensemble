PROGRAM GENERATE
! Import MOP_MOD.
USE MOP_MOD_LIGHT
IMPLICIT NONE

! Local variables used for defining the problem.
TYPE(MOP_TYPE) :: MOP
INTEGER :: INB, IERR, LBATCH
REAL(KIND=R8), ALLOCATABLE :: DES_PTS(:,:), OBJ_PTS(:,:), BATCHX(:,:)

! Recover MOP data.
CALL MOP_RECOVER_DATA(INB, DES_PTS, OBJ_PTS, IERR)
IF (IERR .GE. 10) THEN
   PRINT *, 'data recovery error'; GO TO 999; END IF
! Recover MOP object.
CALL MOP_CHKPT_RECOVER(MOP, IERR)
IF (IERR .GE. 10) THEN
   PRINT *, 'checkpoint recovery error'; GO TO 999; END IF
! Call MOP generator.
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

END PROGRAM GENERATE
