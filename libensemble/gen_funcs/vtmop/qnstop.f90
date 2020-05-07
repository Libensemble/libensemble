MODULE QNSTOPS_MOD
! Code for generating a latin hypercube experimental design from the QNSTOP
! software package.
!
! Credit to B. D. Amos, D. R. Easterling, L. T. Watson, W. I. Thacker,
! B. S. Castle, and M. W. Trosset. Algorithm XXX: QNSTOP -- Quasi-Newton
! Algorithm for Stochastic Optimization.
!
USE REAL_PRECISION, ONLY : R8
CONTAINS

SUBROUTINE LATINDESIGN(P, LNSTART, LB, UB, XI, LHSDESIGN)
! LATINDESIGN generates a Latin hypercube experimental design with
! LNSTART points, one of which is XI, in the P-dimensional box
! B = { x | LB <= x <= UB }.  Partition each interval [LB(J),UB(J)]
! into LNSTART equal length subintervals, which partitions the box B
! into cells. A Latin hypercube experimental design has the property
! that if the LNSTART points are projected onto any coordinate
! direction K, for 1 <= K <= P, the projection has exactly one point
! in each of the subintervals of [LB(K),UB(K)].  The cells centered
! along the diameter [LB,UB] of the box B are called the diagonal cells.
!
! The algorithm is to first generate LNSTART points in the diagonal
! cells, stored as the columns of the array LHSDESIGN(1:P,1:LNSTART).
! Then swap appropriate coordinates with the last column so that the
! given point XI can be inserted as the last column.  Finally, permute
! the coordinates within rows of LHSDESIGN(1:P,1:LNSTART-1) by doing
! LNSTART-2 random interchanges in each row.
!
! On input:
!
! P is the dimension.
!
! LNSTART is the integer number of points in the experimental design.
!
! LB(1:P) is a real array giving the lower bounds.
!
! UB(1:P) is a real array giving the upper bounds, defining the box
!   { x | LB <= x <= UB } in which design is generated.
!
! XI(1:P) is a real array containing the start point to be inserted in the
!   Latin hypercube design.
!
! On output:
!
! LHSDESIGN(1:P,1:LNSTART) is a real array whose columns are the Latin
!   hypercube experimental design, with XI being the last column.

INTEGER, INTENT(IN):: P, LNSTART
REAL(KIND=R8), DIMENSION(P), INTENT(IN):: LB, UB, XI
REAL(KIND=R8), DIMENSION(P,LNSTART), INTENT(OUT):: LHSDESIGN

! Local variables.
INTEGER:: I, J, L   ! Temporary loop variable.
REAL(KIND=R8):: TEMP  ! Used for swapping coordinates.
REAL(KIND=R8), DIMENSION(P):: WORK  ! Holds the bin sizes.
! SWAP is an array of random values for picking swap rows.
REAL(KIND=R8), DIMENSION(P,LNSTART-2):: SWAP


CALL RANDOM_NUMBER(HARVEST=LHSDESIGN(1:P,1:LNSTART))
WORK = (UB - LB)/REAL(LNSTART, KIND=R8)    ! Bin sizes.

! Start with all the points in diagonal cells.
DO J=1,LNSTART
  LHSDESIGN(1:P,J) = LB(1:P) + WORK(1:P)*(REAL(J-1, KIND=R8) + &
    LHSDESIGN(1:P,J))
END DO

! Now place the given start point XI into the last column by swapping
! coordinates.
DO J=1,P
  L = MIN(INT((XI(J) - LB(J))/WORK(J)) + 1, LNSTART )
  LHSDESIGN(J,L) = LHSDESIGN(J,LNSTART)
  LHSDESIGN(J,LNSTART) = XI(J)
END DO

CALL RANDOM_NUMBER(HARVEST=SWAP(1:P, 1:LNSTART-2))
! Permute coordinates within each row of LHSDESIGN(1:P,1:LNSTART-1) by
! doing LNSTART - 2 interchanges.
DO I=1,P
  DO J=LNSTART-1,2,-1
    L = INT(SWAP(I,J-1)*REAL(J,KIND=R8)) + 1
    TEMP = LHSDESIGN(I,J)
    LHSDESIGN(I,J) = LHSDESIGN(I,L)
    LHSDESIGN(I,L) = TEMP
  END DO
END DO
RETURN
END SUBROUTINE LATINDESIGN

END MODULE QNSTOPS_MOD
