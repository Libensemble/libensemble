*DECK DLSEI
      SUBROUTINE DLSEI (W, MDW, ME, MA, MG, N, PRGOPT, X, RNORME,
     +   RNORML, MODE, WS, IP)
C***BEGIN PROLOGUE  DLSEI
C***PURPOSE  Solve a linearly constrained least squares problem with
C            equality and inequality constraints, and optionally compute
C            a covariance matrix.
C***LIBRARY   SLATEC
C***CATEGORY  K1A2A, D9
C***TYPE      REAL(KIND=R8) (LSEI-S, DLSEI-D)
C***KEYWORDS  CONSTRAINED LEAST SQUARES, CURVE FITTING, DATA FITTING,
C             EQUALITY CONSTRAINTS, INEQUALITY CONSTRAINTS,
C             QUADRATIC PROGRAMMING
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     Abstract
C
C     This subprogram solves a linearly constrained least squares
C     problem with both equality and inequality constraints, and, if the
C     user requests, obtains a covariance matrix of the solution
C     parameters.
C
C     Suppose there are given matrices E, A and G of respective
C     dimensions ME by N, MA by N and MG by N, and vectors F, B and H of
C     respective lengths ME, MA and MG.  This subroutine solves the
C     linearly constrained least squares problem
C
C                   EX = F, (E ME by N) (equations to be exactly
C                                       satisfied)
C                   AX = B, (A MA by N) (equations to be
C                                       approximately satisfied,
C                                       least squares sense)
C                   GX .GE. H,(G MG by N) (inequality constraints)
C
C     The inequalities GX .GE. H mean that every component of the
C     product GX must be .GE. the corresponding component of H.
C
C     In case the equality constraints cannot be satisfied, a
C     generalized inverse solution residual vector length is obtained
C     for F-EX.  This is the minimal length possible for F-EX.
C
C     Any values ME .GE. 0, MA .GE. 0, or MG .GE. 0 are permitted.  The
C     rank of the matrix E is estimated during the computation.  We call
C     this value KRANKE.  It is an output parameter in IP(1) defined
C     below.  Using a generalized inverse solution of EX=F, a reduced
C     least squares problem with inequality constraints is obtained.
C     The tolerances used in these tests for determining the rank
C     of E and the rank of the reduced least squares problem are
C     given in Sandia Tech. Rept. SAND-78-1290.  They can be
C     modified by the user if new values are provided in
C     the option list of the array PRGOPT(*).
C
C     The user must dimension all arrays appearing in the call list..
C     W(MDW,N+1),PRGOPT(*),X(N),WS(2*(ME+N)+K+(MG+2)*(N+7)),IP(MG+2*N+2)
C     where K=MAX(MA+MG,N).  This allows for a solution of a range of
C     problems in the given working space.  The dimension of WS(*)
C     given is a necessary overestimate.  Once a particular problem
C     has been run, the output parameter IP(3) gives the actual
C     dimension required for that problem.
C
C     The parameters for DLSEI( ) are
C
C     Input.. All TYPE REAL variables are REAL(KIND=R8)
C
C     W(*,*),MDW,   The array W(*,*) is doubly subscripted with
C     ME,MA,MG,N    first dimensioning parameter equal to MDW.
C                   For this discussion let us call M = ME+MA+MG.  Then
C                   MDW must satisfy MDW .GE. M.  The condition
C                   MDW .LT. M is an error.
C
C                   The array W(*,*) contains the matrices and vectors
C
C                                  (E  F)
C                                  (A  B)
C                                  (G  H)
C
C                   in rows and columns 1,...,M and 1,...,N+1
C                   respectively.
C
C                   The integers ME, MA, and MG are the
C                   respective matrix row dimensions
C                   of E, A and G.  Each matrix has N columns.
C
C     PRGOPT(*)    This real-valued array is the option vector.
C                  If the user is satisfied with the nominal
C                  subprogram features set
C
C                  PRGOPT(1)=1 (or PRGOPT(1)=1.0)
C
C                  Otherwise PRGOPT(*) is a linked list consisting of
C                  groups of data of the following form
C
C                  LINK
C                  KEY
C                  DATA SET
C
C                  The parameters LINK and KEY are each one word.
C                  The DATA SET can be comprised of several words.
C                  The number of items depends on the value of KEY.
C                  The value of LINK points to the first
C                  entry of the next group of data within
C                  PRGOPT(*).  The exception is when there are
C                  no more options to change.  In that
C                  case, LINK=1 and the values KEY and DATA SET
C                  are not referenced.  The general layout of
C                  PRGOPT(*) is as follows.
C
C               ...PRGOPT(1) = LINK1 (link to first entry of next group)
C               .  PRGOPT(2) = KEY1 (key to the option change)
C               .  PRGOPT(3) = data value (data value for this change)
C               .       .
C               .       .
C               .       .
C               ...PRGOPT(LINK1)   = LINK2 (link to the first entry of
C               .                       next group)
C               .  PRGOPT(LINK1+1) = KEY2 (key to the option change)
C               .  PRGOPT(LINK1+2) = data value
C               ...     .
C               .       .
C               .       .
C               ...PRGOPT(LINK) = 1 (no more options to change)
C
C                  Values of LINK that are nonpositive are errors.
C                  A value of LINK .GT. NLINK=100000 is also an error.
C                  This helps prevent using invalid but positive
C                  values of LINK that will probably extend
C                  beyond the program limits of PRGOPT(*).
C                  Unrecognized values of KEY are ignored.  The
C                  order of the options is arbitrary and any number
C                  of options can be changed with the following
C                  restriction.  To prevent cycling in the
C                  processing of the option array, a count of the
C                  number of options changed is maintained.
C                  Whenever this count exceeds NOPT=1000, an error
C                  message is printed and the subprogram returns.
C
C                  Options..
C
C                  KEY=1
C                         Compute in W(*,*) the N by N
C                  covariance matrix of the solution variables
C                  as an output parameter.  Nominally the
C                  covariance matrix will not be computed.
C                  (This requires no user input.)
C                  The data set for this option is a single value.
C                  It must be nonzero when the covariance matrix
C                  is desired.  If it is zero, the covariance
C                  matrix is not computed.  When the covariance matrix
C                  is computed, the first dimensioning parameter
C                  of the array W(*,*) must satisfy MDW .GE. MAX(M,N).
C
C                  KEY=10
C                         Suppress scaling of the inverse of the
C                  normal matrix by the scale factor RNORM**2/
C                  MAX(1, no. of degrees of freedom).  This option
C                  only applies when the option for computing the
C                  covariance matrix (KEY=1) is used.  With KEY=1 and
C                  KEY=10 used as options the unscaled inverse of the
C                  normal matrix is returned in W(*,*).
C                  The data set for this option is a single value.
C                  When it is nonzero no scaling is done.  When it is
C                  zero scaling is done.  The nominal case is to do
C                  scaling so if option (KEY=1) is used alone, the
C                  matrix will be scaled on output.
C
C                  KEY=2
C                         Scale the nonzero columns of the
C                         entire data matrix.
C                  (E)
C                  (A)
C                  (G)
C
C                  to have length one.  The data set for this
C                  option is a single value.  It must be
C                  nonzero if unit length column scaling
C                  is desired.
C
C                  KEY=3
C                         Scale columns of the entire data matrix
C                  (E)
C                  (A)
C                  (G)
C
C                  with a user-provided diagonal matrix.
C                  The data set for this option consists
C                  of the N diagonal scaling factors, one for
C                  each matrix column.
C
C                  KEY=4
C                         Change the rank determination tolerance for
C                  the equality constraint equations from
C                  the nominal value of SQRT(DRELPR).  This quantity can
C                  be no smaller than DRELPR, the arithmetic-
C                  storage precision.  The quantity DRELPR is the
C                  largest positive number such that T=1.+DRELPR
C                  satisfies T .EQ. 1.  The quantity used
C                  here is internally restricted to be at
C                  least DRELPR.  The data set for this option
C                  is the new tolerance.
C
C                  KEY=5
C                         Change the rank determination tolerance for
C                  the reduced least squares equations from
C                  the nominal value of SQRT(DRELPR).  This quantity can
C                  be no smaller than DRELPR, the arithmetic-
C                  storage precision.  The quantity used
C                  here is internally restricted to be at
C                  least DRELPR.  The data set for this option
C                  is the new tolerance.
C
C                  For example, suppose we want to change
C                  the tolerance for the reduced least squares
C                  problem, compute the covariance matrix of
C                  the solution parameters, and provide
C                  column scaling for the data matrix.  For
C                  these options the dimension of PRGOPT(*)
C                  must be at least N+9.  The Fortran statements
C                  defining these options would be as follows:
C
C                  PRGOPT(1)=4 (link to entry 4 in PRGOPT(*))
C                  PRGOPT(2)=1 (covariance matrix key)
C                  PRGOPT(3)=1 (covariance matrix wanted)
C
C                  PRGOPT(4)=7 (link to entry 7 in PRGOPT(*))
C                  PRGOPT(5)=5 (least squares equas.  tolerance key)
C                  PRGOPT(6)=... (new value of the tolerance)
C
C                  PRGOPT(7)=N+9 (link to entry N+9 in PRGOPT(*))
C                  PRGOPT(8)=3 (user-provided column scaling key)
C
C                  CALL DCOPY (N, D, 1, PRGOPT(9), 1)  (Copy the N
C                    scaling factors from the user array D(*)
C                    to PRGOPT(9)-PRGOPT(N+8))
C
C                  PRGOPT(N+9)=1 (no more options to change)
C
C                  The contents of PRGOPT(*) are not modified
C                  by the subprogram.
C                  The options for WNNLS( ) can also be included
C                  in this array.  The values of KEY recognized
C                  by WNNLS( ) are 6, 7 and 8.  Their functions
C                  are documented in the usage instructions for
C                  subroutine WNNLS( ).  Normally these options
C                  do not need to be modified when using DLSEI( ).
C
C     IP(1),       The amounts of working storage actually
C     IP(2)        allocated for the working arrays WS(*) and
C                  IP(*), respectively.  These quantities are
C                  compared with the actual amounts of storage
C                  needed by DLSEI( ).  Insufficient storage
C                  allocated for either WS(*) or IP(*) is an
C                  error.  This feature was included in DLSEI( )
C                  because miscalculating the storage formulas
C                  for WS(*) and IP(*) might very well lead to
C                  subtle and hard-to-find execution errors.
C
C                  The length of WS(*) must be at least
C
C                  LW = 2*(ME+N)+K+(MG+2)*(N+7)
C
C                  where K = max(MA+MG,N)
C                  This test will not be made if IP(1).LE.0.
C
C                  The length of IP(*) must be at least
C
C                  LIP = MG+2*N+2
C                  This test will not be made if IP(2).LE.0.
C
C     Output.. All TYPE REAL variables are REAL(KIND=R8)
C
C     X(*),RNORME,  The array X(*) contains the solution parameters
C     RNORML        if the integer output flag MODE = 0 or 1.
C                   The definition of MODE is given directly below.
C                   When MODE = 0 or 1, RNORME and RNORML
C                   respectively contain the residual vector
C                   Euclidean lengths of F - EX and B - AX.  When
C                   MODE=1 the equality constraint equations EX=F
C                   are contradictory, so RNORME .NE. 0.  The residual
C                   vector F-EX has minimal Euclidean length.  For
C                   MODE .GE. 2, none of these parameters is defined.
C
C     MODE          Integer flag that indicates the subprogram
C                   status after completion.  If MODE .GE. 2, no
C                   solution has been computed.
C
C                   MODE =
C
C                   0  Both equality and inequality constraints
C                      are compatible and have been satisfied.
C
C                   1  Equality constraints are contradictory.
C                      A generalized inverse solution of EX=F was used
C                      to minimize the residual vector length F-EX.
C                      In this sense, the solution is still meaningful.
C
C                   2  Inequality constraints are contradictory.
C
C                   3  Both equality and inequality constraints
C                      are contradictory.
C
C                   The following interpretation of
C                   MODE=1,2 or 3 must be made.  The
C                   sets consisting of all solutions
C                   of the equality constraints EX=F
C                   and all vectors satisfying GX .GE. H
C                   have no points in common.  (In
C                   particular this does not say that
C                   each individual set has no points
C                   at all, although this could be the
C                   case.)
C
C                   4  Usage error occurred.  The value
C                      of MDW is .LT. ME+MA+MG, MDW is
C                      .LT. N and a covariance matrix is
C                      requested, or the option vector
C                      PRGOPT(*) is not properly defined,
C                      or the lengths of the working arrays
C                      WS(*) and IP(*), when specified in
C                      IP(1) and IP(2) respectively, are not
C                      long enough.
C
C     W(*,*)        The array W(*,*) contains the N by N symmetric
C                   covariance matrix of the solution parameters,
C                   provided this was requested on input with
C                   the option vector PRGOPT(*) and the output
C                   flag is returned with MODE = 0 or 1.
C
C     IP(*)         The integer working array has three entries
C                   that provide rank and working array length
C                   information after completion.
C
C                      IP(1) = rank of equality constraint
C                              matrix.  Define this quantity
C                              as KRANKE.
C
C                      IP(2) = rank of reduced least squares
C                              problem.
C
C                      IP(3) = the amount of storage in the
C                              working array WS(*) that was
C                              actually used by the subprogram.
C                              The formula given above for the length
C                              of WS(*) is a necessary overestimate.
C                              If exactly the same problem matrices
C                              are used in subsequent executions,
C                              the declared dimension of WS(*) can
C                              be reduced to this output value.
C     User Designated
C     Working Arrays..
C
C     WS(*),IP(*)              These are respectively type real
C                              and type integer working arrays.
C                              Their required minimal lengths are
C                              given above.
C
C***REFERENCES  K. H. Haskell and R. J. Hanson, An algorithm for
C                 linear least squares problems with equality and
C                 nonnegativity constraints, Report SAND77-0552, Sandia
C                 Laboratories, June 1978.
C               K. H. Haskell and R. J. Hanson, Selected algorithms for
C                 the linearly constrained least squares problem - a
C                 users guide, Report SAND78-1290, Sandia Laboratories,
C                 August 1979.
C               K. H. Haskell and R. J. Hanson, An algorithm for
C                 linear least squares problems with equality and
C                 nonnegativity constraints, Mathematical Programming
C                 21 (1981), pp. 98-118.
C               R. J. Hanson and K. H. Haskell, Two algorithms for the
C                 linearly constrained least squares problem, ACM
C                 Transactions on Mathematical Software, September 1982.
C***ROUTINES CALLED  D1MACH, DASUM, DAXPY, DCOPY, DDOT, DH12, DLSI,
C                    DNRM2, DSCAL, DSWAP, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890618  Completely restructured and extensively revised (WRB & RWC)
C   890831  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900510  Convert XERRWV calls to XERMSG calls.  (RWC)
C   900604  DP version created from SP version.  (RWC)
C   920501  Reformatted the REFERENCES section.  (WRB)
C   180613  Removed prints and replaced DP --> REAL(KIND=R8). (THC)
C***END PROLOGUE  DLSEI
      USE REAL_PRECISION

      INTEGER IP(3), MA, MDW, ME, MG, MODE, N
      REAL(KIND=R8) PRGOPT(*), RNORME, RNORML, W(MDW,*), WS(*), X(*)
C
      EXTERNAL D1MACH, DASUM, DAXPY, DCOPY, DDOT, DH12, DLSI, DNRM2,
     *   DSCAL, DSWAP
      REAL(KIND=R8) D1MACH, DASUM, DDOT, DNRM2
C
      REAL(KIND=R8) DRELPR, ENORM, FNORM, GAM, RB, RN, RNMAX, SIZE,
     *   SN, SNMAX, T, TAU, UJ, UP, VJ, XNORM, XNRME
      INTEGER I, IMAX, J, JP1, K, KEY, KRANKE, LAST, LCHK, LINK, M,
     *   MAPKE1, MDEQC, MEND, MEP1, N1, N2, NEXT, NLINK, NOPT, NP1,
     *   NTIMES
      LOGICAL COV, FIRST
C      CHARACTER*8 XERN1, XERN2, XERN3, XERN4
      SAVE FIRST, DRELPR
C
      DATA FIRST /.TRUE./
C***FIRST EXECUTABLE STATEMENT  DLSEI
C
C     Set the nominal tolerance used in the code for the equality
C     constraint equations.
C
      IF (FIRST) DRELPR = D1MACH(4)
      FIRST = .FALSE.
      TAU = SQRT(DRELPR)
C
C     Check that enough storage was allocated in WS(*) and IP(*).
C
      MODE = 4
      IF (MIN(N,ME,MA,MG) .LT. 0) THEN
C         WRITE (XERN1, '(I8)') N
C         WRITE (XERN2, '(I8)') ME
C         WRITE (XERN3, '(I8)') MA
C         WRITE (XERN4, '(I8)') MG
C         CALL XERMSG ('SLATEC', 'LSEI', 'ALL OF THE VARIABLES N, ME,' //
C     *      ' MA, MG MUST BE .GE. 0$$ENTERED ROUTINE WITH' //
C     *      '$$N  = ' // XERN1 //
C     *      '$$ME = ' // XERN2 //
C     *      '$$MA = ' // XERN3 //
C     *      '$$MG = ' // XERN4, 2, 1)
         RETURN
      ENDIF
C
      IF (IP(1).GT.0) THEN
         LCHK = 2*(ME+N) + MAX(MA+MG,N) + (MG+2)*(N+7)
         IF (IP(1).LT.LCHK) THEN
C            WRITE (XERN1, '(I8)') LCHK
C            CALL XERMSG ('SLATEC', 'DLSEI', 'INSUFFICIENT STORAGE ' //
C     *         'ALLOCATED FOR WS(*), NEED LW = ' // XERN1, 2, 1)
            RETURN
         ENDIF
      ENDIF
C
      IF (IP(2).GT.0) THEN
         LCHK = MG + 2*N + 2
         IF (IP(2).LT.LCHK) THEN
C            WRITE (XERN1, '(I8)') LCHK
C            CALL XERMSG ('SLATEC', 'DLSEI', 'INSUFFICIENT STORAGE ' //
C     *         'ALLOCATED FOR IP(*), NEED LIP = ' // XERN1, 2, 1)
            RETURN
         ENDIF
      ENDIF
C
C     Compute number of possible right multiplying Householder
C     transformations.
C
      M = ME + MA + MG
      IF (N.LE.0 .OR. M.LE.0) THEN
         MODE = 0
         RNORME = 0
         RNORML = 0
         RETURN
      ENDIF
C
      IF (MDW.LT.M) THEN
C         CALL XERMSG ('SLATEC', 'DLSEI', 'MDW.LT.ME+MA+MG IS AN ERROR',
C     +      2, 1)
         RETURN
      ENDIF
C
      NP1 = N + 1
      KRANKE = MIN(ME,N)
      N1 = 2*KRANKE + 1
      N2 = N1 + N
C
C     Set nominal values.
C
C     The nominal column scaling used in the code is
C     the identity scaling.
C
      CALL DCOPY (N, 1.D0, 0, WS(N1), 1)
C
C     No covariance matrix is nominally computed.
C
      COV = .FALSE.
C
C     Process option vector.
C     Define bound for number of options to change.
C
      NOPT = 1000
      NTIMES = 0
C
C     Define bound for positive values of LINK.
C
      NLINK = 100000
      LAST = 1
      LINK = PRGOPT(1)
      IF (LINK.EQ.0 .OR. LINK.GT.NLINK) THEN
C         CALL XERMSG ('SLATEC', 'DLSEI',
C     +      'THE OPTION VECTOR IS UNDEFINED', 2, 1)
         RETURN
      ENDIF
C
  100 IF (LINK.GT.1) THEN
         NTIMES = NTIMES + 1
         IF (NTIMES.GT.NOPT) THEN
C            CALL XERMSG ('SLATEC', 'DLSEI',
C     +         'THE LINKS IN THE OPTION VECTOR ARE CYCLING.', 2, 1)
            RETURN
         ENDIF
C
         KEY = PRGOPT(LAST+1)
         IF (KEY.EQ.1) THEN
            COV = PRGOPT(LAST+2) .NE. 0.D0
         ELSEIF (KEY.EQ.2 .AND. PRGOPT(LAST+2).NE.0.D0) THEN
            DO 110 J = 1,N
               T = DNRM2(M,W(1,J),1)
               IF (T.NE.0.D0) T = 1.D0/T
               WS(J+N1-1) = T
  110       CONTINUE
         ELSEIF (KEY.EQ.3) THEN
            CALL DCOPY (N, PRGOPT(LAST+2), 1, WS(N1), 1)
         ELSEIF (KEY.EQ.4) THEN
            TAU = MAX(DRELPR,PRGOPT(LAST+2))
         ENDIF
C
         NEXT = PRGOPT(LINK)
         IF (NEXT.LE.0 .OR. NEXT.GT.NLINK) THEN
C         CALL XERMSG ('SLATEC', 'DLSEI',
C     +      'THE OPTION VECTOR IS UNDEFINED', 2, 1)
            RETURN
         ENDIF
C
         LAST = LINK
         LINK = NEXT
         GO TO 100
      ENDIF
C
      DO 120 J = 1,N
         CALL DSCAL (M, WS(N1+J-1), W(1,J), 1)
  120 CONTINUE
C
      IF (COV .AND. MDW.LT.N) THEN
C         CALL XERMSG ('SLATEC', 'DLSEI',
C     +      'MDW .LT. N WHEN COV MATRIX NEEDED, IS AN ERROR', 2, 1)
         RETURN
      ENDIF
C
C     Problem definition and option vector OK.
C
      MODE = 0
C
C     Compute norm of equality constraint matrix and right side.
C
      ENORM = 0.D0
      DO 130 J = 1,N
         ENORM = MAX(ENORM,DASUM(ME,W(1,J),1))
  130 CONTINUE
C
      FNORM = DASUM(ME,W(1,NP1),1)
      SNMAX = 0.D0
      RNMAX = 0.D0
      DO 150 I = 1,KRANKE
C
C        Compute maximum ratio of vector lengths. Partition is at
C        column I.
C
         DO 140 K = I,ME
            SN = DDOT(N-I+1,W(K,I),MDW,W(K,I),MDW)
            RN = DDOT(I-1,W(K,1),MDW,W(K,1),MDW)
            IF (RN.EQ.0.D0 .AND. SN.GT.SNMAX) THEN
               SNMAX = SN
               IMAX = K
            ELSEIF (K.EQ.I .OR. SN*RNMAX.GT.RN*SNMAX) THEN
               SNMAX = SN
               RNMAX = RN
               IMAX = K
            ENDIF
  140    CONTINUE
C
C        Interchange rows if necessary.
C
         IF (I.NE.IMAX) CALL DSWAP (NP1, W(I,1), MDW, W(IMAX,1), MDW)
         IF (SNMAX.GT.RNMAX*TAU**2) THEN
C
C        Eliminate elements I+1,...,N in row I.
C
            CALL DH12 (1, I, I+1, N, W(I,1), MDW, WS(I), W(I+1,1), MDW,
     +                1, M-I)
         ELSE
            KRANKE = I - 1
            GO TO 160
         ENDIF
  150 CONTINUE
C
C     Save diagonal terms of lower trapezoidal matrix.
C
  160 CALL DCOPY (KRANKE, W, MDW+1, WS(KRANKE+1), 1)
C
C     Use Householder transformation from left to achieve
C     KRANKE by KRANKE upper triangular form.
C
      IF (KRANKE.LT.ME) THEN
         DO 170 K = KRANKE,1,-1
C
C           Apply transformation to matrix cols. 1,...,K-1.
C
            CALL DH12 (1, K, KRANKE+1, ME, W(1,K), 1, UP, W, 1, MDW,
     *         K-1)
C
C           Apply to rt side vector.
C
            CALL DH12 (2, K, KRANKE+1, ME, W(1,K), 1, UP, W(1,NP1), 1,
     +         1, 1)
  170    CONTINUE
      ENDIF
C
C     Solve for variables 1,...,KRANKE in new coordinates.
C
      CALL DCOPY (KRANKE, W(1, NP1), 1, X, 1)
      DO 180 I = 1,KRANKE
         X(I) = (X(I)-DDOT(I-1,W(I,1),MDW,X,1))/W(I,I)
  180 CONTINUE
C
C     Compute residuals for reduced problem.
C
      MEP1 = ME + 1
      RNORML = 0.D0
      DO 190 I = MEP1,M
         W(I,NP1) = W(I,NP1) - DDOT(KRANKE,W(I,1),MDW,X,1)
         SN = DDOT(KRANKE,W(I,1),MDW,W(I,1),MDW)
         RN = DDOT(N-KRANKE,W(I,KRANKE+1),MDW,W(I,KRANKE+1),MDW)
         IF (RN.LE.SN*TAU**2 .AND. KRANKE.LT.N)
     *      CALL DCOPY (N-KRANKE, 0.D0, 0, W(I,KRANKE+1), MDW)
  190 CONTINUE
C
C     Compute equality constraint equations residual length.
C
      RNORME = DNRM2(ME-KRANKE,W(KRANKE+1,NP1),1)
C
C     Move reduced problem data upward if KRANKE.LT.ME.
C
      IF (KRANKE.LT.ME) THEN
         DO 200 J = 1,NP1
            CALL DCOPY (M-ME, W(ME+1,J), 1, W(KRANKE+1,J), 1)
  200    CONTINUE
      ENDIF
C
C     Compute solution of reduced problem.
C
      CALL DLSI(W(KRANKE+1, KRANKE+1), MDW, MA, MG, N-KRANKE, PRGOPT,
     +         X(KRANKE+1), RNORML, MODE, WS(N2), IP(2))
C
C     Test for consistency of equality constraints.
C
      IF (ME.GT.0) THEN
         MDEQC = 0
         XNRME = DASUM(KRANKE,W(1,NP1),1)
         IF (RNORME.GT.TAU*(ENORM*XNRME+FNORM)) MDEQC = 1
         MODE = MODE + MDEQC
C
C        Check if solution to equality constraints satisfies inequality
C        constraints when there are no degrees of freedom left.
C
         IF (KRANKE.EQ.N .AND. MG.GT.0) THEN
            XNORM = DASUM(N,X,1)
            MAPKE1 = MA + KRANKE + 1
            MEND = MA + KRANKE + MG
            DO 210 I = MAPKE1,MEND
               SIZE = DASUM(N,W(I,1),MDW)*XNORM + ABS(W(I,NP1))
               IF (W(I,NP1).GT.TAU*SIZE) THEN
                  MODE = MODE + 2
                  GO TO 290
               ENDIF
  210       CONTINUE
         ENDIF
      ENDIF
C
C     Replace diagonal terms of lower trapezoidal matrix.
C
      IF (KRANKE.GT.0) THEN
         CALL DCOPY (KRANKE, WS(KRANKE+1), 1, W, MDW+1)
C
C        Reapply transformation to put solution in original coordinates.
C
         DO 220 I = KRANKE,1,-1
            CALL DH12 (2, I, I+1, N, W(I,1), MDW, WS(I), X, 1, 1, 1)
  220    CONTINUE
C
C        Compute covariance matrix of equality constrained problem.
C
         IF (COV) THEN
            DO 270 J = MIN(KRANKE,N-1),1,-1
               RB = WS(J)*W(J,J)
               IF (RB.NE.0.D0) RB = 1.D0/RB
               JP1 = J + 1
               DO 230 I = JP1,N
                  W(I,J) = RB*DDOT(N-J,W(I,JP1),MDW,W(J,JP1),MDW)
  230          CONTINUE
C
               GAM = 0.5D0*RB*DDOT(N-J,W(JP1,J),1,W(J,JP1),MDW)
               CALL DAXPY (N-J, GAM, W(J,JP1), MDW, W(JP1,J), 1)
               DO 250 I = JP1,N
                  DO 240 K = I,N
                     W(I,K) = W(I,K) + W(J,I)*W(K,J) + W(I,J)*W(J,K)
                     W(K,I) = W(I,K)
  240             CONTINUE
  250          CONTINUE
               UJ = WS(J)
               VJ = GAM*UJ
               W(J,J) = UJ*VJ + UJ*VJ
               DO 260 I = JP1,N
                  W(J,I) = UJ*W(I,J) + VJ*W(J,I)
  260          CONTINUE
               CALL DCOPY (N-J, W(J, JP1), MDW, W(JP1,J), 1)
  270       CONTINUE
         ENDIF
      ENDIF
C
C     Apply the scaling to the covariance matrix.
C
      IF (COV) THEN
         DO 280 I = 1,N
            CALL DSCAL (N, WS(I+N1-1), W(I,1), MDW)
            CALL DSCAL (N, WS(I+N1-1), W(1,I), 1)
  280    CONTINUE
      ENDIF
C
C     Rescale solution vector.
C
  290 IF (MODE.LE.1) THEN
         DO 300 J = 1,N
            X(J) = X(J)*WS(N1+J-1)
  300    CONTINUE
      ENDIF
C
      IP(1) = KRANKE
      IP(3) = IP(3) + 2*KRANKE + N
      RETURN
      END
*DECK DLSI
      SUBROUTINE DLSI (W, MDW, MA, MG, N, PRGOPT, X, RNORM, MODE, WS,
     +   IP)
C***BEGIN PROLOGUE  DLSI
C***SUBSIDIARY
C***PURPOSE  Subsidiary to DLSEI
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (LSI-S, DLSI-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C***DESCRIPTION
C
C     This is a companion subprogram to DLSEI.  The documentation for
C     DLSEI has complete usage instructions.
C
C     Solve..
C              AX = B,  A  MA by N  (least squares equations)
C     subject to..
C
C              GX.GE.H, G  MG by N  (inequality constraints)
C
C     Input..
C
C      W(*,*) contains  (A B) in rows 1,...,MA+MG, cols 1,...,N+1.
C                       (G H)
C
C     MDW,MA,MG,N
C              contain (resp) var. dimension of W(*,*),
C              and matrix dimensions.
C
C     PRGOPT(*),
C              Program option vector.
C
C     OUTPUT..
C
C      X(*),RNORM
C
C              Solution vector(unless MODE=2), length of AX-B.
C
C      MODE
C              =0   Inequality constraints are compatible.
C              =2   Inequality constraints contradictory.
C
C      WS(*),
C              Working storage of dimension K+N+(MG+2)*(N+7),
C              where K=MAX(MA+MG,N).
C      IP(MG+2*N+1)
C              Integer working storage
C
C***ROUTINES CALLED  D1MACH, DASUM, DAXPY, DCOPY, DDOT, DH12, DHFTI,
C                    DLPDP, DSCAL, DSWAP
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890618  Completely restructured and extensively revised (WRB & RWC)
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900328  Added TYPE section.  (WRB)
C   900604  DP version created from SP version.  (RWC)
C   920422  Changed CALL to DHFTI to include variable MA.  (WRB)
C***END PROLOGUE  DLSI
      USE REAL_PRECISION

      INTEGER IP(*), MA, MDW, MG, MODE, N
      REAL(KIND=R8) PRGOPT(*), RNORM, W(MDW,*), WS(*), X(*)
C
      EXTERNAL D1MACH, DASUM, DAXPY, DCOPY, DDOT, DH12, DHFTI, DLPDP,
     *   DSCAL, DSWAP
      REAL(KIND=R8) D1MACH, DASUM, DDOT
C
      REAL(KIND=R8) ANORM, DRELPR, FAC, GAM, RB, TAU, TOL, XNORM,
     *   TMP_NORM(1)
      INTEGER I, J, K, KEY, KRANK, KRM1, KRP1, L, LAST, LINK, M, MAP1,
     *   MDLPDP, MINMAN, N1, N2, N3, NEXT, NP1
      LOGICAL COV, FIRST, SCLCOV
C
      SAVE DRELPR, FIRST
      DATA FIRST /.TRUE./
C
C***FIRST EXECUTABLE STATEMENT  DLSI
C
C     Set the nominal tolerance used in the code.
C
      IF (FIRST) DRELPR = D1MACH(4)
      FIRST = .FALSE.
      TOL = SQRT(DRELPR)
C
      MODE = 0
      RNORM = 0.D0
      M = MA + MG
      NP1 = N + 1
      KRANK = 0
      IF (N.LE.0 .OR. M.LE.0) GO TO 370
C
C     To process option vector.
C
      COV = .FALSE.
      SCLCOV = .TRUE.
      LAST = 1
      LINK = PRGOPT(1)
C
  100 IF (LINK.GT.1) THEN
         KEY = PRGOPT(LAST+1)
         IF (KEY.EQ.1) COV = PRGOPT(LAST+2) .NE. 0.D0
         IF (KEY.EQ.10) SCLCOV = PRGOPT(LAST+2) .EQ. 0.D0
         IF (KEY.EQ.5) TOL = MAX(DRELPR,PRGOPT(LAST+2))
         NEXT = PRGOPT(LINK)
         LAST = LINK
         LINK = NEXT
         GO TO 100
      ENDIF
C
C     Compute matrix norm of least squares equations.
C
      ANORM = 0.D0
      DO 110 J = 1,N
         ANORM = MAX(ANORM,DASUM(MA,W(1,J),1))
  110 CONTINUE
C
C     Set tolerance for DHFTI( ) rank test.
C
      TAU = TOL*ANORM
C
C     Compute Householder orthogonal decomposition of matrix.
C
      CALL DCOPY (N, 0.D0, 0, WS, 1)
      CALL DCOPY (MA, W(1, NP1), 1, WS, 1)
      K = MAX(M,N)
      MINMAN = MIN(MA,N)
      N1 = K + 1
      N2 = N1 + N
      CALL DHFTI (W, MDW, MA, N, WS, MA, 1, TAU, KRANK, TMP_NORM,
     +           WS(N2), WS(N1), IP)
      RNORM = TMP_NORM(1)
      FAC = 1.D0
      GAM = MA - KRANK
      IF (KRANK.LT.MA .AND. SCLCOV) FAC = RNORM**2/GAM
C
C     Reduce to DLPDP and solve.
C
      MAP1 = MA + 1
C
C     Compute inequality rt-hand side for DLPDP.
C
      IF (MA.LT.M) THEN
         IF (MINMAN.GT.0) THEN
            DO 120 I = MAP1,M
               W(I,NP1) = W(I,NP1) - DDOT(N,W(I,1),MDW,WS,1)
  120       CONTINUE
C
C           Apply permutations to col. of inequality constraint matrix.
C
            DO 130 I = 1,MINMAN
               CALL DSWAP (MG, W(MAP1,I), 1, W(MAP1,IP(I)), 1)
  130       CONTINUE
C
C           Apply Householder transformations to constraint matrix.
C
            IF (KRANK.GT.0 .AND. KRANK.LT.N) THEN
               DO 140 I = KRANK,1,-1
                  CALL DH12 (2, I, KRANK+1, N, W(I,1), MDW, WS(N1+I-1),
     +                      W(MAP1,1), MDW, 1, MG)
  140          CONTINUE
            ENDIF
C
C           Compute permuted inequality constraint matrix times r-inv.
C
            DO 160 I = MAP1,M
               DO 150 J = 1,KRANK
                  W(I,J) = (W(I,J)-DDOT(J-1,W(1,J),1,W(I,1),MDW))/W(J,J)
  150          CONTINUE
  160       CONTINUE
         ENDIF
C
C        Solve the reduced problem with DLPDP algorithm,
C        the least projected distance problem.
C
         CALL DLPDP(W(MAP1,1), MDW, MG, KRANK, N-KRANK, PRGOPT, X,
     +             XNORM, MDLPDP, WS(N2), IP(N+1))
C
C        Compute solution in original coordinates.
C
         IF (MDLPDP.EQ.1) THEN
            DO 170 I = KRANK,1,-1
               X(I) = (X(I)-DDOT(KRANK-I,W(I,I+1),MDW,X(I+1),1))/W(I,I)
  170       CONTINUE
C
C           Apply Householder transformation to solution vector.
C
            IF (KRANK.LT.N) THEN
               DO 180 I = 1,KRANK
                  CALL DH12 (2, I, KRANK+1, N, W(I,1), MDW, WS(N1+I-1),
     +                      X, 1, 1, 1)
  180          CONTINUE
            ENDIF
C
C           Repermute variables to their input order.
C
            IF (MINMAN.GT.0) THEN
               DO 190 I = MINMAN,1,-1
                  CALL DSWAP (1, X(I), 1, X(IP(I)), 1)
  190          CONTINUE
C
C              Variables are now in original coordinates.
C              Add solution of unconstrained problem.
C
               DO 200 I = 1,N
                  X(I) = X(I) + WS(I)
  200          CONTINUE
C
C              Compute the residual vector norm.
C
               RNORM = SQRT(RNORM**2+XNORM**2)
            ENDIF
         ELSE
            MODE = 2
         ENDIF
      ELSE
         CALL DCOPY (N, WS, 1, X, 1)
      ENDIF
C
C     Compute covariance matrix based on the orthogonal decomposition
C     from DHFTI( ).
C
      IF (.NOT.COV .OR. KRANK.LE.0) GO TO 370
      KRM1 = KRANK - 1
      KRP1 = KRANK + 1
C
C     Copy diagonal terms to working array.
C
      CALL DCOPY (KRANK, W, MDW+1, WS(N2), 1)
C
C     Reciprocate diagonal terms.
C
      DO 210 J = 1,KRANK
         W(J,J) = 1.D0/W(J,J)
  210 CONTINUE
C
C     Invert the upper triangular QR factor on itself.
C
      IF (KRANK.GT.1) THEN
         DO 230 I = 1,KRM1
            DO 220 J = I+1,KRANK
               W(I,J) = -DDOT(J-I,W(I,I),MDW,W(I,J),1)*W(J,J)
  220       CONTINUE
  230    CONTINUE
      ENDIF
C
C     Compute the inverted factor times its transpose.
C
      DO 250 I = 1,KRANK
         DO 240 J = I,KRANK
            W(I,J) = DDOT(KRANK+1-J,W(I,J),MDW,W(J,J),MDW)
  240    CONTINUE
  250 CONTINUE
C
C     Zero out lower trapezoidal part.
C     Copy upper triangular to lower triangular part.
C
      IF (KRANK.LT.N) THEN
         DO 260 J = 1,KRANK
            CALL DCOPY (J, W(1,J), 1, W(J,1), MDW)
  260    CONTINUE
C
         DO 270 I = KRP1,N
            CALL DCOPY (I, 0.D0, 0, W(I,1), MDW)
  270    CONTINUE
C
C        Apply right side transformations to lower triangle.
C
         N3 = N2 + KRP1
         DO 330 I = 1,KRANK
            L = N1 + I
            K = N2 + I
            RB = WS(L-1)*WS(K-1)
C
C           If RB.GE.0.D0, transformation can be regarded as zero.
C
            IF (RB.LT.0.D0) THEN
               RB = 1.D0/RB
C
C              Store unscaled rank one Householder update in work array.
C
               CALL DCOPY (N, 0.D0, 0, WS(N3), 1)
               L = N1 + I
               K = N3 + I
               WS(K-1) = WS(L-1)
C
               DO 280 J = KRP1,N
                  WS(N3+J-1) = W(I,J)
  280          CONTINUE
C
               DO 290 J = 1,N
                  WS(J) = RB*(DDOT(J-I,W(J,I),MDW,WS(N3+I-1),1)+
     +                    DDOT(N-J+1,W(J,J),1,WS(N3+J-1),1))
  290          CONTINUE
C
               L = N3 + I
               GAM = 0.5D0*RB*DDOT(N-I+1,WS(L-1),1,WS(I),1)
               CALL DAXPY (N-I+1, GAM, WS(L-1), 1, WS(I), 1)
               DO 320 J = I,N
                  DO 300 L = 1,I-1
                     W(J,L) = W(J,L) + WS(N3+J-1)*WS(L)
  300             CONTINUE
C
                  DO 310 L = I,J
                     W(J,L) = W(J,L) + WS(J)*WS(N3+L-1)+WS(L)*WS(N3+J-1)
  310             CONTINUE
  320          CONTINUE
            ENDIF
  330    CONTINUE
C
C        Copy lower triangle to upper triangle to symmetrize the
C        covariance matrix.
C
         DO 340 I = 1,N
            CALL DCOPY (I, W(I,1), MDW, W(1,I), 1)
  340    CONTINUE
      ENDIF
C
C     Repermute rows and columns.
C
      DO 350 I = MINMAN,1,-1
         K = IP(I)
         IF (I.NE.K) THEN
            CALL DSWAP (1, W(I,I), 1, W(K,K), 1)
            CALL DSWAP (I-1, W(1,I), 1, W(1,K), 1)
            CALL DSWAP (K-I-1, W(I,I+1), MDW, W(I+1,K), 1)
            CALL DSWAP (N-K, W(I, K+1), MDW, W(K, K+1), MDW)
         ENDIF
  350 CONTINUE
C
C     Put in normalized residual sum of squares scale factor
C     and symmetrize the resulting covariance matrix.
C
      DO 360 J = 1,N
         CALL DSCAL (J, FAC, W(1,J), 1)
         CALL DCOPY (J, W(1,J), 1, W(J,1), MDW)
  360 CONTINUE
C
  370 IP(1) = KRANK
      IP(2) = N + MAX(M,N) + (MG+2)*(N+7)
      RETURN
      END
*DECK D1MACH
      REAL(KIND=R8) FUNCTION D1MACH (I)
C***BEGIN PROLOGUE  D1MACH
C***PURPOSE  Return floating point machine dependent constants.
C***LIBRARY   SLATEC
C***CATEGORY  R1
C***TYPE      REAL(KIND=R8) (R1MACH-S, D1MACH-D)
C***KEYWORDS  MACHINE CONSTANTS
C***AUTHOR  Fox, P. A., (Bell Labs)
C           Hall, A. D., (Bell Labs)
C           Schryer, N. L., (Bell Labs)
C***DESCRIPTION
C
C   D1MACH can be used to obtain machine-dependent parameters for the
C   local machine environment.  It is a function subprogram with one
C   (input) argument, and can be referenced as follows:
C
C        D = D1MACH(I)
C
C   where I=1,...,5.  The (output) value of D above is determined by
C   the (input) value of I.  The results for various values of I are
C   discussed below.
C
C   D1MACH( 1) = B**(EMIN-1), the smallest positive magnitude.
C   D1MACH( 2) = B**EMAX*(1 - B**(-T)), the largest magnitude.
C   D1MACH( 3) = B**(-T), the smallest relative spacing.
C   D1MACH( 4) = B**(1-T), the largest relative spacing.
C   D1MACH( 5) = LOG10(B)
C
C   Assume double precision numbers are represented in the T-digit,
C   base-B form
C
C              sign (B**E)*( (X(1)/B) + ... + (X(T)/B**T) )
C
C   where 0 .LE. X(I) .LT. B for I=1,...,T, 0 .LT. X(1), and
C   EMIN .LE. E .LE. EMAX.
C
C   The values of B, T, EMIN and EMAX are provided in I1MACH as
C   follows:
C   I1MACH(10) = B, the base.
C   I1MACH(14) = T, the number of base-B digits.
C   I1MACH(15) = EMIN, the smallest exponent E.
C   I1MACH(16) = EMAX, the largest exponent E.
C
C   To alter this function for a particular environment, the desired
C   set of DATA statements should be activated by removing the C from
C   column 1.  Also, the values of D1MACH(1) - D1MACH(4) should be
C   checked for consistency with the local operating system.
C
C***REFERENCES  P. A. Fox, A. D. Hall and N. L. Schryer, Framework for
C                 a portable library, ACM Transactions on Mathematical
C                 Software 4, 2 (June 1978), pp. 177-188.
C***ROUTINES CALLED  XERMSG
C***REVISION HISTORY  (YYMMDD)
C   750101  DATE WRITTEN
C   890213  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900618  Added DEC RISC constants.  (WRB)
C   900723  Added IBM RS 6000 constants.  (WRB)
C   900911  Added SUN 386i constants.  (WRB)
C   910710  Added HP 730 constants.  (SMR)
C   911114  Added Convex IEEE constants.  (WRB)
C   920121  Added SUN -r8 compiler option constants.  (WRB)
C   920229  Added Touchstone Delta i860 constants.  (WRB)
C   920501  Reformatted the REFERENCES section.  (WRB)
C   920625  Added CONVEX -p8 and -pd8 compiler option constants.
C           (BKS, WRB)
C   930201  Added DEC Alpha and SGI constants.  (RWC and WRB)
C   010817  Elevated IEEE to highest importance; see next set of
C           comments below.  (DWL)
C***END PROLOGUE  D1MACH
C
      USE REAL_PRECISION

      INTEGER SMALL(4)
      INTEGER LARGE(4)
      INTEGER RIGHT(4)
      INTEGER DIVER(4)
      INTEGER LOG10(4)
C
C Initial data here correspond to the IEEE standard.  The values for
C DMACH(1), DMACH(3) and DMACH(4) are slight upper bounds.  The value
C for DMACH(2) is a slight lower bound.  The value for DMACH(5) is
C a 20-digit approximation.  If one of the sets of initial data below
C is preferred, do the necessary commenting and uncommenting. (DWL)
      REAL(KIND=R8) DMACH(5)
      DATA DMACH / 2.23D-308, 1.79D+308, 1.111D-16, 2.222D-16,
     1             0.30102999566398119521D0 /
      SAVE DMACH
C
      EQUIVALENCE (DMACH(1),SMALL(1))
      EQUIVALENCE (DMACH(2),LARGE(1))
      EQUIVALENCE (DMACH(3),RIGHT(1))
      EQUIVALENCE (DMACH(4),DIVER(1))
      EQUIVALENCE (DMACH(5),LOG10(1))
C
C     MACHINE CONSTANTS FOR THE AMIGA
C     ABSOFT FORTRAN COMPILER USING THE 68020/68881 COMPILER OPTION
C
C     DATA SMALL(1), SMALL(2) / Z'00100000', Z'00000000' /
C     DATA LARGE(1), LARGE(2) / Z'7FEFFFFF', Z'FFFFFFFF' /
C     DATA RIGHT(1), RIGHT(2) / Z'3CA00000', Z'00000000' /
C     DATA DIVER(1), DIVER(2) / Z'3CB00000', Z'00000000' /
C     DATA LOG10(1), LOG10(2) / Z'3FD34413', Z'509F79FF' /
C
C     MACHINE CONSTANTS FOR THE AMIGA
C     ABSOFT FORTRAN COMPILER USING SOFTWARE FLOATING POINT
C
C     DATA SMALL(1), SMALL(2) / Z'00100000', Z'00000000' /
C     DATA LARGE(1), LARGE(2) / Z'7FDFFFFF', Z'FFFFFFFF' /
C     DATA RIGHT(1), RIGHT(2) / Z'3CA00000', Z'00000000' /
C     DATA DIVER(1), DIVER(2) / Z'3CB00000', Z'00000000' /
C     DATA LOG10(1), LOG10(2) / Z'3FD34413', Z'509F79FF' /
C
C     MACHINE CONSTANTS FOR THE APOLLO
C
C     DATA SMALL(1), SMALL(2) / 16#00100000, 16#00000000 /
C     DATA LARGE(1), LARGE(2) / 16#7FFFFFFF, 16#FFFFFFFF /
C     DATA RIGHT(1), RIGHT(2) / 16#3CA00000, 16#00000000 /
C     DATA DIVER(1), DIVER(2) / 16#3CB00000, 16#00000000 /
C     DATA LOG10(1), LOG10(2) / 16#3FD34413, 16#509F79FF /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 1700 SYSTEM
C
C     DATA SMALL(1) / ZC00800000 /
C     DATA SMALL(2) / Z000000000 /
C     DATA LARGE(1) / ZDFFFFFFFF /
C     DATA LARGE(2) / ZFFFFFFFFF /
C     DATA RIGHT(1) / ZCC5800000 /
C     DATA RIGHT(2) / Z000000000 /
C     DATA DIVER(1) / ZCC6800000 /
C     DATA DIVER(2) / Z000000000 /
C     DATA LOG10(1) / ZD00E730E7 /
C     DATA LOG10(2) / ZC77800DC0 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 5700 SYSTEM
C
C     DATA SMALL(1) / O1771000000000000 /
C     DATA SMALL(2) / O0000000000000000 /
C     DATA LARGE(1) / O0777777777777777 /
C     DATA LARGE(2) / O0007777777777777 /
C     DATA RIGHT(1) / O1461000000000000 /
C     DATA RIGHT(2) / O0000000000000000 /
C     DATA DIVER(1) / O1451000000000000 /
C     DATA DIVER(2) / O0000000000000000 /
C     DATA LOG10(1) / O1157163034761674 /
C     DATA LOG10(2) / O0006677466732724 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 6700/7700 SYSTEMS
C
C     DATA SMALL(1) / O1771000000000000 /
C     DATA SMALL(2) / O7770000000000000 /
C     DATA LARGE(1) / O0777777777777777 /
C     DATA LARGE(2) / O7777777777777777 /
C     DATA RIGHT(1) / O1461000000000000 /
C     DATA RIGHT(2) / O0000000000000000 /
C     DATA DIVER(1) / O1451000000000000 /
C     DATA DIVER(2) / O0000000000000000 /
C     DATA LOG10(1) / O1157163034761674 /
C     DATA LOG10(2) / O0006677466732724 /
C
C     MACHINE CONSTANTS FOR THE CDC 170/180 SERIES USING NOS/VE
C
C     DATA SMALL(1) / Z"3001800000000000" /
C     DATA SMALL(2) / Z"3001000000000000" /
C     DATA LARGE(1) / Z"4FFEFFFFFFFFFFFE" /
C     DATA LARGE(2) / Z"4FFE000000000000" /
C     DATA RIGHT(1) / Z"3FD2800000000000" /
C     DATA RIGHT(2) / Z"3FD2000000000000" /
C     DATA DIVER(1) / Z"3FD3800000000000" /
C     DATA DIVER(2) / Z"3FD3000000000000" /
C     DATA LOG10(1) / Z"3FFF9A209A84FBCF" /
C     DATA LOG10(2) / Z"3FFFF7988F8959AC" /
C
C     MACHINE CONSTANTS FOR THE CDC 6000/7000 SERIES
C
C     DATA SMALL(1) / 00564000000000000000B /
C     DATA SMALL(2) / 00000000000000000000B /
C     DATA LARGE(1) / 37757777777777777777B /
C     DATA LARGE(2) / 37157777777777777777B /
C     DATA RIGHT(1) / 15624000000000000000B /
C     DATA RIGHT(2) / 00000000000000000000B /
C     DATA DIVER(1) / 15634000000000000000B /
C     DATA DIVER(2) / 00000000000000000000B /
C     DATA LOG10(1) / 17164642023241175717B /
C     DATA LOG10(2) / 16367571421742254654B /
C
C     MACHINE CONSTANTS FOR THE CELERITY C1260
C
C     DATA SMALL(1), SMALL(2) / Z'00100000', Z'00000000' /
C     DATA LARGE(1), LARGE(2) / Z'7FEFFFFF', Z'FFFFFFFF' /
C     DATA RIGHT(1), RIGHT(2) / Z'3CA00000', Z'00000000' /
C     DATA DIVER(1), DIVER(2) / Z'3CB00000', Z'00000000' /
C     DATA LOG10(1), LOG10(2) / Z'3FD34413', Z'509F79FF' /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -fn OR -pd8 COMPILER OPTION
C
C     DATA DMACH(1) / Z'0010000000000000' /
C     DATA DMACH(2) / Z'7FFFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3CC0000000000000' /
C     DATA DMACH(4) / Z'3CD0000000000000' /
C     DATA DMACH(5) / Z'3FF34413509F79FF' /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -fi COMPILER OPTION
C
C     DATA DMACH(1) / Z'0010000000000000' /
C     DATA DMACH(2) / Z'7FEFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3CA0000000000000' /
C     DATA DMACH(4) / Z'3CB0000000000000' /
C     DATA DMACH(5) / Z'3FD34413509F79FF' /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -p8 COMPILER OPTION
C
C     DATA DMACH(1) / Z'00010000000000000000000000000000' /
C     DATA DMACH(2) / Z'7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3F900000000000000000000000000000' /
C     DATA DMACH(4) / Z'3F910000000000000000000000000000' /
C     DATA DMACH(5) / Z'3FFF34413509F79FEF311F12B35816F9' /
C
C     MACHINE CONSTANTS FOR THE CRAY
C
C     DATA SMALL(1) / 201354000000000000000B /
C     DATA SMALL(2) / 000000000000000000000B /
C     DATA LARGE(1) / 577767777777777777777B /
C     DATA LARGE(2) / 000007777777777777774B /
C     DATA RIGHT(1) / 376434000000000000000B /
C     DATA RIGHT(2) / 000000000000000000000B /
C     DATA DIVER(1) / 376444000000000000000B /
C     DATA DIVER(2) / 000000000000000000000B /
C     DATA LOG10(1) / 377774642023241175717B /
C     DATA LOG10(2) / 000007571421742254654B /
C
C     MACHINE CONSTANTS FOR THE DATA GENERAL ECLIPSE S/200
C     NOTE - IT MAY BE APPROPRIATE TO INCLUDE THE FOLLOWING CARD -
C     STATIC DMACH(5)
C
C     DATA SMALL /    20K, 3*0 /
C     DATA LARGE / 77777K, 3*177777K /
C     DATA RIGHT / 31420K, 3*0 /
C     DATA DIVER / 32020K, 3*0 /
C     DATA LOG10 / 40423K, 42023K, 50237K, 74776K /
C
C     MACHINE CONSTANTS FOR THE DEC ALPHA
C     USING G_FLOAT
C
C     DATA DMACH(1) / '0000000000000010'X /
C     DATA DMACH(2) / 'FFFFFFFFFFFF7FFF'X /
C     DATA DMACH(3) / '0000000000003CC0'X /
C     DATA DMACH(4) / '0000000000003CD0'X /
C     DATA DMACH(5) / '79FF509F44133FF3'X /
C
C     MACHINE CONSTANTS FOR THE DEC ALPHA
C     USING IEEE_FORMAT
C
C     DATA DMACH(1) / '0010000000000000'X /
C     DATA DMACH(2) / '7FEFFFFFFFFFFFFF'X /
C     DATA DMACH(3) / '3CA0000000000000'X /
C     DATA DMACH(4) / '3CB0000000000000'X /
C     DATA DMACH(5) / '3FD34413509F79FF'X /
C
C     MACHINE CONSTANTS FOR THE DEC RISC
C
C     DATA SMALL(1), SMALL(2) / Z'00000000', Z'00100000'/
C     DATA LARGE(1), LARGE(2) / Z'FFFFFFFF', Z'7FEFFFFF'/
C     DATA RIGHT(1), RIGHT(2) / Z'00000000', Z'3CA00000'/
C     DATA DIVER(1), DIVER(2) / Z'00000000', Z'3CB00000'/
C     DATA LOG10(1), LOG10(2) / Z'509F79FF', Z'3FD34413'/
C
C     MACHINE CONSTANTS FOR THE DEC VAX
C     USING D_FLOATING
C     (EXPRESSED IN INTEGER AND HEXADECIMAL)
C     THE HEX FORMAT BELOW MAY NOT BE SUITABLE FOR UNIX SYSTEMS
C     THE INTEGER FORMAT SHOULD BE OK FOR UNIX SYSTEMS
C
C     DATA SMALL(1), SMALL(2) /        128,           0 /
C     DATA LARGE(1), LARGE(2) /     -32769,          -1 /
C     DATA RIGHT(1), RIGHT(2) /       9344,           0 /
C     DATA DIVER(1), DIVER(2) /       9472,           0 /
C     DATA LOG10(1), LOG10(2) /  546979738,  -805796613 /
C
C     DATA SMALL(1), SMALL(2) / Z00000080, Z00000000 /
C     DATA LARGE(1), LARGE(2) / ZFFFF7FFF, ZFFFFFFFF /
C     DATA RIGHT(1), RIGHT(2) / Z00002480, Z00000000 /
C     DATA DIVER(1), DIVER(2) / Z00002500, Z00000000 /
C     DATA LOG10(1), LOG10(2) / Z209A3F9A, ZCFF884FB /
C
C     MACHINE CONSTANTS FOR THE DEC VAX
C     USING G_FLOATING
C     (EXPRESSED IN INTEGER AND HEXADECIMAL)
C     THE HEX FORMAT BELOW MAY NOT BE SUITABLE FOR UNIX SYSTEMS
C     THE INTEGER FORMAT SHOULD BE OK FOR UNIX SYSTEMS
C
C     DATA SMALL(1), SMALL(2) /         16,           0 /
C     DATA LARGE(1), LARGE(2) /     -32769,          -1 /
C     DATA RIGHT(1), RIGHT(2) /      15552,           0 /
C     DATA DIVER(1), DIVER(2) /      15568,           0 /
C     DATA LOG10(1), LOG10(2) /  1142112243, 2046775455 /
C
C     DATA SMALL(1), SMALL(2) / Z00000010, Z00000000 /
C     DATA LARGE(1), LARGE(2) / ZFFFF7FFF, ZFFFFFFFF /
C     DATA RIGHT(1), RIGHT(2) / Z00003CC0, Z00000000 /
C     DATA DIVER(1), DIVER(2) / Z00003CD0, Z00000000 /
C     DATA LOG10(1), LOG10(2) / Z44133FF3, Z79FF509F /
C
C     MACHINE CONSTANTS FOR THE ELXSI 6400
C     (ASSUMING REAL*8 IS THE DEFAULT REAL(KIND=R8))
C
C     DATA SMALL(1), SMALL(2) / '00100000'X,'00000000'X /
C     DATA LARGE(1), LARGE(2) / '7FEFFFFF'X,'FFFFFFFF'X /
C     DATA RIGHT(1), RIGHT(2) / '3CB00000'X,'00000000'X /
C     DATA DIVER(1), DIVER(2) / '3CC00000'X,'00000000'X /
C     DATA LOG10(1), LOG10(2) / '3FD34413'X,'509F79FF'X /
C
C     MACHINE CONSTANTS FOR THE HARRIS 220
C
C     DATA SMALL(1), SMALL(2) / '20000000, '00000201 /
C     DATA LARGE(1), LARGE(2) / '37777777, '37777577 /
C     DATA RIGHT(1), RIGHT(2) / '20000000, '00000333 /
C     DATA DIVER(1), DIVER(2) / '20000000, '00000334 /
C     DATA LOG10(1), LOG10(2) / '23210115, '10237777 /
C
C     MACHINE CONSTANTS FOR THE HONEYWELL 600/6000 SERIES
C
C     DATA SMALL(1), SMALL(2) / O402400000000, O000000000000 /
C     DATA LARGE(1), LARGE(2) / O376777777777, O777777777777 /
C     DATA RIGHT(1), RIGHT(2) / O604400000000, O000000000000 /
C     DATA DIVER(1), DIVER(2) / O606400000000, O000000000000 /
C     DATA LOG10(1), LOG10(2) / O776464202324, O117571775714 /
C
C     MACHINE CONSTANTS FOR THE HP 730
C
C     DATA DMACH(1) / Z'0010000000000000' /
C     DATA DMACH(2) / Z'7FEFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3CA0000000000000' /
C     DATA DMACH(4) / Z'3CB0000000000000' /
C     DATA DMACH(5) / Z'3FD34413509F79FF' /
C
C     MACHINE CONSTANTS FOR THE HP 2100
C     THREE WORD REAL(KIND=R8) OPTION WITH FTN4
C
C     DATA SMALL(1), SMALL(2), SMALL(3) / 40000B,       0,       1 /
C     DATA LARGE(1), LARGE(2), LARGE(3) / 77777B, 177777B, 177776B /
C     DATA RIGHT(1), RIGHT(2), RIGHT(3) / 40000B,       0,    265B /
C     DATA DIVER(1), DIVER(2), DIVER(3) / 40000B,       0,    276B /
C     DATA LOG10(1), LOG10(2), LOG10(3) / 46420B,  46502B,  77777B /
C
C     MACHINE CONSTANTS FOR THE HP 2100
C     FOUR WORD REAL(KIND=R8) OPTION WITH FTN4
C
C     DATA SMALL(1), SMALL(2) /  40000B,       0 /
C     DATA SMALL(3), SMALL(4) /       0,       1 /
C     DATA LARGE(1), LARGE(2) /  77777B, 177777B /
C     DATA LARGE(3), LARGE(4) / 177777B, 177776B /
C     DATA RIGHT(1), RIGHT(2) /  40000B,       0 /
C     DATA RIGHT(3), RIGHT(4) /       0,    225B /
C     DATA DIVER(1), DIVER(2) /  40000B,       0 /
C     DATA DIVER(3), DIVER(4) /       0,    227B /
C     DATA LOG10(1), LOG10(2) /  46420B,  46502B /
C     DATA LOG10(3), LOG10(4) /  76747B, 176377B /
C
C     MACHINE CONSTANTS FOR THE HP 9000
C
C     DATA SMALL(1), SMALL(2) / 00040000000B, 00000000000B /
C     DATA LARGE(1), LARGE(2) / 17737777777B, 37777777777B /
C     DATA RIGHT(1), RIGHT(2) / 07454000000B, 00000000000B /
C     DATA DIVER(1), DIVER(2) / 07460000000B, 00000000000B /
C     DATA LOG10(1), LOG10(2) / 07764642023B, 12047674777B /
C
C     MACHINE CONSTANTS FOR THE IBM 360/370 SERIES,
C     THE XEROX SIGMA 5/7/9, THE SEL SYSTEMS 85/86, AND
C     THE PERKIN ELMER (INTERDATA) 7/32.
C
C     DATA SMALL(1), SMALL(2) / Z00100000, Z00000000 /
C     DATA LARGE(1), LARGE(2) / Z7FFFFFFF, ZFFFFFFFF /
C     DATA RIGHT(1), RIGHT(2) / Z33100000, Z00000000 /
C     DATA DIVER(1), DIVER(2) / Z34100000, Z00000000 /
C     DATA LOG10(1), LOG10(2) / Z41134413, Z509F79FF /
C
C     MACHINE CONSTANTS FOR THE IBM PC
C     ASSUMES THAT ALL ARITHMETIC IS DONE IN REAL(KIND=R8)
C     ON 8088, I.E., NOT IN 80 BIT FORM FOR THE 8087.
C
C     DATA SMALL(1) / 2.23D-308  /
C     DATA LARGE(1) / 1.79D+308  /
C     DATA RIGHT(1) / 1.11D-16   /
C     DATA DIVER(1) / 2.22D-16   /
C     DATA LOG10(1) / 0.301029995663981195D0 /
C
C     MACHINE CONSTANTS FOR THE IBM RS 6000
C
C     DATA DMACH(1) / Z'0010000000000000' /
C     DATA DMACH(2) / Z'7FEFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3CA0000000000000' /
C     DATA DMACH(4) / Z'3CB0000000000000' /
C     DATA DMACH(5) / Z'3FD34413509F79FF' /
C
C     MACHINE CONSTANTS FOR THE INTEL i860
C
C     DATA DMACH(1) / Z'0010000000000000' /
C     DATA DMACH(2) / Z'7FEFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3CA0000000000000' /
C     DATA DMACH(4) / Z'3CB0000000000000' /
C     DATA DMACH(5) / Z'3FD34413509F79FF' /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KA PROCESSOR)
C
C     DATA SMALL(1), SMALL(2) / "033400000000, "000000000000 /
C     DATA LARGE(1), LARGE(2) / "377777777777, "344777777777 /
C     DATA RIGHT(1), RIGHT(2) / "113400000000, "000000000000 /
C     DATA DIVER(1), DIVER(2) / "114400000000, "000000000000 /
C     DATA LOG10(1), LOG10(2) / "177464202324, "144117571776 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KI PROCESSOR)
C
C     DATA SMALL(1), SMALL(2) / "000400000000, "000000000000 /
C     DATA LARGE(1), LARGE(2) / "377777777777, "377777777777 /
C     DATA RIGHT(1), RIGHT(2) / "103400000000, "000000000000 /
C     DATA DIVER(1), DIVER(2) / "104400000000, "000000000000 /
C     DATA LOG10(1), LOG10(2) / "177464202324, "476747767461 /
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRAN SUPPORTING
C     32-BIT INTEGERS (EXPRESSED IN INTEGER AND OCTAL).
C
C     DATA SMALL(1), SMALL(2) /    8388608,           0 /
C     DATA LARGE(1), LARGE(2) / 2147483647,          -1 /
C     DATA RIGHT(1), RIGHT(2) /  612368384,           0 /
C     DATA DIVER(1), DIVER(2) /  620756992,           0 /
C     DATA LOG10(1), LOG10(2) / 1067065498, -2063872008 /
C
C     DATA SMALL(1), SMALL(2) / O00040000000, O00000000000 /
C     DATA LARGE(1), LARGE(2) / O17777777777, O37777777777 /
C     DATA RIGHT(1), RIGHT(2) / O04440000000, O00000000000 /
C     DATA DIVER(1), DIVER(2) / O04500000000, O00000000000 /
C     DATA LOG10(1), LOG10(2) / O07746420232, O20476747770 /
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRAN SUPPORTING
C     16-BIT INTEGERS (EXPRESSED IN INTEGER AND OCTAL).
C
C     DATA SMALL(1), SMALL(2) /    128,      0 /
C     DATA SMALL(3), SMALL(4) /      0,      0 /
C     DATA LARGE(1), LARGE(2) /  32767,     -1 /
C     DATA LARGE(3), LARGE(4) /     -1,     -1 /
C     DATA RIGHT(1), RIGHT(2) /   9344,      0 /
C     DATA RIGHT(3), RIGHT(4) /      0,      0 /
C     DATA DIVER(1), DIVER(2) /   9472,      0 /
C     DATA DIVER(3), DIVER(4) /      0,      0 /
C     DATA LOG10(1), LOG10(2) /  16282,   8346 /
C     DATA LOG10(3), LOG10(4) / -31493, -12296 /
C
C     DATA SMALL(1), SMALL(2) / O000200, O000000 /
C     DATA SMALL(3), SMALL(4) / O000000, O000000 /
C     DATA LARGE(1), LARGE(2) / O077777, O177777 /
C     DATA LARGE(3), LARGE(4) / O177777, O177777 /
C     DATA RIGHT(1), RIGHT(2) / O022200, O000000 /
C     DATA RIGHT(3), RIGHT(4) / O000000, O000000 /
C     DATA DIVER(1), DIVER(2) / O022400, O000000 /
C     DATA DIVER(3), DIVER(4) / O000000, O000000 /
C     DATA LOG10(1), LOG10(2) / O037632, O020232 /
C     DATA LOG10(3), LOG10(4) / O102373, O147770 /
C
C     MACHINE CONSTANTS FOR THE SILICON GRAPHICS
C
C     DATA SMALL(1), SMALL(2) / Z'00100000', Z'00000000' /
C     DATA LARGE(1), LARGE(2) / Z'7FEFFFFF', Z'FFFFFFFF' /
C     DATA RIGHT(1), RIGHT(2) / Z'3CA00000', Z'00000000' /
C     DATA DIVER(1), DIVER(2) / Z'3CB00000', Z'00000000' /
C     DATA LOG10(1), LOG10(2) / Z'3FD34413', Z'509F79FF' /
C
C     MACHINE CONSTANTS FOR THE SUN
C
C     DATA DMACH(1) / Z'0010000000000000' /
C     DATA DMACH(2) / Z'7FEFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3CA0000000000000' /
C     DATA DMACH(4) / Z'3CB0000000000000' /
C     DATA DMACH(5) / Z'3FD34413509F79FF' /
C
C     MACHINE CONSTANTS FOR THE SUN
C     USING THE -r8 COMPILER OPTION
C
C     DATA DMACH(1) / Z'00010000000000000000000000000000' /
C     DATA DMACH(2) / Z'7FFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF' /
C     DATA DMACH(3) / Z'3F8E0000000000000000000000000000' /
C     DATA DMACH(4) / Z'3F8F0000000000000000000000000000' /
C     DATA DMACH(5) / Z'3FFD34413509F79FEF311F12B35816F9' /
C
C     MACHINE CONSTANTS FOR THE SUN 386i
C
C     DATA SMALL(1), SMALL(2) / Z'FFFFFFFD', Z'000FFFFF' /
C     DATA LARGE(1), LARGE(2) / Z'FFFFFFB0', Z'7FEFFFFF' /
C     DATA RIGHT(1), RIGHT(2) / Z'000000B0', Z'3CA00000' /
C     DATA DIVER(1), DIVER(2) / Z'FFFFFFCB', Z'3CAFFFFF'
C     DATA LOG10(1), LOG10(2) / Z'509F79E9', Z'3FD34413' /
C
C     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES FTN COMPILER
C
C     DATA SMALL(1), SMALL(2) / O000040000000, O000000000000 /
C     DATA LARGE(1), LARGE(2) / O377777777777, O777777777777 /
C     DATA RIGHT(1), RIGHT(2) / O170540000000, O000000000000 /
C     DATA DIVER(1), DIVER(2) / O170640000000, O000000000000 /
C     DATA LOG10(1), LOG10(2) / O177746420232, O411757177572 /
C
C***FIRST EXECUTABLE STATEMENT  D1MACH
C      IF (I .LT. 1 .OR. I .GT. 5) CALL XERMSG ('SLATEC', 'D1MACH',
C     +   'I OUT OF BOUNDS', 1, 2)
C
      D1MACH = DMACH(I)
      RETURN
C
      END
*DECK I1MACH
      INTEGER FUNCTION I1MACH (I)
C***BEGIN PROLOGUE  I1MACH
C***PURPOSE  Return integer machine dependent constants.
C***LIBRARY   SLATEC
C***CATEGORY  R1
C***TYPE      INTEGER (I1MACH-I)
C***KEYWORDS  MACHINE CONSTANTS
C***AUTHOR  Fox, P. A., (Bell Labs)
C           Hall, A. D., (Bell Labs)
C           Schryer, N. L., (Bell Labs)
C***DESCRIPTION
C
C   I1MACH can be used to obtain machine-dependent parameters for the
C   local machine environment.  It is a function subprogram with one
C   (input) argument and can be referenced as follows:
C
C        K = I1MACH(I)
C
C   where I=1,...,16.  The (output) value of K above is determined by
C   the (input) value of I.  The results for various values of I are
C   discussed below.
C
C   I/O unit numbers:
C     I1MACH( 1) = the standard input unit.
C     I1MACH( 2) = the standard output unit.
C     I1MACH( 3) = the standard punch unit.
C     I1MACH( 4) = the standard error message unit.
C
C   Words:
C     I1MACH( 5) = the number of bits per integer storage unit.
C     I1MACH( 6) = the number of characters per integer storage unit.
C
C   Integers:
C     assume integers are represented in the S-digit, base-A form
C
C                sign ( X(S-1)*A**(S-1) + ... + X(1)*A + X(0) )
C
C                where 0 .LE. X(I) .LT. A for I=0,...,S-1.
C     I1MACH( 7) = A, the base.
C     I1MACH( 8) = S, the number of base-A digits.
C     I1MACH( 9) = A**S - 1, the largest magnitude.
C
C   Floating-Point Numbers:
C     Assume floating-point numbers are represented in the T-digit,
C     base-B form
C                sign (B**E)*( (X(1)/B) + ... + (X(T)/B**T) )
C
C                where 0 .LE. X(I) .LT. B for I=1,...,T,
C                0 .LT. X(1), and EMIN .LE. E .LE. EMAX.
C     I1MACH(10) = B, the base.
C
C   Single-Precision:
C     I1MACH(11) = T, the number of base-B digits.
C     I1MACH(12) = EMIN, the smallest exponent E.
C     I1MACH(13) = EMAX, the largest exponent E.
C
C   Double-Precision:
C     I1MACH(14) = T, the number of base-B digits.
C     I1MACH(15) = EMIN, the smallest exponent E.
C     I1MACH(16) = EMAX, the largest exponent E.
C
C   To alter this function for a particular environment, the desired
C   set of DATA statements should be activated by removing the C from
C   column 1.  Also, the values of I1MACH(1) - I1MACH(4) should be
C   checked for consistency with the local operating system.
C
C***REFERENCES  P. A. Fox, A. D. Hall and N. L. Schryer, Framework for
C                 a portable library, ACM Transactions on Mathematical
C                 Software 4, 2 (June 1978), pp. 177-188.
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   750101  DATE WRITTEN
C   891012  Added VAX G-floating constants.  (WRB)
C   891012  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900618  Added DEC RISC constants.  (WRB)
C   900723  Added IBM RS 6000 constants.  (WRB)
C   901009  Correct I1MACH(7) for IBM Mainframes. Should be 2 not 16.
C           (RWC)
C   910710  Added HP 730 constants.  (SMR)
C   911114  Added Convex IEEE constants.  (WRB)
C   920121  Added SUN -r8 compiler option constants.  (WRB)
C   920229  Added Touchstone Delta i860 constants.  (WRB)
C   920501  Reformatted the REFERENCES section.  (WRB)
C   920625  Added Convex -p8 and -pd8 compiler option constants.
C           (BKS, WRB)
C   930201  Added DEC Alpha and SGI constants.  (RWC and WRB)
C   930618  Corrected I1MACH(5) for Convex -p8 and -pd8 compiler
C           options.  (DWL, RWC and WRB).
C   010817  Elevated IEEE to highest importance; see next set of
C           comments below.  (DWL)
C***END PROLOGUE  I1MACH
C
C Initial data here correspond to the IEEE standard.  If one of the
C sets of initial data below is preferred, do the necessary commenting
C and uncommenting. (DWL)
      INTEGER IMACH(16),OUTPUT
      DATA IMACH( 1) /          5 /
      DATA IMACH( 2) /          6 /
      DATA IMACH( 3) /          6 /
      DATA IMACH( 4) /          6 /
      DATA IMACH( 5) /         32 /
      DATA IMACH( 6) /          4 /
      DATA IMACH( 7) /          2 /
      DATA IMACH( 8) /         31 /
      DATA IMACH( 9) / 2147483647 /
      DATA IMACH(10) /          2 /
      DATA IMACH(11) /         24 /
      DATA IMACH(12) /       -126 /
      DATA IMACH(13) /        127 /
      DATA IMACH(14) /         53 /
      DATA IMACH(15) /      -1022 /
      DATA IMACH(16) /       1023 /
      SAVE IMACH
      EQUIVALENCE (IMACH(4),OUTPUT)
C
C     MACHINE CONSTANTS FOR THE AMIGA
C     ABSOFT COMPILER
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -126 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1022 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE APOLLO
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        129 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1025 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 1700 SYSTEM
C
C     DATA IMACH( 1) /          7 /
C     DATA IMACH( 2) /          2 /
C     DATA IMACH( 3) /          2 /
C     DATA IMACH( 4) /          2 /
C     DATA IMACH( 5) /         36 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         33 /
C     DATA IMACH( 9) / Z1FFFFFFFF /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -256 /
C     DATA IMACH(13) /        255 /
C     DATA IMACH(14) /         60 /
C     DATA IMACH(15) /       -256 /
C     DATA IMACH(16) /        255 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 5700 SYSTEM
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         48 /
C     DATA IMACH( 6) /          6 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         39 /
C     DATA IMACH( 9) / O0007777777777777 /
C     DATA IMACH(10) /          8 /
C     DATA IMACH(11) /         13 /
C     DATA IMACH(12) /        -50 /
C     DATA IMACH(13) /         76 /
C     DATA IMACH(14) /         26 /
C     DATA IMACH(15) /        -50 /
C     DATA IMACH(16) /         76 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 6700/7700 SYSTEMS
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         48 /
C     DATA IMACH( 6) /          6 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         39 /
C     DATA IMACH( 9) / O0007777777777777 /
C     DATA IMACH(10) /          8 /
C     DATA IMACH(11) /         13 /
C     DATA IMACH(12) /        -50 /
C     DATA IMACH(13) /         76 /
C     DATA IMACH(14) /         26 /
C     DATA IMACH(15) /     -32754 /
C     DATA IMACH(16) /      32780 /
C
C     MACHINE CONSTANTS FOR THE CDC 170/180 SERIES USING NOS/VE
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         64 /
C     DATA IMACH( 6) /          8 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         63 /
C     DATA IMACH( 9) / 9223372036854775807 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         47 /
C     DATA IMACH(12) /      -4095 /
C     DATA IMACH(13) /       4094 /
C     DATA IMACH(14) /         94 /
C     DATA IMACH(15) /      -4095 /
C     DATA IMACH(16) /       4094 /
C
C     MACHINE CONSTANTS FOR THE CDC 6000/7000 SERIES
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /    6LOUTPUT/
C     DATA IMACH( 5) /         60 /
C     DATA IMACH( 6) /         10 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         48 /
C     DATA IMACH( 9) / 00007777777777777777B /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         47 /
C     DATA IMACH(12) /       -929 /
C     DATA IMACH(13) /       1070 /
C     DATA IMACH(14) /         94 /
C     DATA IMACH(15) /       -929 /
C     DATA IMACH(16) /       1069 /
C
C     MACHINE CONSTANTS FOR THE CELERITY C1260
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          0 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / Z'7FFFFFFF' /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -126 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1022 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -fn COMPILER OPTION
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1023 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -fi COMPILER OPTION
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -p8 COMPILER OPTION
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         64 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         63 /
C     DATA IMACH( 9) / 9223372036854775807 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         53 /
C     DATA IMACH(12) /      -1023 /
C     DATA IMACH(13) /       1023 /
C     DATA IMACH(14) /        113 /
C     DATA IMACH(15) /     -16383 /
C     DATA IMACH(16) /      16383 /
C
C     MACHINE CONSTANTS FOR THE CONVEX
C     USING THE -pd8 COMPILER OPTION
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         64 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         63 /
C     DATA IMACH( 9) / 9223372036854775807 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         53 /
C     DATA IMACH(12) /      -1023 /
C     DATA IMACH(13) /       1023 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1023 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE CRAY
C     USING THE 46 BIT INTEGER COMPILER OPTION
C
C     DATA IMACH( 1) /        100 /
C     DATA IMACH( 2) /        101 /
C     DATA IMACH( 3) /        102 /
C     DATA IMACH( 4) /        101 /
C     DATA IMACH( 5) /         64 /
C     DATA IMACH( 6) /          8 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         46 /
C     DATA IMACH( 9) / 1777777777777777B /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         47 /
C     DATA IMACH(12) /      -8189 /
C     DATA IMACH(13) /       8190 /
C     DATA IMACH(14) /         94 /
C     DATA IMACH(15) /      -8099 /
C     DATA IMACH(16) /       8190 /
C
C     MACHINE CONSTANTS FOR THE CRAY
C     USING THE 64 BIT INTEGER COMPILER OPTION
C
C     DATA IMACH( 1) /        100 /
C     DATA IMACH( 2) /        101 /
C     DATA IMACH( 3) /        102 /
C     DATA IMACH( 4) /        101 /
C     DATA IMACH( 5) /         64 /
C     DATA IMACH( 6) /          8 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         63 /
C     DATA IMACH( 9) / 777777777777777777777B /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         47 /
C     DATA IMACH(12) /      -8189 /
C     DATA IMACH(13) /       8190 /
C     DATA IMACH(14) /         94 /
C     DATA IMACH(15) /      -8099 /
C     DATA IMACH(16) /       8190 /
C
C     MACHINE CONSTANTS FOR THE DATA GENERAL ECLIPSE S/200
C
C     DATA IMACH( 1) /         11 /
C     DATA IMACH( 2) /         12 /
C     DATA IMACH( 3) /          8 /
C     DATA IMACH( 4) /         10 /
C     DATA IMACH( 5) /         16 /
C     DATA IMACH( 6) /          2 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         15 /
C     DATA IMACH( 9) /      32767 /
C     DATA IMACH(10) /         16 /
C     DATA IMACH(11) /          6 /
C     DATA IMACH(12) /        -64 /
C     DATA IMACH(13) /         63 /
C     DATA IMACH(14) /         14 /
C     DATA IMACH(15) /        -64 /
C     DATA IMACH(16) /         63 /
C
C     MACHINE CONSTANTS FOR THE DEC ALPHA
C     USING G_FLOAT
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1023 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE DEC ALPHA
C     USING IEEE_FLOAT
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE DEC RISC
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE DEC VAX
C     USING D_FLOATING
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         56 /
C     DATA IMACH(15) /       -127 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE DEC VAX
C     USING G_FLOATING
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1023 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE ELXSI 6400
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         32 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -126 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1022 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE HARRIS 220
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          0 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         24 /
C     DATA IMACH( 6) /          3 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         23 /
C     DATA IMACH( 9) /    8388607 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         23 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         38 /
C     DATA IMACH(15) /       -127 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE HONEYWELL 600/6000 SERIES
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /         43 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         36 /
C     DATA IMACH( 6) /          6 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         35 /
C     DATA IMACH( 9) / O377777777777 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         27 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         63 /
C     DATA IMACH(15) /       -127 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE HP 730
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE HP 2100
C     3 WORD REAL(KIND=R8) OPTION WITH FTN4
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          4 /
C     DATA IMACH( 4) /          1 /
C     DATA IMACH( 5) /         16 /
C     DATA IMACH( 6) /          2 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         15 /
C     DATA IMACH( 9) /      32767 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         23 /
C     DATA IMACH(12) /       -128 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         39 /
C     DATA IMACH(15) /       -128 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE HP 2100
C     4 WORD REAL(KIND=R8) OPTION WITH FTN4
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          4 /
C     DATA IMACH( 4) /          1 /
C     DATA IMACH( 5) /         16 /
C     DATA IMACH( 6) /          2 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         15 /
C     DATA IMACH( 9) /      32767 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         23 /
C     DATA IMACH(12) /       -128 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         55 /
C     DATA IMACH(15) /       -128 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE HP 9000
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          7 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         32 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -126 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1015 /
C     DATA IMACH(16) /       1017 /
C
C     MACHINE CONSTANTS FOR THE IBM 360/370 SERIES,
C     THE XEROX SIGMA 5/7/9, THE SEL SYSTEMS 85/86, AND
C     THE PERKIN ELMER (INTERDATA) 7/32.
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          7 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) /  Z7FFFFFFF /
C     DATA IMACH(10) /         16 /
C     DATA IMACH(11) /          6 /
C     DATA IMACH(12) /        -64 /
C     DATA IMACH(13) /         63 /
C     DATA IMACH(14) /         14 /
C     DATA IMACH(15) /        -64 /
C     DATA IMACH(16) /         63 /
C
C     MACHINE CONSTANTS FOR THE IBM PC
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          0 /
C     DATA IMACH( 4) /          0 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE IBM RS 6000
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          0 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE INTEL i860
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KA PROCESSOR)
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         36 /
C     DATA IMACH( 6) /          5 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         35 /
C     DATA IMACH( 9) / "377777777777 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         27 /
C     DATA IMACH(12) /       -128 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         54 /
C     DATA IMACH(15) /       -101 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KI PROCESSOR)
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         36 /
C     DATA IMACH( 6) /          5 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         35 /
C     DATA IMACH( 9) / "377777777777 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         27 /
C     DATA IMACH(12) /       -128 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         62 /
C     DATA IMACH(15) /       -128 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRAN SUPPORTING
C     32-BIT INTEGER ARITHMETIC.
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         56 /
C     DATA IMACH(15) /       -127 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRAN SUPPORTING
C     16-BIT INTEGER ARITHMETIC.
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          5 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         16 /
C     DATA IMACH( 6) /          2 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         15 /
C     DATA IMACH( 9) /      32767 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         56 /
C     DATA IMACH(15) /       -127 /
C     DATA IMACH(16) /        127 /
C
C     MACHINE CONSTANTS FOR THE SILICON GRAPHICS
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE SUN
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -125 /
C     DATA IMACH(13) /        128 /
C     DATA IMACH(14) /         53 /
C     DATA IMACH(15) /      -1021 /
C     DATA IMACH(16) /       1024 /
C
C     MACHINE CONSTANTS FOR THE SUN
C     USING THE -r8 COMPILER OPTION
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          6 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         32 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         31 /
C     DATA IMACH( 9) / 2147483647 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         53 /
C     DATA IMACH(12) /      -1021 /
C     DATA IMACH(13) /       1024 /
C     DATA IMACH(14) /        113 /
C     DATA IMACH(15) /     -16381 /
C     DATA IMACH(16) /      16384 /
C
C     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES FTN COMPILER
C
C     DATA IMACH( 1) /          5 /
C     DATA IMACH( 2) /          6 /
C     DATA IMACH( 3) /          1 /
C     DATA IMACH( 4) /          6 /
C     DATA IMACH( 5) /         36 /
C     DATA IMACH( 6) /          4 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         35 /
C     DATA IMACH( 9) / O377777777777 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         27 /
C     DATA IMACH(12) /       -128 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         60 /
C     DATA IMACH(15) /      -1024 /
C     DATA IMACH(16) /       1023 /
C
C     MACHINE CONSTANTS FOR THE Z80 MICROPROCESSOR
C
C     DATA IMACH( 1) /          1 /
C     DATA IMACH( 2) /          1 /
C     DATA IMACH( 3) /          0 /
C     DATA IMACH( 4) /          1 /
C     DATA IMACH( 5) /         16 /
C     DATA IMACH( 6) /          2 /
C     DATA IMACH( 7) /          2 /
C     DATA IMACH( 8) /         15 /
C     DATA IMACH( 9) /      32767 /
C     DATA IMACH(10) /          2 /
C     DATA IMACH(11) /         24 /
C     DATA IMACH(12) /       -127 /
C     DATA IMACH(13) /        127 /
C     DATA IMACH(14) /         56 /
C     DATA IMACH(15) /       -127 /
C     DATA IMACH(16) /        127 /
C
C***FIRST EXECUTABLE STATEMENT  I1MACH
      IF (I .LT. 1  .OR.  I .GT. 16) GO TO 10
C
      I1MACH = IMACH(I)
      RETURN
C
   10 CONTINUE
      WRITE (UNIT = OUTPUT, FMT = 9000)
 9000 FORMAT ('1ERROR    1 IN I1MACH - I OUT OF BOUNDS')
C
C     CALL FDUMP
C
      STOP
      END
*DECK DH12
      SUBROUTINE DH12 (MODE, LPIVOT, L1, M, U, IUE, UP, C, ICE, ICV,
     +   NCV)
C***BEGIN PROLOGUE  DH12
C***SUBSIDIARY
C***PURPOSE  Subsidiary to DHFTI, DLSEI and DWNNLS
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (H12-S, DH12-D)
C***AUTHOR  (UNKNOWN)
C***DESCRIPTION
C
C      *** REAL(KIND=R8) VERSION OF H12 ******
C
C     C.L.Lawson and R.J.Hanson, Jet Propulsion Laboratory, 1973 Jun 12
C     to appear in 'Solving Least Squares Problems', Prentice-Hall, 1974
C
C     Construction and/or application of a single
C     Householder transformation..     Q = I + U*(U**T)/B
C
C     MODE    = 1 or 2   to select algorithm  H1  or  H2 .
C     LPIVOT is the index of the pivot element.
C     L1,M   If L1 .LE. M   the transformation will be constructed to
C            zero elements indexed from L1 through M.   If L1 GT. M
C            THE SUBROUTINE DOES AN IDENTITY TRANSFORMATION.
C     U(),IUE,UP    On entry to H1 U() contains the pivot vector.
C                   IUE is the storage increment between elements.
C                                       On exit from H1 U() and UP
C                   contain quantities defining the vector U of the
C                   Householder transformation.   On entry to H2 U()
C                   and UP should contain quantities previously computed
C                   by H1.  These will not be modified by H2.
C     C()    On entry to H1 or H2 C() contains a matrix which will be
C            regarded as a set of vectors to which the Householder
C            transformation is to be applied.  On exit C() contains the
C            set of transformed vectors.
C     ICE    Storage increment between elements of vectors in C().
C     ICV    Storage increment between vectors in C().
C     NCV    Number of vectors in C() to be transformed. If NCV .LE. 0
C            no operations will be done on C().
C
C***SEE ALSO  DHFTI, DLSEI, DWNNLS
C***ROUTINES CALLED  DAXPY, DDOT, DSWAP
C***REVISION HISTORY  (YYMMDD)
C   790101  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890831  Modified array declarations.  (WRB)
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900328  Added TYPE section.  (WRB)
C   900911  Added DDOT to REAL(KIND=R8) statement.  (WRB)
C***END PROLOGUE  DH12
      USE REAL_PRECISION

      INTEGER I, I2, I3, I4, ICE, ICV, INCR, IUE, J, KL1, KL2, KLP,
     *     L1, L1M1, LPIVOT, M, MML1P2, MODE, NCV
      REAL(KIND=R8) B, C, CL, CLINV, ONE, UL1M1, SM, U, UP, DDOT
      DIMENSION U(IUE,*), C(*)
C     BEGIN BLOCK PERMITTING ...EXITS TO 140
C***FIRST EXECUTABLE STATEMENT  DH12
         ONE = 1.0D0
C
C     ...EXIT
         IF (0 .GE. LPIVOT .OR. LPIVOT .GE. L1 .OR. L1 .GT. M) GO TO 140
         CL = ABS(U(1,LPIVOT))
         IF (MODE .EQ. 2) GO TO 40
C           ****** CONSTRUCT THE TRANSFORMATION. ******
            DO 10 J = L1, M
               CL = MAX(ABS(U(1,J)),CL)
   10       CONTINUE
            IF (CL .GT. 0.0D0) GO TO 20
C     .........EXIT
               GO TO 140
   20       CONTINUE
            CLINV = ONE/CL
            SM = (U(1,LPIVOT)*CLINV)**2
            DO 30 J = L1, M
               SM = SM + (U(1,J)*CLINV)**2
   30       CONTINUE
            CL = CL*SQRT(SM)
            IF (U(1,LPIVOT) .GT. 0.0D0) CL = -CL
            UP = U(1,LPIVOT) - CL
            U(1,LPIVOT) = CL
         GO TO 50
   40    CONTINUE
C        ****** APPLY THE TRANSFORMATION  I+U*(U**T)/B  TO C. ******
C
         IF (CL .GT. 0.0D0) GO TO 50
C     ......EXIT
            GO TO 140
   50    CONTINUE
C     ...EXIT
         IF (NCV .LE. 0) GO TO 140
         B = UP*U(1,LPIVOT)
C        B  MUST BE NONPOSITIVE HERE.  IF B = 0., RETURN.
C
         IF (B .LT. 0.0D0) GO TO 60
C     ......EXIT
            GO TO 140
   60    CONTINUE
         B = ONE/B
         MML1P2 = M - L1 + 2
         IF (MML1P2 .LE. 20) GO TO 80
            L1M1 = L1 - 1
            KL1 = 1 + (L1M1 - 1)*ICE
            KL2 = KL1
            KLP = 1 + (LPIVOT - 1)*ICE
            UL1M1 = U(1,L1M1)
            U(1,L1M1) = UP
            IF (LPIVOT .NE. L1M1) CALL DSWAP(NCV,C(KL1),ICV,C(KLP),ICV)
            DO 70 J = 1, NCV
               SM = DDOT(MML1P2,U(1,L1M1),IUE,C(KL1),ICE)
               SM = SM*B
               CALL DAXPY(MML1P2,SM,U(1,L1M1),IUE,C(KL1),ICE)
               KL1 = KL1 + ICV
   70       CONTINUE
            U(1,L1M1) = UL1M1
C     ......EXIT
            IF (LPIVOT .EQ. L1M1) GO TO 140
            KL1 = KL2
            CALL DSWAP(NCV,C(KL1),ICV,C(KLP),ICV)
         GO TO 130
   80    CONTINUE
            I2 = 1 - ICV + ICE*(LPIVOT - 1)
            INCR = ICE*(L1 - LPIVOT)
            DO 120 J = 1, NCV
               I2 = I2 + ICV
               I3 = I2 + INCR
               I4 = I3
               SM = C(I2)*UP
               DO 90 I = L1, M
                  SM = SM + C(I3)*U(1,I)
                  I3 = I3 + ICE
   90          CONTINUE
               IF (SM .EQ. 0.0D0) GO TO 110
                  SM = SM*B
                  C(I2) = C(I2) + SM*UP
                  DO 100 I = L1, M
                     C(I4) = C(I4) + SM*U(1,I)
                     I4 = I4 + ICE
  100             CONTINUE
  110          CONTINUE
  120       CONTINUE
  130    CONTINUE
  140 CONTINUE
      RETURN
      END
*DECK DHFTI
      SUBROUTINE DHFTI (A, MDA, M, N, B, MDB, NB, TAU, KRANK, RNORM, H,
     +   G, IP)
C***BEGIN PROLOGUE  DHFTI
C***PURPOSE  Solve a least squares problem for banded matrices using
C            sequential accumulation of rows of the data matrix.
C            Exactly one right-hand side vector is permitted.
C***LIBRARY   SLATEC
C***CATEGORY  D9
C***TYPE      REAL(KIND=R8) (HFTI-S, DHFTI-D)
C***KEYWORDS  CURVE FITTING, LEAST SQUARES
C***AUTHOR  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C***DESCRIPTION
C
C     DIMENSION A(MDA,N),(B(MDB,NB) or B(M)),RNORM(NB),H(N),G(N),IP(N)
C
C     This subroutine solves a linear least squares problem or a set of
C     linear least squares problems having the same matrix but different
C     right-side vectors.  The problem data consists of an M by N matrix
C     A, an M by NB matrix B, and an absolute tolerance parameter TAU
C     whose usage is described below.  The NB column vectors of B
C     represent right-side vectors for NB distinct linear least squares
C     problems.
C
C     This set of problems can also be written as the matrix least
C     squares problem
C
C                       AX = B,
C
C     where X is the N by NB solution matrix.
C
C     Note that if B is the M by M identity matrix, then X will be the
C     pseudo-inverse of A.
C
C     This subroutine first transforms the augmented matrix (A B) to a
C     matrix (R C) using premultiplying Householder transformations with
C     column interchanges.  All subdiagonal elements in the matrix R are
C     zero and its diagonal elements satisfy
C
C                       ABS(R(I,I)).GE.ABS(R(I+1,I+1)),
C
C                       I = 1,...,L-1, where
C
C                       L = MIN(M,N).
C
C     The subroutine will compute an integer, KRANK, equal to the number
C     of diagonal terms of R that exceed TAU in magnitude. Then a
C     solution of minimum Euclidean length is computed using the first
C     KRANK rows of (R C).
C
C     To be specific we suggest that the user consider an easily
C     computable matrix norm, such as, the maximum of all column sums of
C     magnitudes.
C
C     Now if the relative uncertainty of B is EPS, (norm of uncertainty/
C     norm of B), it is suggested that TAU be set approximately equal to
C     EPS*(norm of A).
C
C     The user must dimension all arrays appearing in the call list..
C     A(MDA,N),(B(MDB,NB) or B(M)),RNORM(NB),H(N),G(N),IP(N).  This
C     permits the solution of a range of problems in the same array
C     space.
C
C     The entire set of parameters for DHFTI are
C
C     INPUT.. All TYPE REAL variables are REAL(KIND=R8)
C
C     A(*,*),MDA,M,N    The array A(*,*) initially contains the M by N
C                       matrix A of the least squares problem AX = B.
C                       The first dimensioning parameter of the array
C                       A(*,*) is MDA, which must satisfy MDA.GE.M
C                       Either M.GE.N or M.LT.N is permitted.  There
C                       is no restriction on the rank of A.  The
C                       condition MDA.LT.M is considered an error.
C
C     B(*),MDB,NB       If NB = 0 the subroutine will perform the
C                       orthogonal decomposition but will make no
C                       references to the array B(*).  If NB.GT.0
C                       the array B(*) must initially contain the M by
C                       NB matrix B of the least squares problem AX =
C                       B.  If NB.GE.2 the array B(*) must be doubly
C                       subscripted with first dimensioning parameter
C                       MDB.GE.MAX(M,N).  If NB = 1 the array B(*) may
C                       be either doubly or singly subscripted.  In
C                       the latter case the value of MDB is arbitrary
C                       but it should be set to some valid integer
C                       value such as MDB = M.
C
C                       The condition of NB.GT.1.AND.MDB.LT. MAX(M,N)
C                       is considered an error.
C
C     TAU               Absolute tolerance parameter provided by user
C                       for pseudorank determination.
C
C     H(*),G(*),IP(*)   Arrays of working space used by DHFTI.
C
C     OUTPUT.. All TYPE REAL variables are REAL(KIND=R8)
C
C     A(*,*)            The contents of the array A(*,*) will be
C                       modified by the subroutine. These contents
C                       are not generally required by the user.
C
C     B(*)              On return the array B(*) will contain the N by
C                       NB solution matrix X.
C
C     KRANK             Set by the subroutine to indicate the
C                       pseudorank of A.
C
C     RNORM(*)          On return, RNORM(J) will contain the Euclidean
C                       norm of the residual vector for the problem
C                       defined by the J-th column vector of the array
C                       B(*,*) for J = 1,...,NB.
C
C     H(*),G(*)         On return these arrays respectively contain
C                       elements of the pre- and post-multiplying
C                       Householder transformations used to compute
C                       the minimum Euclidean length solution.
C
C     IP(*)             Array in which the subroutine records indices
C                       describing the permutation of column vectors.
C                       The contents of arrays H(*),G(*) and IP(*)
C                       are not generally required by the user.
C
C***REFERENCES  C. L. Lawson and R. J. Hanson, Solving Least Squares
C                 Problems, Prentice-Hall, Inc., 1974, Chapter 14.
C***ROUTINES CALLED  D1MACH, DH12, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   790101  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   891006  Cosmetic changes to prologue.  (WRB)
C   891006  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   901005  Replace usage of DDIFF with usage of D1MACH.  (RWC)
C   920501  Reformatted the REFERENCES section.  (WRB)
C***END PROLOGUE  DHFTI
      USE REAL_PRECISION

      INTEGER I, II, IOPT, IP(*), IP1, J, JB, JJ, K, KP1, KRANK, L,
     *     LDIAG, LMAX, M, MDA, MDB, N, NB, NERR
      REAL(KIND=R8) A, B, D1MACH, DZERO, FACTOR,
     *     G, H, HMAX, RELEPS, RNORM, SM, SM1, SZERO, TAU, TMP
      DIMENSION A(MDA,*),B(MDB,*),H(*),G(*),RNORM(*)
      SAVE RELEPS
      DATA RELEPS /0.D0/
C     BEGIN BLOCK PERMITTING ...EXITS TO 360
C***FIRST EXECUTABLE STATEMENT  DHFTI
         IF (RELEPS.EQ.0.D0) RELEPS = D1MACH(4)
         SZERO = 0.0D0
         DZERO = 0.0D0
         FACTOR = 0.001D0
C
         K = 0
         LDIAG = MIN(M,N)
         IF (LDIAG .LE. 0) GO TO 350
C           BEGIN BLOCK PERMITTING ...EXITS TO 130
C              BEGIN BLOCK PERMITTING ...EXITS TO 120
                  IF (MDA .GE. M) GO TO 10
                     NERR = 1
                     IOPT = 2
C                     CALL XERMSG ('SLATEC', 'DHFTI',
C     +                  'MDA.LT.M, PROBABLE ERROR.',
C     +                  NERR, IOPT)
C     ...............EXIT
                     GO TO 360
   10             CONTINUE
C
                  IF (NB .LE. 1 .OR. MAX(M,N) .LE. MDB) GO TO 20
                     NERR = 2
                     IOPT = 2
C                     CALL XERMSG ('SLATEC', 'DHFTI',
C     +                  'MDB.LT.MAX(M,N).AND.NB.GT.1. PROBABLE ERROR.',
C     +                  NERR, IOPT)
C     ...............EXIT
                     GO TO 360
   20             CONTINUE
C
                  DO 100 J = 1, LDIAG
C                    BEGIN BLOCK PERMITTING ...EXITS TO 70
                        IF (J .EQ. 1) GO TO 40
C
C                           UPDATE SQUARED COLUMN LENGTHS AND FIND LMAX
C                          ..
                           LMAX = J
                           DO 30 L = J, N
                              H(L) = H(L) - A(J-1,L)**2
                              IF (H(L) .GT. H(LMAX)) LMAX = L
   30                      CONTINUE
C                    ......EXIT
                           IF (FACTOR*H(LMAX) .GT. HMAX*RELEPS) GO TO 70
   40                   CONTINUE
C
C                        COMPUTE SQUARED COLUMN LENGTHS AND FIND LMAX
C                       ..
                        LMAX = J
                        DO 60 L = J, N
                           H(L) = 0.0D0
                           DO 50 I = J, M
                              H(L) = H(L) + A(I,L)**2
   50                      CONTINUE
                           IF (H(L) .GT. H(LMAX)) LMAX = L
   60                   CONTINUE
                        HMAX = H(LMAX)
   70                CONTINUE
C                    ..
C                     LMAX HAS BEEN DETERMINED
C
C                     DO COLUMN INTERCHANGES IF NEEDED.
C                    ..
                     IP(J) = LMAX
                     IF (IP(J) .EQ. J) GO TO 90
                        DO 80 I = 1, M
                           TMP = A(I,J)
                           A(I,J) = A(I,LMAX)
                           A(I,LMAX) = TMP
   80                   CONTINUE
                        H(LMAX) = H(J)
   90                CONTINUE
C
C                     COMPUTE THE J-TH TRANSFORMATION AND APPLY IT TO A
C                     AND B.
C                    ..
                     CALL DH12(1,J,J+1,M,A(1,J),1,H(J),A(1,J+1),1,MDA,
     *                         N-J)
                     CALL DH12(2,J,J+1,M,A(1,J),1,H(J),B,1,MDB,NB)
  100             CONTINUE
C
C                  DETERMINE THE PSEUDORANK, K, USING THE TOLERANCE,
C                  TAU.
C                 ..
                  DO 110 J = 1, LDIAG
C              ......EXIT
                     IF (ABS(A(J,J)) .LE. TAU) GO TO 120
  110             CONTINUE
                  K = LDIAG
C           ......EXIT
                  GO TO 130
  120          CONTINUE
               K = J - 1
  130       CONTINUE
            KP1 = K + 1
C
C           COMPUTE THE NORMS OF THE RESIDUAL VECTORS.
C
            IF (NB .LT. 1) GO TO 170
            DO 160 JB = 1, NB
               TMP = SZERO
               IF (M .LT. KP1) GO TO 150
               DO 140 I = KP1, M
                  TMP = TMP + B(I,JB)**2
  140          CONTINUE
  150          CONTINUE
               RNORM(JB) = SQRT(TMP)
  160       CONTINUE
  170       CONTINUE
C           SPECIAL FOR PSEUDORANK = 0
            IF (K .GT. 0) GO TO 210
               IF (NB .LT. 1) GO TO 200
               DO 190 JB = 1, NB
                  DO 180 I = 1, N
                     B(I,JB) = SZERO
  180             CONTINUE
  190          CONTINUE
  200          CONTINUE
            GO TO 340
  210       CONTINUE
C
C               IF THE PSEUDORANK IS LESS THAN N COMPUTE HOUSEHOLDER
C               DECOMPOSITION OF FIRST K ROWS.
C              ..
               IF (K .EQ. N) GO TO 230
                  DO 220 II = 1, K
                     I = KP1 - II
                     CALL DH12(1,I,KP1,N,A(I,1),MDA,G(I),A,MDA,1,I-1)
  220             CONTINUE
  230          CONTINUE
C
C
               IF (NB .LT. 1) GO TO 330
               DO 320 JB = 1, NB
C
C                  SOLVE THE K BY K TRIANGULAR SYSTEM.
C                 ..
                  DO 260 L = 1, K
                     SM = DZERO
                     I = KP1 - L
                     IP1 = I + 1
                     IF (K .LT. IP1) GO TO 250
                     DO 240 J = IP1, K
                        SM = SM + A(I,J)*B(J,JB)
  240                CONTINUE
  250                CONTINUE
                     SM1 = SM
                     B(I,JB) = (B(I,JB) - SM1)/A(I,I)
  260             CONTINUE
C
C                  COMPLETE COMPUTATION OF SOLUTION VECTOR.
C                 ..
                  IF (K .EQ. N) GO TO 290
                     DO 270 J = KP1, N
                        B(J,JB) = SZERO
  270                CONTINUE
                     DO 280 I = 1, K
                        CALL DH12(2,I,KP1,N,A(I,1),MDA,G(I),B(1,JB),1,
     *                            MDB,1)
  280                CONTINUE
  290             CONTINUE
C
C                   RE-ORDER THE SOLUTION VECTOR TO COMPENSATE FOR THE
C                   COLUMN INTERCHANGES.
C                 ..
                  DO 310 JJ = 1, LDIAG
                     J = LDIAG + 1 - JJ
                     IF (IP(J) .EQ. J) GO TO 300
                        L = IP(J)
                        TMP = B(L,JB)
                        B(L,JB) = B(J,JB)
                        B(J,JB) = TMP
  300                CONTINUE
  310             CONTINUE
  320          CONTINUE
  330          CONTINUE
  340       CONTINUE
  350    CONTINUE
C        ..
C         THE SOLUTION VECTORS, X, ARE NOW
C         IN THE FIRST  N  ROWS OF THE ARRAY B(,).
C
         KRANK = K
  360 CONTINUE
      RETURN
      END
*DECK DLPDP
      SUBROUTINE DLPDP (A, MDA, M, N1, N2, PRGOPT, X, WNORM, MODE, WS,
     +   IS)
C***BEGIN PROLOGUE  DLPDP
C***SUBSIDIARY
C***PURPOSE  Subsidiary to DLSEI
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (LPDP-S, DLPDP-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C  **** Double Precision version of LPDP ****
C     DIMENSION A(MDA,N+1),PRGOPT(*),X(N),WS((M+2)*(N+7)),IS(M+N+1),
C     where N=N1+N2.  This is a slight overestimate for WS(*).
C
C     Determine an N1-vector W, and
C               an N2-vector Z
C     which minimizes the Euclidean length of W
C     subject to G*W+H*Z .GE. Y.
C     This is the least projected distance problem, LPDP.
C     The matrices G and H are of respective
C     dimensions M by N1 and M by N2.
C
C     Called by subprogram DLSI( ).
C
C     The matrix
C                (G H Y)
C
C     occupies rows 1,...,M and cols 1,...,N1+N2+1 of A(*,*).
C
C     The solution (W) is returned in X(*).
C                  (Z)
C
C     The value of MODE indicates the status of
C     the computation after returning to the user.
C
C          MODE=1  The solution was successfully obtained.
C
C          MODE=2  The inequalities are inconsistent.
C
C***SEE ALSO  DLSEI
C***ROUTINES CALLED  DCOPY, DDOT, DNRM2, DSCAL, DWNNLS
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900328  Added TYPE section.  (WRB)
C   910408  Updated the AUTHOR section.  (WRB)
C***END PROLOGUE  DLPDP
      USE REAL_PRECISION

C
      INTEGER I, IS(*), IW, IX, J, L, M, MDA, MODE, MODEW, N, N1, N2,
     *     NP1
      REAL(KIND=R8) A(MDA,*), DDOT, DNRM2, FAC, ONE,
     *     PRGOPT(*), RNORM, SC, WNORM, WS(*), X(*), YNORM, ZERO
      SAVE ZERO, ONE, FAC
      DATA ZERO,ONE /0.0D0,1.0D0/, FAC /0.1D0/
C***FIRST EXECUTABLE STATEMENT  DLPDP
      N = N1 + N2
      MODE = 1
      IF (M .GT. 0) GO TO 20
         IF (N .LE. 0) GO TO 10
            X(1) = ZERO
            CALL DCOPY(N,X,0,X,1)
   10    CONTINUE
         WNORM = ZERO
      GO TO 200
   20 CONTINUE
C        BEGIN BLOCK PERMITTING ...EXITS TO 190
            NP1 = N + 1
C
C           SCALE NONZERO ROWS OF INEQUALITY MATRIX TO HAVE LENGTH ONE.
            DO 40 I = 1, M
               SC = DNRM2(N,A(I,1),MDA)
               IF (SC .EQ. ZERO) GO TO 30
                  SC = ONE/SC
                  CALL DSCAL(NP1,SC,A(I,1),MDA)
   30          CONTINUE
   40       CONTINUE
C
C           SCALE RT.-SIDE VECTOR TO HAVE LENGTH ONE (OR ZERO).
            YNORM = DNRM2(M,A(1,NP1),1)
            IF (YNORM .EQ. ZERO) GO TO 50
               SC = ONE/YNORM
               CALL DSCAL(M,SC,A(1,NP1),1)
   50       CONTINUE
C
C           SCALE COLS OF MATRIX H.
            J = N1 + 1
   60       IF (J .GT. N) GO TO 70
               SC = DNRM2(M,A(1,J),1)
               IF (SC .NE. ZERO) SC = ONE/SC
               CALL DSCAL(M,SC,A(1,J),1)
               X(J) = SC
               J = J + 1
            GO TO 60
   70       CONTINUE
            IF (N1 .LE. 0) GO TO 130
C
C              COPY TRANSPOSE OF (H G Y) TO WORK ARRAY WS(*).
               IW = 0
               DO 80 I = 1, M
C
C                 MOVE COL OF TRANSPOSE OF H INTO WORK ARRAY.
                  CALL DCOPY(N2,A(I,N1+1),MDA,WS(IW+1),1)
                  IW = IW + N2
C
C                 MOVE COL OF TRANSPOSE OF G INTO WORK ARRAY.
                  CALL DCOPY(N1,A(I,1),MDA,WS(IW+1),1)
                  IW = IW + N1
C
C                 MOVE COMPONENT OF VECTOR Y INTO WORK ARRAY.
                  WS(IW+1) = A(I,NP1)
                  IW = IW + 1
   80          CONTINUE
               WS(IW+1) = ZERO
               CALL DCOPY(N,WS(IW+1),0,WS(IW+1),1)
               IW = IW + N
               WS(IW+1) = ONE
               IW = IW + 1
C
C              SOLVE EU=F SUBJECT TO (TRANSPOSE OF H)U=0, U.GE.0.  THE
C              MATRIX E = TRANSPOSE OF (G Y), AND THE (N+1)-VECTOR
C              F = TRANSPOSE OF (0,...,0,1).
               IX = IW + 1
               IW = IW + M
C
C              DO NOT CHECK LENGTHS OF WORK ARRAYS IN THIS USAGE OF
C              DWNNLS( ).
               IS(1) = 0
               IS(2) = 0
               CALL DWNNLS(WS,NP1,N2,NP1-N2,M,0,PRGOPT,WS(IX),RNORM,
     *                     MODEW,IS,WS(IW+1))
C
C              COMPUTE THE COMPONENTS OF THE SOLN DENOTED ABOVE BY W.
               SC = ONE - DDOT(M,A(1,NP1),1,WS(IX),1)
               IF (ONE + FAC*ABS(SC) .EQ. ONE .OR. RNORM .LE. ZERO)
     *            GO TO 110
                  SC = ONE/SC
                  DO 90 J = 1, N1
                     X(J) = SC*DDOT(M,A(1,J),1,WS(IX),1)
   90             CONTINUE
C
C                 COMPUTE THE VECTOR Q=Y-GW.  OVERWRITE Y WITH THIS
C                 VECTOR.
                  DO 100 I = 1, M
                     A(I,NP1) = A(I,NP1) - DDOT(N1,A(I,1),MDA,X,1)
  100             CONTINUE
               GO TO 120
  110          CONTINUE
                  MODE = 2
C        .........EXIT
                  GO TO 190
  120          CONTINUE
  130       CONTINUE
            IF (N2 .LE. 0) GO TO 180
C
C              COPY TRANSPOSE OF (H Q) TO WORK ARRAY WS(*).
               IW = 0
               DO 140 I = 1, M
                  CALL DCOPY(N2,A(I,N1+1),MDA,WS(IW+1),1)
                  IW = IW + N2
                  WS(IW+1) = A(I,NP1)
                  IW = IW + 1
  140          CONTINUE
               WS(IW+1) = ZERO
               CALL DCOPY(N2,WS(IW+1),0,WS(IW+1),1)
               IW = IW + N2
               WS(IW+1) = ONE
               IW = IW + 1
               IX = IW + 1
               IW = IW + M
C
C              SOLVE RV=S SUBJECT TO V.GE.0.  THE MATRIX R =(TRANSPOSE
C              OF (H Q)), WHERE Q=Y-GW.  THE (N2+1)-VECTOR S =(TRANSPOSE
C              OF (0,...,0,1)).
C
C              DO NOT CHECK LENGTHS OF WORK ARRAYS IN THIS USAGE OF
C              DWNNLS( ).
               IS(1) = 0
               IS(2) = 0
               CALL DWNNLS(WS,N2+1,0,N2+1,M,0,PRGOPT,WS(IX),RNORM,MODEW,
     *                     IS,WS(IW+1))
C
C              COMPUTE THE COMPONENTS OF THE SOLN DENOTED ABOVE BY Z.
               SC = ONE - DDOT(M,A(1,NP1),1,WS(IX),1)
               IF (ONE + FAC*ABS(SC) .EQ. ONE .OR. RNORM .LE. ZERO)
     *            GO TO 160
                  SC = ONE/SC
                  DO 150 J = 1, N2
                     L = N1 + J
                     X(L) = SC*DDOT(M,A(1,L),1,WS(IX),1)*X(L)
  150             CONTINUE
               GO TO 170
  160          CONTINUE
                  MODE = 2
C        .........EXIT
                  GO TO 190
  170          CONTINUE
  180       CONTINUE
C
C           ACCOUNT FOR SCALING OF RT.-SIDE VECTOR IN SOLUTION.
            CALL DSCAL(N,YNORM,X,1)
            WNORM = DNRM2(N1,X,1)
  190    CONTINUE
  200 CONTINUE
      RETURN
      END
*DECK DWNNLS
      SUBROUTINE DWNNLS (W, MDW, ME, MA, N, L, PRGOPT, X, RNORM, MODE,
     +   IWORK, WORK)
C***BEGIN PROLOGUE  DWNNLS
C***PURPOSE  Solve a linearly constrained least squares problem with
C            equality constraints and nonnegativity constraints on
C            selected variables.
C***LIBRARY   SLATEC
C***CATEGORY  K1A2A
C***TYPE      REAL(KIND=R8) (WNNLS-S, DWNNLS-D)
C***KEYWORDS  CONSTRAINED LEAST SQUARES, CURVE FITTING, DATA FITTING,
C             EQUALITY CONSTRAINTS, INEQUALITY CONSTRAINTS,
C             NONNEGATIVITY CONSTRAINTS, QUADRATIC PROGRAMMING
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     Abstract
C
C     This subprogram solves a linearly constrained least squares
C     problem.  Suppose there are given matrices E and A of
C     respective dimensions ME by N and MA by N, and vectors F
C     and B of respective lengths ME and MA.  This subroutine
C     solves the problem
C
C               EX = F, (equations to be exactly satisfied)
C
C               AX = B, (equations to be approximately satisfied,
C                        in the least squares sense)
C
C               subject to components L+1,...,N nonnegative
C
C     Any values ME.GE.0, MA.GE.0 and 0.LE. L .LE.N are permitted.
C
C     The problem is reposed as problem DWNNLS
C
C               (WT*E)X = (WT*F)
C               (   A)    (   B), (least squares)
C               subject to components L+1,...,N nonnegative.
C
C     The subprogram chooses the heavy weight (or penalty parameter) WT.
C
C     The parameters for DWNNLS are
C
C     INPUT.. All TYPE REAL variables are REAL(KIND=R8)
C
C     W(*,*),MDW,  The array W(*,*) is double subscripted with first
C     ME,MA,N,L    dimensioning parameter equal to MDW.  For this
C                  discussion let us call M = ME + MA.  Then MDW
C                  must satisfy MDW.GE.M.  The condition MDW.LT.M
C                  is an error.
C
C                  The array W(*,*) contains the matrices and vectors
C
C                       (E  F)
C                       (A  B)
C
C                  in rows and columns 1,...,M and 1,...,N+1
C                  respectively.  Columns 1,...,L correspond to
C                  unconstrained variables X(1),...,X(L).  The
C                  remaining variables are constrained to be
C                  nonnegative. The condition L.LT.0 or L.GT.N is
C                  an error.
C
C     PRGOPT(*)    This double precision array is the option vector.
C                  If the user is satisfied with the nominal
C                  subprogram features set
C
C                  PRGOPT(1)=1 (or PRGOPT(1)=1.0)
C
C                  Otherwise PRGOPT(*) is a linked list consisting of
C                  groups of data of the following form
C
C                  LINK
C                  KEY
C                  DATA SET
C
C                  The parameters LINK and KEY are each one word.
C                  The DATA SET can be comprised of several words.
C                  The number of items depends on the value of KEY.
C                  The value of LINK points to the first
C                  entry of the next group of data within
C                  PRGOPT(*).  The exception is when there are
C                  no more options to change.  In that
C                  case LINK=1 and the values KEY and DATA SET
C                  are not referenced. The general layout of
C                  PRGOPT(*) is as follows.
C
C               ...PRGOPT(1)=LINK1 (link to first entry of next group)
C               .  PRGOPT(2)=KEY1 (key to the option change)
C               .  PRGOPT(3)=DATA VALUE (data value for this change)
C               .       .
C               .       .
C               .       .
C               ...PRGOPT(LINK1)=LINK2 (link to the first entry of
C               .                       next group)
C               .  PRGOPT(LINK1+1)=KEY2 (key to the option change)
C               .  PRGOPT(LINK1+2)=DATA VALUE
C               ...     .
C               .       .
C               .       .
C               ...PRGOPT(LINK)=1 (no more options to change)
C
C                  Values of LINK that are nonpositive are errors.
C                  A value of LINK.GT.NLINK=100000 is also an error.
C                  This helps prevent using invalid but positive
C                  values of LINK that will probably extend
C                  beyond the program limits of PRGOPT(*).
C                  Unrecognized values of KEY are ignored.  The
C                  order of the options is arbitrary and any number
C                  of options can be changed with the following
C                  restriction.  To prevent cycling in the
C                  processing of the option array a count of the
C                  number of options changed is maintained.
C                  Whenever this count exceeds NOPT=1000 an error
C                  message is printed and the subprogram returns.
C
C                  OPTIONS..
C
C                  KEY=6
C                         Scale the nonzero columns of the
C                  entire data matrix
C                  (E)
C                  (A)
C                  to have length one. The DATA SET for
C                  this option is a single value.  It must
C                  be nonzero if unit length column scaling is
C                  desired.
C
C                  KEY=7
C                         Scale columns of the entire data matrix
C                  (E)
C                  (A)
C                  with a user-provided diagonal matrix.
C                  The DATA SET for this option consists
C                  of the N diagonal scaling factors, one for
C                  each matrix column.
C
C                  KEY=8
C                         Change the rank determination tolerance from
C                  the nominal value of SQRT(SRELPR).  This quantity
C                  can be no smaller than SRELPR, The arithmetic-
C                  storage precision.  The quantity used
C                  here is internally restricted to be at
C                  least SRELPR.  The DATA SET for this option
C                  is the new tolerance.
C
C                  KEY=9
C                         Change the blow-up parameter from the
C                  nominal value of SQRT(SRELPR).  The reciprocal of
C                  this parameter is used in rejecting solution
C                  components as too large when a variable is
C                  first brought into the active set.  Too large
C                  means that the proposed component times the
C                  reciprocal of the parameter is not less than
C                  the ratio of the norms of the right-side
C                  vector and the data matrix.
C                  This parameter can be no smaller than SRELPR,
C                  the arithmetic-storage precision.
C
C                  For example, suppose we want to provide
C                  a diagonal matrix to scale the problem
C                  matrix and change the tolerance used for
C                  determining linear dependence of dropped col
C                  vectors.  For these options the dimensions of
C                  PRGOPT(*) must be at least N+6.  The FORTRAN
C                  statements defining these options would
C                  be as follows.
C
C                  PRGOPT(1)=N+3 (link to entry N+3 in PRGOPT(*))
C                  PRGOPT(2)=7 (user-provided scaling key)
C
C                  CALL DCOPY(N,D,1,PRGOPT(3),1) (copy the N
C                  scaling factors from a user array called D(*)
C                  into PRGOPT(3)-PRGOPT(N+2))
C
C                  PRGOPT(N+3)=N+6 (link to entry N+6 of PRGOPT(*))
C                  PRGOPT(N+4)=8 (linear dependence tolerance key)
C                  PRGOPT(N+5)=... (new value of the tolerance)
C
C                  PRGOPT(N+6)=1 (no more options to change)
C
C
C     IWORK(1),    The amounts of working storage actually allocated
C     IWORK(2)     for the working arrays WORK(*) and IWORK(*),
C                  respectively.  These quantities are compared with
C                  the actual amounts of storage needed for DWNNLS( ).
C                  Insufficient storage allocated for either WORK(*)
C                  or IWORK(*) is considered an error.  This feature
C                  was included in DWNNLS( ) because miscalculating
C                  the storage formulas for WORK(*) and IWORK(*)
C                  might very well lead to subtle and hard-to-find
C                  execution errors.
C
C                  The length of WORK(*) must be at least
C
C                  LW = ME+MA+5*N
C                  This test will not be made if IWORK(1).LE.0.
C
C                  The length of IWORK(*) must be at least
C
C                  LIW = ME+MA+N
C                  This test will not be made if IWORK(2).LE.0.
C
C     OUTPUT.. All TYPE REAL variables are REAL(KIND=R8)
C
C     X(*)         An array dimensioned at least N, which will
C                  contain the N components of the solution vector
C                  on output.
C
C     RNORM        The residual norm of the solution.  The value of
C                  RNORM contains the residual vector length of the
C                  equality constraints and least squares equations.
C
C     MODE         The value of MODE indicates the success or failure
C                  of the subprogram.
C
C                  MODE = 0  Subprogram completed successfully.
C
C                       = 1  Max. number of iterations (equal to
C                            3*(N-L)) exceeded. Nearly all problems
C                            should complete in fewer than this
C                            number of iterations. An approximate
C                            solution and its corresponding residual
C                            vector length are in X(*) and RNORM.
C
C                       = 2  Usage error occurred.  The offending
C                            condition is noted with the error
C                            processing subprogram, XERMSG( ).
C
C     User-designated
C     Working arrays..
C
C     WORK(*)      A double precision working array of length at least
C                  M + 5*N.
C
C     IWORK(*)     An integer-valued working array of length at least
C                  M+N.
C
C***REFERENCES  K. H. Haskell and R. J. Hanson, An algorithm for
C                 linear least squares problems with equality and
C                 nonnegativity constraints, Report SAND77-0552, Sandia
C                 Laboratories, June 1978.
C               K. H. Haskell and R. J. Hanson, Selected algorithms for
C                 the linearly constrained least squares problem - a
C                 users guide, Report SAND78-1290, Sandia Laboratories,
C                 August 1979.
C               K. H. Haskell and R. J. Hanson, An algorithm for
C                 linear least squares problems with equality and
C                 nonnegativity constraints, Mathematical Programming
C                 21 (1981), pp. 98-118.
C               R. J. Hanson and K. H. Haskell, Two algorithms for the
C                 linearly constrained least squares problem, ACM
C                 Transactions on Mathematical Software, September 1982.
C               C. L. Lawson and R. J. Hanson, Solving Least Squares
C                 Problems, Prentice-Hall, Inc., 1974.
C***ROUTINES CALLED  DWNLSM, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890618  Completely restructured and revised.  (WRB & RWC)
C   891006  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900510  Convert XERRWV calls to XERMSG calls, change Prologue
C           comments to agree with WNNLS.  (RWC)
C   920501  Reformatted the REFERENCES section.  (WRB)
C   180613  Removed prints and replaced DP --> REAL(KIND=R8). (THC)
C***END PROLOGUE  DWNNLS
      USE REAL_PRECISION

      INTEGER IWORK(*), L, L1, L2, L3, L4, L5, LIW, LW, MA, MDW, ME,
     *     MODE, N
      REAL(KIND=R8)  PRGOPT(*), RNORM, W(MDW,*), WORK(*), X(*)
C      CHARACTER*8 XERN1
C***FIRST EXECUTABLE STATEMENT  DWNNLS
      MODE = 0
      IF (MA+ME.LE.0 .OR. N.LE.0) RETURN
C
      IF (IWORK(1).GT.0) THEN
         LW = ME + MA + 5*N
         IF (IWORK(1).LT.LW) THEN
C            WRITE (XERN1, '(I8)') LW
C            CALL XERMSG ('SLATEC', 'DWNNLS', 'INSUFFICIENT STORAGE ' //
C     *         'ALLOCATED FOR WORK(*), NEED LW = ' // XERN1, 2, 1)
            MODE = 2
            RETURN
         ENDIF
      ENDIF
C
      IF (IWORK(2).GT.0) THEN
         LIW = ME + MA + N
         IF (IWORK(2).LT.LIW) THEN
C            WRITE (XERN1, '(I8)') LIW
C            CALL XERMSG ('SLATEC', 'DWNNLS', 'INSUFFICIENT STORAGE ' //
C     *         'ALLOCATED FOR IWORK(*), NEED LIW = ' // XERN1, 2, 1)
            MODE = 2
            RETURN
         ENDIF
      ENDIF
C
      IF (MDW.LT.ME+MA) THEN
C         CALL XERMSG ('SLATEC', 'DWNNLS',
C     *      'THE VALUE MDW.LT.ME+MA IS AN ERROR', 1, 1)
         MODE = 2
         RETURN
      ENDIF
C
      IF (L.LT.0 .OR. L.GT.N) THEN
C         CALL XERMSG ('SLATEC', 'DWNNLS',
C     *      'L.GE.0 .AND. L.LE.N IS REQUIRED', 2, 1)
         MODE = 2
         RETURN
      ENDIF
C
C     THE PURPOSE OF THIS SUBROUTINE IS TO BREAK UP THE ARRAYS
C     WORK(*) AND IWORK(*) INTO SEPARATE WORK ARRAYS
C     REQUIRED BY THE MAIN SUBROUTINE DWNLSM( ).
C
      L1 = N + 1
      L2 = L1 + N
      L3 = L2 + ME + MA
      L4 = L3 + N
      L5 = L4 + N
C
      CALL DWNLSM(W, MDW, ME, MA, N, L, PRGOPT, X, RNORM, MODE, IWORK,
     *            IWORK(L1), WORK(1), WORK(L1), WORK(L2), WORK(L3),
     *            WORK(L4), WORK(L5))
      RETURN
      END
*DECK DWNLSM
      SUBROUTINE DWNLSM (W, MDW, MME, MA, N, L, PRGOPT, X, RNORM, MODE,
     +   IPIVOT, ITYPE, WD, H, SCALE, Z, TEMP, D)
C***BEGIN PROLOGUE  DWNLSM
C***SUBSIDIARY
C***PURPOSE  Subsidiary to DWNNLS
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (WNLSM-S, DWNLSM-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     This is a companion subprogram to DWNNLS.
C     The documentation for DWNNLS has complete usage instructions.
C
C     In addition to the parameters discussed in the prologue to
C     subroutine DWNNLS, the following work arrays are used in
C     subroutine DWNLSM  (they are passed through the calling
C     sequence from DWNNLS for purposes of variable dimensioning).
C     Their contents will in general be of no interest to the user.
C
C     Variables of type REAL are REAL(KIND=R8).
C
C         IPIVOT(*)
C            An array of length N.  Upon completion it contains the
C         pivoting information for the cols of W(*,*).
C
C         ITYPE(*)
C            An array of length M which is used to keep track
C         of the classification of the equations.  ITYPE(I)=0
C         denotes equation I as an equality constraint.
C         ITYPE(I)=1 denotes equation I as a least squares
C         equation.
C
C         WD(*)
C            An array of length N.  Upon completion it contains the
C         dual solution vector.
C
C         H(*)
C            An array of length N.  Upon completion it contains the
C         pivot scalars of the Householder transformations performed
C         in the case KRANK.LT.L.
C
C         SCALE(*)
C            An array of length M which is used by the subroutine
C         to store the diagonal matrix of weights.
C         These are used to apply the modified Givens
C         transformations.
C
C         Z(*),TEMP(*)
C            Working arrays of length N.
C
C         D(*)
C            An array of length N that contains the
C         column scaling for the matrix (E).
C                                       (A)
C
C***SEE ALSO  DWNNLS
C***ROUTINES CALLED  D1MACH, DASUM, DAXPY, DCOPY, DH12, DNRM2,
C                     SLATEC_DROTM, SLATEC_DROTMG, DSCAL, DSWAP,
C                     DWNLIT, IDAMAX, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890618  Completely restructured and revised.  (WRB & RWC)
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900328  Added TYPE section.  (WRB)
C   900510  Fixed an error message.  (RWC)
C   900604  DP version created from SP version.  (RWC)
C   900911  Restriction on value of ALAMDA included.  (WRB)
C***END PROLOGUE  DWNLSM
      USE REAL_PRECISION

      INTEGER IPIVOT(*), ITYPE(*), L, MA, MDW, MME, MODE, N
      REAL(KIND=R8) D(*), H(*), PRGOPT(*), RNORM, SCALE(*), TEMP(*),
     *   W(MDW,*), WD(*), X(*), Z(*)
C
      EXTERNAL D1MACH, DASUM, DAXPY, DCOPY, DH12, DNRM2, SLATEC_DROTM,
     *   SLATEC_DROTMG, DSCAL, DSWAP, DWNLIT, IDAMAX, XERMSG
      REAL(KIND=R8) D1MACH, DASUM, DNRM2
      INTEGER IDAMAX
C
      REAL(KIND=R8) ALAMDA, ALPHA, ALSQ, AMAX, BLOWUP, BNORM,
     *   DOPE(3), DRELPR, EANORM, FAC, SM, SPARAM(5), T, TAU, WMAX, Z2,
     *   ZZ
      INTEGER I, IDOPE(3), IMAX, ISOL, ITEMP, ITER, ITMAX, IWMAX, J,
     *   JCON, JP, KEY, KRANK, L1, LAST, LINK, M, ME, NEXT, NIV, NLINK,
     *   NOPT, NSOLN, NTIMES
      LOGICAL DONE, FEASBL, FIRST, HITCON, POS
C
      SAVE DRELPR, FIRST
      DATA FIRST /.TRUE./
C***FIRST EXECUTABLE STATEMENT  DWNLSM
C
C     Initialize variables.
C     DRELPR is the precision for the particular machine
C     being used.  This logic avoids resetting it every entry.
C
      IF (FIRST) DRELPR = D1MACH(4)
      FIRST = .FALSE.
C
C     Set the nominal tolerance used in the code.
C
      TAU = SQRT(DRELPR)
C
      M = MA + MME
      ME = MME
      MODE = 2
C
C     To process option vector
C
      FAC = 1.D-4
C
C     Set the nominal blow up factor used in the code.
C
      BLOWUP = TAU
C
C     The nominal column scaling used in the code is
C     the identity scaling.
C
      CALL DCOPY (N, 1.D0, 0, D, 1)
C
C     Define bound for number of options to change.
C
      NOPT = 1000
C
C     Define bound for positive value of LINK.
C
      NLINK = 100000
      NTIMES = 0
      LAST = 1
      LINK = PRGOPT(1)
      IF (LINK.LE.0 .OR. LINK.GT.NLINK) THEN
C         CALL XERMSG ('SLATEC', 'DWNLSM',
C     +      'IN DWNNLS, THE OPTION VECTOR IS UNDEFINED', 3, 1)
         RETURN
      ENDIF
C
  100 IF (LINK.GT.1) THEN
         NTIMES = NTIMES + 1
         IF (NTIMES.GT.NOPT) THEN
C         CALL XERMSG ('SLATEC', 'DWNLSM',
C     +      'IN DWNNLS, THE LINKS IN THE OPTION VECTOR ARE CYCLING.',
C     +      3, 1)
            RETURN
         ENDIF
C
         KEY = PRGOPT(LAST+1)
         IF (KEY.EQ.6 .AND. PRGOPT(LAST+2).NE.0.D0) THEN
            DO 110 J = 1,N
               T = DNRM2(M,W(1,J),1)
               IF (T.NE.0.D0) T = 1.D0/T
               D(J) = T
  110       CONTINUE
         ENDIF
C
         IF (KEY.EQ.7) CALL DCOPY (N, PRGOPT(LAST+2), 1, D, 1)
         IF (KEY.EQ.8) TAU = MAX(DRELPR,PRGOPT(LAST+2))
         IF (KEY.EQ.9) BLOWUP = MAX(DRELPR,PRGOPT(LAST+2))
C
         NEXT = PRGOPT(LINK)
         IF (NEXT.LE.0 .OR. NEXT.GT.NLINK) THEN
C            CALL XERMSG ('SLATEC', 'DWNLSM',
C     +         'IN DWNNLS, THE OPTION VECTOR IS UNDEFINED', 3, 1)
            RETURN
         ENDIF
C
         LAST = LINK
         LINK = NEXT
         GO TO 100
      ENDIF
C
      DO 120 J = 1,N
         CALL DSCAL (M, D(J), W(1,J), 1)
  120 CONTINUE
C
C     Process option vector
C
      DONE = .FALSE.
      ITER = 0
      ITMAX = 3*(N-L)
      MODE = 0
      NSOLN = L
      L1 = MIN(M,L)
C
C     Compute scale factor to apply to equality constraint equations.
C
      DO 130 J = 1,N
         WD(J) = DASUM(M,W(1,J),1)
  130 CONTINUE
C
      IMAX = IDAMAX(N,WD,1)
      EANORM = WD(IMAX)
      BNORM = DASUM(M,W(1,N+1),1)
      ALAMDA = EANORM/(DRELPR*FAC)
C
C     On machines, such as the VAXes using D floating, with a very
C     limited exponent range for double precision values, the previously
C     computed value of ALAMDA may cause an overflow condition.
C     Therefore, this code further limits the value of ALAMDA.
C
      ALAMDA = MIN(ALAMDA,SQRT(D1MACH(2)))
C
C     Define scaling diagonal matrix for modified Givens usage and
C     classify equation types.
C
      ALSQ = ALAMDA**2
      DO 140 I = 1,M
C
C        When equation I is heavily weighted ITYPE(I)=0,
C        else ITYPE(I)=1.
C
         IF (I.LE.ME) THEN
            T = ALSQ
            ITEMP = 0
         ELSE
            T = 1.D0
            ITEMP = 1
         ENDIF
         SCALE(I) = T
         ITYPE(I) = ITEMP
  140 CONTINUE
C
C     Set the solution vector X(*) to zero and the column interchange
C     matrix to the identity.
C
      CALL DCOPY (N, 0.D0, 0, X, 1)
      DO 150 I = 1,N
         IPIVOT(I) = I
  150 CONTINUE
C
C     Perform initial triangularization in the submatrix
C     corresponding to the unconstrained variables.
C     Set first L components of dual vector to zero because
C     these correspond to the unconstrained variables.
C
      CALL DCOPY (L, 0.D0, 0, WD, 1)
C
C     The arrays IDOPE(*) and DOPE(*) are used to pass
C     information to DWNLIT().  This was done to avoid
C     a long calling sequence or the use of COMMON.
C
      IDOPE(1) = ME
      IDOPE(2) = NSOLN
      IDOPE(3) = L1
C
      DOPE(1) = ALSQ
      DOPE(2) = EANORM
      DOPE(3) = TAU
      CALL DWNLIT (W, MDW, M, N, L, IPIVOT, ITYPE, H, SCALE, RNORM,
     +            IDOPE, DOPE, DONE)
      ME    = IDOPE(1)
      KRANK = IDOPE(2)
      NIV   = IDOPE(3)
C
C     Perform WNNLS algorithm using the following steps.
C
C     Until(DONE)
C        compute search direction and feasible point
C        when (HITCON) add constraints
C        else perform multiplier test and drop a constraint
C        fin
C     Compute-Final-Solution
C
C     To compute search direction and feasible point,
C     solve the triangular system of currently non-active
C     variables and store the solution in Z(*).
C
C     To solve system
C     Copy right hand side into TEMP vector to use overwriting method.
C
  160 IF (DONE) GO TO 330
      ISOL = L + 1
      IF (NSOLN.GE.ISOL) THEN
         CALL DCOPY (NIV, W(1,N+1), 1, TEMP, 1)
         DO 170 J = NSOLN,ISOL,-1
            IF (J.GT.KRANK) THEN
               I = NIV - NSOLN + J
            ELSE
               I = J
            ENDIF
C
            IF (J.GT.KRANK .AND. J.LE.L) THEN
               Z(J) = 0.D0
            ELSE
               Z(J) = TEMP(I)/W(I,J)
               CALL DAXPY (I-1, -Z(J), W(1,J), 1, TEMP, 1)
            ENDIF
  170    CONTINUE
      ENDIF
C
C     Increment iteration counter and check against maximum number
C     of iterations.
C
      ITER = ITER + 1
      IF (ITER.GT.ITMAX) THEN
         MODE = 1
         DONE = .TRUE.
      ENDIF
C
C     Check to see if any constraints have become active.
C     If so, calculate an interpolation factor so that all
C     active constraints are removed from the basis.
C
      ALPHA = 2.D0
      HITCON = .FALSE.
      DO 180 J = L+1,NSOLN
         ZZ = Z(J)
         IF (ZZ.LE.0.D0) THEN
            T = X(J)/(X(J)-ZZ)
            IF (T.LT.ALPHA) THEN
               ALPHA = T
               JCON = J
            ENDIF
            HITCON = .TRUE.
         ENDIF
  180 CONTINUE
C
C     Compute search direction and feasible point
C
      IF (HITCON) THEN
C
C        To add constraints, use computed ALPHA to interpolate between
C        last feasible solution X(*) and current unconstrained (and
C        infeasible) solution Z(*).
C
         DO 190 J = L+1,NSOLN
            X(J) = X(J) + ALPHA*(Z(J)-X(J))
  190    CONTINUE
         FEASBL = .FALSE.
C
C        Remove column JCON and shift columns JCON+1 through N to the
C        left.  Swap column JCON into the N th position.  This achieves
C        upper Hessenberg form for the nonactive constraints and
C        leaves an upper Hessenberg matrix to retriangularize.
C
  200    DO 210 I = 1,M
            T = W(I,JCON)
            CALL DCOPY (N-JCON, W(I, JCON+1), MDW, W(I, JCON), MDW)
            W(I,N) = T
  210    CONTINUE
C
C        Update permuted index vector to reflect this shift and swap.
C
         ITEMP = IPIVOT(JCON)
         DO 220 I = JCON,N - 1
            IPIVOT(I) = IPIVOT(I+1)
  220    CONTINUE
         IPIVOT(N) = ITEMP
C
C        Similarly permute X(*) vector.
C
         CALL DCOPY (N-JCON, X(JCON+1), 1, X(JCON), 1)
         X(N) = 0.D0
         NSOLN = NSOLN - 1
         NIV = NIV - 1
C
C        Retriangularize upper Hessenberg matrix after adding
C        constraints.
C
         I = KRANK + JCON - L
         DO 230 J = JCON,NSOLN
            IF (ITYPE(I).EQ.0 .AND. ITYPE(I+1).EQ.0) THEN
C
C              Zero IP1 to I in column J
C
               IF (W(I+1,J).NE.0.D0) THEN
                  CALL SLATEC_DROTMG (SCALE(I), SCALE(I+1), W(I,J),
     +                         W(I+1,J), SPARAM)
                  W(I+1,J) = 0.D0
                  CALL SLATEC_DROTM (N+1-J, W(I,J+1), MDW, W(I+1,J+1),
     +                        MDW, SPARAM)
               ENDIF
            ELSEIF (ITYPE(I).EQ.1 .AND. ITYPE(I+1).EQ.1) THEN
C
C              Zero IP1 to I in column J
C
               IF (W(I+1,J).NE.0.D0) THEN
                  CALL SLATEC_DROTMG (SCALE(I), SCALE(I+1), W(I,J),
     +                         W(I+1,J), SPARAM)
                  W(I+1,J) = 0.D0
                  CALL SLATEC_DROTM (N+1-J, W(I,J+1), MDW, W(I+1,J+1),
     +                        MDW, SPARAM)
               ENDIF
            ELSEIF (ITYPE(I).EQ.1 .AND. ITYPE(I+1).EQ.0) THEN
               CALL DSWAP (N+1, W(I,1), MDW, W(I+1,1), MDW)
               CALL DSWAP (1, SCALE(I), 1, SCALE(I+1), 1)
               ITEMP = ITYPE(I+1)
               ITYPE(I+1) = ITYPE(I)
               ITYPE(I) = ITEMP
C
C              Swapped row was formerly a pivot element, so it will
C              be large enough to perform elimination.
C              Zero IP1 to I in column J.
C
               IF (W(I+1,J).NE.0.D0) THEN
                  CALL SLATEC_DROTMG (SCALE(I), SCALE(I+1), W(I,J),
     +                         W(I+1,J), SPARAM)
                  W(I+1,J) = 0.D0
                  CALL SLATEC_DROTM (N+1-J, W(I,J+1), MDW, W(I+1,J+1),
     +                        MDW, SPARAM)
               ENDIF
            ELSEIF (ITYPE(I).EQ.0 .AND. ITYPE(I+1).EQ.1) THEN
               IF (SCALE(I)*W(I,J)**2/ALSQ.GT.(TAU*EANORM)**2) THEN
C
C                 Zero IP1 to I in column J
C
                  IF (W(I+1,J).NE.0.D0) THEN
                     CALL SLATEC_DROTMG (SCALE(I), SCALE(I+1), W(I,J),
     +                            W(I+1,J), SPARAM)
                     W(I+1,J) = 0.D0
                     CALL SLATEC_DROTM (N+1-J, W(I,J+1), MDW,
     +                           W(I+1,J+1), MDW, SPARAM)
                  ENDIF
               ELSE
                  CALL DSWAP (N+1, W(I,1), MDW, W(I+1,1), MDW)
                  CALL DSWAP (1, SCALE(I), 1, SCALE(I+1), 1)
                  ITEMP = ITYPE(I+1)
                  ITYPE(I+1) = ITYPE(I)
                  ITYPE(I) = ITEMP
                  W(I+1,J) = 0.D0
               ENDIF
            ENDIF
            I = I + 1
  230    CONTINUE
C
C        See if the remaining coefficients in the solution set are
C        feasible.  They should be because of the way ALPHA was
C        determined.  If any are infeasible, it is due to roundoff
C        error.  Any that are non-positive will be set to zero and
C        removed from the solution set.
C
         DO 240 JCON = L+1,NSOLN
            IF (X(JCON).LE.0.D0) GO TO 250
  240    CONTINUE
         FEASBL = .TRUE.
  250    IF (.NOT.FEASBL) GO TO 200
      ELSE
C
C        To perform multiplier test and drop a constraint.
C
         CALL DCOPY (NSOLN, Z, 1, X, 1)
         IF (NSOLN.LT.N) CALL DCOPY (N-NSOLN, 0.D0, 0, X(NSOLN+1), 1)
C
C        Reclassify least squares equations as equalities as necessary.
C
         I = NIV + 1
  260    IF (I.LE.ME) THEN
            IF (ITYPE(I).EQ.0) THEN
               I = I + 1
            ELSE
               CALL DSWAP (N+1, W(I,1), MDW, W(ME,1), MDW)
               CALL DSWAP (1, SCALE(I), 1, SCALE(ME), 1)
               ITEMP = ITYPE(I)
               ITYPE(I) = ITYPE(ME)
               ITYPE(ME) = ITEMP
               ME = ME - 1
            ENDIF
            GO TO 260
         ENDIF
C
C        Form inner product vector WD(*) of dual coefficients.
C
         DO 280 J = NSOLN+1,N
            SM = 0.D0
            DO 270 I = NSOLN+1,M
               SM = SM + SCALE(I)*W(I,J)*W(I,N+1)
  270       CONTINUE
            WD(J) = SM
  280    CONTINUE
C
C        Find J such that WD(J)=WMAX is maximum.  This determines
C        that the incoming column J will reduce the residual vector
C        and be positive.
C
  290    WMAX = 0.D0
         IWMAX = NSOLN + 1
         DO 300 J = NSOLN+1,N
            IF (WD(J).GT.WMAX) THEN
               WMAX = WD(J)
               IWMAX = J
            ENDIF
  300    CONTINUE
         IF (WMAX.LE.0.D0) GO TO 330
C
C        Set dual coefficients to zero for incoming column.
C
         WD(IWMAX) = 0.D0
C
C        WMAX .GT. 0.D0, so okay to move column IWMAX to solution set.
C        Perform transformation to retriangularize, and test for near
C        linear dependence.
C
C        Swap column IWMAX into NSOLN-th position to maintain upper
C        Hessenberg form of adjacent columns, and add new column to
C        triangular decomposition.
C
         NSOLN = NSOLN + 1
         NIV = NIV + 1
         IF (NSOLN.NE.IWMAX) THEN
            CALL DSWAP (M, W(1,NSOLN), 1, W(1,IWMAX), 1)
            WD(IWMAX) = WD(NSOLN)
            WD(NSOLN) = 0.D0
            ITEMP = IPIVOT(NSOLN)
            IPIVOT(NSOLN) = IPIVOT(IWMAX)
            IPIVOT(IWMAX) = ITEMP
         ENDIF
C
C        Reduce column NSOLN so that the matrix of nonactive constraints
C        variables is triangular.
C
         DO 320 J = M,NIV+1,-1
            JP = J - 1
C
C           When operating near the ME line, test to see if the pivot
C           element is near zero.  If so, use the largest element above
C           it as the pivot.  This is to maintain the sharp interface
C           between weighted and non-weighted rows in all cases.
C
            IF (J.EQ.ME+1) THEN
               IMAX = ME
               AMAX = SCALE(ME)*W(ME,NSOLN)**2
               DO 310 JP = J - 1,NIV,-1
                  T = SCALE(JP)*W(JP,NSOLN)**2
                  IF (T.GT.AMAX) THEN
                     IMAX = JP
                     AMAX = T
                  ENDIF
  310          CONTINUE
               JP = IMAX
            ENDIF
C
            IF (W(J,NSOLN).NE.0.D0) THEN
               CALL SLATEC_DROTMG (SCALE(JP), SCALE(J), W(JP,NSOLN),
     +                      W(J,NSOLN), SPARAM)
               W(J,NSOLN) = 0.D0
               CALL SLATEC_DROTM (N+1-NSOLN, W(JP,NSOLN+1), MDW,
     +                     W(J,NSOLN+1), MDW, SPARAM)
            ENDIF
  320    CONTINUE
C
C        Solve for Z(NSOLN)=proposed new value for X(NSOLN).  Test if
C        this is nonpositive or too large.  If this was true or if the
C        pivot term was zero, reject the column as dependent.
C
         IF (W(NIV,NSOLN).NE.0.D0) THEN
            ISOL = NIV
            Z2 = W(ISOL,N+1)/W(ISOL,NSOLN)
            Z(NSOLN) = Z2
            POS = Z2 .GT. 0.D0
            IF (Z2*EANORM.GE.BNORM .AND. POS) THEN
               POS = .NOT. (BLOWUP*Z2*EANORM.GE.BNORM)
            ENDIF
C
C           Try to add row ME+1 as an additional equality constraint.
C           Check size of proposed new solution component.
C           Reject it if it is too large.
C
         ELSEIF (NIV.LE.ME .AND. W(ME+1,NSOLN).NE.0.D0) THEN
            ISOL = ME + 1
            IF (POS) THEN
C
C              Swap rows ME+1 and NIV, and scale factors for these rows.
C
               CALL DSWAP (N+1, W(ME+1,1), MDW, W(NIV,1), MDW)
               CALL DSWAP (1, SCALE(ME+1), 1, SCALE(NIV), 1)
               ITEMP = ITYPE(ME+1)
               ITYPE(ME+1) = ITYPE(NIV)
               ITYPE(NIV) = ITEMP
               ME = ME + 1
            ENDIF
         ELSE
            POS = .FALSE.
         ENDIF
C
         IF (.NOT.POS) THEN
            NSOLN = NSOLN - 1
            NIV = NIV - 1
         ENDIF
         IF (.NOT.(POS.OR.DONE)) GO TO 290
      ENDIF
      GO TO 160
C
C     Else perform multiplier test and drop a constraint.  To compute
C     final solution.  Solve system, store results in X(*).
C
C     Copy right hand side into TEMP vector to use overwriting method.
C
  330 ISOL = 1
      IF (NSOLN.GE.ISOL) THEN
         CALL DCOPY (NIV, W(1,N+1), 1, TEMP, 1)
         DO 340 J = NSOLN,ISOL,-1
            IF (J.GT.KRANK) THEN
               I = NIV - NSOLN + J
            ELSE
               I = J
            ENDIF
C
            IF (J.GT.KRANK .AND. J.LE.L) THEN
               Z(J) = 0.D0
            ELSE
               Z(J) = TEMP(I)/W(I,J)
               CALL DAXPY (I-1, -Z(J), W(1,J), 1, TEMP, 1)
            ENDIF
  340    CONTINUE
      ENDIF
C
C     Solve system.
C
      CALL DCOPY (NSOLN, Z, 1, X, 1)
C
C     Apply Householder transformations to X(*) if KRANK.LT.L
C
      IF (KRANK.LT.L) THEN
         DO 350 I = 1,KRANK
            CALL DH12 (2, I, KRANK+1, L, W(I,1), MDW, H(I), X, 1, 1, 1)
  350    CONTINUE
      ENDIF
C
C     Fill in trailing zeroes for constrained variables not in solution.
C
      IF (NSOLN.LT.N) CALL DCOPY (N-NSOLN, 0.D0, 0, X(NSOLN+1), 1)
C
C     Permute solution vector to natural order.
C
      DO 380 I = 1,N
         J = I
  360    IF (IPIVOT(J).EQ.I) GO TO 370
         J = J + 1
         GO TO 360
C
  370    IPIVOT(J) = IPIVOT(I)
         IPIVOT(I) = J
         CALL DSWAP (1, X(J), 1, X(I), 1)
  380 CONTINUE
C
C     Rescale the solution using the column scaling.
C
      DO 390 J = 1,N
         X(J) = X(J)*D(J)
  390 CONTINUE
C
      DO 400 I = NSOLN+1,M
         T = W(I,N+1)
         IF (I.LE.ME) T = T/ALAMDA
         T = (SCALE(I)*T)*T
         RNORM = RNORM + T
  400 CONTINUE
C
      RNORM = SQRT(RNORM)
      RETURN
      END
*DECK DROTM
      SUBROUTINE SLATEC_DROTM (N, DX, INCX, DY, INCY, DPARAM)
C***BEGIN PROLOGUE  SLATEC_DROTM
C***PURPOSE  Apply a modified Givens transformation.
C***LIBRARY   SLATEC (BLAS)
C***CATEGORY  D1A8
C***TYPE      REAL(KIND=R8) (SROTM-S, DROTM-D)
C***KEYWORDS  BLAS, LINEAR ALGEBRA, MODIFIED GIVENS ROTATION, VECTOR
C***AUTHOR  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***DESCRIPTION
C
C                B L A S  Subprogram
C    Description of Parameters
C
C     --Input--
C        N  number of elements in input vector(s)
C       DX  double precision vector with N elements
C     INCX  storage spacing between elements of DX
C       DY  double precision vector with N elements
C     INCY  storage spacing between elements of DY
C   DPARAM  5-element D.P. vector.  DPARAM(1) is DFLAG described below.
C           Locations 2-5 of SPARAM contain elements of the
C           transformation matrix H described below.
C
C     --Output--
C       DX  rotated vector (unchanged if N .LE. 0)
C       DY  rotated vector (unchanged if N .LE. 0)
C
C     Apply the modified Givens transformation, H, to the 2 by N matrix
C     (DX**T)
C     (DY**T) , where **T indicates transpose.  The elements of DX are
C     in DX(LX+I*INCX), I = 0 to N-1, where LX = 1 if INCX .GE. 0, else
C     LX = 1+(1-N)*INCX, and similarly for DY using LY and INCY.
C
C     With DPARAM(1)=DFLAG, H has one of the following forms:
C
C     DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
C
C       (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
C     H=(          )    (          )    (          )    (          )
C       (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
C
C     See SLATEC_DROTMG for a description of data storage in DPARAM.
C
C***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
C                 Krogh, Basic linear algebra subprograms for Fortran
C                 usage, Algorithm No. 539, Transactions on Mathematical
C                 Software 5, 3 (September 1979), pp. 308-323.
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   791001  DATE WRITTEN
C   861211  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   920310  Corrected definition of LX in DESCRIPTION.  (WRB)
C   920501  Reformatted the REFERENCES section.  (WRB)
C   180613  Renamed SLATEC_DROTM to avoid BLAS naming conflict. (THC)
C***END PROLOGUE  SLATEC_DROTM
      USE REAL_PRECISION

      REAL(KIND=R8) DFLAG, DH12, DH22, DX, TWO, Z, DH11, DH21,
     1                 DPARAM, DY, W, ZERO
      DIMENSION DX(*), DY(*), DPARAM(5)
      SAVE ZERO, TWO
      DATA ZERO, TWO /0.0D0, 2.0D0/
C***FIRST EXECUTABLE STATEMENT  SLATEC_DROTM
      DFLAG=DPARAM(1)
      IF (N.LE.0 .OR. (DFLAG+TWO.EQ.ZERO)) GO TO 140
          IF (.NOT.(INCX.EQ.INCY.AND. INCX .GT.0)) GO TO 70
C
               NSTEPS=N*INCX
C              IF (DFLAG) 50, 10, 30
C              Replaced obsolete code above with an IF-block (THC).
               IF (DFLAG < 0) THEN
                    GO TO 50
               ELSE IF (DFLAG == 0) THEN
                    GO TO 10
               ELSE IF (DFLAG > 0) THEN
                    GO TO 30
               END IF
   10          CONTINUE
               DH12=DPARAM(4)
               DH21=DPARAM(3)
                    DO 20 I = 1,NSTEPS,INCX
                    W=DX(I)
                    Z=DY(I)
                    DX(I)=W+Z*DH12
                    DY(I)=W*DH21+Z
   20               CONTINUE
               GO TO 140
   30          CONTINUE
               DH11=DPARAM(2)
               DH22=DPARAM(5)
                    DO 40 I = 1,NSTEPS,INCX
                    W=DX(I)
                    Z=DY(I)
                    DX(I)=W*DH11+Z
                    DY(I)=-W+DH22*Z
   40               CONTINUE
               GO TO 140
   50          CONTINUE
               DH11=DPARAM(2)
               DH12=DPARAM(4)
               DH21=DPARAM(3)
               DH22=DPARAM(5)
                    DO 60 I = 1,NSTEPS,INCX
                    W=DX(I)
                    Z=DY(I)
                    DX(I)=W*DH11+Z*DH12
                    DY(I)=W*DH21+Z*DH22
   60               CONTINUE
               GO TO 140
   70     CONTINUE
          KX=1
          KY=1
          IF (INCX .LT. 0) KX = 1+(1-N)*INCX
          IF (INCY .LT. 0) KY = 1+(1-N)*INCY
C
C         IF (DFLAG) 120,80,100
C         Replaced obsolete code above with an IF-block (THC).
          IF (DFLAG < 0) THEN
               GO TO 120
          ELSE IF (DFLAG == 0) THEN
               GO TO 80
          ELSE IF (DFLAG > 0) THEN
               GO TO 100
          END IF
   80     CONTINUE
          DH12=DPARAM(4)
          DH21=DPARAM(3)
               DO 90 I = 1,N
               W=DX(KX)
               Z=DY(KY)
               DX(KX)=W+Z*DH12
               DY(KY)=W*DH21+Z
               KX=KX+INCX
               KY=KY+INCY
   90          CONTINUE
          GO TO 140
  100     CONTINUE
          DH11=DPARAM(2)
          DH22=DPARAM(5)
               DO 110 I = 1,N
               W=DX(KX)
               Z=DY(KY)
               DX(KX)=W*DH11+Z
               DY(KY)=-W+DH22*Z
               KX=KX+INCX
               KY=KY+INCY
  110          CONTINUE
          GO TO 140
  120     CONTINUE
          DH11=DPARAM(2)
          DH12=DPARAM(4)
          DH21=DPARAM(3)
          DH22=DPARAM(5)
               DO 130 I = 1,N
               W=DX(KX)
               Z=DY(KY)
               DX(KX)=W*DH11+Z*DH12
               DY(KY)=W*DH21+Z*DH22
               KX=KX+INCX
               KY=KY+INCY
  130          CONTINUE
  140     CONTINUE
          RETURN
      END
*DECK SLATEC_DROTMG
      SUBROUTINE SLATEC_DROTMG (DD1, DD2, DX1, DY1, DPARAM)
C***BEGIN PROLOGUE  SLATEC_DROTMG
C***PURPOSE  Construct a modified Givens transformation.
C***LIBRARY   SLATEC (BLAS)
C***CATEGORY  D1B10
C***TYPE      REAL(KIND=R8) (SROTMG-S, DROTMG-D)
C***KEYWORDS  BLAS, LINEAR ALGEBRA, MODIFIED GIVENS ROTATION, VECTOR
C***AUTHOR  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***DESCRIPTION
C
C                B L A S  Subprogram
C    Description of Parameters
C
C     --Input--
C      DD1  double precision scalar
C      DD2  double precision scalar
C      DX1  double precision scalar
C      DX2  double precision scalar
C   DPARAM  D.P. 5-vector. DPARAM(1)=DFLAG defined below.
C           Locations 2-5 contain the rotation matrix.
C
C     --Output--
C      DD1  changed to represent the effect of the transformation
C      DD2  changed to represent the effect of the transformation
C      DX1  changed to represent the effect of the transformation
C      DX2  unchanged
C
C     Construct the modified Givens transformation matrix H which zeros
C     the second component of the 2-vector  (SQRT(DD1)*DX1,SQRT(DD2)*
C     DY2)**T.
C     With DPARAM(1)=DFLAG, H has one of the following forms:
C
C     DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
C
C       (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
C     H=(          )    (          )    (          )    (          )
C       (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
C
C     Locations 2-5 of DPARAM contain DH11, DH21, DH12, and DH22,
C     respectively.  (Values of 1.D0, -1.D0, or 0.D0 implied by the
C     value of DPARAM(1) are not stored in DPARAM.)
C
C***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
C                 Krogh, Basic linear algebra subprograms for Fortran
C                 usage, Algorithm No. 539, Transactions on Mathematical
C                 Software 5, 3 (September 1979), pp. 308-323.
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   780301  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890531  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   920316  Prologue corrected.  (WRB)
C   920501  Reformatted the REFERENCES section.  (WRB)
C   180613  Renamed SLATEC_DROTMG to avoid BLAS naming conflict. (THC)
C***END PROLOGUE  SLATEC_DROTMG
      USE REAL_PRECISION

      REAL(KIND=R8) GAM, ONE, RGAMSQ, DD1, DD2, DH11, DH12, DH21,
     1                 DH22, DPARAM, DP1, DP2, DQ1, DQ2, DU, DY1, ZERO,
     2                 GAMSQ, DFLAG, DTEMP, DX1, TWO
      DIMENSION DPARAM(5)
      SAVE ZERO, ONE, TWO, GAM, GAMSQ, RGAMSQ
      DATA ZERO, ONE, TWO /0.0D0, 1.0D0, 2.0D0/
      DATA GAM, GAMSQ, RGAMSQ /4096.0D0, 16777216.D0, 5.9604645D-8/
C***FIRST EXECUTABLE STATEMENT  SLATEC_DROTMG
      IF (.NOT. DD1 .LT. ZERO) GO TO 10
C       GO ZERO-H-D-AND-DX1..
          GO TO 60
   10 CONTINUE
C     CASE-DD1-NONNEGATIVE
      DP2=DD2*DY1
      IF (.NOT. DP2 .EQ. ZERO) GO TO 20
          DFLAG=-TWO
          GO TO 260
C     REGULAR-CASE..
   20 CONTINUE
      DP1=DD1*DX1
      DQ2=DP2*DY1
      DQ1=DP1*DX1
C
      IF (.NOT. ABS(DQ1) .GT. ABS(DQ2)) GO TO 40
          DH21=-DY1/DX1
          DH12=DP2/DP1
C
          DU=ONE-DH12*DH21
C
          IF (.NOT. DU .LE. ZERO) GO TO 30
C         GO ZERO-H-D-AND-DX1..
               GO TO 60
   30     CONTINUE
               DFLAG=ZERO
               DD1=DD1/DU
               DD2=DD2/DU
               DX1=DX1*DU
C         GO SCALE-CHECK..
               GO TO 100
   40 CONTINUE
          IF (.NOT. DQ2 .LT. ZERO) GO TO 50
C         GO ZERO-H-D-AND-DX1..
               GO TO 60
   50     CONTINUE
               DFLAG=ONE
               DH11=DP1/DP2
               DH22=DX1/DY1
               DU=ONE+DH11*DH22
               DTEMP=DD2/DU
               DD2=DD1/DU
               DD1=DTEMP
               DX1=DY1*DU
C         GO SCALE-CHECK
               GO TO 100
C     PROCEDURE..ZERO-H-D-AND-DX1..
   60 CONTINUE
          DFLAG=-ONE
          DH11=ZERO
          DH12=ZERO
          DH21=ZERO
          DH22=ZERO
C
          DD1=ZERO
          DD2=ZERO
          DX1=ZERO
C         RETURN..
          GO TO 220
C     PROCEDURE..FIX-H..
   70 CONTINUE
      IF (.NOT. DFLAG .GE. ZERO) GO TO 90
C
          IF (.NOT. DFLAG .EQ. ZERO) GO TO 80
          DH11=ONE
          DH22=ONE
          DFLAG=-ONE
          GO TO 90
   80     CONTINUE
          DH21=-ONE
          DH12=ONE
          DFLAG=-ONE
   90 CONTINUE
C     GO TO IGO,(120,150,180,210)
C     Replaced the above obsolete code with modern alternative (THC).
      SELECT CASE(IGO)
      CASE(120)
          GO TO 120
      CASE(150)
          GO TO 150
      CASE(180)
          GO TO 180
      CASE(210)
          GO TO 210
      END SELECT
C     PROCEDURE..SCALE-CHECK
  100 CONTINUE
  110     CONTINUE
          IF (.NOT. DD1 .LE. RGAMSQ) GO TO 130
               IF (DD1 .EQ. ZERO) GO TO 160
               IGO = 120
C              FIX-H..
               GO TO 70
  120          CONTINUE
               DD1=DD1*GAM**2
               DX1=DX1/GAM
               DH11=DH11/GAM
               DH12=DH12/GAM
          GO TO 110
  130 CONTINUE
  140     CONTINUE
          IF (.NOT. DD1 .GE. GAMSQ) GO TO 160
               IGO = 150
C              FIX-H..
               GO TO 70
  150          CONTINUE
               DD1=DD1/GAM**2
               DX1=DX1*GAM
               DH11=DH11*GAM
               DH12=DH12*GAM
          GO TO 140
  160 CONTINUE
  170     CONTINUE
          IF (.NOT. ABS(DD2) .LE. RGAMSQ) GO TO 190
               IF (DD2 .EQ. ZERO) GO TO 220
               IGO = 180
C              FIX-H..
               GO TO 70
  180          CONTINUE
               DD2=DD2*GAM**2
               DH21=DH21/GAM
               DH22=DH22/GAM
          GO TO 170
  190 CONTINUE
  200     CONTINUE
          IF (.NOT. ABS(DD2) .GE. GAMSQ) GO TO 220
               IGO = 210
C              FIX-H..
               GO TO 70
  210          CONTINUE
               DD2=DD2/GAM**2
               DH21=DH21*GAM
               DH22=DH22*GAM
          GO TO 200
  220 CONTINUE
C         IF (DFLAG) 250,230,240
C         Replaced obsolete code above with an IF-block (THC).
          IF (DFLAG < 0) THEN
               GO TO 250
          ELSE IF (DFLAG == 0) THEN
               GO TO 230
          ELSE IF (DFLAG > 0) THEN
               GO TO 240
          END IF

  230     CONTINUE
               DPARAM(3)=DH21
               DPARAM(4)=DH12
               GO TO 260
  240     CONTINUE
               DPARAM(2)=DH11
               DPARAM(5)=DH22
               GO TO 260
  250     CONTINUE
               DPARAM(2)=DH11
               DPARAM(3)=DH21
               DPARAM(4)=DH12
               DPARAM(5)=DH22
  260 CONTINUE
          DPARAM(1)=DFLAG
          RETURN
      END
*DECK DWNLIT
      SUBROUTINE DWNLIT (W, MDW, M, N, L, IPIVOT, ITYPE, H, SCALE,
     +   RNORM, IDOPE, DOPE, DONE)
C***BEGIN PROLOGUE  DWNLIT
C***SUBSIDIARY
C***PURPOSE  Subsidiary to DWNNLS
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (WNLIT-S, DWNLIT-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     This is a companion subprogram to DWNNLS( ).
C     The documentation for DWNNLS( ) has complete usage instructions.
C
C     Note  The M by (N+1) matrix W( , ) contains the rt. hand side
C           B as the (N+1)st col.
C
C     Triangularize L1 by L1 subsystem, where L1=MIN(M,L), with
C     col interchanges.
C
C***SEE ALSO  DWNNLS
C***ROUTINES CALLED  DCOPY, DH12, SLATEC_DROTM, SLATEC_DROTMG, DSCAL,
C                    DSWAP, DWNLT1, DWNLT2, DWNLT3, IDAMAX
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890618  Completely restructured and revised.  (WRB & RWC)
C   890620  Revised to make WNLT1, WNLT2, and WNLT3 subroutines.  (RWC)
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900328  Added TYPE section.  (WRB)
C   900604  DP version created from SP version. .  (RWC)
C***END PROLOGUE  DWNLIT
      USE REAL_PRECISION

      INTEGER IDOPE(*), IPIVOT(*), ITYPE(*), L, M, MDW, N
      REAL(KIND=R8) DOPE(*), H(*), RNORM, SCALE(*), W(MDW,*)
      LOGICAL DONE
C
      EXTERNAL DCOPY, DH12, SLATEC_DROTM, SLATEC_DROTMG, DSCAL, DSWAP,
     *   DWNLT1, DWNLT2, DWNLT3, IDAMAX
      INTEGER IDAMAX
      LOGICAL DWNLT2
C
      REAL(KIND=R8) ALSQ, AMAX, EANORM, FACTOR, HBAR, RN, SPARAM(5),
     *   T, TAU
      INTEGER I, I1, IMAX, IR, J, J1, JJ, JP, KRANK, L1, LB, LEND, ME,
     *   MEND, NIV, NSOLN
      LOGICAL INDEP, RECALC
C
C***FIRST EXECUTABLE STATEMENT  DWNLIT
      ME    = IDOPE(1)
      NSOLN = IDOPE(2)
      L1    = IDOPE(3)
C
      ALSQ   = DOPE(1)
      EANORM = DOPE(2)
      TAU    = DOPE(3)
C
      LB     = MIN(M-1,L)
      RECALC = .TRUE.
      RNORM  = 0.D0
      KRANK  = 0
C
C     We set FACTOR=1.0 so that the heavy weight ALAMDA will be
C     included in the test for column independence.
C
      FACTOR = 1.D0
      LEND = L
      DO 180 I=1,LB
C
C        Set IR to point to the I-th row.
C
         IR = I
         MEND = M
         CALL DWNLT1 (I, LEND, M, IR, MDW, RECALC, IMAX, HBAR, H, SCALE,
     +                W)
C
C        Update column SS and find pivot column.
C
         CALL DWNLT3 (I, IMAX, M, MDW, IPIVOT, H, W)
C
C        Perform column interchange.
C        Test independence of incoming column.
C
  130    IF (DWNLT2(ME, MEND, IR, FACTOR, TAU, SCALE, W(1,I))) THEN
C
C           Eliminate I-th column below diagonal using modified Givens
C           transformations applied to (A B).
C
C           When operating near the ME line, use the largest element
C           above it as the pivot.
C
            DO 160 J=M,I+1,-1
               JP = J-1
               IF (J.EQ.ME+1) THEN
                  IMAX = ME
                  AMAX = SCALE(ME)*W(ME,I)**2
                  DO 150 JP=J-1,I,-1
                     T = SCALE(JP)*W(JP,I)**2
                     IF (T.GT.AMAX) THEN
                        IMAX = JP
                        AMAX = T
                     ENDIF
  150             CONTINUE
                  JP = IMAX
               ENDIF
C
               IF (W(J,I).NE.0.D0) THEN
                  CALL SLATEC_DROTMG (SCALE(JP), SCALE(J), W(JP,I),
     +                         W(J,I), SPARAM)
                  W(J,I) = 0.D0
                  CALL SLATEC_DROTM (N+1-I, W(JP,I+1), MDW, W(J,I+1),
     +                        MDW, SPARAM)
               ENDIF
  160       CONTINUE
         ELSE IF (LEND.GT.I) THEN
C
C           Column I is dependent.  Swap with column LEND.
C           Perform column interchange,
C           and find column in remaining set with largest SS.
C
            CALL DWNLT3 (I, LEND, M, MDW, IPIVOT, H, W)
            LEND = LEND - 1
            IMAX = IDAMAX(LEND-I+1, H(I), 1) + I - 1
            HBAR = H(IMAX)
            GO TO 130
         ELSE
            KRANK = I - 1
            GO TO 190
         ENDIF
  180 CONTINUE
      KRANK = L1
C
  190 IF (KRANK.LT.ME) THEN
         FACTOR = ALSQ
         DO 200 I=KRANK+1,ME
            CALL DCOPY (L, 0.D0, 0, W(I,1), MDW)
  200    CONTINUE
C
C        Determine the rank of the remaining equality constraint
C        equations by eliminating within the block of constrained
C        variables.  Remove any redundant constraints.
C
         RECALC = .TRUE.
         LB = MIN(L+ME-KRANK, N)
         DO 270 I=L+1,LB
            IR = KRANK + I - L
            LEND = N
            MEND = ME
            CALL DWNLT1 (I, LEND, ME, IR, MDW, RECALC, IMAX, HBAR, H,
     +                   SCALE, W)
C
C           Update col ss and find pivot col
C
            CALL DWNLT3 (I, IMAX, M, MDW, IPIVOT, H, W)
C
C           Perform column interchange
C           Eliminate elements in the I-th col.
C
            DO 240 J=ME,IR+1,-1
               IF (W(J,I).NE.0.D0) THEN
                 CALL SLATEC_DROTMG (SCALE(J-1), SCALE(J), W(J-1,I),
     +                        W(J,I), SPARAM)
                  W(J,I) = 0.D0
                  CALL SLATEC_DROTM (N+1-I, W(J-1,I+1), MDW,W(J,I+1),
     +                        MDW, SPARAM)
               ENDIF
  240       CONTINUE
C
C           I=column being eliminated.
C           Test independence of incoming column.
C           Remove any redundant or dependent equality constraints.
C
            IF (.NOT.DWNLT2(ME, MEND, IR, FACTOR,TAU,SCALE,W(1,I))) THEN
               JJ = IR
               DO 260 IR=JJ,ME
                  CALL DCOPY (N, 0.D0, 0, W(IR,1), MDW)
                  RNORM = RNORM + (SCALE(IR)*W(IR,N+1)/ALSQ)*W(IR,N+1)
                  W(IR,N+1) = 0.D0
                  SCALE(IR) = 1.D0
C
C                 Reclassify the zeroed row as a least squares equation.
C
                  ITYPE(IR) = 1
  260          CONTINUE
C
C              Reduce ME to reflect any discovered dependent equality
C              constraints.
C
               ME = JJ - 1
               GO TO 280
            ENDIF
  270    CONTINUE
      ENDIF
C
C     Try to determine the variables KRANK+1 through L1 from the
C     least squares equations.  Continue the triangularization with
C     pivot element W(ME+1,I).
C
  280 IF (KRANK.LT.L1) THEN
         RECALC = .TRUE.
C
C        Set FACTOR=ALSQ to remove effect of heavy weight from
C        test for column independence.
C
         FACTOR = ALSQ
         DO 350 I=KRANK+1,L1
C
C           Set IR to point to the ME+1-st row.
C
            IR = ME+1
            LEND = L
            MEND = M
            CALL DWNLT1 (I, L, M, IR, MDW, RECALC, IMAX, HBAR, H, SCALE,
     +                   W)
C
C           Update column SS and find pivot column.
C
            CALL DWNLT3 (I, IMAX, M, MDW, IPIVOT, H, W)
C
C           Perform column interchange.
C           Eliminate I-th column below the IR-th element.
C
            DO 320 J=M,IR+1,-1
               IF (W(J,I).NE.0.D0) THEN
                 CALL SLATEC_DROTMG (SCALE(J-1), SCALE(J), W(J-1,I),
     +                        W(J,I), SPARAM)
                  W(J,I) = 0.D0
                  CALL SLATEC_DROTM (N+1-I, W(J-1,I+1),  MDW, W(J,I+1),
     +                        MDW, SPARAM)
               ENDIF
  320       CONTINUE
C
C           Test if new pivot element is near zero.
C           If so, the column is dependent.
C           Then check row norm test to be classified as independent.
C
            T = SCALE(IR)*W(IR,I)**2
            INDEP = T .GT. (TAU*EANORM)**2
            IF (INDEP) THEN
               RN = 0.D0
               DO 340 I1=IR,M
                  DO 330 J1=I+1,N
                     RN = MAX(RN, SCALE(I1)*W(I1,J1)**2)
  330             CONTINUE
  340          CONTINUE
               INDEP = T .GT. RN*TAU**2
            ENDIF
C
C           If independent, swap the IR-th and KRANK+1-th rows to
C           maintain the triangular form.  Update the rank indicator
C           KRANK and the equality constraint pointer ME.
C
            IF (.NOT.INDEP) GO TO 360
            CALL DSWAP(N+1, W(KRANK+1,1), MDW, W(IR,1), MDW)
            CALL DSWAP(1, SCALE(KRANK+1), 1, SCALE(IR), 1)
C
C           Reclassify the least square equation as an equality
C           constraint and rescale it.
C
            ITYPE(IR) = 0
            T = SQRT(SCALE(KRANK+1))
            CALL DSCAL(N+1, T, W(KRANK+1,1), MDW)
            SCALE(KRANK+1) = ALSQ
            ME = ME+1
            KRANK = KRANK+1
  350    CONTINUE
      ENDIF
C
C     If pseudorank is less than L, apply Householder transformation.
C     from right.
C
  360 IF (KRANK.LT.L) THEN
         DO 370 J=KRANK,1,-1
            CALL DH12 (1, J, KRANK+1, L, W(J,1), MDW, H(J), W, MDW, 1,
     +                J-1)
  370    CONTINUE
      ENDIF
C
      NIV = KRANK + NSOLN - L
      IF (L.EQ.N) DONE = .TRUE.
C
C     End of initial triangularization.
C
      IDOPE(1) = ME
      IDOPE(2) = KRANK
      IDOPE(3) = NIV
      RETURN
      END
*DECK DWNLT1
      SUBROUTINE DWNLT1 (I, LEND, MEND, IR, MDW, RECALC, IMAX, HBAR, H,
     +   SCALE, W)
C***BEGIN PROLOGUE  DWNLT1
C***SUBSIDIARY
C***PURPOSE  Subsidiary to WNLIT
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (WNLT1-S, DWNLT1-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     To update the column Sum Of Squares and find the pivot column.
C     The column Sum of Squares Vector will be updated at each step.
C     When numerically necessary, these values will be recomputed.
C
C***SEE ALSO  DWNLIT
C***ROUTINES CALLED  IDAMAX
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890620  Code extracted from WNLIT and made a subroutine.  (RWC))
C   900604  DP version created from SP version.  (RWC)
C***END PROLOGUE  DWNLT1
      USE REAL_PRECISION

      INTEGER I, IMAX, IR, LEND, MDW, MEND
      REAL(KIND=R8) H(*), HBAR, SCALE(*), W(MDW,*)
      LOGICAL RECALC
C
      EXTERNAL IDAMAX
      INTEGER IDAMAX
C
      INTEGER J, K
C
C***FIRST EXECUTABLE STATEMENT  DWNLT1
      IF (IR.NE.1 .AND. (.NOT.RECALC)) THEN
C
C        Update column SS=sum of squares.
C
         DO 10 J=I,LEND
            H(J) = H(J) - SCALE(IR-1)*W(IR-1,J)**2
   10    CONTINUE
C
C        Test for numerical accuracy.
C
         IMAX = IDAMAX(LEND-I+1, H(I), 1) + I - 1
         RECALC = (HBAR+1.E-3*H(IMAX)) .EQ. HBAR
      ENDIF
C
C     If required, recalculate column SS, using rows IR through MEND.
C
      IF (RECALC) THEN
         DO 30 J=I,LEND
            H(J) = 0.D0
            DO 20 K=IR,MEND
               H(J) = H(J) + SCALE(K)*W(K,J)**2
   20       CONTINUE
   30    CONTINUE
C
C        Find column with largest SS.
C
         IMAX = IDAMAX(LEND-I+1, H(I), 1) + I - 1
         HBAR = H(IMAX)
      ENDIF
      RETURN
      END
*DECK DWNLT2
      LOGICAL FUNCTION DWNLT2 (ME, MEND, IR, FACTOR, TAU, SCALE, WIC)
C***BEGIN PROLOGUE  DWNLT2
C***SUBSIDIARY
C***PURPOSE  Subsidiary to WNLIT
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (WNLT2-S, DWNLT2-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     To test independence of incoming column.
C
C     Test the column IC to determine if it is linearly independent
C     of the columns already in the basis.  In the initial tri. step,
C     we usually want the heavy weight ALAMDA to be included in the
C     test for independence.  In this case, the value of FACTOR will
C     have been set to 1.E0 before this procedure is invoked.
C     In the potentially rank deficient problem, the value of FACTOR
C     will have been set to ALSQ=ALAMDA**2 to remove the effect of the
C     heavy weight from the test for independence.
C
C     Write new column as partitioned vector
C           (A1)  number of components in solution so far = NIV
C           (A2)  M-NIV components
C     And compute  SN = inverse weighted length of A1
C                  RN = inverse weighted length of A2
C     Call the column independent when RN .GT. TAU*SN
C
C***SEE ALSO  DWNLIT
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890620  Code extracted from WNLIT and made a subroutine.  (RWC))
C   900604  DP version created from SP version.  (RWC)
C***END PROLOGUE  DWNLT2
      USE REAL_PRECISION

      REAL(KIND=R8) FACTOR, SCALE(*), TAU, WIC(*)
      INTEGER IR, ME, MEND
C
      REAL(KIND=R8) RN, SN, T
      INTEGER J
C
C***FIRST EXECUTABLE STATEMENT  DWNLT2
      SN = 0.E0
      RN = 0.E0
      DO 10 J=1,MEND
         T = SCALE(J)
         IF (J.LE.ME) T = T/FACTOR
         T = T*WIC(J)**2
C
         IF (J.LT.IR) THEN
            SN = SN + T
         ELSE
            RN = RN + T
         ENDIF
   10 CONTINUE
      DWNLT2 = RN .GT. SN*TAU**2
      RETURN
      END
*DECK DWNLT3
      SUBROUTINE DWNLT3 (I, IMAX, M, MDW, IPIVOT, H, W)
C***BEGIN PROLOGUE  DWNLT3
C***SUBSIDIARY
C***PURPOSE  Subsidiary to WNLIT
C***LIBRARY   SLATEC
C***TYPE      REAL(KIND=R8) (WNLT3-S, DWNLT3-D)
C***AUTHOR  Hanson, R. J., (SNLA)
C           Haskell, K. H., (SNLA)
C***DESCRIPTION
C
C     Perform column interchange.
C     Exchange elements of permuted index vector and perform column
C     interchanges.
C
C***SEE ALSO  DWNLIT
C***ROUTINES CALLED  DSWAP
C***REVISION HISTORY  (YYMMDD)
C   790701  DATE WRITTEN
C   890620  Code extracted from WNLIT and made a subroutine.  (RWC))
C   900604  DP version created from SP version.  (RWC)
C***END PROLOGUE  DWNLT3
      USE REAL_PRECISION

      INTEGER I, IMAX, IPIVOT(*), M, MDW
      REAL(KIND=R8) H(*), W(MDW,*)
C
      EXTERNAL DSWAP
C
      REAL(KIND=R8) T
      INTEGER ITEMP
C
C***FIRST EXECUTABLE STATEMENT  DWNLT3
      IF (IMAX.NE.I) THEN
         ITEMP        = IPIVOT(I)
         IPIVOT(I)    = IPIVOT(IMAX)
         IPIVOT(IMAX) = ITEMP
C
         CALL DSWAP(M, W(1,IMAX), 1, W(1,I), 1)
C
         T       = H(IMAX)
         H(IMAX) = H(I)
         H(I)    = T
      ENDIF
      RETURN
      END
