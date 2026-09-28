! Cleveland LOWESS reference implementation.
! Original source code for the LOWESS algorithm by William S. Cleveland.
! Retained as a validation reference.

      SUBROUTINE LOWESS(X,Y,N,F,NSTEPS,DELTA,YS,RW,RES)
!
!  Algorithm:
!
!  1. Compute tricube weights centered at each point.
!  2. Compute initial fitted values with weighted linear regression.
!  3. Compute residuals.
!  4. Repeat NSTEPS times:
!     A. Compute bisquare robustness weights from residuals.
!     B. Recompute fitted values with neighborhood and robustness weights.
!     C. Recompute residuals.
!
!  Arguments:
!  X      Input abscissas in ascending order.
!  Y      Input ordinates.
!  N      Number of data points.
!  F      Span fraction, where 0 < F <= 1.
!  NSTEPS Number of robustness iterations, usually 2 or 3.
!  DELTA  Non-negative linear-interpolation threshold.
!  YS     Output smoothed values.
!  RW     Workspace for robustness weights.
!  RES    Workspace for residuals.
!
         REAL X(N), Y(N), YS(N), RW(N), RES(N), DELTA, F
         INTEGER N, NSTEPS
         INTEGER I, ITER, LAST, M, NLEFT, NRIGHT
         REAL ALPHA, C1, C9, CMAD, CUT, D1, D2, DENOM, RANGE
!
         IF (N .LT. 2) THEN
            YS(1) = Y(1)
            RETURN
         END IF
!
         RANGE = X(N) - X(1)
         DO 10 I = 1, N
            RW(I) = 1.0
   10    CONTINUE
!
         DO 100 ITER = 1, NSTEPS + 1
            LAST = 0
            I = 1
   20       CONTINUE
            IF (I .GT. N) GO TO 90
!
            IF (LAST .EQ. 0) GO TO 30
            IF (X(I) - X(LAST) .GT. DELTA) GO TO 30
            GO TO 40
!
   30       CONTINUE
            CALL LOWEST(X, Y, N, X(I), YS(I), F, RW, RES)
            LAST = I
            GO TO 80
!
   40       CONTINUE
            M = I
   50       CONTINUE
            IF (M .GE. N) GO TO 60
            IF (X(M+1) - X(LAST) .GT. DELTA) GO TO 60
            M = M + 1
            GO TO 50
!
   60       CONTINUE
            IF (M .EQ. I) THEN
               CALL LOWEST(X, Y, N, X(I), YS(I), F, RW, RES)
               LAST = I
               GO TO 80
            END IF
!
            CALL LOWEST(X, Y, N, X(M), YS(M), F, RW, RES)
            DO 70 J = I, M - 1
               ALPHA = (X(J) - X(LAST)) / (X(M) - X(LAST))
               YS(J) = ALPHA * YS(M) + (1.0 - ALPHA) * YS(LAST)
   70       CONTINUE
            LAST = M
            I = M
!
   80       CONTINUE
            I = I + 1
            GO TO 20
!
   90       CONTINUE
            IF (ITER .GT. NSTEPS) GO TO 100
!
!        Compute residuals.
            DO 110 I = 1, N
               RES(I) = ABS(Y(I) - YS(I))
  110       CONTINUE
!
!        Compute six times the median absolute residual.
            DO 120 I = 1, N
               RW(I) = RES(I)
  120       CONTINUE
            M = N / 2 + 1
            CALL SORT(RW, N)
            IF (MOD(N, 2) .EQ. 0) THEN
               CMAD = 3.0 * (RW(M-1) + RW(M))
            ELSE
               CMAD = 6.0 * RW(M)
            END IF
!
            IF (CMAD .LE. 1.0E-7 * RANGE) GO TO 100
!
!        Compute bisquare weights.
            DO 130 I = 1, N
               CUT = RES(I) / CMAD
               IF (CUT .GE. 1.0) THEN
                  RW(I) = 0.0
               ELSE
                  RW(I) = (1.0 - CUT**2)**2
               END IF
  130       CONTINUE
!
  100    CONTINUE
         RETURN
      END


      SUBROUTINE LOWEST(X,Y,N,XS,YS,F,W,RES)
!
!  Compute the weighted linear-regression value at XS.
!
         REAL X(N), Y(N), W(N), RES(N), XS, YS, F
         INTEGER N
         INTEGER H, H9, I, J, NLEFT, NRIGHT
         REAL A, B, C, D, H1, R, SUMW, SUMWX, SUMWXX, SUMWY, SUMWXY
!
         H = INT(F * FLOAT(N))
         IF (H .LT. 2) H = 2
         IF (H .GT. N) H = N
!
!  Find the H nearest neighbors to XS.
         NLEFT = 1
         NRIGHT = H
         DO 10 I = 1, N - H
            IF (ABS(X(I+H) - XS) .LT. ABS(X(I) - XS)) THEN
               NLEFT = NLEFT + 1
               NRIGHT = NRIGHT + 1
            ELSE
               GO TO 20
            END IF
   10    CONTINUE
   20    CONTINUE
!
!  Determine the maximum distance across the window.
         D = AMAX1(ABS(XS - X(NLEFT)), ABS(X(NRIGHT) - XS))
         IF (D .LE. 0.0) THEN
            YS = Y(NLEFT)
            RETURN
         END IF
!
!  Compute the weighted least-squares regression.
         SUMW = 0.0
         SUMWX = 0.0
         SUMWY = 0.0
         SUMWXX = 0.0
         SUMWXY = 0.0
!
         DO 30 J = NLEFT, NRIGHT
            R = ABS(X(J) - XS) / D
            IF (R .LT. 1.0) THEN
               C = (1.0 - R**3)**3
            ELSE
               C = 0.0
            END IF
            A = C * W(J)
            SUMW = SUMW + A
            SUMWX = SUMWX + A * X(J)
            SUMWY = SUMWY + A * Y(J)
            SUMWXX = SUMWXX + A * X(J)**2
            SUMWXY = SUMWXY + A * X(J) * Y(J)
   30    CONTINUE
!
         B = SUMW * SUMWXX - SUMWX**2
         IF (ABS(B) .LE. 1.0E-7) THEN
            YS = SUMWY / SUMW
         ELSE
            A = (SUMWY * SUMWXX - SUMWX * SUMWXY) / B
            B = (SUMW * SUMWXY - SUMWX * SUMWY) / B
            YS = A + B * XS
         END IF
!
         RETURN
      END


      SUBROUTINE SORT(A,N)
!
!  Sort residuals with selection sort.
!
         REAL A(N), T
         INTEGER N, I, J, MIN
!
         DO 20 I = 1, N - 1
            MIN = I
            DO 10 J = I + 1, N
               IF (A(J) .LT. A(MIN)) MIN = J
   10       CONTINUE
            IF (MIN .NE. I) THEN
               T = A(I)
               A(I) = A(MIN)
               A(MIN) = T
            END IF
   20    CONTINUE
         RETURN
      END
