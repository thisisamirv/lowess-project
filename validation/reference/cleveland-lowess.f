!
!     Historical reference implementation of LOWESS.
!     W. S. Cleveland, Bell Laboratories, 30 December 1985.
!
!     This preserves the original fixed-form single-precision code.
!     The only code cleanup replaces n%2 with Fortran mod(n,2).
!
      subroutine lowess(x, y, n, f, nits, delta, ys, rw, res)
!
!     w. s. cleveland
!     bell laboratories
!     murray hill nj 07974
!     mon dec 30 16:55 est 1985
!
!
!     outline of method
!
!     the arrays x and y contain n data points.  a robust locally
!     weighted regression is fit to the data.  the smoothed values
!     are returned in the array ys.  the user chooses the fraction, f,
!     of data points which are to have an influence on the smoothness
!     at each value.  larger values of f give more smoothness.
!     nits is the number of robustifying iterations.  robustness is
!     needed because the local linear fitting method is highly sensitive
!     to outliers.  if nits = 0, the robustifying iterations are
!     omitted.  in most cases, nits = 2 or 3 is sufficient.  delta is a
!     non-negative parameter which can be used to save computations.
!     if delta = 0.0, computations are done for all points.  if
!     delta > 0.0, linear interpolation is used to find smoothed
!     values at points which are within delta of a point at which a
!     locally weighted regression has already been evaluated.  the
!     larger delta is, the more computations are saved.  a good choice
!     for delta is 0.01 times the range of x.
!
!     x(n),y(n) - input data arrays
!     n - number of data points
!     f - smoothing parameter, fraction of points used in each local
!         regression.  0.0 < f < 1.0
!     nits - number of robustifying iterations.  nits >= 0
!     delta - non-negative parameter for skipping computations.  delta >= 0.0
!     ys(n) - output array of smoothed values
!     rw(n),res(n) - working storage arrays for robustness weights and
!                    residuals respectively
!
         real x(n), y(n), ys(n), rw(n), res(n)
         integer n, nits
         real f, delta
         integer i, iter, j, nleft, nright, ns
         real alpha, cut, cmany, c2, c6, h, h1, r
!
         if (n .lt. 2) then
            ys(1) = y(1)
            return
         endif
         ns = max0(min0(int(f * float(n)), n), 2)
!
!     robustifying iterations
!
         do 100 iter = 1, nits + 1
            nleft = 1
            nright = ns
            i = 1
 10         if (i .gt. n) go to 90
            h = x(nright) - x(i)
            h1 = x(i) - x(nleft)
            if (h1 .gt. h) h = h1
            if (h .le. 0.0) h = 1.0
            range = x(n) - x(1)
            if (range .le. 0.0) range = 1.0
            cut = x(i) + delta
            j = i
 20         if (j .gt. n) go to 30
            if (x(j) .gt. cut) go to 30
            if (x(j) .eq. x(i)) then
               call lowest(x, y, n, x(i), ys(j), nleft, nright, rw,
     &         iter .gt. 1, rw)
            else
               h1 = x(j) - x(i)
               alpha = h1 / delta
               ys(j) = alpha * ys(j) + (1.0 - alpha) * ys(i)
            endif
            j = j + 1
            go to 20
 30         i = j
            if (i .gt. n) go to 90
 40         if (nright .eq. n) go to 50
            if (x(nright + 1) - x(i) .gt. x(i) - x(nleft)) go to 50
            nleft = nleft + 1
            nright = nright + 1
            go to 40
 50         go to 10
 90         if (iter .eq. nits + 1) go to 100
            do 95 i = 1, n
               res(i) = y(i) - ys(i)
 95         continue
!
!        calculate robustness weights
!
            do 96 i = 1, n
               rw(i) = abs(res(i))
 96         continue
            call sort(rw, n)
            j = n / 2 + 1
            h = rw(j)
            if (mod(n, 2) .eq. 0) h = (rw(j - 1) + rw(j)) / 2.0
            c6 = 6.0 * h
            c2 = 2.0 * h
            cmany = 1.0e-6 * c2
            do 97 i = 1, n
               r = abs(res(i))
               if (r .le. cmany) then
                  rw(i) = 1.0
               else if (r .le. c6) then
                  rw(i) = (1.0 - (r / c6)**2)**2
               else
                  rw(i) = 0.0
               endif
 97         continue
 100     continue
         return
      end
!
      subroutine lowest(x, y, n, xs, ys, nleft, nright, w, userw, rw)
         real x(n), y(n), w(n), rw(n)
         integer n, nleft, nright
         real xs, ys
         logical userw
         integer j
         real range, h, h1, h9, r, sumwt, sumxwt, sumx2w, sumywt,
     &   sumxyw, meanx, meany
         real varx, covxy, beta
!
         range = x(n) - x(1)
         if (range .le. 0.0) range = 1.0
         h = x(nright) - xs
         h1 = xs - x(nleft)
         if (h1 .gt. h) h = h1
         if (h .le. 0.0) h = 1.0
         h9 = 0.999 * h
         sumwt = 0.0
         do 10 j = nleft, nright
            r = abs(x(j) - xs)
            if (r .le. h9) then
               w(j) = (1.0 - (r / h)**3)**3
               if (userw) w(j) = rw(j) * w(j)
               sumwt = sumwt + w(j)
            else
               w(j) = 0.0
            endif
 10      continue
         if (sumwt .le. 0.0) then
            ys = y(nleft)
            return
         endif
         sumxwt = 0.0
         sumywt = 0.0
         do 20 j = nleft, nright
            sumxwt = sumxwt + x(j) * w(j)
            sumywt = sumywt + y(j) * w(j)
 20      continue
         meanx = sumxwt / sumwt
         meany = sumywt / sumwt
         sumx2w = 0.0
         sumxyw = 0.0
         do 30 j = nleft, nright
            h1 = x(j) - meanx
            sumx2w = sumx2w + h1 * h1 * w(j)
            sumxyw = sumxyw + h1 * (y(j) - meany) * w(j)
 30      continue
         varx = sumx2w / sumwt
         covxy = sumxyw / sumwt
         if (varx .le. 1.0e-7 * range * range) then
            ys = meany
         else
            beta = covxy / varx
            ys = meany + beta * (xs - meanx)
         endif
         return
      end
!
      subroutine sort(a, n)
         real a(n)
         integer n
         integer i, j, k, m
         real t
!
         m = n
 10      m = m / 2
         if (m .eq. 0) return
         k = n - m
         do 30 i = 1, k
            j = i
 20         if (j .lt. 1) go to 30
            if (a(j) .le. a(j + m)) go to 30
            t = a(j)
            a(j) = a(j + m)
            a(j + m) = t
            j = j - m
            go to 20
 30      continue
         go to 10
      end
