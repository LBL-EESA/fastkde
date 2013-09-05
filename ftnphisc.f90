      !Calculates the Fourier representation of the optimal,
      !self-consistent distribution of a given set of data points.
      !Based on Bernacchia and Pigolotti (2011, J. R. Statistic Soc. B.).
      subroutine calculatephisc(  &
                                  datapoints, &   !The random data points on which to base the distribution
                                  ndatapoints,  & !The number of data points
                                  dataaverage,  & !The average of the data 
                                  datastd,  &     !The stddev of the data 
                                  tpoints,    &   !The frequency points at which to calculate the optimal distribution
                                  ntpoints,   &   !The number of frequency points
                                  ithresh, &      !The threshold number of consecutive below-threshold points before breaking the loop
                                  phisc, &        !The optimal distribution
                                  ecf  &          !The empirical characteristic function
                                )
      implicit none
      !*******************************
      ! Input variables
      !*******************************
      !f2py integer,intent(hide),depend(tpoints) :: ntpoints = len(tpoints)
      integer, intent(in) :: ntpoints
      !f2py integer,intent(hide),depend(datapoints) :: ndatapoints = len(datapoints)
      integer, intent(in) :: ndatapoints
      !f2py double precision,intent(in),dimension(ndatapoints) :: datapoints
      double precision, intent(in), dimension(ndatapoints) :: datapoints
      !f2py double precision,intent(in) :: dataaverage, datastd
      double precision,intent(in) :: dataaverage,datastd
      !f2py double precision,intent(in),dimension(ntpoints) :: tpoints
      double precision, intent(in), dimension(ntpoints) :: tpoints
      !f2py integer,intent(in) :: ithresh
      integer, intent(in) :: ithresh
      !*******************************
      ! Output variables
      !*******************************
      !f2py complex(kind=8),intent(out),dimension(ntpoints) :: phisc
      complex(kind=8),intent(out),dimension(ntpoints) :: phisc
      !f2py complex(kind=8),intent(out),dimension(ntpoints) :: ecf
      complex(kind=8),intent(out),dimension(ntpoints) :: ecf
      !*******************************
      ! Local variables
      !*******************************
      complex(kind=8),parameter :: ii = (0.0d0, 1.0d0), &
                                            c1 = (1.0d0, 0.0d0), &
                                            c2 = (2.0d0, 0.0d0), &
                                            c4 = (4.0d0, 0.0d0)
      integer :: i,j
      complex(kind=8) :: t,x
      double precision :: N
      complex(kind=8) :: cN
      integer :: numConsecutive
      logical :: bCalculateECF,bCalculatePhiSC
      complex(kind=8) :: myECF
      complex(kind=8) :: myECFsq
      complex(kind=8) :: myECFThreshold
      double precision :: dum
      integer,parameter :: idofullcalc = 1

        !Flag that we should calculate the ECF
        bCalculateECF = .true.
        bCalculatePhiSC = .true.
        !Set the counter of consecutive below-threshold frequencies
        !after which to unset the bCalculateECF flag
        numConsecutive = 0

        !Set a double version of ndatapoints
        N = dble(ndatapoints)
        cN = complex(N,0.0d0)

        !Calculate the threshold of stability for the ECF
        myECFThreshold = complex(4.d0*(N - 1.d0)/(N*N),0.0d0)

        tloop:  &
        do i = 1, ntpoints
          !Initialize phisc to 0 at this frequency
          phisc(i) = complex(0.0d0,0.0d0)
          ecf(i) = complex(0.0d0,0.0d0) 
          t = complex(tpoints(i),0.0d0)

          if(bCalculateECF.or.(idofullcalc.ne.0))then
            !********************************************************************
            ! Calculate the empirical characteristic function at this frequency  
            !********************************************************************
            myECF = complex(0.0d0,0.0d0)
            dataloop: &
            do j = 1,ndatapoints
              !Create a complex version of the data points, and
              !standardize the data on the fly
              x = complex((datapoints(j) - dataaverage)/datastd,0.0d0)
              myECF = myECF + exp(ii*t*x)
            end do dataloop
            myECF = myECF/cN
            ecf(i) = myECF
          end if

          if(bCalculateECF)then
            myECFsq = complex(abs(myECF)*abs(myECF),0.0d0)
            !********************************************************************
            ! Determine if the ECF is above the stability threshold
            !********************************************************************
            !Check if we are above the threshold
            if(real(myECFsq).ge.real(myECFThreshold))then
              bCalculatePhiSC = .true.
              numConsecutive = 0
            else
              bCalculatePhiSC = .false.
              numConsecutive = numConsecutive + 1
            end if

            !If we have reached the maximum number of consecutive
            !below-threshold values, then don't do any more ECF or PhiSC
            !calculation (to save compute time).  However, keep looping
            !so that we can make sure tha phisc is set to 0 for the
            !rest of the values.
            !(Note that the bCalculateECF flag is ignored if idofullcalc
            !is set to something other than 0)
            if(numConsecutive.gt.ithresh)then
              bCalculateECF = .false.
              bCalculatePhiSC = .false.
            end if

            !********************************************************************
            ! Calculate the optimal, self-consistent distribution 
            !********************************************************************
            if(bCalculatePhiSC)then
              dum = 1.0d0 + sqrt(1.0d0 - real(myECFThreshold)/real(myECFsq))
              phisc(i) =  (cN*myECF/(c2*(cN - c1)))  &
                        * complex(dum,0.0d0)
            end if
          end if
        end do tloop


      end subroutine calculatephisc


      subroutine calculatekerneldensityestimate(  &
                                  datapoints, &   !The random data points on which to base the distribution
                                  ndatapoints,  & !The number of data points
                                  dataaverage,  & !The average of the data 
                                  datastd,  &     !The stddev of the data 
                                  xpoints,    &   !The values of the x grid
                                  nxpoints,   &   !The number of grid values
                                  nspreadhalf,  & !The number of values to involve in the convolution (divided by 2)
                                  fourtau,    &   !The kernel width parameter
                                  fkde        &   !The kernel density estimate
                                  )
      !*******************************
      ! Input variables
      !*******************************
      !f2py integer,intent(hide),depend(xpoints) :: nxpoints = len(xpoints)
      integer, intent(in) :: nxpoints
      !f2py integer,intent(hide),depend(datapoints) :: ndatapoints = len(datapoints)
      integer, intent(in) :: ndatapoints
      !f2py double precision,intent(in),dimension(ndatapoints) :: datapoints
      double precision, intent(in), dimension(ndatapoints) :: datapoints
      !f2py double precision,intent(in) :: dataaverage, datastd
      double precision,intent(in) :: dataaverage,datastd
      !f2py double precision,intent(in),dimension(nxpoints) :: xpoints
      double precision, intent(in), dimension(nxpoints) :: xpoints
      !*******************************
      ! Output variables
      !*******************************
      !f2py double precision,intent(out),dimension(nxpoints) :: fkde
      double precision, intent(out),dimension(nxpoints) :: fkde
      !*******************************
      ! Local variables
      !*******************************
      integer :: j,m,m0
      double precision :: xmin,deltax,gaussTerm
      double precision :: xj,mprime
      double precision,parameter :: pi = 3.141592653589793

        !Calculate the quantites necessary for estimating 
        !x-indices.
        xmin = xpoints(1)
        deltax = xpoints(2) - xpoints(1)

        fkde = 0.0 

        dataloop: &
        do j = 1,ndatapoints
          !Set the x-point, and standardize the data on-the-fly
          xj = (datapoints(j) - dataaverage)/datastd
          mprime = (xj - xmin)/deltax + 1
          m0 = floor(mprime)

          mmin = max(1,m0-nspreadhalf+1)
          mmax = min(nxpoints,m0+nspreadhalf)
          gridloop: &
          do m = mmin,mmax
            !gaussTerm = exp(-((xpoints(m) - xj)**2)/fourtau)
            gaussTerm = exp(-((m-mprime)**2)/fourtau)
            fkde(m) = fkde(m) + gaussTerm
          end do gridloop
        end do dataloop

      return 

      end subroutine calculateKernelDensityEstimate
