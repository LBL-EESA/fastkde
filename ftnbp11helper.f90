      !
      !calculateecfdirect(datapoints,nvariables,ndatapoints,dataaverage,datastd,
      !                   tpoints,ntpoints,ecf)
      !
      !Calculates the empirical characteristic function of a given set
      !of arbitrarily-dimensioned data points using a direct Fourier
      !transform.  This routine standardizes the data on-the-fly, such
      !that the real-space mean and std-dev for each variable are 0 and
      !1 respectively.
      !
      ! datapoints (in) : array([nvariables,ndatapoints])
      !                 Input data that are assumed to be a set of
      !                 'ndatapoints' groups of 'nvariables' points (so
      !                 the dataset is 'nvariables'-dimensional); in
      !                 otherwords, they are samples of 'nvariables'
      !                 jointly distributed variables.
      !
      !
      ! nvariables (in) : integer
      !                 The number of variables (dimensions) of the
      !                 dataset
      !
      ! ndatapoints(in) : integer
      !                 The number of samples
      !
      ! dataaverage(in) : array([nvariables])
      !                 The average of each of the variables in the
      !                 dataset
      !
      ! datastd    (in) : array([nvariables])
      !                 The standard deviation of each of the variables 
      !                 in the dataset
      !
      ! tpoints    (in) : array([ntpoints])
      !                 The set of frequency points at which to
      !                 calculate the ECF. This same set of frequency
      !                 points is used for all dimenions of the ECF.
      !
      ! ntpoints   (in) : integer
      !                 The number of frequency points
      !
      ! freqspacesize (in) : integer
      !                 The total number of frequency points.  Should
      !                 be equal to ntpoints**nvariables
      !
      ! ecf        (out): array([freqspacesize])
      !                 A flattened array representing the ECF of the
      !                 data points.  It is ordered such that it can
      !                 be reconstructed into a multidimensional array
      !                 in Python by the following code:
      !
      !                   ecf = reshape(ecf,nvariables*[ntpoints])
      !
      !                 The resulting array will have 'nvariables'
      !                 dimensions each of length 'ntpoints'.
      !
      subroutine calculateecfdirect(  &
                                  datapoints, &   !The random data points on which to base the distribution
                                  nvariables, &   !The number of variables
                                  ndatapoints,  & !The number of data points
                                  dataaverage,  & !The average of the data 
                                  datastd,  &     !The stddev of the data 
                                  tpoints,    &   !The frequency points at which to calculate the optimal distribution
                                  ntpoints,   &   !The number of frequency points along each dimensions
                                  freqspacesize,& !The total number of frequency points
                                  ecf  &          !The empirical characteristic function
                                )
      implicit none
      !*******************************
      ! Input variables
      !*******************************
      !f2py integer,intent(hide),depend(tpoints) :: ntpoints = len(tpoints)
      integer, intent(in) :: ntpoints
      !f2py integer,intent(hide),depend(datapoints) :: nvariables = shape(datapoints,0)
      integer, intent(in) :: nvariables
      !f2py integer,intent(hide),depend(datapoints) :: ndatapoints = shape(datapoints,1)
      integer, intent(in) :: ndatapoints
      !f2py double precision,intent(in),dimension(nvariables,ndatapoints) :: datapoints
      double precision, intent(in), dimension(nvariables,ndatapoints) :: datapoints
      !f2py double precision,intent(in),dimension(nvariables) :: dataaverage, datastd
      double precision,intent(in),dimension(nvariables) :: dataaverage,datastd
      !f2py double precision,intent(in),dimension(ntpoints) :: tpoints
      double precision, intent(in), dimension(ntpoints) :: tpoints
      !f2py integer :: freqspacesize
      integer, intent(in) :: freqspacesize
      !*******************************
      ! Output variables
      !*******************************
      !f2py complex(kind=8),intent(out),dimension(freqspacesize) :: ecf
      complex(kind=8),intent(out),dimension(freqspacesize) :: ecf
      !*******************************
      ! Local variables
      !*******************************
      complex(kind=8),parameter :: ii = (0.0d0, 1.0d0), &
                                            c1 = (1.0d0, 0.0d0), &
                                            c2 = (2.0d0, 0.0d0), &
                                            c4 = (4.0d0, 0.0d0)
      integer :: i,j,k
      double precision :: N
      double precision :: x,ecfArg
      complex(kind=8) :: cN
      integer :: numConsecutive
      complex(kind=8) :: myECF
      complex(kind=8) :: myECFsq
      double precision :: dum
      integer,dimension(nvariables) :: iDimCounters

        !Set a double version of ndatapoints
        N = dble(ndatapoints)
        !and a complex version too
        cN = complex(N,0.0d0)

        !Loop over all the frequencies.  Note that the ECF
        ! is flattened, so the determinedimensionindices()
        ! routine is used to convert the 1D frequency index,
        ! i, into a set of multidimensional indices corresponding
        ! to the point in the nvariables-dimensioned frequency space.
        ! These indices are then used in the tpoints variable to access
        ! the acual frequencies of the current point.
        !Frequency-space loop
        floop: &
        do i = 1,freqSpaceSize
          !Calculate the dimension index counters
           call determinedimensionindices(&
                                  i, &
                                  ntpoints, &
                                  iDimCounters, &
                                  nvariables)

          !********************************************************************
          ! Calculate the empirical characteristic function at this frequency  
          !********************************************************************
          myECF = complex(0.0d0,0.0d0)
          !Loop over all the data points
          dataloop: &
          do j = 1,ndatapoints
            ecfArg = 0.0
            !Loop over all the variables
            variableloop: &
            do k = 1,nvariables
              !Standardize the data on the fly
              x = (datapoints(k,j) - dataaverage(k))/datastd(k)


              !Augment the argument of the ECF exponential
              ecfArg = ecfArg + tpoints(iDimCounters(k))*x
              !write(*,*),i,iDimCounters,tpoints(iDimCounters(k)),ecfArg
            end do variableloop
            !Calculate the exponential term of the ECF
            !for this data point, and add it to the running
            !ECF calculation for this frequency point
            myECF = myECF + exp(ii*ecfArg)
          end do dataloop

          !Set the ECF of this frequency point
          ecf(i) = myECF/(nvariables*ndatapoints)
        end do floop

      end subroutine calculateecfdirect


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

      subroutine determinedimensionindices(i,dimSize,iDimCounters,ndims)
      implicit none
        !******************
        ! Input variables
        !******************
        !f2py integer,intent(in) :: i,dimSize
        integer,intent(in) :: i,dimSize
        !f2py integer,intent(in) :: ndims 
        integer,intent(in) :: ndims
        !******************
        ! Output variables
        !******************
        !f2py integer,dimension(ndims),intent(out) :: iDimCounters
        integer,dimension(ndims),intent(out) :: iDimCounters

        !******************
        ! Local variables
        !******************
        integer :: n,np,iDum

        iDum = i
        do n = ndims,1,-1
          np = ndims-n+1
          iDimCounters(np) = floor(float(iDum-1)/float(dimSize**(n-1))) + 1
          iDum = iDum - (iDimCounters(np)-1)*dimSize**(n-1)
        end do
          
        
        ! i =   iDimCounters(1) + dimSize*iDimCounters(2)
        !     + dimSize**2*iDimCounters(3) + ... 
        !     + dimSize**(ndims-1)*iDimCounters(ndims)
      end subroutine determinedimensionindices

      subroutine mapdimensionindices(npoints,dimSize,iDimCounters,ndims)
      implicit none
        !******************
        ! Input variables
        !******************
        !f2py integer,intent(in) :: dimSize,npoints
        integer,intent(in) :: dimSize,npoints
        !f2py integer,intent(in) :: ndims 
        integer,intent(in) :: ndims
        !******************
        ! Output variables
        !******************
        !f2py integer,dimension(npoints,ndims),intent(out) :: iDimCounters
        integer,dimension(npoints,ndims),intent(out) :: iDimCounters

        !******************
        ! Local variables
        !******************
        integer :: i,n,np,iDum

        do i = 1,npoints
          iDum = i
          do n = ndims,1,-1
            np = ndims-n+1
            iDimCounters(i,np) = floor(float(iDum-1)/float(dimSize**(n-1))) + 1
            iDum = iDum - (iDimCounters(i,np)-1)*dimSize**(n-1)
          end do
        end do
          
        
        ! i =   iDimCounters(1) + dimSize*iDimCounters(2)
        !     + dimSize**2*iDimCounters(3) + ... 
        !     + dimSize**(ndims-1)*iDimCounters(ndims)
      end subroutine mapdimensionindices
