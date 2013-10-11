      subroutine lowesthypervolumefilter( ecfsq,  &
                                          ecfthreshold, &
                                          nvariables, &
                                          ntpoints, &
                                          icalcphi, &
                                          imax,     &
                                          freqspacesize )
      implicit none
      !*******************************
      ! Input variables
      !*******************************
      !f2py integer,intent(in) :: nvariables,ntpoints
      integer,intent(in) :: nvariables,ntpoints
      !f2py integer,intent(hide),depend(ecfsq) :: freqspacesize = len(ecfsq)
      integer,intent(in) :: freqspacesize
      !f2py double precision, intent(in), dimension(freqspacesize) :: ecfsq
      double precision,dimension(freqspacesize),intent(in) :: ecfsq
      !f2py double precision, intent(in) :: ecfthreshold
      double precision,intent(in) :: ecfthreshold
      !*******************************
      ! Output variables
      !*******************************
      !f2py integer, intent(out), dimension(freqspacesize) :: icalcphi
      integer,dimension(freqspacesize),intent(out) :: icalcphi
      !f2py integer, intent(out) :: imax
      integer,intent(out) :: imax
      !*******************************
      ! Local variables
      !*******************************
      integer :: ilist,inext,k
      integer,dimension(nvariables) :: idimcounters
      logical,dimension(freqspacesize) :: hasnotbeensearched

        !Initialize the search list to indicate that
        !no cells have yet been searched
        hasnotbeensearched = .true.
        !Initialize the dimension index list to -1 to flag
        !that there are no valid values yet
        icalcphi = -1
        !Set the list index counter to the beginning
        ilist = 1

        !Set the 'next' neighbor index to the mid point (the
        !0-frequency point)
        do k = 1,nvariables
          idimcounters(k) = (ntpoints-1)/2 + 1
          !Determine the flattened index of the center point
          !And put it in inext
          call determineflattenedindex( idimcounters, &
                                        ntpoints, &
                                        nvariables,  &
                                        inext)

        end do 

        !Check if this value is above the ECF threshold.  If not,
        !do nothing and return
        if(ecfsq(inext) >= ecfthreshold) then
          !Flag that this cell has now been searched 
          hasnotbeensearched(inext) = .false.
          !Add this index to the filter list
          icalcphi(ilist) = inext - 1
          !increment the filter list counter
          ilist = ilist + 1
          !Check whether this cell has any neighbors above the threshold
          !(this enters a recursive loop that finishes once all
          !contiguous above-threshold values have been added to
          !icalcphi)
          call aremyneighborsabove( ecfsq,  &
                                    ecfthreshold, &
                                    nvariables, &
                                    ntpoints, &
                                    icalcphi, &
                                    inext,  &
                                    ilist,  &
                                    hasnotbeensearched,  &
                                    freqspacesize &
                                  )
        end if

        !Set that the highest value in the icalcphi list is at the imax
        !position
        imax = ilist-1

        return
      end subroutine lowesthypervolumefilter

      recursive subroutine aremyneighborsabove( ecfsq,  &
                                      ecfthreshold, &
                                      nvariables, &
                                      ntpoints, &
                                      icalcphi, &
                                      icurrent,  &
                                      ilist,  &
                                      hasnotbeensearched,  &
                                      freqspacesize &
                                    )
      implicit none
      !*******************************
      ! Input variables
      !*******************************
      !f2py integer,intent(in) :: nvariables,ntpoints
      integer,intent(in) :: nvariables,ntpoints,icurrent
      !f2py integer,intent(inout) :: ilist
      integer,intent(inout) :: ilist
      !f2py integer,intent(hide),depend(ecfsq) :: freqspacesize = len(ecfsq)
      integer,intent(in) :: freqspacesize
      !f2py double precision, intent(in), dimension(freqspacesize) :: ecfsq
      double precision,dimension(freqspacesize),intent(in) :: ecfsq
      !f2py logical,intent(inout),dimension(freqspacesize) :: hasnotbeensearched
      logical,intent(inout),dimension(freqspacesize) :: hasnotbeensearched
      !f2py double precision, intent(in) :: ecfthreshold
      double precision,intent(in) :: ecfthreshold
      !*******************************
      ! Output variables
      !*******************************
      !f2py integer, intent(out), dimension(freqspacesize) :: icalcphi
      integer,dimension(freqspacesize),intent(out) :: icalcphi


      !*******************************
      ! Local variables
      !*******************************
      integer :: inext,k,j
      integer,dimension(nvariables) :: idimcounters
      integer,parameter,dimension(2) :: plusminus = (/-1,1/)

         !Calculate the dimension index counters
         call determinedimensionindices(&
                                icurrent, &
                                ntpoints, &
                                idimcounters, &
                                nvariables)

        !dimension (variable) loop
        dimloop:  &
        do k = 1,nvariables
          !Direction loop (backward vs forward in the current direction)
          plusminusloop:  &
          do j = 1,2
            !Add 1 to the dimension counter for the current dimension
            !(i.e. search forward in this dimension) to access
            !the neighbor that is idimcounters(k)+1 away in the k
            !direction
            idimcounters(k) = idimcounters(k) + plusminus(j)
            !Check if this dimension counter is within the dimension
            !bounds; do nothing if not, since it would otherwise mean
            !we are already at a dimension boundary
            if(idimcounters(k).ge.1.and.idimcounters(k).le.ntpoints)then
              !Determine the flattened index of this new neighbor point
              call determineflattenedindex( idimcounters, &
                                            ntpoints, &
                                            nvariables,  &
                                            inext)

   

              !Check if this neighbor is above the threshold and hasn't
              !been searched yet
              if((ecfsq(inext).ge.ecfthreshold).and.hasnotbeensearched(inext)) then
                !Flag that it has now been searched
                hasnotbeensearched(inext) = .false.
                !Add this index to the filter list
                icalcphi(ilist) = inext - 1
                !increment the filter list counter
                ilist = ilist + 1
                !Check if this cell has any neighbors that are
                !above the threshold
                call aremyneighborsabove( ecfsq,  &
                                          ecfthreshold, &
                                          nvariables, &
                                          ntpoints, &
                                          icalcphi, &
                                          inext,  &
                                          ilist,  &
                                          hasnotbeensearched,  &
                                          freqspacesize &
                                        )


              end if
            end if

            !decrement the dimension counter;
            idimcounters(k) = idimcounters(k) - plusminus(j)
          end do plusminusloop
        end do dimloop

      end subroutine aremyneighborsabove
