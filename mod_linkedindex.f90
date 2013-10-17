module mod_linkedindex
  implicit none

  type node
    type(node),pointer :: next,prev
    integer :: ind
  end type node

  type(node),pointer :: indexQueue

  logical :: isnotinitialized = .true.


end module mod_linkedindex
