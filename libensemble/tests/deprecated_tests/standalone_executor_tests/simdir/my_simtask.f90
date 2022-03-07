module Fortran_Sleep
use, intrinsic :: iso_c_binding, only: c_int
implicit none
interface
   !should be unsigned int ... not available in Fortran
   !OK until highest bit gets set.
   function FortSleep(seconds)  bind(C, name="sleep")
       import
       integer (c_int) :: FortSleep
       integer (c_int), intent (in), VALUE :: seconds
   end function FortSleep
end interface

end module Fortran_Sleep

program hello_fortran
  use mpi
  use, intrinsic :: iso_c_binding, only: c_int
  use Fortran_Sleep

  implicit none

  integer (kind = 4) :: err
  integer (kind = 4) :: rank
  integer (kind = 4) :: nprocs
  character(len=32)  :: arg
  character(len=32)  :: arglist(2)
  integer (c_int)    :: delay,dummy
  integer            :: i

  delay=5
  dummy = FortSleep(delay)


  !init in case get no args
  DO i = 1, 2
    arglist(i) = "None"
  END DO

  DO i = 1, iargc()
    CALL getarg(i, arg)
    arglist(i) = arg
  END DO

  call mpi_init(err)
  call mpi_comm_size (MPI_COMM_WORLD, nprocs, err)
  call mpi_comm_rank (MPI_COMM_WORLD, rank, err)



  write(*,100) rank, nprocs, trim(arglist(1)), trim(arglist(2))

  call MPI_Finalize (err)

  100 FORMAT('Hello World from proc: ',i3,' of',i3,'  arg1: ',a,'  arg2: ',a)
end program hello_fortran
