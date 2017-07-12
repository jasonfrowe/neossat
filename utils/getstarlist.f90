subroutine getstarlist(nunit,nstar,coo)
use precision
implicit none
!import vars
integer :: nunit,nstar
real(double), dimension(:,:) :: coo
!local vars
integer :: i,filestatus
real(double) :: x,y

i=0 !counter to fill in array
do
   read(nunit,*,iostat=filestatus) x,y

   if(filestatus == 0) then
      i=i+1
      if(i.le.nstar)then
         coo(1,i)=x
         coo(2,i)=y
      else
         write(0,*) "Exceeded NSTAR in getstarlist"
         stop
      endif
   elseif(filestatus == -1) then
      exit  !successively break from data read loop.
   else
      write(0,*) "File Error!! Line: ",nstar
      write(0,900) "iostat: ",filestatus
      900 format(A8,I3)
      stop

   endif
enddo

return
end subroutine getstarlist