!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine openfilelist(nunit,filelist,nfiles)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
!import vars
integer :: nunit,nfiles
character(80) :: filelist
!local vars
integer :: filestatus
character(80) :: fitsfile

open(unit=nunit,file=filelist,iostat=filestatus,status='old')
if(filestatus>0)then !trap missing file errors
   write(0,*) "Cannot open filelist",filelist
   stop
endif

!count number of files.
nfiles=0
do
   read(nunit,'(A)',iostat=filestatus) fitsfile

   if(filestatus == 0) then
      nfiles=nfiles+1
   elseif(filestatus == -1) then
      exit  !successively break from data read loop.

   else
      write(0,*) "File Error!! Line: ",nfiles
      write(0,900) "iostat: ",filestatus
      900 format(A8,I3)
      stop

   endif
enddo

rewind(nunit)

return
end subroutine openfilelist