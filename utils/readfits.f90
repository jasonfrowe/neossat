!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine readfits(unitfits,naxes,fitsdata,bpix)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
!import vars
integer :: unitfits
integer, dimension(:) :: naxes
real(double) :: bpix
real(double), dimension(:,:) :: fitsdata
!local vars
integer :: group,firstpix,nbuf,i,istatus
real(double) :: nullval
real(double), allocatable, dimension(:) :: buffer
logical :: anynull

istatus=0 !initalize istatus to zero
group=1
firstpix=1
nullval=bpix
nbuf=naxes(1)
allocate(buffer(nbuf))

do i=1,naxes(2)
   call ftgpvd(unitfits,group,firstpix,nbuf,nullval,buffer,anynull,istatus)
   !write(0,*) buffer
   fitsdata(1:nbuf,i)=buffer(1:nbuf)
   if(istatus.ne.0)then
      write(0,*) "istatus: ",istatus
      write(0,*) "Error reading in Image"
   endif
   firstpix=firstpix+nbuf
enddo


return
end subroutine readfits

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine readheader(filename,unitfits,ndim,naxes,jddate,jdzero,exptime)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
!import vars
integer :: ndim, naxes(ndim),unitfits
real(double) :: jddate,jdzero,exptime
character(80) :: filename
!local vars
integer :: istatus,readwrite,dumi,nkeys,nspace,i,naxis,nfound
character(80) :: record
character(80), allocatable, dimension(:) :: header

! status will report errors.  No errors means status=0.
! initalize value of status
istatus=0
! gets an unused unit number to open fits file
call ftgiou(unitfits,istatus)
! setting to zero makes fits file readwrite
readwrite=0
! open this fits file
call ftopen(unitfits,filename,readwrite,dumi,istatus)
if(istatus.ne.0)then
   write(0,*) "Status: ",istatus
   write(0,*) "Cannot open "
   write(0,'(A80)') filename
   stop
endif

jddate=-1.0d0 !a negative date means a date is not defined 
exptime=-1.0d0 !a negitive exposure time means exptime is not defined

nkeys=0
! get number of headers in image
call ftghsp(unitfits,nkeys,nspace,istatus)
allocate(header(nkeys))
do i=1,nkeys
   call ftgrec(unitfits,i,record,istatus)
   header(i)=record
   !write(6,'(A80)') header(i)
   if(header(i)(1:6).eq.'JD-OBS')then  
      !write(6,'(A80)') header(i)(12:27)
      read(header(i)(12:27),*) jddate
      jddate=jddate-jdzero !apply MJD zero point
      !write(0,*) 'jddate: ',jddate
   endif
   if(header(i)(1:8).eq.'EXPOSURE')then
      read(header(i)(12:30),*) exptime
   endif
enddo
!read(5,*)

!read in number of dimensions of image
!this should be 2 
call ftgidm(unitfits,naxis,istatus)
!check that naxis is ndim
if(naxis.ne.ndim)then
   write(0,'(A20,I1,A11)') "Image has more than ",ndim," dimensions"
   write(0,'(A6,I1)') "naxis = ",naxis
   stop
endif

!read in image dimensions
call ftgknj(unitfits,'NAXIS',1,ndim,naxes,nfound,istatus)
!write(0,'(A7,3(I5,1X))') "naxes: ",(naxes(i),i=1,ndim)

!Check that it found both NAXIS1 and NAXIS2 keywords.
if (nfound.ne.ndim)then
   write(0,*) 'READIMAGE failed to read the NAXISn keywords.'
   stop
endif

return
end subroutine readheader