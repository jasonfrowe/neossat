subroutine writefits(naxes,parray,fileout)
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
!import vars
integer, dimension(:) :: naxes
real(double), dimension(:,:) :: parray
character(80) :: fileout
!local vars
integer :: nkeys,nstep,status,blocksize,bitpix,naxis,funit, &
   npixels,group,firstpix,nbuf,i,j,nbuffer
integer, dimension(4) :: nr
real(double), allocatable, dimension(:) :: buffer
character(80) :: record
logical simple,extend

status=0
!if file already exists.. delete it.
call deletefile(fileout,status)
!get a unit number
call ftgiou(funit,status)
!Create the new empty FITS file.  The blocksize parameter is a
!historical artifact and the value is ignored by FITSIO.
blocksize=1
status=0
call ftinit(funit,fileout,blocksize,status)
if(status.ne.0)then
   write(0,*) "Status: ",status
   write(0,*) "Critial Error open FITS for writing"
   write(0,'(A80)') fileout
endif

!Initialize parameters about the FITS image.
!BITPIX = 16 means that the image pixels will consist of 16-bit
!integers.  The size of the image is given by the NAXES values.
!The EXTEND = TRUE parameter indicates that the FITS file
!may contain extensions following the primary array.
simple=.true.
bitpix=-32
naxis=2
extend=.true.

!Write the required header keywords to the file
call ftphpr(funit,simple,bitpix,naxis,naxes,0,1,extend,status)

!Write the array to the FITS file.
npixels=naxes(1)*naxes(2)
group=1
firstpix=1
nbuf=naxes(1)
j=0

allocate(buffer(nbuf))
do while (npixels.gt.0)
!read in 1 column at a time
   nbuffer=min(nbuf,npixels)

   j=j+1
!find max and min values
   do i=1,nbuffer
      buffer(i)=parray(i,j)
   enddo

   call ftpprd(funit,group,firstpix,nbuffer,buffer,status)

!update pointers and counters

   npixels=npixels-nbuffer
   firstpix=firstpix+nbuffer

enddo


!close fits file
call ftclos(funit,status)
call ftfiou(funit,status)

return
end subroutine writefits
