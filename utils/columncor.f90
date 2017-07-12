subroutine columncor(naxes,fitsdata,bpix)
use precision
implicit none
!import vars
integer, dimension(:) :: naxes
real(double) :: bpix
real(double), dimension(:,:) :: fitsdata
!local vars
integer :: i,j,npix
integer, allocatable, dimension(:) :: p
real(double) :: colcor,coravg
real(double), allocatable, dimension(:) :: pixels

interface
   function median(n,data)
      use precision
      implicit none
      integer :: n
      real(double) :: median
      real(double), dimension(:) :: data
   end function median
end interface

allocate(pixels(naxes(2))) !stores valid pixels from each column

coravg=0.0d0 !average correction value 
do i=1,naxes(1) 
   npix=0 !counts number of valid pixels in each column
   do j=1,naxes(2)
      if(fitsdata(i,j).gt.bpix)then
         npix=npix+1
         pixels(npix)=fitsdata(i,j)
      endif
   enddo
   colcor=median(npix,pixels)
   fitsdata(i,:)=fitsdata(i,:)-colcor 
   coravg=coravg+colcor
enddo
coravg=coravg/dble(naxes(1))
fitsdata=fitsdata+coravg !add back 'sky'

return
end subroutine columncor