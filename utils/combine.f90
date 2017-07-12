subroutine combine(naxes,nfiles,allfitsdata,fitsdata,bpix,ilow,ihigh)
use precision
implicit none
!import vars
integer :: nfiles,ilow,ihigh
integer, dimension(:) :: naxes
real(double) :: bpix
real(double), dimension(:,:) :: fitsdata
real(double), dimension(:,:,:) :: allfitsdata
!local vars
integer :: i,j,k,np,i1,i2
integer, allocatable, dimension(:) :: p
real(double), allocatable, dimension(:) :: pixels

allocate(pixels(nfiles),p(nfiles))

do i=1,naxes(1)
   do j=1,naxes(2)
      np=0
      do k=1,nfiles
         if(allfitsdata(i,j,k).gt.bpix)then
            np=np+1
            pixels(np)=allfitsdata(i,j,k)
         endif
      enddo
      if(np.lt.1)then
         fitsdata(i,j)=bpix
      else if (np.eq.1)then
         fitsdata(i,j)=pixels(np)
      else
         call rqsort(np,pixels,p)
         i1=1+ilow
         i2=np-ihigh
         if(i1.gt.i2)then
            i1=(np+1)/2
            i2=(np+1)/2
         endif
         !write(0,*) i1,i2,Sum(pixels(i1:i2))/dble(i2-i1+1)
         !read(5,*)
         fitsdata(i,j)=0.0d0
         do k=i1,i2
            fitsdata(i,j)=fitsdata(i,j)+pixels(p(k)) !average of pixels
         enddo
         fitsdata(i,j)=fitsdata(i,j)/dble(i2-i1+1)
      endif
   enddo
enddo

return
end subroutine combine