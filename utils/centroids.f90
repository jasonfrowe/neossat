subroutine centroids(naxes,fitsdata,bpix,nstar,coo,ndp)
use precision
implicit none
!import vars
integer :: nstar,ndp
integer, dimension(:) :: naxes
real(double) :: bpix
real(double), dimension(:,:) :: fitsdata,coo
!local vars
integer :: i,j,k
real(double) :: sumx,sumy,fsum,dj,dk,xc,yc

!loop over all stars and update centroids
do i=1,nstar
   sumx=0.0d0 !initialize to zero
   sumy=0.0d0
   fsum=0.0d0
   do j=max(1,int(coo(1,i))-ndp),min(naxes(1),int(coo(1,i))+ndp)
      dj=dble(j)
      do k=max(1,int(coo(2,i))-ndp),min(naxes(2),int(coo(2,i))+ndp)
         dk=dble(k)
         if(fitsdata(j,k).gt.bpix)then
            sumx=sumx+fitsdata(j,k)*dj
            sumy=sumy+fitsdata(j,k)*dk
            fsum=fsum+fitsdata(j,k)
         endif
      enddo
   enddo
   xc=sumx/fsum
   yc=sumy/fsum

   coo(1,i)=xc
   coo(2,i)=yc
enddo

return
end subroutine centroids