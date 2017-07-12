subroutine darkcorrect(naxes,fitsdata,darkdata,bpix,ans,idplot)
use precision
implicit none
!import vars
integer :: idplot
integer, dimension(:) :: naxes
real(double) :: bpix,maxd,mind,ans(2)
real(double), dimension(:,:) :: fitsdata,darkdata
!local vars
integer npixmax,npix,i,j,k
real(double) :: a,b,abdev
real(double), allocatable, dimension(:) :: x,y
!pgplot vars
real :: bb(4),xl(2),yl(2)
real, allocatable, dimension(:) :: xp,yp

npixmax=naxes(1)*naxes(2)
allocate(xp(npixmax),yp(npixmax),x(npixmax),y(npixmax))


!get bounds of dark image
k=0
do i=1,naxes(1)
   do j=1,naxes(2)
      if(darkdata(i,j).gt.bpix)then
         k=k+1
         if(k.eq.1)then
            maxd=darkdata(i,j)
            mind=darkdata(i,j)
         else
            maxd=max(maxd,darkdata(i,j))
            mind=min(mind,darkdata(i,j))
         endif
      endif
   enddo
enddo

!get valid pixels to compare dark and science image
npix=0
do i=1,naxes(1)
   do j=1,naxes(2)
      if((fitsdata(i,j).gt.bpix).and.(darkdata(i,j).gt.bpix))then
         if((fitsdata(i,j).ge.mind).and.(fitsdata(i,j).le.maxd))then 
            npix=npix+1
            x(npix)=darkdata(i,j)
            y(npix)=fitsdata(i,j)
         endif
      endif
   enddo
enddo


call medfit(x,y,npix,a,b,abdev)
ans(1)=a
ans(2)=b

!write(6,*) "Dark (zero,slope): ",ans(1),ans(2)


if(idplot.eq.0)then
   !convert dble -> real for PGPLOT
   xp=real(x)
   yp=real(y)
   !bound for plotting
   bb(1)=minval(xp)
   bb(2)=maxval(xp)
   bb(3)=minval(yp)
   bb(4)=maxval(yp)
   !set up plot
   call pgwindow(bb(1),bb(2),bb(3),bb(4))
   call pgbox('BCNTS1',0.0,0,'BCNTSV1',0.0,0)
   call pglabel("Dark","Object","")
   !plot data 
   call pgsci(2)
   call pgpt(npix,xp,yp,1)
   call pgsci(1)
   !plot fit
   xl(1)=bb(1)
   xl(2)=bb(2)
   yl(1)=real(ans(1)+ans(2)*bb(1))
   yl(2)=real(ans(1)+ans(2)*bb(2))
   call pgsci(2)
   call pgline(2,xl,yl)
   call pgsci(1)
endif

!apply Dark Correction
fitsdata=fitsdata-(ans(1)+ans(2)*darkdata)

return
end