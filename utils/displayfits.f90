subroutine displayfits(naxes,parray,bpix,sigscalel,sigscaleh)
use precision
implicit none
!import vars
integer, dimension(:) :: naxes
real(double) :: bpix,sigscalel,sigscaleh
real(double), dimension(:,:) :: parray
!local vars
integer :: npt,i,j,k,nxmax,nymax,nr(4),maxpix,nsample,ncol
integer, allocatable, dimension(:) :: p
integer, allocatable, dimension(:,:) :: ia
real(double) :: maxp,minp,hcut,med,std,stdev2,minlp,lmin,lmax,z1,z2
real(double), allocatable, dimension(:) :: a
real(double), allocatable, dimension(:,:) :: lparray
!PGPLOT vars
integer :: nlw
real :: rj(4),fontsize,r,g,b,xr,x2,y2

!local configuration options
nsample=5 !to calculate stats, 1 pixels in a nsample x nsample grid is selected
          !this greatly speeds up the program.
hcut=1.0d10 !ignore pixels values greater than hcut when calculating stats 
maxpix=1000000 !only number a maximum of maxpix pixels when calculating stats 
ncol=64 !number of colours for display
fontsize=1.0 !size of font for display (real)
nlw=2 !line width (integer)

!store size of image for display
nxmax=naxes(1)
nymax=naxes(2)

!local arrays
allocate(lparray(nxmax,nymax)) !used for making a log-scale plot
allocate(ia(nxmax,nymax))      !contains integer colour mapping 

!get full range of valid data
minp=0.0d0 !init minp,maxp incase of no data.
maxp=0.0d0
npt=0
do i=1,nxmax
   do j=1,nymax
      if(parray(i,j).gt.bpix)then
         npt=npt+1
         if(npt.eq.1)then
            maxp=parray(i,j)
            minp=parray(i,j)
         else
            maxp=max(maxp,parray(i,j))
            minp=min(minp,parray(i,j))
         endif
      endif
   enddo
enddo

!image display range
nr(1)=1
nr(3)=1
nr(2)=nxmax
nr(4)=nymax
!make real for PGPLOT
rj=real(nr)

!if no valid data in array, then do nothing and return
if(npt.le.1)then
   write(0,*) "Min/Max:",minp,maxp
   write(0,*) "Warning: No Valid data to display"
   return
endif

!calculate stats about the image
allocate(a(npt),p(npt))
k=0
do i=nr(1),nr(2),nsample
   do j=nr(3),nr(4),nsample
      if((parray(i,j).gt.bpix).and.(abs(parray(i,j)).lt.hcut))then
         k=k+1
         a(k)=parray(i,j)
         if(k.ge.maxpix) goto 10 !break from loop if we have enough pixles for stats 
      endif
   enddo
enddo
10 continue

!calculate median and standard deviation around the median
if(k.ge.3)then
   call rqsort(k,a,p)
   med=a(p(k/2))
   std=stdev2(k,a,med)
else
   med=0.0
   std=1.0
endif
deallocate(a,p)
!write(0,*) "med,std: ",med,std

!use a log scaling for display
minlp=minp
lmin=1000.0  !lmin and lmax will be greater or equal to 0.
lmax=-1000.0
do i=nr(1),nr(2)
   do j=nr(3),nr(4)
      if(parray(i,j).gt.bpix)then
         if(parray(i,j)-minlp+1.0.le.0.0d0)then
            lparray(i,j)=0.0
         else
            lparray(i,j)=log10(parray(i,j)-minlp+1.0)
         endif
         lmin=min(lparray(i,j),lmin)
         lmax=max(lparray(i,j),lmax)
      endif
   enddo
enddo

!calculate display range based on log scale
z1=0.0
z2=log10(maxp-minlp+1.0)
if(sigscalel.gt.0.0d0)then
   z1=log10(max(1.0,med-sigscalel*std-minlp)+1.0)
endif
if(sigscaleh.gt.0.0d0)then
   z2=log10(max(1.0,med+sigscaleh*std-minlp)+1.0)
endif

!uncomment for a sqrt scale opposed to log
!lparray=sqrt(parray-minp)
!z1=sqrt(med-std-minp)
!z2=sqrt(maxp-minp)

!write(0,*) "zscale: ",z1,z2

!map pixel values to display scale
do i=nr(1),nr(2)
   do j=nr(3),nr(4)
      if(parray(i,j).gt.bpix)then
         IA(i,j)=int((lparray(i,j)-z1)/(z2-z1)*dble(NCOL-1))+16
         if(lparray(i,j).le.z1) then
            ia(i,j)=16
         endif
      else
         ia(i,j)=15
      endif
      if(lparray(i,j).gt.z2) ia(i,j)=ncol+15
   enddo
enddo

!set up pgplot window
!call pgscr(0,1.0,1.0,1.0)
call pgscr(15,0.0,0.3,0.2)

call pgsch(fontsize) !make the font a bit bigger
call pgslw(nlw)  !make the lines a bit thicker

!call pgscr(1,0.0,0.0,0.0)
call pgsci(1)
call pgvport(0.0,1.00,0.0,1.0)
call pgwindow(0.0,1.0,0.0,1.0)
!call PGRECT (0.0, 1.0, 0.0, 1.0)

call pgvport(0.10,0.95,0.15,0.95) !make room around the edges for labels
call pgsci(1)
call pgwindow(rj(1),rj(2),rj(3),rj(4)) !plot scale
call pgbox("BCNTS1",0.0,0,"BCNTS1",0.0,0)
!call pglabel("X (pixels)","Y (pixels)","")
call pgptxt((rj(1)+rj(2))/2.0,rj(3)-0.16*(rj(4)-rj(3)),0.0,0.5,         &
   "X (pixels)")
call pgptxt(rj(1)-0.06*(rj(2)-rj(1)),(rj(4)+rj(3))/2,90.0,0.5,          &
   "Y (pixels)")
call pgsci(1)

!read in and set display scale
do i=1,ncol
   call heatlut(i*4-3,r,g,b)
   call heatlut(i*4-2,r,g,b)
   call heatlut(i*4-1,r,g,b)
   call heatlut(i*4  ,r,g,b)
   CALL PGSCR(I+15, R, G, B)
enddo

!display the pixels
xr=real(max(nr(2)-nr(1),nr(4)-nr(3)))
x2=real(nr(2)-nr(1))/xr
y2=real(nr(4)-nr(3))/xr
call pgpixl(ia,nxmax,nymax,nr(1),nr(2),nr(3),nr(4),rj(1),rj(2),rj(3),rj(4))
CALL PGSCR(ncol+15, 1.0, 1.0, 1.0)
call pgsci(ncol+15)
!redraw axes
call pgbox("BCTS",0.0,0,"BCTS",0.0,0)
call pgsci(1)

call pgwindow(rj(1),rj(2),rj(3),rj(4)) !plot scale

!deallocate arrays
deallocate(ia,lparray)

return
end subroutine displayfits