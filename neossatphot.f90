program neossatphot
use precision
implicit none
integer iargc,iusedark,ndim,unitfits,i,idplot,isplot,iplot,nstar,nunit, &
 ndp
real(double) :: bpix,sigscalel,sigscaleh,jddate,jdzero,ans(2),sphot, &
 starloc(4),exptime,sbox
integer, allocatable, dimension(:) :: naxes,naxesd
real(double), allocatable, dimension(:,:) :: darkdata,fitsdata,coo
character(80) :: fitsfile,darkfile,starfile

interface
   subroutine readfits(unitfits,naxes,fitsdata,bpix)
      use precision
      implicit none
      integer :: unitfits
      integer, dimension(:) :: naxes
      real(double) :: bpix
      real(double), dimension(:,:) :: fitsdata
   end subroutine readfits
   subroutine displayfits(naxes,parray,bpix,sigscalel,sigscaleh)
      use precision
      implicit none
      integer, dimension(:) :: naxes
      real(double) :: bpix,sigscalel,sigscaleh
      real(double), dimension(:,:) :: parray
   end subroutine displayfits
   subroutine darkcorrect(naxes,fitsdata,darkdata,bpix,ans,idplot)
      use precision
      implicit none
      integer :: idplot
      integer, dimension(:) :: naxes
      real(double) :: bpix,ans(2)
      real(double), dimension(:,:) :: fitsdata,darkdata
   end subroutine darkcorrect
   subroutine columncor(naxes,fitsdata,bpix)
      use precision
      implicit none
      integer, dimension(:) :: naxes
      real(double) :: bpix
      real(double), dimension(:,:) :: fitsdata
   end subroutine columncor
   subroutine getstarlist(nunit,nstar,coo)
      use precision
      implicit none
      integer :: nunit,nstar
      real(double), dimension(:,:) :: coo
   end subroutine getstarlist
   subroutine centroids(naxes,fitsdata,bpix,nstar,coo,ndp)
      use precision
      implicit none
      integer :: nstar,ndp
      integer, dimension(:) :: naxes
      real(double) :: bpix
      real(double), dimension(:,:) :: fitsdata,coo
   end subroutine centroids
end interface

if (iargc().lt.2) then
   write(0,*) 'Usage: neossatphot <FITS> <STARLIST> [DARK_FITS]'
   write(0,*) '  <FITS>     : NEOSSat FITS file'
   write(0,*) '  <STARLIST> : Text file containing CCD co-ordinates of star locations'
   write(0,*) '  [DARK_FITS]: optional dark frame'
   stop 900
endif

!global vars
ndim=2 !number of dimensions in FITS image. If you change ndim then you also
       !need to change the allocation of fitsdata to match.  All NEOSSat data
       !appears to be ndim=2
bpix=-1.0d10 !value to mark bad pixels.  Any pixel with a value *below* bpix
             !is considered invalid.  
sigscalel=0.01 !low bounds for clipping.  Keep small to keep background 'dark'
sigscaleh=3.0  !high bounds for clipping for display scale used by displayfits
idplot=1 !0= plot dark correction calculation, 1= do not plot
isplot=0 !0= plot science image, 1= do not plot
jdzero=2457900.0d0 !MJD zero point.
ndp=5 !ndp=2 produces a 5x5 window to find centroid center
sbox=10.0 !simple box size for easy photometry 

if((idplot.eq.1).and.(isplot.eq.1))then
   iplot=1
else
   iplot=0
endif

if(iplot.eq.0)then
   !display fits file
   !call pgopen('?')
   call pgopen('/xserve')
   call PGPAP (8.0 ,1.0) !use a square 8" across
   call pgpage()
endif

!allocate arrays
allocate(naxes(ndim),naxesd(ndim))


call getarg(1,fitsfile) !store commandline parameter for FITS file name

iusedark=1 !default is to not use a dark frame
if(iargc().gt.2)then
   call getarg(3,darkfile)
   iusedark=0 !set flag that a dark frame is available 
   call readheader(darkfile,unitfits,ndim,naxes,jddate,jdzero,exptime)
   naxesd=naxes
   allocate(darkdata(naxesd(1),naxesd(2)))
   call readfits(unitfits,naxesd,darkdata,bpix)
   close(unitfits) !close FITS file
   !call displayfits(naxes,darkdata,bpix,sigscalel,sigscaleh)
endif

!read in Science image
call readheader(fitsfile,unitfits,ndim,naxes,jddate,jdzero,exptime)
if(((naxesd(1).ne.naxes(1)).or.(naxesd(2).ne.naxes(2))).and.(iusedark.eq.0))then
   write(0,*) "Error: Dimension mismatch"
   write(0,'(A8,3(I5,1X))') "naxes : ",(naxes(i),i=1,ndim)
   write(0,'(A8,3(I5,1X))') "naxesd: ",(naxesd(i),i=1,ndim)
   stop 901
endif
allocate(fitsdata(naxes(1),naxes(2)))
call readfits(unitfits,naxes,fitsdata,bpix)
close(unitfits) !close FITS file 

!read in Star list
nunit=10
call getarg(2,starfile)
call openfilelist(nunit,starfile,nstar) !get nstar so we can allocate arrays
allocate(coo(2,nstar))
call getstarlist(nunit,nstar,coo)
close(nunit)

!Dark correction
ans=0.0d0
if(iusedark.eq.0)then
   call darkcorrect(naxes,fitsdata,darkdata,bpix,ans,idplot)
endif

!Column correction
!subtract the median from each column
call columncor(naxes,fitsdata,bpix)

!display science image
if(isplot.eq.0)then
   call displayfits(naxes,fitsdata,bpix,sigscalel,sigscaleh)
endif

!Update centroids
!write(0,*) coo(1,1),coo(2,1)
do i=1,3
   call centroids(naxes,fitsdata,bpix,nstar,coo,ndp)
enddo
!write(0,*) coo(1,1),coo(2,1)
starloc(1)=coo(1,1)-sbox
starloc(2)=coo(1,1)+sbox
starloc(3)=coo(2,1)-sbox
starloc(4)=coo(2,1)+sbox

if(isplot.eq.0)then
   !overlay square for beta pic
   call pgsci(2)
   call pgsfs(2) !outline of box instead of fill
   call pgrect(real(starloc(1)),real(starloc(2)),real(starloc(3)),real(starloc(4)))
   call pgsci(1)
endif

!extract photometry
!call photometry()

sphot=Sum(fitsdata(starloc(1):starloc(2),starloc(3):starloc(4)))

write(6,500) jddate,ans(1),ans(2),sphot,exptime,coo(1,1),coo(2,1)
500 format(F16.10,2(1X,F10.5),1X,1PE17.10,0P,1X,F8.4,2(1X,F17.10))

if(iplot.eq.0) call pgclos()

end program neossatphot