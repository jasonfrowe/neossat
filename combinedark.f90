program combinedarks
use precision
implicit none
integer iargc,ndim,i,unitfits,nunit,ifiles,filestatus,nfiles,ilow,ihigh
integer, allocatable, dimension(:) :: naxes,naxes1
real(double) :: bpix,sigscalel,sigscaleh
real(double), allocatable, dimension(:,:) :: fitsdata
real(double), allocatable, dimension(:,:,:) :: allfitsdata
character(80) :: fitsfile,filelist,darkfile

!interfaces to routines with implicit array sizes 
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
   subroutine combine(naxes,nfiles,allfitsdata,fitsdata,bpix,ilow,ihigh)
      use precision
      implicit none
      integer :: nfiles,ilow,ihigh
      integer, dimension(:) :: naxes
      real(double) :: bpix
      real(double), dimension(:,:) :: fitsdata
      real(double), dimension(:,:,:) :: allfitsdata
   end subroutine combine
   subroutine writefits(naxes,parray,fileout)
      use precision
      implicit none
      integer, dimension(:) :: naxes
      real(double), dimension(:,:) :: parray
      character(80) :: fileout
   end subroutine writefits
end interface

!global vars
ndim=2 !number of dimensions in FITS image. If you change ndim then you also
       !need to change the allocation of fitsdata to match.  All NEOSSat data
       !appears to be ndim=2
nunit=10 !unit number used to read in file list.
bpix=-1.0d10 !value to mark bad pixels.  Any pixel with a value *below* bpix
             !is considered invalid.  
sigscalel=0.01 !low bounds for clipping.  Keep small to keep background 'dark'
sigscaleh=3.0  !high bounds for clipping for display scale used by displayfits
ilow=1 !number of low value frames to reject
ihigh=1 !number of high value frames to reject
darkfile="dark.fits" !name of combined dark frame


if (iargc().lt.1) then
   write(0,*) 'Usage: neossatphot <FITSLIST>'
   write(0,*) '  <FITSLIST> : List of NEOSSat FITS file'
endif

call getarg(1,filelist) !store commandline parameter for FITS file name

call openfilelist(nunit,filelist,nfiles)

!subroutine readheader can be found in utils/readfits.f90
allocate(naxes(ndim),naxes1(ndim))

!display fits file
!call pgopen('?')
call pgopen('/xserve')
call PGPAP (8.0 ,1.0) !use a square 8" across
call pgpage()

ifiles=0
do

   !read in filename
   read(nunit,'(A)',iostat=filestatus) fitsfile
   !write(0,'(A10,A80)') "fitsfile: ",fitsfile

   if(filestatus == 0) then

      ifiles=ifiles+1

      call readheader(fitsfile,unitfits,ndim,naxes) !openfile, get header info, including size of FITS

      if(ifiles.eq.1)then
         naxes1=naxes
         allocate(fitsdata(naxes1(1),naxes1(2)))
         allocate(allfitsdata(naxes1(1),naxes1(2),nfiles))
      else
         if((naxes1(1).ne.naxes(1)).or.(naxes1(2).ne.naxes(2)))then
            write(0,*) "Error: Dimension mismatch"
            write(0,'(A8,3(I5,1X))') "naxes : ",(naxes(i),i=1,ndim)
            write(0,'(A8,3(I5,1X))') "naxes1: ",(naxes1(i),i=1,ndim)
         endif
      endif

      call readfits(unitfits,naxes,fitsdata,bpix)
      !write(0,*) "read..",minval(fitsdata),maxval(fitsdata)
      allfitsdata(:,:,ifiles)=fitsdata(:,:)

      !if (ifiles.gt.1) call pgpage()
      !call displayfits(naxes,fitsdata,bpix,sigscalel,sigscaleh)

      close(unitfits) !close FITS file

   elseif(filestatus == -1) then
      exit  !successively break from data read loop.

   else
      write(0,*) "File Error!! Line: ",ifiles
      write(0,900) "iostat: ",filestatus
      900 format(A8,I3)
      stop

   endif

enddo
close(nunit) !close filelist

!combine allfitsdata -> fitsdata
call combine(naxes,nfiles,allfitsdata,fitsdata,bpix,ilow,ihigh)

!display the combined image
call displayfits(naxes,fitsdata,bpix,sigscalel,sigscaleh)

!write the combined Image to a new file
call writefits(naxes,fitsdata,darkfile)

!deallcate arrays
deallocate(fitsdata,allfitsdata)

call pgclos() !close plotting unit

end program combinedarks
