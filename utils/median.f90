function median(n,data)
use precision
implicit none
!import vars
integer :: n
real(double) :: median
real(double), dimension(:) :: data
!local vars
integer, allocatable, dimension(:) :: p

allocate(p(n))
call rqsort(n,data,p) !sort data to find the median
median=data(p(n/2)) !middle value

return
end function median