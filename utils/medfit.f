CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE medfit(x,y,ndata,a,b,abdev)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      INTEGER ndata,NMAX,ndatat
      PARAMETER (NMAX=5045328)
      REAL*8 a,abdev,b,x(ndata),y(ndata),
     *     arr(NMAX),xt(NMAX),yt(NMAX),aa,abdevt
      COMMON /arrays/ xt,yt,arr,aa,abdevt,ndatat
C USES rofunc
      INTEGER j
      REAL*8 b1,b2,bb,chisq,del,f,f1,f2,sigb,sx,sxx,sxy,sy,rofunc
      sx=0.
      sy=0.
      sxy=0.
      sxx=0.
      do 11 j=1,ndata
         xt(j)=x(j)
         yt(j)=y(j)
         sx=sx+x(j)
         sy=sy+y(j)
         sxy=sxy+x(j)*y(j)
         sxx=sxx+x(j)**2
 11   enddo
      ndatat=ndata
      del=ndata*sxx-sx**2
      aa=(sxx*sy-sx*sxy)/del
      bb=(ndata*sxy-sx*sy)/del
      chisq=0.
      do 12 j=1,ndata
         chisq=chisq+(y(j)-(aa+bb*x(j)))**2
 12   enddo
      sigb=sqrt(chisq/del)
      b1=bb
      f1=rofunc(b1)
      if(sigb.gt.0.)then
         b2=bb+sign(3.*sigb,f1)
         f2=rofunc(b2)
         if(b2.eq.b1)then
            a=aa
            b=bb
            abdev=abdevt/ndata
            return
         endif
 1       if(f1*f2.gt.0.)then
            bb=b2+1.6*(b2-b1)
            b1=b2
            f1=f2
            b2=bb
            f2=rofunc(b2)
            goto 1
         endif
         sigb=0.01*sigb
 2       if(abs(b2-b1).gt.sigb)then
            bb=b1+0.5*(b2-b1)
            if(bb.eq.b1.or.bb.eq.b2)goto 3
            f=rofunc(bb)
            if(f*f1.ge.0.)then
               f1=f
               b1=bb
            else
               f2=f
               b2=bb
            endif
            goto 2
         endif
      endif
 3    a=aa
      b=bb
      abdev=abdevt/ndata
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      FUNCTION rofunc(b)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      INTEGER NMAX
      REAL*8 rofunc,b,EPS
      PARAMETER (NMAX=5045328,EPS=1.d-7)
C USES select
      INTEGER j,ndata
      REAL*8 aa,abdev,d,sum,arr(NMAX),x(NMAX),y(NMAX),select
      COMMON /arrays/ x,y,arr,aa,abdev,ndata
      do 11 j=1,ndata
         arr(j)=y(j)-b*x(j)
 11   enddo
      if (mod(ndata,2).eq.0) then
         j=ndata/2
         aa=0.5*(select(j,ndata,arr)+select(j+1,ndata,arr))
      else
         aa=select((ndata+1)/2,ndata,arr)
      endif
      sum=0.
      abdev=0.
      do 12 j=1,ndata
         d=y(j)-(b*x(j)+aa)
         abdev=abdev+abs(d)
         if (y(j).ne.0.) d=d/abs(y(j))
         if (abs(d).gt.EPS) sum=sum+x(j)*sign(1.0d0,d)
 12   enddo
      rofunc=sum
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      FUNCTION select(k,n,arr)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      INTEGER k,n
      REAL*8 select,arr(n)
      INTEGER i,ir,j,l,mid
      REAL*8 a,temp
      l=1
      ir=n
 1    if(ir-l.le.1)then
         if(ir-l.eq.1)then
            if(arr(ir).lt.arr(l))then
               temp=arr(l)
               arr(l)=arr(ir)
               arr(ir)=temp
            endif
         endif
         select=arr(k)
         return
      else
         mid=(l+ir)/2
         temp=arr(mid)
         arr(mid)=arr(l+1)
         arr(l+1)=temp
         if(arr(l).gt.arr(ir))then
            temp=arr(l)
            arr(l)=arr(ir)
            arr(ir)=temp
         endif
         if(arr(l+1).gt.arr(ir))then
            temp=arr(l+1)
            arr(l+1)=arr(ir)
            arr(ir)=temp
         endif
         if(arr(l).gt.arr(l+1))then
            temp=arr(l)
            arr(l)=arr(l+1)
            arr(l+1)=temp
         endif
         i=l+1
         j=ir
         a=arr(l+1)
 3       continue
         i=i+1
         if(arr(i).lt.a)goto 3
 4       continue
         j=j-1
         if(arr(j).gt.a)goto 4
         if(j.lt.i)goto 5
         temp=arr(i)
         arr(i)=arr(j)
         arr(j)=temp
         goto 3
 5       arr(l+1)=arr(j)
         arr(j)=a
         if(j.ge.k)ir=j-1
         if(j.le.k)l=i
      endif
      goto 1
      END