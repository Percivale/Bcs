
	parameter(nmax=10000)
        parameter(nhdim=930)
	real x(nmax),y(nmax),z(nmax)
	real xo(nmax),yo(nmax),zo(nmax)
	real boxx,boxy,boxz,beta
        real gx,gy,gz,apu,bz,dr,rcc
        real gt(nhdim),gg(nhdim),tt(nhdim)
        real sg(nhdim),st(nhdim),ss(nhdim)
        real tg(nhdim),gs(nhdim),ts(nhdim),all(nhdim)  
        real adhoc1,adhoc2,adhoc3,adhoc4,adhoc5,adhoc6
        real adhoc7,adhoc8,adhoc9,adhocc
	integer nions,ntot,count
        integer gete(nhdim),gege(nhdim),tete(nhdim),kk,ll,mm
        integer tege(nhdim),gesb(nhdim),tesb(nhdim)
        integer sbge(nhdim),sbte(nhdim),sbsb(nhdim),aa(nhdim)
        integer nge,nte,nsb
        character*2 sp(nmax),co         

 	open(10,file='TRAJEC.xyz')     ! INPUT
        open(11,file='gofr_0.01.hist') ! OUTPUT = partial distribution functions (PDFs)

        au=0.529177        ! Bohr radius
        boxx=35.17239*au   ! Box size
        boxy=35.17239*au
        boxz=35.17239*au
        beta=90.0          ! Box angle (cubic)
        pi=acos(-1.0)
        bz=0.01            ! Bin size for PDFs
        niter=8717         ! Number of frames in the xyz-file (trajectory)
        rcc=9.3            ! max. distance = 0.5*box

        nge=108            ! #Ge
        nte=108            ! #Te
        nions=216          ! #all

!       V/[(N_A*N_B-N_AB)*dr]

        adhoc1=boxx*boxy*boxz/(nge*(nge-1))/bz       ! cofactors for each PDF
        adhoc3=boxx*boxy*boxz/(nge*nte-(nge+nte))/bz
        adhoc7=boxx*boxy*boxz/(nte*nge-(nte+nge))/bz
        adhoc9=boxx*boxy*boxz/(nte*(nte-1))/bz
        adhocc=boxx*boxy*boxz/(nions*(nions-1))/bz

        do mm=1,niter         ! Frames
!	
	read(10,*)nions
	read(10,*)co
	do i=1,nions
	read(10,*) sp(i),x(i),y(i),z(i)
        x(i)=x(i)-boxx*nint(x(i)/boxx)
        y(i)=y(i)-boxy*nint(y(i)/boxy)
        z(i)=z(i)-boxz*nint(z(i)/boxz)
        end do	

        do i=1,nions
         do j=1,nions
         dx=x(i)-x(j)-boxx*nint((x(i)-x(j))/boxx) ! simple trick to include
         dy=y(i)-y(j)-boxy*nint((y(i)-y(j))/boxy) ! peridic boundary conditions (PBCs)
         dz=z(i)-z(j)-boxz*nint((z(i)-z(j))/boxz) ! <-- takes the images into account
         dr=sqrt(dx*dx+dy*dy+dz*dz)
!  Ge
	 if (dr.le.rcc.and.sp(i).eq.'Ge'.and.sp(j).eq.'Ge'.and.i.ne.j)
     &     then
          ll=nint(dr/bz)
          gege(ll)=gege(ll)+1   ! increases the distance bin for Ge-Ge PDF
          end if
!
          if (dr.le.rcc.and.sp(i).eq.'Ge'.and.sp(j).eq.'Te') then
          ll=nint(dr/bz)
          gete(ll)=gete(ll)+1
          end if
!  Te
          if (dr.le.rcc.and.sp(i).eq.'Te'.and.sp(j).eq.'Ge') then
          ll=nint(dr/bz)
          tege(ll)=tege(ll)+1
          end if
!
          if (dr.le.rcc.and.sp(i).eq.'Te'.and.sp(j).eq.'Te'.and.i.ne.j)
     &     then
          ll=nint(dr/bz)
          tete(ll)=tete(ll)+1
          end if
!  ALL
          if (dr.le.rcc.and.i.ne.j) then
          ll=nint(dr/bz)
          aa(ll)=aa(ll)+1
          end if
!
         end do
        end do
!
	end do          !  Frames

        
!-------------------------------------------------------

        do i=1,nhdim
        rbz=bz*i
        gg(i)=adhoc1*gege(i)/(4*pi*rbz*rbz)/niter ! Add cofactors and normalize
        gt(i)=adhoc3*gete(i)/(4*pi*rbz*rbz)/niter
        tg(i)=adhoc7*tege(i)/(4*pi*rbz*rbz)/niter
        tt(i)=adhoc9*tete(i)/(4*pi*rbz*rbz)/niter
        all(i)=adhocc*aa(i)/(4*pi*rbz*rbz)/niter 
!
        write(11,8999)rbz,gg(i),gt(i),tg(i),tt(i),all(i) ! Final PDFs and total RDF
        end do

 8989	format(x,(a),6f12.5)
 8999	format(x,f12.5,3x,10f12.5)

	stop
	end

