# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:56:10 2013

@author: josip
"""

def read_avg_file(filein):    
    """ 
    Ucitavam razne ascii datoteke koje su dobivene bin2ascii_GG programom pretvorbom iz .bin CAMx formata 

    """
    from itertools import islice
    import numpy as np  
    import math 
    data={}
    f=open(filein,'r')
    h1=f.readline()
    print "Working on "+ h1[0:10] + " type !"
    h2=np.fromstring(f.readline(),sep=' ')    
    broj_mreza,broj_specija,idate_start,itime_start,idate_end,itime_end = h2    
    dt=24 # koliko imam ispisa u 24 sata (svaki sat -> dt=24)
    tmp=f.readline()
    dx,dy,nx,ny,nz = np.fromstring(tmp,sep=' ')[5:10]    
    f.readline()
    data['ime']=[]
    for i in np.arange(broj_specija):
        data['ime'].append(f.readline().strip())
    broj_recorda=np.floor(((idate_end+itime_end/24)-(idate_start+itime_start/24))*dt) # prije je bilo ovo: broj_recorda=np.floor((idate_end+itime_end)-(idate_start+itime_start/24))
    data['konc']=np.zeros((broj_recorda,broj_specija,ny,nx))
    data['record_time']=[]
    data['t0']=[]
    data['t1']=[]
    print "File ima " + str(broj_recorda) + " sati"
    for i in np.arange(broj_recorda): #time loop
        print i
        #tmp=f.next()
        tmp=f.readline()
        tmp=np.fromstring(tmp,sep=' ') #fscanf(f,'%d %f %d %f\n',[1 4]);
        print tmp
        data['record_time'].append((tmp[0]+tmp[1]/24.0+tmp[2]+tmp[3]/24.0)*0.5)
        data['t0'].append(tmp[0]+tmp[1]/24.0)
        data['t1'].append(tmp[2]+tmp[3]/24.0)
        for j in np.arange(broj_specija): #  specija loop
            tmp=f.readline()
            #print tmp
            #tmp=list(islice(f,ny*nx/5))            
            tmp=np.fromfile(f,count=int(ny*nx),sep=' ')
            dd=np.loadtxt(tmp,dtype='float').reshape(ny,nx)
            data['konc'][i,j,:,:]=dd
        del tmp
        print 'Working on record ' + str(i)
    return data

def get_avg_info(filein):    
    from itertools import islice
    import numpy as np  
    data={}
    f=open(filein,'r')
    h1=f.readline()
    h2=np.fromstring(f.readline(),sep=' ')    
    broj_mreza,broj_specija,idate_start,itime_start,idate_end,itime_end = h2    
    dt=24 # koliko imam ispisa u 24 sata (svaki sat -> dt=24)
    tmp=f.readline()
    dx,dy,nx,ny,nz = np.fromstring(tmp,sep=' ')[5:10]  
    tmp=f.readline()
    x0,y0,x1,y1 = np.fromstring(tmp,sep=' ')
    data['ime']=[]
    for i in np.arange(broj_specija):
        data['ime'].append(f.readline().strip())
    broj_recorda=np.floor(((idate_end+itime_end/24)-(idate_start+itime_start/24))*dt) # prije je bilo ovo: broj_recorda=np.floor((idate_end+itime_end)-(idate_start+itime_start/24))
    print "Ispisujem [dx,dy,nx,ny,nz,broj_recorda]"
    return [x0,y0,x1,y1,dx,dy,nx,ny,nz,broj_recorda]
   
def yyjjj2yymmdd(datein):
    from datetime import datetime,timedelta
    tmp=str(datein) 
    if int(tmp[0:2])>50:
        #ovo ako je godin veća od 50 ne bu radil kod..do tad ludilo        
        y=int('20' + '0' + tmp[0])
    else:        
        y=int('20' + tmp[0:2])
    d=float(tmp[2:])
    dateout=datetime(y,1,1)+timedelta(d-1)
    return dateout
           
def getWRFij(WRFin,xlon,ylat):
    import numpy as np
    from netCDF4 import Dataset,num2date
    nc=Dataset(WRFin,'r')
    lon=nc.variables['XLONG'][0,1,:]
    lat=nc.variables['XLAT'][0,:,1]
    indX=np.argmin((abs(lon-xlon)))
    indY=np.argmin((abs(lat-ylat)))
    return [indX,indY]

def getCAMxij(WRFin,xlon,ylat,x0,nx,y0,ny):
    import numpy as np
    from netCDF4 import Dataset,num2date
    nc=Dataset(WRFin,'r')
    lon=nc.variables['XLONG'][0,y0:ny+y0,x0:nx+x0]
    lat=nc.variables['XLAT'][0,y0:ny+y0,x0:nx+x0]    
    print lon.shape
    #print lat.shape
    indX=np.argmin((abs(lon[1,:]-xlon)))
    indY=np.argmin((abs(lat[:,1]-ylat)))
    return [indX,indY]

def getCAMgrid(WRFin,xlon,ylat,x0,nx,y0,ny):
    import numpy as np
    from netCDF4 import Dataset,num2date
    nc=Dataset(WRFin,'r')
    lon=nc.variables['XLONG'][0,y0:ny+y0,x0:nx+x0]
    lat=nc.variables['XLAT'][0,y0:ny+y0,x0:nx+x0]    
    return lon,lat  
    