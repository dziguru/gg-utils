# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:03:28 2014

@author: goran gasparac

"""

import numpy as np
from datetime import datetime,timedelta

def sigma2height(indata,info):
    """

    Prebacujem sigma nivoe u visinu u metrima
    indata je tekst_file ili wrfout, to specificiraj u info
    info = 'file'
         = 'wrfout'
    
    """
    import pandas
    from netCDF4 import Dataset,num2date
    import numpy as np
    
    if info=='dataframe':
        df=pandas.read_csv(indata,names=['sigma'])
    elif info=='wrfout':
        nc=Dataset(indata,'r')
        sigma_lev=nc.variables['ZNW'][1,:]
        df=pandas.DataFrame();
        df.sigma=np.flipud(sigma_lev)
    elif info=='nc':
        nc=indata
        sigma_lev=nc.variables['ZNW'][1,:]
        df=pandas.DataFrame();
        df.sigma=np.flipud(sigma_lev)

    p_surf=100000.
    p_top=5000.
    p_star=95000.
    ts0=290.
    tlp=50.

    p={}
    dp={}
    tk={}
    dz={}
    z={}
    z_half={}
    
    br_nivoa=len(df.sigma)
    
    for ii in range(br_nivoa):
        p[ii]=p_top+df.sigma[ii]*p_star
    
    for jj in range(br_nivoa-1):
        dp[jj]=p[jj]-p[jj+1]
    
    tk[br_nivoa-1]=285.
    for ii in range(br_nivoa-2,-1,-1):
        tk[ii]=tk[ii+1]+tlp*(p[ii]-p[ii+1])/((p[ii]+p[ii+1])/2.)
        
    for ii in range(br_nivoa-2,-1,-1):
        dz[ii]=-287/9.81*dp[ii]*(tk[ii]+tk[ii+1])/(p[ii]+p[ii+1])
        
    z[br_nivoa-1]=0.
    for ii in range(br_nivoa-2,-1,-1):
        z[ii]=z[ii+1]+dz[ii]
    
    z[br_nivoa-1]=0.
    for ii in range(br_nivoa-2,-1,-1):
        z_half[ii]=(z[ii]+z[ii+1])/2.
    
    z_half[len(z_half)]=0 # dodamo jos nulu jer to će biti sloj gdje je sigma=1
    out=np.flipud(z_half.values())
    return out

def mixratio2rh(mix_ratio,pres,temp):
    """
    Calculating RH from mixing ration, pressure and temperauture.

    mixratio(Q2,PSFC,T2)
    
    INPUTS
    mix_ratio     [kg/kg]
    pressure      [Pa]
    temp          [C]
    
    OUTPUT
    rh  relative humidity [percentage]
    """
    
    import numpy as np
    
    #CONVERSIONS
    temp=temp+273.15;          #C to kelvin
    mix_ratio=mix_ratio*1000;  #kg/kg to g/kg
    pres=pres/100;             #pa to hPa

    """
    %-----------------------------------
    %Calculate saturation vapor pressure
    %Paul van Delst, CIMSS/SSEC, 18-Mar-1998
    %     IDL v5 langueage
    %-----------------------------------
    % in mb (over water as compared to ice)
    """
    
    #data init
    t_sat=373.16;
    t_ratio=t_sat/temp;
    rt_ratio=1.0/t_ratio;
    sl_pressure=1013.246;

    c1=7.90298;
    c2=5.02808;
    c3=1.3816e-7;
    c4=11.344;
    c5=8.1328e-3;
    c6=3.49149;

    #calcs
    svp = -1.0*c1*(t_ratio-1.0)+c2*np.log10(t_ratio)-c3*(10.0**(c4*(1.0-rt_ratio))-1.0)+c5*(10.0**(-1.0*c6*(t_ratio-1.0))-1.0)+np.log10(sl_pressure);
    svp = 10.0**svp; # in mb

    """
    Calculate saturation mixing ratio
    from the Cooperative Institute for Meteorological Satellite Studies, 1999
         IDL v5 langueage
    -- in mb
    
    The saturation mixing ratio is caluclated using:
    
                             Mw         e
              ws = 1000.0 * ---- * -----------
                             Md     ( p - e )
    
           where Mw = molecular weight of water (kg),
                 Md = effective molecular weight of dry air (kg),
                 e  = saturation vapour pressure over water (hPa), and
                 p  = total atmospheric pressure (hPa).
    
           The value used for Mw/Md is hardwired to 0.621970585.

    """
    water_air_mass_ratio = 0.621970585;
    vapor_pressure_ratio = svp/(pres-svp);
    
    smr = 1000.0*water_air_mass_ratio*vapor_pressure_ratio;

    #Calculate relative humidity
   
    rh=mix_ratio/smr*100;
    return rh

def uv2wd(u,v):
    """
    Uzimam u i v i bacam van wd
    Problem je kaj math.atan2 moze racunat samo sa dva broja, a ne sa vektor stupcem
    Zato idu 2 petlje, zasad dok se ne nađe bolje rjesenje....
    """
    import math
    wd=[]
    for ind in range(0,len(u)):
        wd.append(57.3*math.atan2(u[ind],v[ind])+180)
    return wd
    
def geo_distance_np(lat1,lon1,lat2,lon2):
    """
    lat1,lon1 - koordinate točke
    lat2,lon2 - koordinate mreže (2D array)
    """
    import numpy as np
    R = 6373.0
    
    lat1 = np.pi*lat1/180.0
    lon1 = np.pi*lon1/180.0
    lat2 = np.pi*lat2/180.0
    lon2 = np.pi*lon2/180.0
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1    
    
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    #Kilometri!!
    return distance
    
    
def geo_distance(lat1,lon1,lat2,lon2):

    from math import sin, cos, sqrt, atan2, radians
    
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    # Note: unit is km!
    
    return(distance)
        
def openWRF(fileIN):
    from netCDF4 import Dataset,num2date
    try:
        nc=Dataset(fileIN,'r')
        return nc
    except Exception:        
        print 'Found no WRF-file for. Looked for: '+ fileIN
        return None

def getDimensions(nc):
    Nx = nc.getncattr('WEST-EAST_GRID_DIMENSION')-1
    Ny = nc.getncattr('SOUTH-NORTH_GRID_DIMENSION')-1
    Nz = nc.getncattr('BOTTOM-TOP_GRID_DIMENSION')-1    
    dx = nc.getncattr('DX')
    dy = nc.getncattr('DY')
    lons = nc.variables['XLONG'][0]
    lats = nc.variables['XLAT'][0]
    return Nx,Ny,Nz,lons,lats,dx,dy
    
def getVar(nc,var):
    import numpy as np
    from datetime import datetime,timedelta
    try:
        if var=='Times':
            varIN=nc.variables['Times'][:]
            wt=np.array([datetime.strptime("".join(t),'%Y-%m-%d_%H:%M:%S') for t in varIN])
        else:
            wt=nc.variables[var][:]
        return wt
    except Exception:        
        print 'There is no variable in nc file!'
        return None    

def getXY(nc,lon,lat,station_list,slat,slong):
    import numpy as np
    indX={};indY={} # koordinate u WRF mreži    
    for s in station_list:
        indX[s]=np.argmin((abs(lon[1,:]-slong[s])))
        indY[s]=np.argmin((abs(lat[:,1]-slat[s])))
    return indX,indY


def getXY_simple(lon,lat,stat_long,stat_lat):
    import numpy as np
    indX=np.argmin((abs(lon[1,:]-stat_long)))
    indY=np.argmin((abs(lat[:,1]-stat_lat)))
    return indX,indY
    
    
def getXY_p2p(fileIN,sp,lon,lat, nms,stID):
    """
    Gleda udaljenost točke obzirom na x,y koordinate
    
    lon,lat su 2D arrayi grida (EMEP, WRF, ...)
    filein je file koji sadrži station_name,lon,lat
    nms - imena kolona u fileu, standard je ['sname','lon','lat'],
    stID - kolona na kojoj je ime stanice
    
    npr:
    indXY=getXY_p2p(fileIN,lon,lat,['sname','lon','lat'],0)
          
    """
    
    import numpy as np
    import pandas as pd
    df=pd.read_csv(fileIN,sep=sp,header=0,names=nms)            
    
    ex=lon.flatten(); ey=lat.flatten()
    sn=str(df.columns[stID])
    indXY={}
    for ind,s in enumerate(df[sn]):
        if lat.min() < df['lat'][ind] and df['lat'][ind] < lat.max():
            if lon.min() < df['lon'][ind] and df['lon'][ind] < lon.max():
                dd=geo_distance_np(df['lat'][ind],df['lon'][ind],ey,ex)
                indXY[s]=dd.argsort()[0]
                a=np.c_[ex[indXY[s]],ey[indXY[s]],dd[indXY[s]]]
                print s
                print "Model grid:" + str(a[0][0]) + ',' + str(a[0][1])
                print "Point:" + str(df['lon'][ind]) + ',' + str(df['lat'][ind])
                print "Distance " + str(a[0][2])
            else:
                print s 
                print
    return indXY
 
def getXY_p2p_ver2(wlon,wlat,points):
    """
    Gleda udaljenost točke obzirom na x,y koordinate
    
    wlon,wlat su 2D arrayi grida (EMEP, WRF, ...)
    points=[lon,lat] - array od lon,lat točaka    
        
    npr:
    indXY=getXY_p2p(wlon,wlat,points)
          
    """
    
    import numpy as np
    
    plon=points[0];plat=points[1]
    
    ex=wlon.flatten(); ey=wlat.flatten()
    indXY={}
    for ind,ss in enumerate(points[0]):        
        if wlat.min() < plat[ind] and plat[ind] < wlat.max():
            if wlon.min() < plon[ind] and plon[ind] < wlon.max():
                dd=geo_distance_np(plat[ind],plon[ind],ey,ex)
                indXY[ind]=dd.argsort()[0]
                a=np.c_[ex[indXY[ind]],ey[indXY[ind]],dd[indXY[ind]]]
                #print "Model grid:" + str(a[0][0]) + ',' + str(a[0][1])
                #print "Point:" + str(plon[ind]) + ',' + str(plat[ind])
                #print "Distance " + str(a[0][2])
    return indXY   

def setexec(fpath):
    import os, stat;
    sm=os.stat(fpath).st_mode;
    sm = sm | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(fpath,sm);
    
    
def cdl2nc(nc,mdl,cdl_dir,cdl_tmp,init_day):
    """
    Kreiram iz headera (cdl_tmp) nc file
    Korigiram nc,ny i start vrijeme
    Taj file je prazan i spreman za punjenje sa vrijednostima
    """
    
    import numpy as np
    import os
    import subprocess

    cdl_out=os.path.join(cdl_dir,'WRFChem.cdl')    
    dom='01'
    Nx,Ny,Nz,lons,lats,dx,dy=getDimensions(nc)
    cdl=open(cdl_tmp).read()
    cdl=cdl.replace('wrf_name','wrfchem_' + init_day)
    cdl=cdl.replace('no_y_points',str(Ny+1))
    print "Please note: yDIM je +1 " # iz nekog razloga!
    
    cdl=cdl.replace('no_x_points',str(Nx))
    cdl=cdl.replace('no_z_points',str(Nz))
    if os.path.isfile(cdl_out)==True:
        os.remove(cdl_out)    
    open(cdl_out,'w').write(cdl)
    
    anthro_out=os.path.join(cdl_dir,mdl + '_d' + dom + '_EMEP.nc')
            
    if os.path.isfile(anthro_out)==False:
        print "Kreiram : " + anthro_out + "...."
    else:
        print anthro_out + " vec postoji, brisem ga...."
        os.remove(anthro_out)
    setexec(cdl_out)
    subprocess.call(["ncgen","-o",anthro_out,cdl_out])#,shell=True);
    os.remove(cdl_out)   
    return anthro_out    
    
def fillNC(nc,var1,data,var2):
    """
    """
    
    import numpy as np

    for var in nc.variables :
        print var
        nc.variables[var1][:]=data[var2][:]
    return nc
    

def wind_chill(ws,tc):
    """
    ws - brzina u m/s
    tc - temp u °C
    vraća van wind chill u fahrenheitima (wf),
    odnosno u celsiusima (wc)

    npr wind_chill(1,14)    
    """
    ws_mph=ws*2.236936 # m/s u mph
    tf=tc*1.8+32 # °C u F
    wf=0.0817*(3.71*ws_mph**0.5 + 5.81 - 0.25*ws_mph)*(tf-91.4)+91.4
    wc=(wf-32)/1.8 # F u °C 
    return wc

def get_closest_points(lon,lat,slat,slong,np):
    """
    lon, lat - mreza modela
    slat,slong - koordinate tocke
    np - koliko najblizih tocaka da izbaci
    """    

    import numpy as np    
    ex=lon.flatten(); ey=lat.flatten()
    dd=geo_distance_np(slat,slong,ey,ex)
    indXY=dd.argsort()[:4]
    return indXY,dd[:4]  
    

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:22:28 2017

@author: goran
"""

def getRi(zz,Th,u,v):
    """
    Calculating Ri number, Amela (Tellus)
    
    """
    import numpy as np
    
    g=9.81
    Ri=np.zeros((Th.shape[0],Th.shape[1]-1,))
    for tt in range(1,Th.shape[0]):        
        uu=u[tt,:-1]+0.00001
        vv=v[tt,:]+0.00001
        zi=zz[tt]
        Th0=Th[0,:-1]
        Th1=Th[tt,:-1]
        Th_avg=(Th0+Th1)/2.
        a=g*(zi-zz[0])/Th_avg
        b=(Th1-Th0)/(uu**2+vv**2)
        Ri[tt,:]=a/b
        a=uu**2+vv**2
    return Ri
#ri=getRi(z[:,.0],Th,u,v)

def idw4(in_matrix):
    """"
    Based on 
    http://www.gitta.info/ContiSpatVar/en/html/Interpolatio_learningObject2.xhtml

    """        
    a=[];b=[]
    for i1,m in enumerate(in_matrix):
        a.append(m[0]/m[1])
        b.append(1/m[1])
    value=np.sum(a)/np.sum(b)
    return value

def get_crossSection_axes(nc,g0,g1):
    """
    lon,lat koordinate wrf mreze (x,y)
    g0 - pocetna tocka presjeka
    g1 - krajnja tocka presjeka
    npr.
        g0=[10,45]
        g1=[20,49]
    na crossectionu radimo broj tocaka kao cca broj tocaka u x smjeru od g0,g1
    kako je udaljenost najbliza na horizontalnom il vert. pravcu, kada idemo u koso 
    """
    import numpy as np

    
    #nc=openWRF(fileIN)
    lon=getVar(nc,'XLONG')[0]
    lat=getVar(nc,'XLAT')[0]
    hgt=getVar(nc,'HGT')[0]
    Nx,Ny,Nz,lons,lats,dx,dy=getDimensions(nc)

    dist_g=geo_distance(g0[1],g0[0],g1[1],g1[0]) 
    dist_gx=geo_distance(g0[1],g0[0],g0[1],g1[0]) # udaljenost u x smjeru
    dist_gy=geo_distance(g0[1],g0[0],g1[1],g0[0]) # udaljenost u y smjeru
    dist_w=geo_distance(lat[10][0],lon[0][10],lat[10][0],lon[0][9]) #udaljenost izmedju dvije najblize tocke
    a=int(np.round_(dist_gx/dist_w,0)) #hipotenuza
    b=int(np.round_(dist_gy/dist_w,0)) #hipotenuza
    cp=np.sqrt(a**2+b**2)
    cp=int(cp)
    
    print "Udaljenost izmedju dvije odabrane tocke: " + str(dist_g) + " km"
    print "Korak mreze u cca sredini: " + str(dist_w) + " km"
    print "Okvirni broj tocaka prema Pitag. poucku: " + str(cp)
    
    # nas pravac je
    gx=np.linspace(g0[0],g1[0],cp)
    gy=np.linspace(g0[1],g1[1],cp)

    xos=np.zeros((Nz-1,len(gx)))
    for hh in range(0,Nz-1):
        xos[hh,:]=gx
    return gx,gy,xos,lon,lat,hgt


def get_indXY_vert_cross(gx,gy,lon,lat):
	"""
	sluzi za interp_vert_cross
	"""
	indXY=np.zeros((len(gx),4))
	dd=np.zeros((len(gx),4))
	for ind,x in enumerate(gx):    
	    y=gy[ind]
	    #print x,y
	    indXY_0,dd_0=get_closest_points(lon,lat,y,x,4)    
	    indXY[ind]=indXY_0
	    dd[ind]=dd_0
	return indXY,dd


def interp_vert_cross(nc,gx,gy,dt):

    import numpy as np
    from datetime import datetime,timedelta
    
    """
    Interpoliram vrijednosti iz mreze na definirani presjek
    u,v - komponente vjetra
    z - vert koordinate:
        phb_stag=getVar(nc,'PHB')
        phb=0.5*(phb_stag[:,0:-1,:]+phb_stag[:,1::,:])
        z=phb/10000.
    gx - linija tocaka na koje treba interpolirati
    dt - vremenski korak
    
    moze se podesiti da cita samo odredjene trenutke...
    """
    print "Ucitavam varijable ..."    
    ustag=getVar(nc,'U')
    vstag=getVar(nc,'V')
    u=0.5*(ustag[:,:,:,0:-1]+ustag[:,:,:,1::])        # VALJDA?!?!
    v=0.5*(vstag[:,:,0:-1,:]+vstag[:,:,1::,:])
    
    wtimes=getVar(nc,'Times')
    
    lon=getVar(nc,'XLONG')[0]
    lat=getVar(nc,'XLAT')[0]
    
    phb_stag=getVar(nc,'PHB')
    phb=0.5*(phb_stag[:,0:-1,:]+phb_stag[:,1::,:])
    z=phb/10000.
    
    ucomp=np.zeros((u.shape[0]/dt,u.shape[1],len(gx)))
    vcomp=np.zeros((u.shape[0]/dt,u.shape[1],len(gx)))
    zcomp=np.zeros((u.shape[0]/dt,u.shape[1],len(gx)))
    print "sdate " + datetime.strftime(wtimes[0],'%Y-%m-%d %H')
    print "edate " + datetime.strftime(wtimes[-1],'%Y-%m-%d %H')
    #print "...cca 20sek/time_int"

    
    for tt in np.arange(0,u.shape[0],dt):
        tta=tt/dt        
        #print tt,t1

        print "Working on " + datetime.strftime(wtimes[tt],'%Y-%m-%d %H')
        for lev in np.arange(u.shape[1]):
            #print "Level " + str(lev)
            for ind,x in enumerate(gx):    
                y=gy[ind]
                #print x,y
                indXY,dd=get_closest_points(lon,lat,y,x,4)    
                uu=np.zeros((4,2)) # array [value, distance] potrebno za interpolaciju
                vv=np.zeros((4,2)) # array [value, distance]
                zz=np.zeros((4,2)) # array [value, distance]
                for i1,ii in enumerate(indXY):
                    #print "\t " , lon.flatten()[ii],lat.flatten()[ii]
                    uu[i1,0]=u[tt,lev,:].flatten()[ii]
                    uu[i1,1]=dd[i1]
                    vv[i1,0]=v[tt,lev,:].flatten()[ii]
                    vv[i1,1]=dd[i1]
                    zz[i1,0]=z[tt,lev,:].flatten()[ii]
                    zz[i1,1]=dd[i1]                
                ucomp[tta,lev,ind]=idw4(uu)
                vcomp[tta,lev,ind]=idw4(vv)
                zcomp[tta,lev,ind]=idw4(zz)    
    ucomp=ucomp[:,1:,:]
    vcomp=vcomp[:,1:,:]
    zcomp=zcomp[:,1:,:]
    return ucomp,vcomp,zcomp,wtimes[::dt]
    
def interp_vert_cross_v2(nc,gx,gy,dt):

    import numpy as np
    from datetime import datetime,timedelta
    
    """
    Interpoliram vrijednosti iz mreze na definirani presjek
    u,v - komponente vjetra
    z - vert koordinate:
        phb_stag=getVar(nc,'PHB')
        phb=0.5*(phb_stag[:,0:-1,:]+phb_stag[:,1::,:])
        z=phb/10000.
    gx - linija tocaka na koje treba interpolirati
    dt - vremenski korak
    
    moze se podesiti da cita samo odredjene trenutke...
    v2 - Ubrzano.. (27.02.2017)
    
    """
    print "Ucitavam varijable ..."    
    ustag=getVar(nc,'U')
    vstag=getVar(nc,'V')
    u=0.5*(ustag[:,:,:,0:-1]+ustag[:,:,:,1::])        # VALJDA?!?!
    v=0.5*(vstag[:,:,0:-1,:]+vstag[:,:,1::,:])
    
    wtimes=getVar(nc,'Times')
    
    lon=getVar(nc,'XLONG')[0]
    lat=getVar(nc,'XLAT')[0]
    
    phb_stag=getVar(nc,'PHB')
    phb=0.5*(phb_stag[:,0:-1,:]+phb_stag[:,1::,:])
    z=phb/10000.
    
    ucomp=np.zeros((u.shape[0]/dt,u.shape[1],len(gx)))
    vcomp=np.zeros((u.shape[0]/dt,u.shape[1],len(gx)))
    zcomp=np.zeros((u.shape[0]/dt,u.shape[1],len(gx)))
    print "sdate " + datetime.strftime(wtimes[0],'%Y-%m-%d %H')
    print "edate " + datetime.strftime(wtimes[-1],'%Y-%m-%d %H')
    #print "...cca 20sek/time_int"

    indXY,dd=get_indXY_vert_cross(gx,gy,lon,lat)
    for tt in np.arange(0,u.shape[0],dt):
        tta=tt/dt        
        #print tt,t1

        print "Working on " + datetime.strftime(wtimes[tt],'%Y-%m-%d %H')
        for lev in np.arange(u.shape[1]):
            #print "Level " + str(lev)
            for ind,x in enumerate(gx):    
                y=gy[ind]
                #print x,y
                #indXY,dd=get_closest_points(lon,lat,y,x,4)    
                uu=np.zeros((4,2)) # array [value, distance] potrebno za interpolaciju
                vv=np.zeros((4,2)) # array [value, distance]
                zz=np.zeros((4,2)) # array [value, distance]
                for i1,ii in enumerate(indXY[ind]):
                    #print "\t " , lon.flatten()[ii],lat.flatten()[ii]
                    uu[i1,0]=u[tt,lev,:].flatten()[ii]
                    uu[i1,1]=dd[ind][i1]
                    vv[i1,0]=v[tt,lev,:].flatten()[ii]
                    vv[i1,1]=dd[ind][i1]
                    zz[i1,0]=z[tt,lev,:].flatten()[ii]
                    zz[i1,1]=dd[ind][i1]                
                ucomp[tta,lev,ind]=idw4(uu)
                vcomp[tta,lev,ind]=idw4(vv)
                zcomp[tta,lev,ind]=idw4(zz)    
    ucomp=ucomp[:,1:,:]
    vcomp=vcomp[:,1:,:]
    zcomp=zcomp[:,1:,:]
    return ucomp,vcomp,zcomp,wtimes[::dt]
    
def interp_vert_cross_getVar(nc,gx,gy,dt,param):

    import numpy as np
    from datetime import datetime,timedelta
    
    """
    isto kao i interp_vert_cross - samo ovdje cupam bilo koju varijablu van    
    Interpoliram vrijednosti iz mreze na definirani presjek
    u,v - komponente vjetra
    z - vert koordinate:
        phb_stag=getVar(nc,'PHB')
        phb=0.5*(phb_stag[:,0:-1,:]+phb_stag[:,1::,:])
        z=phb/10000.
    gx - linija tocaka na koje treba interpolirati
    dt - vremenski korak
    
    moze se podesiti da cita samo odredjene trenutke...
    
    28.02 - podeseno da cita 3D ili 2D varijable
    """

    print "Ucitavam varijable ..."    
    varIN=getVar(nc,param)
    if np.size(varIN)>10:
        print "Getting -- " + param
        wtimes=getVar(nc,'Times')
        
        lon=getVar(nc,'XLONG')[0]
        lat=getVar(nc,'XLAT')[0]
        
        phb_stag=getVar(nc,'PHB')
        phb=0.5*(phb_stag[:,0:-1,:]+phb_stag[:,1::,:])
        z=phb/10000.
        
        if len(varIN.shape)==4: # citam 3D
            var=np.zeros((varIN.shape[0]/dt,varIN.shape[1],len(gx)))
            indXY,dd=get_indXY_vert_cross(gx,gy,lon,lat)
            for tt in np.arange(0,var.shape[0],dt):
                tta=tt/dt        
                #print tt,t1    
                print "Working on " + datetime.strftime(wtimes[tt],'%Y-%m-%d %H')
                for lev in np.arange(var.shape[1]):
                    #print "Level " + str(lev)
                    for ind,x in enumerate(gx):    
                        y=gy[ind]
                        #print x,y
    #                    indXY,dd=get_closest_points(lon,lat,y,x,4)    
                        vv=np.zeros((4,2)) # array [value, distance] potrebno za interpolaciju
                        for i1,ii in enumerate(indXY[ind]):
                            vv[i1,0]=varIN[tt,lev,:].flatten()[ii]
                            vv[i1,1]=dd[ind][i1]
                        var[tta,lev,ind]=idw4(vv)
            var=var[:,1:,:]
        elif len(varIN.shape)==3:  # citam 2D
            var=np.zeros((varIN.shape[0]/dt,len(gx)))
            indXY,dd=get_indXY_vert_cross(gx,gy,lon,lat)
            for tt in np.arange(0,var.shape[0],dt):
                tta=tt/dt        
                #print tt,t1    
                print "Working on " + datetime.strftime(wtimes[tt],'%Y-%m-%d %H')
                for ind,x in enumerate(gx):    
                    y=gy[ind]
                    #print x,y
#                    indXY,dd=get_closest_points(lon,lat,y,x,4)    
                    vv=np.zeros((4,2)) # array [value, distance] potrebno za interpolaciju
                    for i1,ii in enumerate(indXY[ind]):
                        vv[i1,0]=varIN[tt,:].flatten()[ii]
                        vv[i1,1]=dd[ind][i1]
                    var[tta,ind]=idw4(vv)
            var=var[:,:]           
        return var

