# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:03:28 2014

@author: goran gasparac

"""

def makeNamelist_WRF(sim_dur,ROOTdir,DIRpre,broj_domena,init_hour,sdate):
    """
    Sredjujem namelist.input i namelist.wps,
    mjenjam u templateima tekst sa odgovarajucim datumima
    """
    
    import os
    from datetime import datetime,timedelta
    
#    sim_dur=7
#    ROOTdir='/home/projekti/WRF/OPERATIVA'
#    DIRpre='/home/projekti/WRF/OPERATIVA/1_pre_v2'
#    broj_domena=3
#    init_hour=12
#    sdate='20150910'
    
    namin_IN=os.path.join(DIRpre,'namelist.input_template_18-6-2_0703')
    namin_OUT=os.path.join(DIRpre,'namelist.input')
    namwps_IN=os.path.join(DIRpre,'namelist.wps_template_18-6-2_0609')
    namwps_OUT=os.path.join(DIRpre,'namelist.wps')
    
    sdate_num=datetime.strptime(sdate,'%Y%m%d')
    edate_num=sdate_num+timedelta(sim_dur)
    y0=sdate_num.year;y1=edate_num.year;
    m0=sdate_num.month;m1=edate_num.month;
    d0=sdate_num.day;d1=edate_num.day
    
    sdate_str=datetime.strftime(sdate_num,'%Y-%m-%d_12:00:00')
    edate_str=datetime.strftime(edate_num,'%Y-%m-%d_12:00:00')

   
    fIN=open(namwps_IN, "rt")
    tekst = fIN.read()
    tekst1 = tekst.replace('kreni', sdate_str)
    tekst2= tekst1.replace('stani', edate_str)
    tekst3= tekst2.replace('broj_domena', str(broj_domena))
    fIN.close()
    fIN=open(namwps_OUT, "w")
    fIN.write(tekst3)
    fIN.close()
    
    fIN=open(namin_IN, "rt")
    tekst = fIN.read()
    tekst1 = tekst.replace('starty', str(y0))
    tekst2= tekst1.replace('startm', str(m0))
    tekst3 = tekst2.replace('startd', str(d0))
    tekst4= tekst3.replace('starthr', str(12))
    tekst5 = tekst4.replace('endy', str(y1))
    tekst6= tekst5.replace('endm', str(m1))
    tekst7 = tekst6.replace('endd',str(d1))
    tekst8= tekst7.replace('endhr', str(12))
    tekst9= tekst8.replace('broj_domena', str(broj_domena))
    tekst10= tekst9.replace('duration', str(sim_dur))
    fIN.close()
    fIN=open(namin_OUT, "w")
    fIN.write(tekst10)
    fIN.close()

def wrfout2arwpost_v3R(Wout2Wbad,domain):
    """
    Nnapadam WRFout s ARWpostom i gotovo file šaljem u ./ncs direktorij
    služi kasnije punjenje wrfout_*.nc fileova (za Bademu)

    Wout2Wbad = '/home/projekti/WRF/Wout2Wbad'
    domain=2
        
    """
    import os 
    from datetime import datetime
    import sys
    sys.path.append("/home/goran/Dropbox/WORK/python/__PYscripts/")    
    
    ARWobrada = os.path.join(Wout2Wbad, 'ARWobrada')
    ncs = os.path.join(Wout2Wbad, 'ncs')
    os.chdir(ARWobrada)
    
    namin_IN=os.path.join(ARWobrada,'namelist.ARWpost_template_d0' + str(domain))
    namin_OUT=os.path.join(ARWobrada,'namelist.ARWpost')
    
    for wf in os.listdir(ncs):
        if 'wrfout_d0' + str(domain) in wf:
            wrfIN=os.path.join(ncs,wf)
            wfile=wf
            
    nc=openWRF(wrfIN)
    wtimes=getVar(nc,'Times')
    
    sdate=datetime.strftime(wtimes[0],'%Y-%m-%d_%H:00:00')
    edate=datetime.strftime(wtimes[-1],'%Y-%m-%d_%H:00:00')
    dt=wtimes[1]-wtimes[0]
    dt_sec=dt.seconds
    
    fIN=open(namin_IN, "rt")
    tekst = fIN.read()
    tekst1 = tekst.replace('YYY1-M1-D1_HH1:00:00', sdate)
    tekst2= tekst1.replace('YYY2-M2_D2_HH2:00:00', edate)
    tekst3= tekst2.replace('int_sec', str(dt_sec))
    tekst4= tekst3.replace('wrfout_d0' + str(domain) + '_name', wfile)
    fIN.close()
    fIN=open(namin_OUT, "w")
    fIN.write(tekst4)
    fIN.close()

    
    
def getGrib_GFS(ROOTdir,DIRpre,init_hour,sdate):
    """
    Kopiram GFS input u DIRpre i untaram ih...
    """

    import os
    from datetime import datetime,timedelta
    import shutil as su
    import tarfile as tf
    
    lck_file=os.path.join(ROOTdir,'lock','run_' + sdate + '_12')
    touch(lck_file)

    # brisem lock file od prije 7 dana (npr)
    rdate=datetime.strftime(datetime.strptime(sdate,'%Y%m%d')-timedelta(5),'%Y%m%d')    
    remove_lock=os.path.join(ROOTdir,'lock','run_' + rdate + '_12')
    if os.path.exists(remove_lock):
        print remove_lock
        print "Brisem lock file"
        os.remove(remove_lock)

    sdate_num = datetime.strptime(sdate,'%Y%m%d')
    gfs_date = datetime.strftime(sdate_num+timedelta(hours=init_hour),'%Y%m%d%H')
    gfs_file = gfs_date + '_025_grib2.tgz'
    GFSin = os.path.join('/home/projekti/GFS/archive_grib2/',gfs_file)
    #print GFSin

    su.copy(GFSin,DIRpre)
    os.chdir(DIRpre)    
    tar=tf.open(gfs_file)
    tar.extractall()
    tar.close()
    os.remove(gfs_file)
    

def touch(fname):
    import os
    
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()
        

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
        print 'There is no variable '+ fileIN + ' in nc file!'
        return None    

def getXY(nc,lon,lat,station_list,slat,slong):
    import numpy as np
    indX={};indY={} # koordinate u WRF mreži    
    for s in station_list:
        indX[s]=np.argmin((abs(lon[1,:]-slong[s])))
        indY[s]=np.argmin((abs(lat[:,1]-slat[s])))
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
            if lon.min() < df['lon'][ind] and df['lon'][ind] < lat.max():
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

