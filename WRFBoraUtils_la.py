# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:03:28 2014

@author: goran gasparac

"""


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
    

