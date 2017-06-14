# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:17:01 2013

@author: wrfuser
"""

import numpy as np
import os
from datetime import datetime,timedelta
from netCDF4 import Dataset
from bunch import Bunch
import commands

wdir='/home/projekti/WRF/Wout2Wbad/pyTest/' # working directory
#wrfIN='/home/projekti/WRF/OPERATIVA/arhiva_master/'
#oznaka='operativa' #oznaka na txt fileu
wrfIN='/disk2/WRF/gtest_operativa/wrfout/GG3_18km_4domains/'
oznaka='GG3' #oznaka na txt fileu
#domain=str(2)

for i in [2,3,4]: # domene:

    comm='ls ' + wrfIN + 'wrfout_d0' + str(i) + '*'
    out=commands.getoutput(comm)
    indate=datetime.strptime(out[-1-18:],'%Y-%m-%d_%H:%M:%S')
    domain=str(i)    
    wrfout= 'wrfout_d0' + domain + '_' + str(indate.strftime('%Y-%m-%d_%H:%M:%S'))
    wrfin=os.path.join(wrfIN,wrfout)
    print 'Vucem podatke iz ' + wrfin
    os.chdir(wdir)
    
    # linkam wrfout
    wrfdest=os.path.join(wdir,wrfout)
    if not os.path.lexists(wrfdest):
        os.symlink(wrfin,wrfdest) # linkam wrofut u wdir
    
    # ucitavam nc file
    nc=Dataset(wrfdest,'r')
    wrfVar=Bunch()
    tmp=Bunch()
    tmp.times=nc.variables['Times'][:]
    wrfVar.times=np.array([datetime.strptime("".join(t),'%Y-%m-%d_%H:%M:%S') for t in tmp.times])
    lon=nc.variables['XLONG']
    lat=nc.variables['XLAT']
    landmask=nc.variables['LANDMASK']
    
    #plt.scatter(lon[0,:,:].flatten(),lat[0,:,:].flatten())
    
    x=lon[0,:,:].flatten()
    y=lat[0,:,:].flatten()
    lm=landmask[0,:,:].flatten()    
    f=open('./DOMAIN_' + oznaka + '_' + domain + '_LatLon.txt', 'w')
    f.write('"WRF_lon",')
    f.write('"WRF_lat",')
    f.write('"WRF_landmask"\n')
    for i in range(len(lon[0,:,:].flatten())):
        f.write(str(x[i])+',')
        f.write(str(y[i])+',')
        f.write(str(lm[i])+'\n')
    f.close()
    #
    # ili ovak    
    #np.savetxt('proba.txt',np.array([x,y,lm]).T)

commands.getoutput('rm ' + wdir + 'wrfout*') # obrisem sve iz wdir