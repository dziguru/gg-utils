# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:19:14 2013

@author: wrfuser

"""

import numpy as np
import os
from datetime import datetime,timedelta
from netCDF4 import Dataset
from bunch import Bunch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from subprocess import call,Popen,PIPE,STDOUT

wdir='/home/projekti/WRF/Wout2Wbad/pyTest/'
wrfIN='/home/projekti/WRF/OPERATIVA/arhiva_master/'
domain=str(2)
indate=datetime.now() + timedelta(-1)
#print(indate.strftime('%Y-%m-%d'))

# spajanje stringa ovako
#seq=('wrfout_d0', domain,'_',str(indate.strftime('%Y-%m-%d')),'_','12:00:00')
#wrfout=''.join(seq) # ovdje stavljam razdjelnik
#print(wrfout)

# ili ovako
wrfout= 'wrfout_d0' + domain + '_' + str(indate.strftime('%Y-%m-%d')) + '_' + '12:00:00'
wrfin=os.path.join(wrfIN,wrfout)
#print(wrfin)
os.chdir(wdir)

# linkam wrfout
wrfdest=os.path.join(wdir,wrfout)
#print(wrfdest)
if not os.path.lexists(wrfdest):
    os.symlink(wrfin,wrfdest)

# ucitavam nc file
nc=Dataset(wrfdest,'r')
wrfVar=Bunch()
tmp=Bunch()
tmp.times=nc.variables['Times'][:]
wrfVar.times=np.array([datetime.strptime("".join(t),'%Y-%m-%d_%H:%M:%S') for t in tmp.times])
lon=nc.variables['XLONG']
lat=nc.variables['XLAT']
# tražim najbližu točku
Sisak=Bunch()
Sisak.lat=45.489502
Sisak.lon=16.379242
x_os=lon[1,:,:].flatten()
y_os=lat[1,:,:].flatten()
dist=(x_os-Sisak.lon)**2+(y_os-Sisak.lat)**2
ind=dist.argmin()


#x_os.flatten()[dist.argmin()]
#y_os.flatten()[dist.argmin()]
# sad zapravo [dist.argmin()] daje poziciju Siska u matrici

# met parametri:
wrfVar.t2=nc.variables['T2']
wrfVar.u10=nc.variables['U10']
wrfVar.v10=nc.variables['V10']

# trpanje vrijednosti iz nc-a, na više načina:
"""
t2=np.array([mreza.flatten()[ind] for mreza in wrfVar.t2[:]])

t2=wrfVar.t2[:].reshape((169,111*126))[:,ind]

t2=[]
ws=[]
for i in range(len(wrfVar.t2)):
    t2.append(wrfVar.t2[i,:,:].flatten()[dist.argmin()])
    #ws.append(np.sqrt(wrfVar.u10[i,:,:].flatten()[dist.argmin()]**2+wrfVar.v10[i,:,:].flatten()[dist.argmin()]**2))
"""
   
t2=wrfVar.t2[:].reshape((169,111*126))[:,ind]
ws=np.sqrt(wrfVar.u10[:].reshape((169,111*126))[:,ind]**2+wrfVar.u10[:].reshape((169,111*126))[:,ind]**2)

#plt.plot(wrfVar.times[:],wrfVar.t2[:,40,40]-273.15)
plt.plot(wrfVar.times[:],t2-273.15)
plt.gcf().autofmt_xdate()
loc = mdates.DayLocator()
fmt = mdates.DateFormatter('%Y%m%d')
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(fmt)

plt.plot(wrfVar.times[:],ws)
plt.gcf().autofmt_xdate()
loc = mdates.DayLocator()
fmt = mdates.DateFormatter('%d/%m')
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(fmt)
`
"""
for i in range(len(wrfVar.t2)):
    plt.clf()
    plt.contourf(wrfVar.t2[i])
    plt.show()
    raw_input('Press enter.')
"""