# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:27:51 2013

@author: wrfuser

PRIMJERI!
"""

os.path.join
datetime.now()
str(indate)
indate.strftime('%Y-%m')
datetime.strptime('2013-04-10','%Y-%m-%d')
from netCDF4 import num2date
from datetime import timedelta
indate + timedelta(2)
indate + timedelta(2,hour=-24)
indate + timedelta(2,hours=-24)
timedelta(2,hours=-24)
timedelta(2.5)

bashCommand = "ls -la ../"
process = Popen(bashCommand.split(), stdout=PIPE)
output = process.communicate()[0]
print(output)
wrfVar.t2[:]
wrfVar.t2[:].shape
wrfVar.t2[:,0,0]
wrfVar.t2[:,0,0].shape
plt.plot(wrfVar.times,wrfVar.t2[:,0,0].shape,'.')
wrfVar.t2[:,0,0].shape
wrfVar.times
wrfVar.times.shape
wrfVar.t2[:,0,0].shape
plt.plot(wrfVar.times,wrfVar.t2[:,0,0],'.')
plt.plot(wrfVar.times,wrfVar.t2[:,0,0],'.',rot=30)