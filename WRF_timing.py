# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:43:23 2013

@author: wrfuser
"""

import numpy as np
from datetime import datetime,timedelta
from bunch import Bunch
import commands

runDIR='/mnt/PandoraDisk2/RITEH_run/2_run_wrf3.3/' #'/disk2/WRF/gtest_operativa/GG3_18km_4domains/'
namelistWPSdir='/mnt/PandoraDisk2/RITEH_run/1_pre_wrf3.3/' #'/disk2/WRF/gtest_operativa/GG3_18km_4domains/'
rsl=runDIR + 'rsl.error.0000'
wps=namelistWPSdir + 'namelist.wps'

sim=Bunch()
vrijeme=Bunch()
dt=Bunch()

# definiram sve vremenske varijable
f=open(rsl,'r')
line=f.readlines()[-1]
f.close()
sim.cdate=datetime.strptime(line[22:41],'%Y-%m-%d_%H:%M:%S')
#print sim.cdate

comm="ls -la " + runDIR + 'wrfi* --time-style="+%Y-%m-%d %H:%M" ' + " | awk '{print $6, $7, $9}'" 
out=commands.getoutput(comm)
vrijeme.krenuo=datetime.strptime(out[0:16],'%Y-%m-%d %H:%M')
#print vrijeme.krenuo 
vrijeme.sad=datetime.now()
#print vrijeme.sad
f=open(wps,'r')
line=f.readlines()[3]
sim.sdate=datetime.strptime(line[15:34],'%Y-%m-%d_%H:%M:%S')
f.close()
f=open(wps,'r')
line=f.readlines()[4]
sim.edate=datetime.strptime(line[15:34],'%Y-%m-%d_%H:%M:%S')
f.close()
#print sim.edate

# izracun koliko radim
vrijeme.proslo=vrijeme.sad-vrijeme.krenuo; 
sim.proslo=sim.cdate-sim.sdate; 
sim.duration=sim.edate-sim.sdate; 
dt.h=(vrijeme.proslo.total_seconds()/sim.proslo.total_seconds())*3600; 
dt.total_sim=dt.h*sim.duration.total_seconds()
#print dt.h/60/60
vrijeme.zavrsavam=vrijeme.krenuo+timedelta(hours=sim.duration.total_seconds()/60/60*dt.h/60/60)

# ispis i slanje na mail
WRFstatINF='WRF_timing.txt'
f=open(WRFstatINF,'w')
f.write('Statictic info za WRF run:\n')
f.write('\t\t\t ' + runDIR + ' \n')
f.write('************************************************************************\n')
f.write('\t\t Krenuo sam raditi u: ' + vrijeme.krenuo.strftime('%H:%M (%d.%m.%Y)') + '\n')
f.write('\t\t Za 1h simulacije mi treba: ' + str(round(dt.h/60,2)) + ' minuta \n')
f.write('\t\t Ova simulacija ima: ' + str(round(sim.duration.total_seconds()/60/60,0)) + ' sati \n')
f.write('\t\t Prema tome radim cca: ' + str(round(sim.duration.total_seconds()/60/60*dt.h/60/60,2)) + ' sati \n')
f.write('\t\t Okvirno zavrsavam u: ' + vrijeme.zavrsavam.strftime('%H:%M (%d.%m.%Y)') + ' sati \n')
f.write('************************************************************************\n')
f.close()
# šaljem mail
mail_comm='cat ' + WRFstatINF + ' | mail -s "WRF timing info" "gorangas@gmail.com"'
out=commands.getoutput(mail_comm)

print ('************************************************************************\n')
print ('\t\t Krenuo sam raditi u: ' + vrijeme.krenuo.strftime('%H:%M (%d.%m.%Y)') + '\n')
print ('\t\t Za 1h simulacije mi treba: ' + str(round(dt.h/60,2)) + ' minuta \n')
print ('\t\t Radim cca: ' + str(round(sim.duration.total_seconds()/60/60*dt.h/60/60,2)) + ' sati \n')
print ('\t\t Okvirno zavrsavam u: ' + vrijeme.zavrsavam.strftime('%H:%M (%d.%m.%Y)') + ' sati \n')
print ('************************************************************************\n')

comm='rm -f ' + WRFstatINF
out=commands.getoutput(comm)
