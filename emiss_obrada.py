# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:18:55 2013

@author: wrfuser
"""

import numpy as np
import pandas as p
from pandas import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime,timedelta
from jkUtils import PandasLoadFromPG

eDIR='/mnt/Titan/projekti/EUREKA/emiss/';
pol=['co','no','so2','pm']
seg=['h6101','k12']

data={}
for i in range(len(seg)):
    for j in range(len(pol)):
        plt.close('all')
        fileIN=seg[i] + '_' + pol[j]
        inFILE=eDIR + fileIN + '.txt'
        data[fileIN]=p.read_table(inFILE)        
        df_time=np.array([datetime.strptime("".join(t),'%d-%m-%Y-%H:%M') for t in data[fileIN].vrijeme])
        sort_time=np.sort(df_time)
        sort_time_ind=np.argsort(df_time)
        plt.plot(sort_time,data[fileIN].konc[sort_time_ind]) 
        plt.gcf().autofmt_xdate()
        ha=plt.axes()
        loc = mdates.DayLocator(bymonthday=1) # stavljam tikove prvi dan u mj
        fmt = mdates.DateFormatter('%d.%m.%Y')
        ha.xaxis.set_major_locator(loc) #plt.gca().xaxis.set_major_locator(loc)
        ha.xaxis.set_major_formatter(fmt)
        plt.title(seg[i] + ' ' + pol[j],{'fontsize':'large'})
        plt.xlabel('Dani');plt.ylabel('Koncentracija')
        plt.savefig('sisak1_2010_' + fileIN + '_python.png')
        plt.close()    
#
# uzimam iz baze
for i in range(len(seg)):
    for j in range(len(pol)):   
        FILEin=str(seg[i]) + '_' + str(pol[j])
        print FILEin
        comm= """
            SELECT 
            datum_mjerenja ||''|| substring(vrijeme_mjerenja,1,5) as vrijeme, 
            replace(vrijednost_normalizirana,',','')::float as konc 
            FROM emiss.
            """
        df=PandasLoadFromPG('titan','zrak','postgres','anscd11',comm + FILEin)        
        f=open('./OUT/INA_RNS_' + FILEin + '.txt','w')        
        f.write('Time \t Concentration \n')
        for k in range(len(df.vrijeme)):
            f.write(df.vrijeme[k] + '\t')
            f.write(df.konc[k] + '\n')
        f.close()
        #np.savetxt('./OUT/' + FILEin + '.txt',np.array([df.vrijeme,df.konc]).T)
# importam TE podatke
df=pandas.read_csv('./inData/2009Blok1kotaoA.CSV')