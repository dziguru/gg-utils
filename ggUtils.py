# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:46:22 2013

@author: wrfuser
"""
 
def grbgText2nan_TE(table_name): # garbage text to nan
    """
    Ucitam table /dakle mora biti dataframe u igri/,
    i zamjenim NULL, Greška, Iskljuc, Održ/Gre.....
    u NaN
    """
    import numpy as np
    table_name[table_name=='NULL']=np.nan
    table_name[table_name==u'Greška']=np.nan
    table_name[table_name=='Iskljuc']=np.nan
    table_name[table_name==u'Održ/Gre']=np.nan
    return table_name
        
def plt(xx,yy,legenda=None,color_str=None,x_label=None,y_label=None,xtick_loc=None,time_fmt=None,xlim=None,ylim=None,tick_position=None,**kwargs):
    """ 
    Za crtanje nečega što ima na x osi vrijeme

    AKO ZADAJEŠ xtick_loc :: defaultni time_fmt='%d.%m.%Y.',           
    
    npr:    
    ggUtils.plt(x,y,color_str='b',y_label='$Temperatura \ ^{\circ}C$',xtick_loc='day',time_fmt='%m.%d.',xlim=[dates[0],dates[-1]],tick_position='left')
    
    INFO:
    - ako zadajem boju kao string ('b','r',...), stavljam color_str='b'
    - ako koristim RGB boje u tuple-u, onda koristi npr. color=[1,1,1]
    
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    if color_str is None:
        gfig=plt.plot(xx,yy,**kwargs)
    else:
        gfig=plt.plot(xx,yy,color_str,**kwargs)
    gfig=plt.grid()
    gfig=plt.tick_params(axis='both',labelsize=8)        
    gfig=plt.gca().xaxis.grid(True,which='major')
    gfig=plt.gca().yaxis.grid(True,which='major')
    if tick_position==None:
        gfig=plt.gca().yaxis.set_label_position('left')
    else:
        gfig=plt.gca().yaxis.set_label_position(tick_position)        
    if xlim is not None:
        gfig=plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        gfig=plt.xlim(ylim[0],ylim[1])
    if time_fmt==None:
        time_fmt='%d.%m.%Y.'
    if legenda is not None:
        gfig=plt.legend((legenda,),prop={'size':8},loc=0)  
    if x_label is not None:
        gfig=plt.xlabel(x_label,{'fontsize':'small'})           
    if y_label is not None:    
        gfig=plt.ylabel(y_label,{'fontsize':'small'})         
    if xtick_loc=='day':
        loc = mdates.DayLocator()
    elif xtick_loc=='hour':
        loc=mdates.HourLocator()
    elif xtick_loc=='week':
        loc=mdates.WeekdayLocator(byweekday=1)
    elif xtick_loc=='month':
        loc = mdates.MonthLocator()
    elif xtick_loc=='year':
        loc = mdates.YearLocator()      
    elif xtick_loc==None:
        return gfig
    fmt = mdates.DateFormatter(time_fmt)
    gfig=plt.gca().xaxis.set_major_locator(loc)     
    gfig=plt.gca().xaxis.set_major_formatter(fmt)  
    gfig=plt.tick_params(axis='both',labelsize=14)         
    return gfig

#def gpcolor(xx,yy,zz):
#    import matplotlib.pyplot as plt
#    import matplotlib.dates as mdates
#    gfig=plt.plot(xx,yy,zz)
#    return gfig    

def draw_arrows(ax, x, y, u, v, width, scale, units='x', color='#494949'):        
    ax.quiver(x, y, u, v, units=units, width=width, scale=scale, pivot='middle', 
              headlength=2.7, headwidth=2.3, headaxislength=2.7, color=color, zorder=5)
#
#def get_coastline_seg(fname):
#    from bunch import Bunch
#    from netCDF4 import Dataset
#    import numpy as np
#    coast_seg = Bunch()
#    c = Dataset(fname)    
#    for l in ['adria', 'kopno', 'more']:
#        lat = c.variables["lat_" + l][:]
#        lon = c.variables["lon_" + l][:]
#        ind = np.argwhere(lon==-999).flatten()
#        segments = [(v+1,ind[i+1]) for i, v in enumerate(ind[:-1])]
#        xp, yp = [], []
#        for s in segments:
#            xp.append(lon[slice(*s)])
#            yp.append(lat[slice(*s)])
#        setattr(coast_seg, l, (xp, yp))
#    return coast_seg
#  
#def draw_coastline(coast_seg, wks_name):
#    import Ngl        
#    wks = open_wks(wks_name, tip='png', cmap="BlAqGrYeOrRevi200" )
#    plres = Ngl.Resources()
#    plres.gsLineThicknessF  = 2.0
#    plres.gsLineColor       = "black"
#    xp, yp = coast_seg.adria
#    for i in range(len(xp)):
#        Ngl.add_polyline(wks, xp[i], yp[i], plres)
#    xp, yp = coast_seg.kopno
#    for i in range(len(xp)):
#        Ngl.add_polyline(wks, xp[i], yp[i], plres)
#        
#def open_wks(name, tip, cmap="precip3_16lev"): # Open a PostScript workstation
#    import Ngl      
#    wkres = Ngl.Resources()
#    wkres.wkColorMap = cmap
#    wkres.wkWidth = 650
#    wkres.wkHeight = 650
#    wks = Ngl.open_wks(tip, name, wkres)
#    return wks
              
              
def scatter_plot(o,p,title=None,outfile=None,xlabel_txt=None,ylabel_txt=None): #o=observed, p=predicted
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    pm=(p-p.mean())
    om=(o-o.mean())
    R=np.sqrt(sum(pm*om)**2/(sum(om**2)*sum(pm**2)))
    ind=~np.isnan(o) & ~np.isnan(p) #izbacuje null vrijednosti (missinge) 
    o=o[ind]
    p=p[ind]
    fit=np.polyfit(o,p,1)
    fitf=np.poly1d(fit)
    mx=o.max(); my=p.max()
    minx=o.min(); miny=p.min()
    (mx,my)=(max(mx,my),)*2  #Ovo uskla\u0111uje maximume po x-u i y-onu
    (minx,miny)=(min(minx,miny),)*2
    plt.plot(o,p,'.'); plt.gca().set_aspect('equal'); 
    plt.gca().set_xlim(minx+0.5,mx+0.5)
    plt.gca().set_ylim(miny+0.5,my+0.5)
    plt.plot([minx,mx],fitf([minx,mx]),'r')
    plt.ylabel(ylabel_txt,{'fontsize':'xx-small'})
    plt.xlabel(xlabel_txt,{'fontsize':'xx-small'})
    plt.title(title,{'fontsize':'xx-small'})
    s=r'$Y=%.2fX+%.2f$'%tuple(fit)+'\n'+'$R=%.2f$'%R+'\n'+'$N=%d$'%len(p)
    ann_ax=plt.gcf().add_subplot(111)
    ann_ax.annotate(s, (0.05, 0.9), xycoords="axes fraction", va="center", ha="left",fontsize='xx-small',
                    bbox=dict(boxstyle="round, pad=1", fc="w")) #plt.suptitle(s,x=0.15,y=0.85,horizontalalignment='left')
    #plt.savefig(outfile,dpi=400,bbox_inches='tight')         

def baza2df(station,param):
    """
    Cupam podatke iz baze npr:     
    baza2df('Sisak%1','NO2')
    stations: Sisak%1, Kutina%1,Osijek%1
    ... trebalo bi neki error bacit npr. ako nema te postaje il tog param..
    """
    
    import jkUtils 
   
    comm1="""
        select 
        	station_id,
        	name
        from harvest.station
        where name like '"""
    comm2="""' and source NOT LIKE 'azo%'"""
    stationIN=jkUtils.PandasLoadFromPG('titan','monitoring','postgres','anscd11',comm1+station+comm2)
    
    comm1="""
        select 
        	param_id,
        	param_code
        from harvest.param
        where param_code like '"""
    comm2="""'"""
    paramIN=jkUtils.PandasLoadFromPG('titan','monitoring','postgres','anscd11',comm1+param+comm2)
    
    comm1=r"""
        select
        	d.time_id,
        	param_code,
        	d.value
        from
        (
        	select 
        		time_id,
        		value,
        		param_id
        	from harvest.data_hour
        		where  param_id = """
    comm2=r"""
                and station_id= """
    comm3="""                
                and time_id > '2011-11-01'
                and time_id < '2011-11-20'
          
        ) d
        		left join harvest.param p using (param_id)
          order by time_id
          """
    df=jkUtils.PandasLoadFromPG('titan','monitoring','postgres','anscd11',comm1+paramIN.param_id[0]+comm2+stationIN.station_id[0]+comm3)
    print "PGADMIN3:: " + stationIN.name[0] + ', ' + paramIN.param_code[0] 
    return df

def mkdir(path):
    import os    
    if os.path.exists(path)==False:
        os.makedirs(path)
        

def setXtick(ax,xtick_loc,time_fmt,interval,**kwargs):
    """
    Ulazni parametri:

    ax=plt.gca()

    xtick_loc - 'day','month',...

    int - 1,2,3,...
    """
    import matplotlib.dates as mdates    
        
    if xtick_loc=='day':
        loc = mdates.DayLocator()
    elif xtick_loc=='hour':
        loc=mdates.HourLocator(interval=interval)
    elif xtick_loc=='week':
        loc=mdates.WeekdayLocator(byweekday=1)
    elif xtick_loc=='month':
        loc = mdates.MonthLocator(interval=interval)
    elif xtick_loc=='year':
        loc = mdates.YearLocator(interval=interval)      
        
    fmt = mdates.DateFormatter(time_fmt)
    ax.xaxis.set_major_locator(loc)     
    ax.xaxis.set_major_formatter(fmt)      
    return ax
    
        