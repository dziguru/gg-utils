# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:30:13 2015

@author: wrfuser
"""

def plot_coast(lon=None,lat=None,m=None):
    import scipy as sc
    import matplotlib.pyplot as plt
    if m=='linux':
        matIN=r'C:\Users\Goran\Dropbox\DOKTORAT\coast_rps5.mat'
    else:
        matIN='/home/goran/Dropbox/DOKTORAT/coast_rps5.mat'
    coast=sc.io.loadmat(matIN,squeeze_me=False,mat_dtype=0,chars_as_strings=1)
    plt.plot(coast['coast'][:,0],coast['coast'][:,1],'k')
    if lon!=None:
        plt.xlim(lon.min(),lon.max())
        plt.ylim(lat.min(),lat.max())
    fig=plt.gcf()
    return fig
    
def basemap_plot(lon=None,lat=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Polygon    
    if lon!=None:
        m = Basemap(llcrnrlon=lon.min() ,llcrnrlat=lat.min(), urcrnrlon=lon.max() ,urcrnrlat=lat.max(), resolution='h', area_thresh=10000)
    else:
        m = Basemap(resolution='h', area_thresh=10000)    
    m.drawstates()
    m.drawmapboundary()                
    m.drawcoastlines(linewidth=1)        
    m.drawcountries(linewidth=1)        
    return m

def draw_poly( lats, lons, m, fc, alfa,**kwargs):
    """
    Crtam poligon na mapu
    lats_d02 = [ lat.min(), lat.max(), lat.max(), lat.min() ]
    lons_d02 = [ lon.min(), lon.min(), lon.max(), lon.max() ]

    m = Basemap(llcrnrlon=-10 ,llcrnrlat=33, urcrnrlon=40 ,urcrnrlat=70, resolution='h', area_thresh=10000)
    m.drawstates()
    m.drawmapboundary()                
    m.drawcoastlines(linewidth=1)        
    m.drawcountries(linewidth=1)        
    draw_poly( lats_d01, lons_d01, m, 'red', 0.5)
    """
    from matplotlib.patches import Polygon   
    import matplotlib.pyplot as plt

    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor=fc, alpha=alfa )
    plt.gca().add_patch(poly)
    fig_out=plt.gcf()
    return fig_out    
    

#%%
def plot_xy(nc,params,tms,lev=None):
    """
    RADI ZA WRF, WRFChem
    nc - nc file
    params=OrderedDict([('T2','pcolor'),('UV10','quiver')])
    tms- vremena u kojem crtamo file [0,10,20...]    
    lev level na kojoj visini uzimamo
    npr
    
    from collections import OrderedDict
    plot_xy(nc,OrderedDict([('T2','pcolor'),('UV10','quiver')]),2,1)

    """
    
    import matplotlib.pyplot as plt
    import ggWRFutils as gW
    from datetime import datetime
    import numpy as np
    wvar={}
    for p in params:
        if p != 'Times':
            if p=='WS10':
                wvar[p]=np.sqrt(nc.variables['U10'][:]**2+nc.variables['U10'][:]**2)
            elif p=='UV10': 
                wvar['U10']=nc.variables['U10'][:,:,:]    
                wvar['V10']=nc.variables['V10'][:,:,:]    
            elif p=='UV':
                wvar['U']=nc.variables['U'][:,lev,:,:]     
                wvar['V']=nc.variables['V'][:,lev,:,:]     
            elif len(nc.variables[p].shape) > 3:
                wvar[p]=nc.variables[p][:,lev,:,:]     
            else:                
                wvar[p]=nc.variables[p][:]  
    Nx,Ny,Nz,lon,lat,dx,dy=gW.getDimensions(nc)
    for p in params:
        if params[p]=='pcolor':
            plt.pcolor(lon,lat,wvar[p][tms,:,:],shading='flat')
            plt.colorbar()
        if params[p]=='contourf':
            plt.contourf(lon,lat,wvar[p][tms,:,:],50)
            plt.colorbar()
        if params[p]=='contour':
            plt.contourf(lon,lat,wvar[p][tms,:,:])
            plt.colorbar()
        if params[p]=='quiver':
            if p=='UV10':
                plt.quiver(lon[::10,::10],lat[::10,::10],wvar['U10'][tms,::10,::10],wvar['V10'][tms,::10,::10],units='width')
            elif p=='UV':
                plt.quiver(lon,lat,wvar['U'][tms,:,:],wvar['V'][tms,:,:])
        plt.hold(True)
    plt.xlim(lon.min(),lon.max())
    plt.ylim(lat.min(),lat.max())
    fig=plt.gcf()
    return fig

#%%
def diff_xy(nc1,nc2,params,tms,lev=None, v1=None, v2=None):
    """
    DIFF plot crta
    RADI ZA WRF, WRFChem
    nc1,nc2 - nc file
    params='U10' - za sad radi samo sa jednim parametrom (problem kod returna)
    tms- vremena u kojem crtamo file [0,10,20...]    
    lev level na kojoj visini uzimamo
    v1,v2 - color range
    npr
    
    from collections import OrderedDict
    plot_xy(nc,OrderedDict([('T2','pcolor'),('UV10','quiver')]),2,1)

    """
    
    import matplotlib.pyplot as plt
    import ggWRFutils as gW
    from datetime import datetime
    import numpy as np   
    from pylab import size
    if size(params)>1:
        wvar1={}
        for p in params:
            if p=='WS10':
                wvar1[p]=np.sqrt(nc1.variables['U10'][:]**2+nc1.variables['U10'][:]**2)
            elif p=='UV10': 
                wvar1['U10']=nc1.variables['U10'][:,:,:]    
                wvar1['V10']=nc1.variables['V10'][:,:,:]    
            elif p=='UV':
                wvar1['U']=nc1.variables['U'][:,lev,:,:]     
                wvar1['V']=nc1.variables['V'][:,lev,:,:]     
            elif len(nc1.variables[p].shape) > 3:
                wvar1[p]=nc1.variables[p][:,lev,:,:]     
            else:                
                wvar1[p]=nc1.variables[p][:]  
        wvar2={}
        for p in params:
            if p=='WS10':
                wvar2[p]=np.sqrt(nc2.variables['U10'][:]**2+nc2.variables['U10'][:]**2)
            elif p=='UV10': 
                wvar2['U10']=nc2.variables['U10'][:,:,:]    
                wvar2['V10']=nc2.variables['V10'][:,:,:]    
            elif p=='UV':
                wvar2['U']=nc2.variables['U'][:,lev,:,:]     
                wvar2['V']=nc2.variables['V'][:,lev,:,:]     
            elif len(nc2.variables[p].shape) > 3:
                wvar2[p]=nc2.variables[p][:,lev,:,:]     
            else:                
                wvar2[p]=nc2.variables[p][:]  
    elif size(params)==1:
        p=params                
        wvar1={}
        if p=='WS10':
            wvar1[p]=np.sqrt(nc1.variables['U10'][:]**2+nc1.variables['U10'][:]**2)
        elif p=='UV10': 
            wvar1['U10']=nc1.variables['U10'][:,:,:]    
            wvar1['V10']=nc1.variables['V10'][:,:,:]    
        elif p=='UV':
            wvar1['U']=nc1.variables['U'][:,lev,:,:]     
            wvar1['V']=nc1.variables['V'][:,lev,:,:]     
        elif len(nc1.variables[p].shape) > 3:
            wvar1[p]=nc1.variables[p][:,lev,:,:]     
        else:                
            wvar1[p]=nc1.variables[p][:]  
        wvar2={}
        if p=='WS10':
            wvar2[p]=np.sqrt(nc2.variables['U10'][:]**2+nc2.variables['U10'][:]**2)
        elif p=='UV10': 
            wvar2['U10']=nc2.variables['U10'][:,:,:]    
            wvar2['V10']=nc2.variables['V10'][:,:,:]    
        elif p=='UV':
            wvar2['U']=nc2.variables['U'][:,lev,:,:]     
            wvar2['V']=nc2.variables['V'][:,lev,:,:]     
        elif len(nc2.variables[p].shape) > 3:
            wvar2[p]=nc2.variables[p][:,lev,:,:]     
        else:                
            wvar2[p]=nc2.variables[p][:]  
       
    Nx,Ny,Nz,lon,lat,dx,dy=gW.getDimensions(nc1)
#    fig_out=[]
#    for p in params:    
#        varIN=wvar1[p][tms,:,:] - wvar2[p][tms,:,:]      
#        fig=plt.figure()          
#        plt.pcolor(lon,lat,varIN, cmap='RdBu',vmin=varIN.min(),vmax=varIN.max(), shading='flat')
#        plt.colorbar()
#        plt.xlim(lon.min(),lon.max())
#        plt.ylim(lat.min(),lat.max())
#        fig_out.append(fig)
    varIN=wvar1[p][tms,:,:] - wvar2[p][tms,:,:]  
    if v1==None:
        plt.pcolor(lon,lat,varIN, cmap='RdBu',vmin=varIN.min(),vmax=varIN.max(), shading='flat')
    else:
        plt.pcolor(lon,lat,varIN, cmap='RdBu',vmin=v1,vmax=v2, shading='flat')
    plt.colorbar()
    plt.xlim(lon.min(),lon.max())
    plt.ylim(lat.min(),lat.max())
    fig_out=plt.gcf()
    return fig_out
    
def dateINdateFNM(dIN,INfmt,FNMfmt):
    """
    Ucitavam datum i izbacujem u određenim formatima 
    """
    from datetime import datetime,timedelta
    dateIN=datetime.strftime(dIN,INfmt)
    dateFNM=datetime.strftime(dIN,FNMfmt)
    return dateIN,dateFNM
    
def plot2D(xx,yy,par,shd,clim,xos,yos):
    import matplotlib.pyplot as plt
    import ggWRFutils as gW
    from datetime import datetime
    import numpy as np   
    plt.pcolor(xx,yy,par,shading=shd)
    plt.colorbar()
    plt.clim(clim)
    plt.xlim(xos)
    plt.ylim(yos)
    fig_out=plt.gcf()
    return fig_out
    
    

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap    

def scpl(o,p): #xlabel_txt=None,ylabel_txt=None,title_txt=None):
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
    plt.plot([minx,mx],fitf([minx,mx]),'gray')
    ax=plt.gca();lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);plt.xlim(0,lim);plt.ylim(0,lim)
    plt.plot([0,lim],[0,lim],'--r',alpha=0.5)
    #plt.ylabel('$' + ylabel_txt + '$',{'fontsize':'xx-small'})
    #plt.xlabel('$' + xlabel_txt + '$',{'fontsize':'xx-small'})
    #plt.title(title_txt,{'fontsize':'xx-small'})
    plt.tick_params(axis='both',labelsize=10)   
    s=r'$Y=%.2fX+%.2f$'%tuple(fit)+'\n'+'$R=%.2f$'%R+'\n'+'$N=%d$'%len(p)
    ann_ax=plt.gcf().add_subplot(111)
    ann_ax.annotate(s, (0.05, 0.9), xycoords="axes fraction", va="center", ha="left",fontsize='xx-small',
                    bbox=dict(boxstyle="round, pad=1", fc="w")) #plt.suptitle(s,x=0.15,y=0.85,horizontalalignment='left')
    #plt.savefig(outfile,dpi=400,bbox_inches='tight')    
    fig=plt.gcf()
    return fig

def sbn_dual_HS(x,y):
    """
    dual hist + scatter kroz seborn paket
    x - vektor
    y - vektor
    x = rs.gamma(2, size=1000)
    y = -.5 * x + rs.normal(size=1000)
    
    """
    from scipy.stats import kendalltau
    import seaborn as sns
    sns.set(style="ticks")
    fig=sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")    
    return fig