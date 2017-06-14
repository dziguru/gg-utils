# -*- coding: utf-8 -*-
"""
@author: Goran Gašparac

DESC: utility script for APG course
"""
        
def plt(xx,yy,color=None,legenda=None,x_label=None,y_label=None,xtick_loc=None,time_fmt=None,xlim=None,ylim=None,tick_position=None,**kwargs):
    """ 
    Prilagođena plot funkcija.    
    
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    gfig=plt.plot(xx,yy,color,**kwargs)
    gfig=plt.grid()
    gfig=plt.tick_params(axis='both',labelsize=9)        
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
        gfig=plt.legend((legenda,),prop={'size':6},loc=0)  
    if x_label is not None:
        gfig=plt.xlabel(x_label,{'fontsize':'xx-small'})           
    if y_label is not None:
        gfig=plt.ylabel(y_label,{'fontsize':'xx-small'})         
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
    gfig=plt.tick_params(axis='both',labelsize=6)         
    return gfig

def hist(xx, bins, range=None, normed=False, weights=None, 
         cumulative=False, bottom=None, histtype='bar', align='mid', 
         orientation='vertical', rwidth=None, log=False, colorIN=None, 
         label=None, stacked=False, hold=None,
         legenda=None,x_label=None,y_label=None,
         xlim=None,ylim=None,
         tick_position=None,**kwargs):
    """ 
    Prilagođena histogram funkcija

    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    gfig=plt.hist(xx,bins,color=colorIN)
    gfig=plt.grid()
    gfig=plt.tick_params(axis='both',labelsize=9)        
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
    if legenda is not None:
        gfig=plt.legend((legenda,),prop={'size':4},loc=0)  
    if x_label is not None:
        gfig=plt.xlabel(x_label,{'fontsize':'xx-small'})           
    if y_label is not None:
        gfig=plt.ylabel(y_label,{'fontsize':'xx-small'})         
    gfig=plt.tick_params(axis='both',labelsize=6)         
    return gfig

def scatter_plot(o,p,xlabel_txt=None,ylabel_txt=None): #o=observed, p=predicted
    """ 
    Prilagođena histogram funkcija

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fit=np.polyfit(o,p,1)
    fitf=np.poly1d(fit)
    mx=o.max(); my=p.max()
    minx=o.min(); miny=p.min()
    (mx,my)=(max(mx,my),)*2  #Ovo uskla\u0111uje maximume po x-u i y-onu
    (minx,miny)=(min(minx,miny),)*2
    plt.plot(o,p,'.')
    plt.gca().set_xlim(minx+0.5,mx+0.5)
    plt.gca().set_ylim(miny+0.5,my+0.5)#plt.plot([minx,mx],fitf([minx,mx]),'r')
    plt.ylabel(ylabel_txt,{'fontsize':'xx-small'})
    plt.xlabel(xlabel_txt,{'fontsize':'xx-small'})
    gfig=plt.gcf()
    return gfig
    
def svd(data,dec,sOUT):
    """
    Singular value decomposition, Ako imam potpuni rastav, decomposition=1
    
    Ako imam djelomicni rastav, decomposition=0
    
    Vraca s u arrayu ako je flag = 1, u suprotnome kao vector
    
    u,s,v=svd(A,1,1),
    
    gdje je npr A=np.random.randint(1,99,(5,3)) # integers od 0 do 99

    """
    import numpy as np
    if dec==0:
        u0, s0, v0 = np.linalg.svd(data,full_matrices=False)
    elif dec==1:
        u1, s1, v1 = np.linalg.svd(data,full_matrices=True)
    
    if sOUT==0:
        if dec==0:
            return u0,s0,v0
        elif dec==1:
            return u1,s1,v1
    elif sOUT==1:
        if dec==0:
            return u0,np.diag(s0),v0
        elif dec==1:
            s1a=np.zeros((data.shape[0],data.shape[1]))
            for ii in range(0,min(data.shape)):
                s1a[ii,ii]=s1[ii]
            return u1,s1a,v1
                
def def_out(ROOTdir,out):
    """ 
    Kreiram output direktorij
    ROOTdir je glavni direktorij koji već postoji
    out - direktorij koji želim kreirati
    """    
    import os
    outdir=os.path.join(ROOTdir,out)
    os.path.join(outdir)
    if os.path.isdir(outdir)==False:    
        os.mkdir(outdir)
        print "Direktorij ne postoji, kreiram ga\n"
    print "Bacam sve u: \n " + outdir                
    return outdir
    
def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
    """ 
    # Source: scipy-central.org/item/23/1/plot-an-ellipse
    # License: Creative Commons Zero (almost public domain) http://scpyce.org/cc0

    Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    import numpy as np
    from pylab import plot, show, grid

    pts = np.zeros((360*k+1, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])
 
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    
    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)
    return pts    
