# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:21:30 2015

@author: wrfuser
"""


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


def plotZM(data, x, y, plotOpt=None, modelLevels=None, surfacePressure=None):
    """Create a zonal mean contour plot of one variable
    plotOpt is a dictionary with plotting options:
      'scale_factor': multiply values with this factor before plotting
      'units': a units label for the colorbar
      'levels': use list of values as contour intervals
      'title': a title for the plot
    modelLevels: a list of pressure values indicating the model vertical resolution. If present,
        a small side panel will be drawn with lines for each model level
    surfacePressure: a list (dimension len(x)) of surface pressure values. If present, these will
        be used to mask out regions below the surface
    """
    # explanation of axes:
    #   ax1: primary coordinate system latitude vs. pressure (left ticks on y axis)
    #   ax2: twinned axes for altitude coordinates on right y axis
    #   axm: small side panel with shared y axis from ax2 for display of model levels
    # right y ticks and y label will be drawn on axr if modelLevels are given, else on ax2
    #   axr: pointer to "right axis", either ax2 or axm

    if plotOpt is None: plotOpt = {}
    labelFontSize = "small"
    # create figure and axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # scale data if requested
    scale_factor = plotOpt.get('scale_factor', 1.0)
    pdata = data * scale_factor
    # determine contour levels to be used; default: linear spacing, 20 levels
    clevs = plotOpt.get('levels', np.linspace(data.min(), data.max(), 20))
    # map contour values to colors
    norm=matplotlib.colors.BoundaryNorm(clevs, ncolors=256, clip=False)
    # draw the (filled) contours
    contour = ax1.contourf(x, y, pdata, levels=clevs, norm=norm) 
    # mask out surface pressure if given
    if not surfacePressure is None: 
        ax1.fill_between(x, surfacePressure, surfacePressure.max(), color="white")    
    # add a title
    title = plotOpt.get('title', 'Vertical cross section')
    ax1.set_title(title)
    # add colorbar
    # Note: use of the ticks keyword forces colorbar to draw all labels
    fmt = matplotlib.ticker.FormatStrFormatter("%g")
    cbar = fig.colorbar(contour, ax=ax1, orientation='horizontal', shrink=0.8,
                        ticks=clevs, format=fmt)
    cbar.set_label(plotOpt.get('units', ''))
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(labelFontSize)
    # set up y axes: log pressure labels on the left y axis, altitude labels
    # according to model levels on the right y axis
    ax1.set_ylabel("Pressure [hPa]")
    ax1.set_yscale('log')
    ax1.set_ylim(10.*np.ceil(y.max()/10.), y.min()) # avoid truncation of 1000 hPa
    subs = [1,2,5]
    if y.max()/y.min() < 30.:
        subs = [1,2,3,4,5,6,7,8,9]
    y1loc = matplotlib.ticker.LogLocator(base=10., subs=subs)
    ax1.yaxis.set_major_locator(y1loc)
    fmt = matplotlib.ticker.FormatStrFormatter("%g")
    ax1.yaxis.set_major_formatter(fmt)
    for t in ax1.get_yticklabels():
        t.set_fontsize(labelFontSize)
    # calculate altitudes from pressure values (use fixed scale height)
    z0 = 8.400    # scale height for pressure_to_altitude conversion [km]
    altitude = z0 * np.log(1015.23/y)
    # add second y axis for altitude scale 
    ax2 = ax1.twinx()
    # change values and font size of x labels
    ax1.set_xlabel('Latitude [degrees]')
    xloc = matplotlib.ticker.FixedLocator(np.arange(-90.,91.,30.))
    ax1.xaxis.set_major_locator(xloc)
    for t in ax1.get_xticklabels():
        t.set_fontsize(labelFontSize)
    # draw horizontal lines to the right to indicate model levels
    if not modelLevels is None:
        pos = ax1.get_position()
        axm = fig.add_axes([pos.x1,pos.y0,0.02,pos.height], sharey=ax2)
        axm.set_xlim(0., 1.)
        axm.xaxis.set_visible(False)
        modelLev = axm.hlines(altitude, 0., 1., color='0.5')
        axr = axm     # specify y axis for right tick marks and labels
        # turn off tick labels of ax2
        for t in ax2.get_yticklabels():
            t.set_visible(False)
        label_xcoor = 3.7
    else:
        axr = ax2
        label_xcoor = 1.05
    axr.set_ylabel("Altitude [km]")
    axr.yaxis.set_label_coords(label_xcoor, 0.5)
    axr.set_ylim(altitude.min(), altitude.max())
    yrloc = matplotlib.ticker.MaxNLocator(steps=[1,2,5,10])
    axr.yaxis.set_major_locator(yrloc)
    axr.yaxis.tick_right()
    for t in axr.yaxis.get_majorticklines():
        t.set_visible(False)
    for t in axr.get_yticklabels():
        t.set_fontsize(labelFontSize)
    # show plot
    plt.show()