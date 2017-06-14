# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:44:25 2013

@author: goran

Needed statistical parameters for PhD content: WRF, CAMx models.
For now, it offers only comparison between two vectors.
"""

def coef_corr(x, y): #, rowvar=None, bias=None, ddof=None,**kwargs): 
    """ koeficijent korelacije """
    import numpy as np
#    if bias==None:
#        bias=0
#    if rowvar==None:
#        rowvar=1
    r=np.corrcoef(x, y)
    return r

def jkR(p,o):
   import numpy as np
   #return np.sqrt(1-(o-p).var()/o.var())
   pm=(p-p.mean())
   om=(o-o.mean())
   r2=sum(pm*om)**2/(sum(om**2)*sum(pm**2))
   return np.sqrt(r2)
   
def r(x,y):
    """ Coeff. corelation (Amela)  
    		x - modeled
    		y - observed 

                     n*sum( average(O*M) ) - sum( average(O) )*sum( average (M) )
r = __________________________________________________________________________________________________________________________
     sqrt( n*sum( average( O**2) )  - sum( average ( O ) )**2 ) *  sqrt( n*sum( average( M**2) )  - sum( average ( M ) )**2 )
    """
    import numpy as np
    n=min(np.size(x),np.size(y))
#    M=np.average(x,axis=None, weights=None,returned=False)
#    O=np.average(y,axis=None, weights=None,returned=False)    
#    OM=np.average( np.array([ x[t]*y[t] for t in range(n) ]) )
#    O2=np.average( np.array([ x[t]**2 for t in range(n) ]) )
#    M2=np.average( np.array([ y[t]**2 for t in range(n) ]) )
#    r_up=n*np.sum(OM) - np.sum(O)*np.sum(M)
#    r_down=np.sqrt( n*np.sum(O2) - (np.sum(O))**2 )*np.sqrt( n*np.sum(M2) - (np.sum(M))**2 )
    M=x;O=y;
    r_up=n*sum(O*M) - sum(O)*sum(M)
    r_down=np.sqrt( n*sum(O**2) - (sum(O))**2 )*np.sqrt( n*sum(M**2) - (sum(M))**2 )
    r = r_up/r_down
    
    return r
    
def bias(x,y,axis=None, weights=None,returned=False,**kwargs):
    """ Bias 
    		x - modeled
    		y - observed 
    """
    import numpy as np
    #M=np.average(x,axis=None, weights=None,returned=False)
    #O=np.average(y,axis=None, weights=None,returned=False)
    M=x;O=y;
    bs=((M.mean()-O.mean())/O.mean())*100
    return bs

def mae(x,y,**kwargs):
    """ Mean absolute error  
    		x - modeled
    		y - observed 
    """
    import numpy as np
    n=min(np.size(x),np.size(y))
    O_M=np.array([ np.abs(y[t]-x[t]) for t in range(n) ])
    me=1/float(n)*float(np.sum(O_M))
    return me

def mse(x,y,**kwargs):
    """ Mean square error  
    		x - modeled
    		y - observed 
    """
    import numpy as np
    n=min(np.size(x),np.size(y))
    M_O=np.array([ (x[t]-y[t])**2 for t in range(n) ])
    ms=1/float(n)*float(np.sum(M_O))
    return ms

def rmse(x,y,**kwargs):
    """ Root mean square error  
    		x - modeled
    		y - observed 
    """
    import numpy as np
    n=min(np.size(x),np.size(y))
    M_O=np.array([ (x[t]-y[t])**2 for t in range(n) ])
    rms=np.sqrt(1/float(n)*float(np.sum(M_O)))
    return rms    
    
def fb(x,y,axis=None, weights=None,returned=False,**kwargs):
    """ Fractional bias  
    		x - modeled
    		y - observed 
    """
    import numpy as np
    M=x;O=y;    
    #M=np.average(x,axis=None, weights=None,returned=False)
    #O=np.average(y,axis=None, weights=None,returned=False)    
    frb=(O.mean()-M.mean())/(0.5*(O.mean()+M.mean()))
    return frb
    
def nmse(x,y,axis=None, weights=None,returned=False,**kwargs):
    """ Normalized mean square error  
    		x - modeled
    		y - observed 
    """
    import numpy as np
    n=min(np.size(x),np.size(y))
    M=x;O=y;
    #M=np.average(x,axis=None, weights=None,returned=False)
    #O=np.average(y,axis=None, weights=None,returned=False)    
    #O_M=np.average( np.array([ (y[t]-x[t])**2 for t in range(n) ]) )
    O_M=(O-M)**2
    nm=O_M.mean()/(O.mean()*M.mean())
    return nm
    
def nmse_s(x,y,axis=None, weights=None,returned=False,**kwargs):
    """ Normalized mean square error  SYSTEMATIC
    		x - modeled
    		y - observed 
    """
    import numpy as np
    M=x;O=y;
    #M=np.average(x,axis=None, weights=None,returned=False)
    #O=np.average(y,axis=None, weights=None,returned=False)    
    frb=(O.mean()-M.mean())/(0.5*(O.mean()+M.mean()))
    nsys=4*frb**2/(4 - frb**2)
    return nsys
    
def ioa(x,y):
    """ Index of agremeent (Amela rad; oznaka d)  
    		x - modeled
    		y - observed 
    		0<=ioa<=1
    """
    import numpy as np
    n=min(np.size(x),np.size(y))
    M_O=np.array([ (x[t]-y[t])**2 for t in range(n) ])
    abs_M_O=np.array([ ( np.absolute(x[t])+np.absolute(y[t]) )**2 for t in range(n) ])    
    ioa=1-np.sum(M_O)/np.sum(abs_M_O)
    return ioa    