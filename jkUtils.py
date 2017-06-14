# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 11:46:41 2012

@author: josip
"""
#def PGUnstackTable(host,)

def PandasLoadFromPG(host,dbname,user,password,sql):
    import pandas
    import psycopg2
    import numpy as np
    import sys

    conn_str = "host='%s' dbname='%s' user='%s' password='%s'"%(host,dbname,user,password)
    try:
        conn=psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(sql)
        # Ovdje treba izvući imena colona i TIPOVE !!!
        cols=[t[0] for t in cursor.description]
        row_list=cursor.fetchall()
        #return cols,row_list
        apgdata=np.array(row_list)
        #ind_datum=pandas.DatetimeIndex(apgdata[:,0])
        #ds=pandas.DataFrame(data=apgdata[:,1:],index=ind_datum,columns=cols[1:],dtype='float')
        ds=pandas.DataFrame(data=apgdata,columns=cols)
        return ds
    except:
        # Get the most recent exception
        exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
        # Exit the script and print an error telling what happened.
        sys.exit("Error in SaveTableFromPG!\n ->%s" % (exceptionValue))

def PandasUploadToPG(df,table_name,host,dbname,user,password,if_exists='replace'):
    import pandas_dbms
    import psycopg2 as pg
    conn_str="host='%s' dbname='%s' user='%s' password='%s'"%(host,dbname,user,password)
    conn=pg.connect(conn_str)
    pandas_dbms.write_frame(df,name=table_name,con=conn,flavor='postgresql',if_exists=if_exists)
     
def PandasLoadFromPG1(host,dbname,user,password,sql):
    import pandas_dbms
    import psycopg2 as pg
    conn_str="host='%s' dbname='%s' user='%s' password='%s'"%(host,dbname,user,password)
    conn=pg.connect(conn_str)
    return pandas_dbms.read_db(sql, conn)
    
def DataFrameToShape(df,outfile,geomcol='geom'):
    """
    ulazni DataFrame snimi kao shape.
    'geom' je kolona sa geometrijom kao tekstom
    """
    from osgeo import ogr
    import os

    (outfilepath, outfilename) = os.path.split(outfile)    
    (outfileshortname, extension) = os.path.splitext(outfilename)
    
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Create the output shapefile but check first if file exists
    if os.path.exists(outfile):
        driver.DeleteDataSource(outfile)

    outdataset = driver.CreateDataSource(outfile)

    if outdataset is None:
        print ' Could not create file'
        return
    
    outlayer = outdataset.CreateLayer(outfileshortname, geom_type=ogr.wkbPolygon)

    for col in df.columns:
        if str(col)!=geomcol:
            outlayer.CreateField(ogr.FieldDefn(str(col),ogr.OFTReal))
        
    featureDefn = outlayer.GetLayerDefn()

    for ind in df.index:
        #create a new output feature
        outfeature = ogr.Feature(featureDefn)
        geometry=ogr.CreateGeometryFromWkt(df[geomcol][ind])
        
        #set the geometry and attribute
        outfeature.SetGeometry(geometry)
        for col in df.columns:
            if str(col)!=geomcol:
                value=df[col][ind]
                if not isnan(value):
                    outfeature.SetField(str(col),value)
        
        #add the feature to the output shapefile
        outlayer.CreateFeature(outfeature)
 
        #destroy the features and get the next input features
        outfeature.Destroy
    
    outdataset.Destroy()