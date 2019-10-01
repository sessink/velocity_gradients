#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:00:12 2018

@author: seb
"""
import bottleneck as bn
import gsw
import numpy as np
import scipy.linalg as la
from haversine import haversine

def least_square_method(x0, y0, u0, v0, lat0):
    ncc = x0.size
    dlon=np.zeros(ncc)  
    dlat=np.zeros(ncc)  
    mx0=bn.nanmean(x0)
    my0=bn.nanmean(y0)
    for i in range(ncc):
        
        # old haversine(lon,lat)
        #dlon[i] =  haversine( [x0.iloc[i],mx0],[my0,my0] )*1000*np.sign(x0.iloc[i]-mx0)
        #dlat[i] =  haversine( [mx0,mx0],[y0.iloc[i],my0] )*1000*np.sign(y0.iloc[i]-my0)
        # new haversine(p1,p2)
        dlon[i] =  haversine( [x0.iloc[i],my0],[mx0,my0] )*1000*np.sign(x0.iloc[i]-mx0)
        dlat[i] =  haversine( [mx0,my0],[mx0,y0.iloc[i]] )*1000*np.sign(y0.iloc[i]-my0)
        # cartesian
        dlon[i] = mx0-x0.iloc[i]
        dlat[i] = my0-y0.iloc[i]
    f=gsw.f(bn.nanmean(lat0))
    #print(dlon,dlat)
    
    R = np.mat( np.vstack( (np.ones((ncc,)) ,dlon, dlat) ).T )
    u0=np.mat(u0).T/100
    v0=np.mat(v0).T/100
    
    A,_,_,_=la.lstsq(R,u0)
    B,_,_,_=la.lstsq(R,v0)
    
    vort = (B[1]-A[2])/f
    strain = np.sqrt( (A[1]-B[2])**2 + (B[1]+A[2])**2 ) /f
    div = (A[1]+B[2])/f

    return vort, strain, div



#%%
#import bottleneck as bn
#import gsw
#import scipy.linalg as la
#
#x0 = df.lon.iloc[indices[i,:]]
#y0 = df.lat.iloc[indices[i,:]]
#u0 = df.u.iloc[indices[i,:]]/100
#v0 = df.v.iloc[indices[i,:]]/100
#
#ncc = x0.size
#dlon=np.zeros(ncc)  
#dlat=np.zeros(ncc)  
#mx0=bn.nanmean(x0)
#my0=bn.nanmean(y0)
#for i in range(ncc):
#    # should be distance in m in x from COM
#    dlon[i] =  haversine( x0.iloc[i],mx0,my0,my0 )*1000*np.sign(x0.iloc[i]-mx0)
#    dlat[i] =  haversine( mx0,mx0,y0.iloc[i],my0 )*1000*np.sign(y0.iloc[i]-my0)
#f=gsw.f(bn.nanmean(y0))
#
#R = np.mat( np.vstack( (np.ones((ncc,)) ,dlon, dlat) ).T )
#u0=np.mat(u0).T
#v0=np.mat(v0).T
##
#A,_,_,_=la.lstsq(R,u0)
#B,_,_,_=la.lstsq(R,v0)
##
#vort = (B[1]-A[2])/f
#strain = np.sqrt( (A[1]-B[2])**2 + (B[1]+A[2])**2 ) /f
#div = (A[1]+B[2])/f