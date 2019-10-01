import numpy as np
import pandas as pd
import bottleneck as bn
from numba import jit,vectorize
import warnings
warnings.simplefilter('ignore')
from scipy.special import comb
from sklearn.utils import resample # for bootstrapping
from itertools import combinations
from datetime import datetime,timedelta
from nn_vorticity_module import least_square_method
from haversine import haversine
import random
import matplotlib.dates as mdates
import multiprocessing as mp
from time import time
import scipy.linalg as la

class Polygon:
    
    def __init__(self,i,comb,data):
        self.lats=data.lat.values
        self.lons=data.lon.values
        self.com=np.array( [bn.nanmean(data.lon.values),bn.nanmean(data.lat.values)])
        self.length=[]
        self.aspect=[]
        self.angle=[]
    
    def p(self,i):
        # return coordinates of a point
        return [self.lons[i],self.lats[i]]
    
    def calc_lengths(self):
        lengths=[]
        ncc = len(self.lons)
        r = combinations(np.arange(ncc), 2) 
        
        k=0
        for i,j in r:
            lengths.append( haversine( self.p(i),self.p(j) ) )
            k+=1
            
        lengths=np.array(lengths)                   
        if np.sum(np.isfinite(lengths))==k:
            self.length = np.sqrt( bn.nanmean(lengths**2) )
        else:
            self.length = np.nan    
    
    def least_square_method(self):
        #import gsw
        import scipy.linalg as la
        timeseries=True
        ncc = len(self.lons)
        dlon=[]
        dlat=[] 
        for i in range(ncc):
            # haversine(p1,p2)
            dlon.append(haversine( [self.lons[i],self.com[1]],self.com)*1000*np.sign(self.lons[i]-self.com[0]))
            dlat.append(haversine( [self.com[0],self.lats[i]],self.com)*1000*np.sign(self.lats[i]-self.com[1]))
        
        if not timeseries:
            R = np.mat( np.vstack( (np.ones((ncc,)) ,np.array(dlon), np.array(dlat) )).T )
            u0=np.mat(self.us).T
            v0=np.mat(self.vs).T

            A,_,_,_=la.lstsq(R,u0)
            B,_,_,_=la.lstsq(R,v0)
        
            self.A=A[1:]
            self.B=B[1:]

        self.aspect = aspect_ratio(dlon,dlat)

def aspect_ratio(dlon,dlat):   
    points =np.vstack( [dlon,dlat] )
    if np.sum( np.isfinite(points))==2*npol:
            # careful with nans
        cov = np.cov(points)
        w,v = np.linalg.eig(cov)
        aspect = bn.nanmin(w)/bn.nanmax(w)
    else:
        aspect=np.nan
    return aspect        
        
def makePolygons(i):
    criteria2 = data_chosen.particle.isin(combs[i])
    return Polygon(i,combs[i],data_chosen[criteria2])

def calc_properties(i):
    results[i].calc_lengths()
    results[i].least_square_method() 
    return results[i]

def find_percentiles(data):
    alpha = 0.75 # 95% confidence interval
    ordered = np.sort(data)
    lo = np.percentile(ordered,100*(1-alpha)/2,)
    hi = np.percentile(ordered,100*(alpha+(1-alpha)/2))
    return lo,hi

data = pd.read_pickle('./posveldata_all.pkl')
N = data.particle.unique().size
npol=6
Nnpol = comb(N,npol,exact=True)
sampling_times = pd.date_range(data.index.unique()[0], periods=16, freq='2D')
T = len(sampling_times)
print('N=%d, npol=%d, Nnpol=%d, T=%d' %(N,npol,Nnpol,T))

combs=[]
for combi in combinations(np.arange(N),npol):
    combs.append(combi)

results=[]

mean_length=np.zeros(T)
median_length=np.zeros(T)
std_length=np.zeros(T)
lo_length=np.zeros(T)
hi_length=np.zeros(T)

median_aspect=np.zeros(T)
mean_aspect=np.zeros(T)
lo_aspect=np.zeros(T)
hi_aspect=np.zeros(T)

start = time()
for t,tim in enumerate(sampling_times):
    startloop = time()
    results=[]
    if tim in data.index:
              
        # select subset in time
        criteria1 = data.index == tim #data.index.unique()[t]
        data_chosen = data[criteria1]
        
        # make polygons
        results=[]
        pool = mp.Pool(8)     
        results = pool.map(makePolygons, range(Nnpol))
        pool.close()
        pool.join()
        
        # calculate properties
        pool = mp.Pool(8)     
        results = pool.map(calc_properties, range(Nnpol))
        pool.close()
        pool.join()
        
        # read geometry data
        asp = np.array([results[i].aspect for i in range(Nnpol)]).squeeze()
        ang = np.array([results[i].angle for i in range(Nnpol)]).squeeze()
        leng = np.array([results[i].length for i in range(Nnpol)]).squeeze()
        
        mean_length[t] = bn.nanmean(leng)
        std_length[t] = bn.nanstd(leng)
        median_length[t] = bn.nanmedian(leng)
        lo_length[t],hi_length[t] = find_percentiles(leng[np.isfinite(leng)])
        
        mean_aspect[t] = bn.nanmean(asp)
        median_aspect[t] = bn.nanmedian(asp)
        lo_aspect[t],hi_aspect[t] = find_percentiles(asp[np.isfinite(asp)])
        
        if np.mod(t,1)==0:
            print('1 step in %d %3.3f minutes' %(t,(time()-startloop)/60) )
        
    else:
        print('no data at that time.')
        
    if np.mod(t,10)==0 or t==T-1:
        print('save data' )
        df=[]
        df = pd.DataFrame(index=sampling_times,data={'mean_length':mean_length,'median_length':median_length,'lo_length':lo_length,'hi_length':hi_length,
                                             'mean_aspect':mean_aspect,'median_aspect':median_aspect,'lo_aspect':lo_aspect,'hi_aspect':hi_aspect})

        df['dtime']=df.index-df.index[0]
        df['dtime']=df.dtime.dt.days.values
        fname = 'deformation_n_6_T_90d_split_%d.pkl' %t
        df.to_pickle(fname)
    
print('total %3.3f minutes' %((time()-start)/60) )