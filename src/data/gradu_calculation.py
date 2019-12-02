import sys
sys.path.append('../scripts/')
from time import time
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import bottleneck as bn

from scipy.special import gammaln,comb
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr
import scipy.linalg as la
from scipy.spatial import ConvexHull

from deformtools.haversine import haversine
from multiprocessing import Pool
import gsw

warnings.simplefilter("ignore",category=FutureWarning)
warnings.simplefilter("ignore",category=RuntimeWarning)

# %%
def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted( random.sample(range(n), r) )
    return tuple(pool[i] for i in indices)

def least_square_method(dspt):
        npol =6
        com = np.array( [bn.nanmean(dspt.lon),bn.nanmean(dspt.lat)])
        timeseries=False
        ncc = dspt.lon.size
        dlon=[]
        dlat=[]
        for i in range(ncc):
            # haversine(p1,p2)
            dlon.append(haversine( [dspt.lon[i],com[1]],com)*1000*np.sign(dspt.lon[i]-com[0]))
            dlat.append(haversine( [com[0],dspt.lat[i]],com)*1000*np.sign(dspt.lat[i]-com[1]))

        dlon=np.array(dlon)
        dlat=np.array(dlat)
        if not timeseries:
            R = np.mat( np.vstack( (np.ones((ncc)) ,dlon, dlat )).T )
            u0=np.mat(dspt.u.values).T
            v0=np.mat(dspt.v.values).T

            if (np.isnan( u0 ).sum()==0) & (np.isnan( v0 ).sum()==0) & (np.isnan( R ).sum()==0):
                A,_,_,_=la.lstsq(R,u0)
                B,_,_,_=la.lstsq(R,v0)
            else:
                A = np.nan*np.ones(ncc)
                B = np.nan*np.ones(ncc)

        points =np.vstack( [dlon,dlat] )
        if (np.isfinite(dlon).sum()==npol) and (np.isfinite(dlat).sum()==npol):
            # careful with nans
            cov = np.cov(points)
            w,v = np.linalg.eig(cov)
            aspect = bn.nanmin(w)/bn.nanmax(w)

            if aspect<0.99:
                ind = bn.nanargmax(w)
                angle = np.arctan(v[ind,1]/v[ind,0])*180/np.pi
                if (angle < 0):
                    angle += 360.
            else:
                angle=np.nan
        else:
            aspect=np.nan
            angle=np.nan


        dspt['ux'] = float(A[1])
        dspt['uy'] = float(A[2])
        dspt['vx'] = float(B[1])
        dspt['vy'] = float(B[2])
        dspt['aspect'] = aspect
        dspt['angle'] = angle

        return dspt

def compute_vort_etc(ds):
    ds['vort'] = (ds.vx-ds.uy)/gsw.f(40)
    ds['div'] = (ds.ux+ds.vy)/gsw.f(40)
    ds['strain'] = np.sqrt( (ds.ux-ds.vy)**2 +(ds.vx+ds.uy)**2 )/gsw.f(40)
    return ds

def calc_lengths(dspt):
    lengths=[]
    ncc = dspt.lon.size
    r = combinations(np.arange(ncc), 2)

    k=0
    for i,j in r:
        lengths.append( haversine( [dspt.lon[i],dspt.lat[i]], [dspt.lon[j],dspt.lat[j]] ) )
        k+=1
    lengths=np.array(lengths)
    if np.isfinite(lengths).sum()==k:
        length = np.sqrt( np.mean(lengths**2) )
    else:
        length = np.nan
    dspt['length'] = length
    return dspt

def apply_to_ds(i):
    dsp = ds.isel(id=list(i))
    dsp = dsp.dropna(dim='time',how='any',subset=['lat','lon','u','v'])

    timeseries=[]
    for ti in dsp.time:
        timeseries.append(least_square_method(dsp.sel(time=ti)))

    series = [calc_lengths(temp) for temp in timeseries]
    return xr.concat(series, dim='time')

def get_chunksize(iterable,num_process):
    split_div = 2
    chunksize, extra = divmod(len(iterable), num_process * split_div)
    if extra:
        chunksize += 1
    if len(iterable) == 0:
        chunksize = 0
    return chunksize

# %% MAIN
global ds
ds = xr.open_dataset('./data/drifters/posveldata_xr.nc')
res = np.load('./data/clusters/initial_lengths.npy',allow_pickle=True)

# %%
# select polygons of a certain inital size
le = res[:,1]
co = res[:,0]
selected_co = co[le<10]

npol=6
N = 45
combin=[]
for combi in combinations(np.arange(N),npol):
    combin.append(combi)


ds = ds.sel(time=slice('2015-09-02','2015-09-12'))
ds['u'] = ds.u.where(ds.u !=0)
ds['v'] = ds.v.where(ds.v !=0)
ds = ds.dropna('id',how='all')
ds = ds.dropna('time',how='all',thresh=3)

sel_co_split = np.split( selected_co , np.arange(2000,48001,2000))
# %%
num_process = 8
pool = Pool(processes=num_process)

sel_co_split = selected_co[46001:]
for i,split in enumerate(sel_co_split):
    result=[]
    multi = []
    print(f'This is split number {i:02d}.')

    start=time()
    result = pool.map(apply_to_ds, split, chunksize=get_chunksize(split,num_process) )
    # pool.join()
    multi = xr.concat(result, dim='clusters')
    multi = compute_vort_etc(multi)
    multi.to_netcdf(f'./data/clusters/combinations_lt_10_1h_{i:02d}.nc')
    print(f'Saved split number {i:02d}.')
    print((time()-start)/60)

pool.close()
pool.terminate()
