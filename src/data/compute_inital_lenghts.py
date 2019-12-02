import sys
sys.path.append('../scripts/')

import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import gammaln
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr

from deformtools.haversine import haversine
import scipy.linalg as la
from scipy.spatial import ConvexHull
import bottleneck as bn
import gsw
from scipy.special import comb
from itertools import combinations

import warnings
warnings.simplefilter("ignore",category=FutureWarning)
warnings.simplefilter("ignore",category=RuntimeWarning)

# %%
def get_chunksize(iterable,num_process):
    split_div = 2
    chunksize, extra = divmod(len(iterable), num_process * split_div)
    if extra:
        chunksize += 1
    if len(iterable) == 0:
        chunksize = 0
    return chunksize

def calc_inital_length(dspt):
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
    return length

def par_lengths(i):
    dsp = ds.isel(id=list(i))
#     dsp = dsp.dropna(dim='time',how='any',subset=['lat','lon','u','v'])
#     lengths.append( calc_inital_length(dsp.isel(time=0)) )
#     ids.append(i)
    return [i,calc_inital_length(dsp.isel(time=0))]

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted( random.sample(range(n), r) )
    return tuple(pool[i] for i in indices)

# %%
npol=6
N = 45
combin=[]
for combi in combinations(np.arange(N),npol):
    combin.append(combi)

num_process=8
pool = Pool(processes=num_process)

start=time()
iterable = list( combinations(np.arange(45),6) )
res = pool.map(par_lengths, iterable, chunksize=get_chunksize(iterable,num_process), )
pool.close()
print((time()-start)/60)

res = np.array(res)
np.save('initial_lengths.npy',res)
