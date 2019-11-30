import sys
sys.path.append('../scripts/')

import numpy as np
import pandas as pd
import xarray as xr

data = pd.read_pickle('../data/drifters/posveldata_3h.pkl')

liste = []
for drif in data.index.unique():
    temp = data[data.index==drif].set_index('time')
    liste.append(xr.Dataset( temp , coords={'time':temp.index, 'id':drif}))

ds = xr.concat( liste, dim='id')

ds['sal'] = ds.sal.where( (ds.sal>20) &  (ds.sal<35))

# mild QC
ds['u'] = np.real( ds.uv )/100
ds['v'] = np.imag( ds.uv )/100
ds['u'] = ds.u.where( (ds.u>-1.5) &  (ds.u<1.5))
ds['v'] = ds.v.where( (ds.v>-1.5) &  (ds.v<1.5))

ds = ds.drop(['uv_filt','uv','lon_filt','lat_filt'])
ds.to_netcdf('../data/drifters/posveldata_xr.nc')
