#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:35:17 2018

@author: seb
"""

# extract surface slice:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid',context='poster')
from scipy.io import netcdf
import numpy as np
from spectra_module import spec2d
import pandas as pd

path = './output_2013asiri_05b/'
zgrid = pd.read_csv(path+'zgrid.out', skipinitialspace=True, sep=' ', header=None)[1][1:51]

f = netcdf.netcdf_file(path+'full_04700.cdf','r')

ps1D = np.zeros((50,189))
bin_cents = np.zeros((50,189))
for k in range(50):

    u = f.variables['u'][:][k,:,:].copy()
    v = f.variables['v'][:][k,:,:].copy()
    U = 0.5*( u**2 + v**2) 
    
    ps1D[k,:], bin_cents[k,:] = spec2d(U,logspacing=False)

#ps1D[~np.isfinite(ps1D)]=np.nan
#ps1D[ps1D<1e-2]=np.nan
plt.figure()
#plt.contour(bin_cents[0,:],zgrid,np.log(ps1D))
plt.pcolor(bin_cents[0,:],zgrid,np.log(ps1D),cmap='RdBu_r',vmin=1e-2,vmax=0.7e1)
plt.xscale('log')
plt.xlim(5e-3,1e0)
plt.ylim(-1000,0)
plt.colorbar(label='log(PSD)')
plt.ylabel('depth')
plt.xlabel('spatial frequency [1/km]')
plt.savefig('spec_vs_depth.png',dpi=300,bbox_inches='tight')

plt.figure(figsize=(8,8))
plt.loglog(bin_cents[0,:],ps1D[33,:],label='z=%d m' %zgrid[33])
#plt.loglog(bin_cents[0,:],ps1D[15,:],label='z=%d m' %zgrid[15])
plt.loglog(bin_cents[0,:],ps1D[5,:],label='z=%d m' %zgrid[5])
#plt.loglog(bin_cents[0,:],ps1D[0,:],label='z=%d m' %zgrid[0])
plt.loglog(bin_cents[0,:],bin_cents[0,:]**(-2),color='black')
plt.annotate('k~-3',(1e-1,1e3),fontsize=20)
plt.annotate('k~-2',(1e-1,1e2),fontsize=20)
#plt.loglog(bin_cents[0,:],ps1D[0,:],label='z=%d m' %zgrid[0])
plt.loglog(bin_cents[0,:],bin_cents[0,:]**(-3),color='black')
#plt.loglog(bin_cents,np.exp(2)*bin_cents**(-1))
plt.xlim(5e-3,1e0)
plt.xlabel('spatial frequency 1/km')
plt.ylabel(r'power spectral density $m^3/s^2$')
plt.legend()
plt.savefig('spectra_vs_depth.png',dpi=300,bbox_inches='tight')
plt.show()
