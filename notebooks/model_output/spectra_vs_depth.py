#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:28:48 2018

@author: seb
"""

# extract surface slice:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
from scipy.io import netcdf
import numpy as np
f = netcdf.netcdf_file('full_08325.cdf','r')

U = f.variables['u'][:][33,:321,:193]
V = f.variables['v'][:][33,:321,:193]

u=U.copy()
v=V.copy()

plt.pcolor(u)

# parameters
NI=322
NK=194
kI=2*np.pi/NI
kK=2*np.pi/NK

# detrend
u = u-np.mean(u)
v = v-np.mean(v)

# fourier transform
uhat = np.fft.fft2(u)
vhat = np.fft.fft2(v)

# create k,l wavenumber
k = np.arange(-NI/2,NI/2-1,1) # make array of length L, matlab [1:1:L]
l = np.arange(-NK/2,NK/2-1,1)
maxk = np.amax(k);maxl=np.amax(l); # maximum of array

# shift 0 frequency to center of spectrum
uhath = np.fft.fftshift(uhat)
vhath = np.fft.fftshift(vhat)
velhath = np.sqrt(np.square(uhath) + np.square(vhath))
kk,ll = np.meshgrid(l,k) 
kl = np.sqrt( np.square(kk) + np.square(ll) ) 

aa=np.amin( [ kl[: int( np.floor( len(kl)/2)) ]-1 ] )
bb=np.amax( [ kl[: int( np.floor( len(kl)/2))] ] )
dk=1 # this is the bin width in wavenumber space
H1=np.arange(aa,bb,dk)
power_vel = np.abs(velhath)
nk=len(H1)

# binning into total wavenumber bins
power_avg_vel=np.zeros(nk-1)
kl_avg=np.zeros(nk-1)
for jk,var in enumerate(H1[:len(H1)-1]):     
    xx = np.where( np.logical_and( kl>=H1[jk], kl<H1[jk+1] ) )
    power_avg_vel[jk] = np.nanmean( power_vel[xx] )
    kl_avg[jk]=np.nanmean( kl[xx] )
    xx=[]
    
# "high pass filter " 
power_avg_vel[np.where(power_avg_vel < 1e-14)] = 0

fig,ax = plt.subplots(1,1)
ax.scatter(kl_avg, power_avg_vel,marker='.')
#ax.scatter(kl_avg, np.exp(2)*(kl_avg*1000)**(-2),marker='.')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_ylim(1e-5,1e2)
ax.set_title("PSD")
ax.set_xlabel("total wavenumber K")
plt.show()

