#!/usr/bin/env python
# coding: utf-8

# Scientific Computing
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt

# My Stuff
from tools import datenum2datetime, latlon2uv, optimize_df


# %%FUNCTIONS
def butter_lowpass(cutoff, fs, order=5):
    '''create butterworth filter for drifter trajectories'''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''run butterworth filter forward and backward to remove edge-effects'''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def downsample_trajectories(df):
    alldata = pd.DataFrame([])  # empty dataframe
    for i in range(len(df.particle.unique())):
        # loop over drifters
        temp_df = df[df.particle == i]
        tempp = temp_df.resample('1h').bfill(limit=1).interpolate(
            'pchip')  # if resampling has gaps, fill with last entry!
        # reindexing not necessary
        tempp.loc[:, 'time'] = tempp.index
        tempp.set_index('particle', inplace=True)
        tempp.loc[:, 'particle'] = tempp.index
        # concat multiple drifters
        alldata = pd.concat([alldata, tempp])
    return alldata


def load_matfile(infile):
    return sio.loadmat(infile, squeeze_me=True, struct_as_record=False)['data']


def make_dataframe(ddata):
    '''
    Take matlab object and form a pandas dataframe
    '''
    lon = []
    lat = []
    ts = []
    ids = []
    uv = []
    sal = []
    sst = []
    for i, data in enumerate(ddata):
        lon.append(data.lon)
        lat.append(data.lat)
        sal.append(data.salinity)
        sst.append(data.sst)
        ts.append(data.ts)
        uv.append(data.uv)
        ids.append(np.ones_like(data.ts) * i)

    dat = pd.DataFrame()
    for i, data in enumerate(ddata):
        temp = pd.DataFrame({
            'sst': sst[i],
            'sal': sal[i],
            'lat': lat[i],
            'lon': lon[i],
            'uv': uv[i],
            'particle': i,
            'time': ts[i]
        })
        temp['time'] = temp['time'].apply(datenum2datetime)
        temp.set_index('time', inplace=True)
        dat = pd.concat([dat, temp])

    dat = dat.drop(dat[((dat.particle == 7) &
                        (dat.index > '2015-11-20'))].index)

    return dat

def filter_trajectories(dat):
    # p1 = dat[dat['particle'] == 1]
    # ff = gsw.f(np.nanmin(p1['lat']))
    # Tinertial = 2 * np.pi / ff  # seconds
    order = 6
    # cutoff = 1/(1.2*Tinertial)  # desired cutoff frequency of the filter, Hz
    cutoff = 1 / (3 * 60 * 60)  # desired cutoff frequency of the filter, Hz
    # 3h in seconds
    fs = 1 / (3600)

    new = pd.DataFrame()
    for i in range(45):
        p = dat[dat['particle'] == i]
        p.drop(p[~np.isfinite(p.lat)].index, inplace=True)
        p.drop(p[~np.isfinite(p.lon)].index, inplace=True)
        # p.sort_index(ascending=False,inplace=True)
        temp = pd.DataFrame(index=p.index)
        # temp['uv_filt'] = butter_lowpass_filter(p.uv, cutoff, fs, order)
        temp['lat_filt'] = butter_lowpass_filter(p.lat, cutoff, fs, order)
        temp['lon_filt'] = butter_lowpass_filter(p.lon, cutoff, fs, order)
        temp['particle'] = i
        temp['uv'] = p['uv']
        temp['lon'] = p['lon']
        temp['lat'] = p['lat']
        temp['sst'] = p['sst']
        temp['sal'] = p['sal']
        new = pd.concat([new, temp])

    total = pd.DataFrame()
    for i in range(45):
        # print(i)
        temp = new[new.particle == i]
        lon = temp.lon_filt
        lat = temp.lat_filt
        uv = latlon2uv(lon, lat)
        temp['uv_filt'] = uv
        total = pd.concat([total, temp])
    return total


def main_wrapper(infile, outfile):
    # read
    ddata = load_matfile(str(infile))
    dat = make_dataframe(ddata)
    # filter`
    total = filter_trajectories(dat)
    # downsample
    alldata = downsample_trajectories(total)
    # optimize
    alldata['particle'] = np.round(alldata.particle).astype(np.int32)
    alldata.set_index('particle', inplace=True)
    optimized_df = optimize_df(alldata)
    # save
    optimized_df.to_pickle(str(outfile))


# %%
main_wrapper(snakemake.input, snakemake.output)
