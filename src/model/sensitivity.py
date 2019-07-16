# Standard Library
import math
import random

# Scientific Computing
import numpy as np
import pandas as pd
#from model_plotting import plot_snapshot,plot_snapshot_w
import scipy.linalg as la
import xarray as xr
from scipy import interpolate, stats
from scipy.io import netcdf
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import bottleneck as bn
import gsw
from sklearn.utils import resample

sns.set(context='talk', style='whitegrid', font_scale=1.3)


# %%
def make_hex(x0, y0, L, skew, M):
    xx = np.zeros(M)
    yy = np.zeros(M)

    degstart = random.randint(0, 360)
    angle = np.arange(degstart, degstart + 360, 360 / M)
    angle[angle > 360] = angle[angle > 360] - 360

    random_stretch = random.randint(0, 2)
    i = 0
    for j, angle in enumerate(angle):
        if j == random_stretch:
            alpha = skew
        else:
            alpha = 1
        xx[i] = x0 + alpha * L * math.cos(math.radians(angle))
        yy[i] = y0 + alpha * L * math.sin(math.radians(angle))
        i += 1

    return xx, yy


def make_n_hexs(L, skew, N, M):
    x = np.zeros((N, M))
    y = np.zeros((N, M))
    for i in range(N):
        x0 = (180 - 10) * random.random() + 10
        y0 = (300 - 10) * random.random() + 10
        x[i, :], y[i, :] = make_hex(x0, y0, L, skew, M)
    return x, y


def least_square_method(x0, y0, u0, v0, switch):
    # TODO: vectorize this?

    ncc = x0.size

    dlon = np.zeros(ncc)
    dlat = np.zeros(ncc)
    for i in range(ncc):
        dlon[i] = (x0[i] - np.nanmean(x0)) * 1000
        # should be distance in m in x from COM
        dlat[i] = (y0[i] - np.nanmean(y0)) * 1000
    f = gsw.f(17)

    R = np.mat(np.vstack((np.ones((ncc, )), dlon, dlat)).T)
    u0 = np.mat(u0).T - np.nanmean(u0)
    v0 = np.mat(v0).T - np.nanmean(v0)

    if switch is 'lstsq':
        A, _, _, _ = la.lstsq(R, u0)
        B, _, _, _ = la.lstsq(R, v0)
    elif switch is 'inv':
        A = np.linalg.inv(R.T * R) * R.T * u0
        B = np.linalg.inv(R.T * R) * R.T * v0
    elif switch is 'solve':
        A = np.linalg.solve(R, u0)
        B = np.linalg.solve(R, v0)

    vort = (B[1] - A[2]) / f
    strain = np.sqrt((A[1] - B[2])**2 + (B[1] + A[2])**2) / f
    div = (A[1] + B[2]) / f

    return vort, strain, div


def bootstrap_ci(zetai_mean, vorti, N):
    n_iterations = 1000
    n_size = int( N* 0.50)
    alpha = 0.95  # 95% confidence interval

    statss = []
    for i in range(n_iterations):
        sample_ind = resample(np.arange(N), n_samples=n_size)
        #error = np.var(zetai_mean[sample_ind]-vorti[sample_ind])/np.var(zetai_mean[sample_ind])
        error = stats.pearsonr(zetai_mean[sample_ind], vorti[sample_ind])[0]**2
        statss.append(error)

    ordered = np.sort(statss)
    lower = np.percentile(ordered, 100 * (1 - alpha) / 2)
    upper = np.percentile(ordered, 100 * (alpha + (1 - alpha) / 2))
    return lower, upper


def r2(x, y):
    return stats.pearsonr(x, y)[0]**2


def read_model_field(zgrid_path, model_path):
    zgrid = pd.read_csv(str(zgrid_path),
                        skipinitialspace=True,
                        sep=' ',
                        header=None)
    zgrid = zgrid[1].values

    time = model_path.split('full_')[-1].split('.cdf')[0]

    droplist = [
        'h', 'consump', 'tr', 's', 'rho', 'temp', 'p', 'pv', 'conv', 'con100',
        'w', 'zc'
    ]
    dat = xr.open_dataset(str(model_path),
                          drop_variables=droplist).isel(sigma=33)

    return dat


def filter_fields(dat):
    dat['vor_10km'] = (('y', 'x'),
                       gaussian_filter(dat.vor, sigma=5, mode='wrap'))
    dat['u_10km'] = (('y', 'x'), gaussian_filter(dat.u, sigma=5, mode='wrap'))
    dat['v_10km'] = (('y', 'x'), gaussian_filter(dat.v, sigma=5, mode='wrap'))
    dat['vor_20km'] = (('y', 'x'),
                       gaussian_filter(dat.vor, sigma=10, mode='wrap'))
    dat['u_20km'] = (('y', 'x'), gaussian_filter(dat.u, sigma=10, mode='wrap'))
    dat['v_20km'] = (('y', 'x'), gaussian_filter(dat.v, sigma=10, mode='wrap'))

    fu0 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.u)
    fv0 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.v)
    fzeta0 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.vor)
    # 10km
    fu10 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.u_10km)
    fv10 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.v_10km)
    fzeta10 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.vor_10km)
    # 20km
    fu20 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.u_20km)
    fv20 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.v_20km)
    fzeta20 = interpolate.RectBivariateSpline(dat.yc, dat.xc, dat.vor_20km)

    fu = [fu0, fu10, fu20]
    fv = [fv0, fv10, fv20]
    fzeta = [fzeta0, fzeta10, fzeta20]
    return fu, fv, fzeta


def sensitivity_length(filt, fu, fv, fzeta):
    '''
    least square method varying length of hexagons
    '''
    N = 2500
    M = 3
    llist = np.arange(1, 30, 1).astype(float)
    llist = np.insert(llist, 0, 0.5)

    error_l = []
    error_l_ci = []
    for l, L in enumerate(llist):
        if l % 10 == 0:
            print(l)
        xi, yi = make_n_hexs(L, 1, N, M)

        ui = np.zeros((N, M))
        vi = np.zeros((N, M))
        zeta_at_mean = np.zeros(N)
        for i in range(N):
            for j in range(xi[1, :].size):
                ui[i, j] = fu(yi[i, j], xi[i, j])
                vi[i, j] = fv(yi[i, j], xi[i, j])
            zeta_at_mean[i] = fzeta(bn.nanmean(yi[i]), bn.nanmean(xi[i]))

        vort_drifters = np.zeros(N)
        for i in range(N):
            vort_drifters[i], _, _ = least_square_method(
                xi[i, :], yi[i, :], ui[i, :], vi[i, :], 'solve')

        #error_l[l] = np.var(zetai_mean-vorti)/np.var(zetai_mean)
        error_l.append(stats.pearsonr(zeta_at_mean, vort_drifters)[0]**2)
        error_l_ci.append(bootstrap_ci(zeta_at_mean, vort_drifters, N))

    df = pd.DataFrame(index=np.asarray(llist))
    df['error'] = np.asarray(error_l)
    df['ci_low'] = np.asarray(error_l_ci)[:, 0]
    df['ci_high'] = np.asarray(error_l_ci)[:, 1]
    df['filter'] = filt

    return df


# %% MAIN
# zgrid_path = 'data/psom/zgrid.out'
# model_path = 'data/psom/full_08325.cdf'
dat = read_model_field(snakemake.input[0], snakemake.input[1])
fu, fv, fzeta = filter_fields(dat)

length_bucket = []
for i,filt in enumerate([0,10,20]):
    length_bucket.append(sensitivity_length(filt, fu=fu[i], fv=fv[i],
                                            fzeta=fzeta[i]))

pd.concat(length_bucket).reset_index().to_feather(str(snakemake.output))
