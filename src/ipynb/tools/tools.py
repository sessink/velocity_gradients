def read_model_field(zgrid_path, model_path):
    import pandas as pd
    import xarray as xr
    
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

def calc_aspect(xs, ys):
    import numpy as np
    import bottleneck as bn
    
    points = np.vstack([xs, ys])
    cov = np.cov(points)
    w, v = np.linalg.eig(cov)
    return bn.nanmin(w) / bn.nanmax(w)


def least_square_method(x0, y0, u0, v0, switch):
    # TODO: vectorize this?
    import numpy as np
    import scipy.linalg as la
    import gsw

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
    import numpy as np
    from sklearn.utils import resample
    import scipy.stats as stats
    
    n_iterations = 1000
    n_size = int(N * 0.50)
    alpha = 0.95  # 95% confidence interval

    statss = []
    for i in range(n_iterations):
        sample_ind = resample(np.arange(N), n_samples=n_size)
        error = stats.pearsonr(zetai_mean[sample_ind], vorti[sample_ind])[0]**2
        statss.append(error)

    ordered = np.sort(statss)
    lower = np.percentile(ordered, 100 * (1 - alpha) / 2)
    upper = np.percentile(ordered, 100 * (alpha + (1 - alpha) / 2))
    return lower, upper


def r2(x, y):
    import scipy.stats as stats
    return stats.pearsonr(x, y)[0]**2


def filter_fields(dat):
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter
    
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

def make_hex(x0, y0, L, skew, M):
    '''
    make a polygon with M vertices, length L, and skewness skew
    '''
    import numpy as np
    import random
    import math
    
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
    import random
    import numpy as np
    x = np.zeros((N, M))
    y = np.zeros((N, M))
    for i in range(N):
        x0 = (180 - 10) * random.random() + 10
        y0 = (300 - 10) * random.random() + 10
        x[i, :], y[i, :] = make_hex(x0, y0, L, skew, M)
    return x, y