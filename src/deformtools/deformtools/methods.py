def least_square_method(x0, y0, u0, v0, method):
    '''
    Least square method to estimate velocity gradients
    '''
    import numpy as np
    import scipy.linalg as la
    import gsw
    ncc = x0.size
    dlon= (x0- np.nanmean(x0)) * 1000
    dlat= (y0 - np.nanmean(y0)) * 1000
    f = gsw.f(17)

    R = np.mat(np.vstack((np.ones((ncc, )), dlon, dlat)).T)
    u0 = np.mat(u0).T - np.nanmean(u0)
    v0 = np.mat(v0).T - np.nanmean(v0)

    if method is 'lstsq':
        A, _, _, _ = la.lstsq(R, u0)
        B, _, _, _ = la.lstsq(R, v0)
    elif method is 'inv':
        A = np.linalg.inv(R.T * R) * R.T * u0
        B = np.linalg.inv(R.T * R) * R.T * v0
    elif method is 'solve':
        A = np.linalg.solve(R, u0)
        B = np.linalg.solve(R, v0)

    vort = (B[1] - A[2]) / f
    strain = np.sqrt((A[1] - B[2])**2 + (B[1] + A[2])**2) / f
    div = (A[1] + B[2]) / f

    return vort, strain, div


def filter_fields(dat):
    '''
    Filter and interpolate fields
    '''
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter

    dat['vor_10km'] = (('x', 'y'),
                       gaussian_filter(dat.vor, sigma=5, mode='wrap'))
    dat['u_10km'] = (('x', 'y'), gaussian_filter(dat.u, sigma=5, mode='wrap'))
    dat['v_10km'] = (('x', 'y'), gaussian_filter(dat.v, sigma=5, mode='wrap'))
    dat['vor_20km'] = (('x', 'y'),
                       gaussian_filter(dat.vor, sigma=10, mode='wrap'))
    dat['u_20km'] = (('x', 'y'), gaussian_filter(dat.u, sigma=10, mode='wrap'))
    dat['v_20km'] = (('x', 'y'), gaussian_filter(dat.v, sigma=10, mode='wrap'))

    fu0 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.u)
    fv0 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.v)
    fzeta0 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.vor)
    # 10km
    fu10 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.u_10km)
    fv10 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.v_10km)
    fzeta10 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.vor_10km)
    # 20km
    fu20 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.u_20km)
    fv20 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.v_20km)
    fzeta20 = interpolate.RectBivariateSpline(dat.x, dat.y, dat.vor_20km)

    fu = [fu0, fu10, fu20]
    fv = [fv0, fv10, fv20]
    fzeta = [fzeta0, fzeta10, fzeta20]

    return fu, fv, fzeta
