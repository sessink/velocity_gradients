import numpy as np

def calc_aspect(xs, ys):
    import bottleneck as bn

    points = np.vstack([xs, ys])
    cov = np.cov(points)
    w, v = np.linalg.eig(cov)
    return bn.nanmin(w) / bn.nanmax(w)

def bootstrap_ci(array):
    import bottleneck as bn
    from sklearn.utils import resample
    import scipy.stats as stats

    N = array.size
    n_iterations = 1000
    n_size = int(N)
    alpha = 0.95  # 95% confidence interval

    statss = []
    for i in range(n_iterations):
        sample_ind = resample(np.arange(N), n_samples=n_size)
        stat = bn.nanmean(array[sample_ind])
        statss.append(stat)

    ordered = np.sort(statss)
    lower = np.percentile(ordered, 100 * (1 - alpha) / 2)
    upper = np.percentile(ordered, 100 * (alpha + (1 - alpha) / 2))
    return lower, upper

def plot_ci(array,ax,**kwargs):
    return ax.fill_between(dat.time.values,dat[array].mean(axis=0)-dat[array].std(axis=0),
                          dat[array].mean(axis=0)+dat[array].std(axis=0),**kwargs)

def alphabet(ax):
    for j, axx in enumerate(ax):
        axx.annotate(chr(j+65), (0, 1.02),
                     xycoords='axes fraction',
                     weight='bold')

def get_percentiles(array):
    alpha = 0.95
    return np.nanpercentile(array, 100 * (1 - alpha) / 2), np.nanpercentile(array, 100 * (alpha + (1 - alpha) / 2))

def get_ci(dat,array):
    per = []
    for t in dat.time:
        per.append(get_percentiles(dat[array].sel(time=t)))
    per = np.array(per)
    return per[:,0],per[:,1]

def da_median(dat,array):
    ''' Compute median of chunked datarray
    '''
    per = []
    for t in dat.time:
        # per.append(np.nanpercentile(dat[array].sel(time=t), 50))
        per.append(dat[array].sel(time=t).chunk({'clusters': -1}).quantile(0.5))
    return np.array(per)


def r2(x, y):
    import scipy.stats as stats
    return stats.pearsonr(x, y)[0]**2

def make_hex(x0, y0, L, skew, M):
    '''
    make a polygon with M vertices, length L, and skewness skew
    '''
    import random
    import math

    xx = np.zeros(M)
    yy = np.zeros(M)

    degstart = random.randint(0, 360)
    angle = np.arange(degstart, degstart + 360, 360 / M)
    angle[angle > 360] = angle[angle > 360] - 360

#     random_stretch = random.randint(0, 2)
    i = 0
#     offset =  random.randint(0, 180)
    for j, angle in enumerate(angle):
        La = L*skew
        Lb = L
        xx[i] = La * math.cos(math.radians(angle))
        yy[i] = Lb * math.sin(math.radians(angle))
        i += 1


    theta = np.radians(random.randint(0, 360))
    r = np.array(( (np.cos(theta), -np.sin(theta)),
                 (np.sin(theta),  np.cos(theta)) ))

    A = r.dot([xx,yy])

    return A[0,:]+x0, A[1,:]+y0

def make_n_hexs(L, skew, N, M):
    import random
    x = np.zeros((N, M))
    y = np.zeros((N, M))
    for i in range(N):
        x0 = (180 - 10) * random.random() + 10
        y0 = (300 - 10) * random.random() + 10
        x[i, :], y[i, :] = make_hex(x0, y0, L, skew, M)
    return x, y
