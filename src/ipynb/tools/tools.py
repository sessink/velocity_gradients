def calc_aspect(xs, ys):
    import numpy as np
    import bottleneck as bn

    points = np.vstack([xs, ys])
    cov = np.cov(points)
    w, v = np.linalg.eig(cov)
    return bn.nanmin(w) / bn.nanmax(w)

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
