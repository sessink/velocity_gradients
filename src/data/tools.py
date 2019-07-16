def latlon2uv_dir(lon, lat, direction):
    ''' forward/backward differences

        convert time series of lon and lat into time series of u and v
        using: haversine

        lat,lon: pandas columns,degrees
        u,v: vectores, m/s
    '''
    import numpy as np
    from haversine import haversine

    dt = 60 * 60  # now in second
    lon2 = lon.reindex(index=np.roll(lon.index, -1)).values
    lat2 = lat.reindex(index=np.roll(lat.index, -1)).values
    lon = lon.values
    lat = lat.values
    if direction == 'back':
        lon = np.flipud(lon)
        lon2 = np.flipud(lon2)
        lat = np.flipud(lat)
        lat2 = np.flipud(lat2)

    dr = np.array([
        haversine([lon[i], lat[i]], [lon2[i], lat2[i]])
        for i in range(len(lon))
    ])

    xx = np.sin(np.deg2rad(lon2 - lon)) * np.cos(np.deg2rad(lat2))
    yy = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lat2)) - np.sin(
        np.deg2rad(lat)) * np.cos(np.deg2rad(lat2)) * np.cos(
            np.deg2rad(lon - lon))

    gamma = np.arctan2(yy, xx)

    if direction == 'back':
        dr = np.flipud(dr)
        gamma = np.flipud(gamma)

    c = 1000
    u = c * dr / dt * np.cos(gamma)
    v = c * dr / dt * np.sin(gamma)
    return u, v


def latlon2uv(lon, lat):
    '''
    apply both forward and backward differences to get centered difference
    '''
    u1, v1 = latlon2uv_dir(lon, lat, 'for')
    u2, v2 = latlon2uv_dir(lon, lat, 'back')

    # correct the edges!
    u1[-1] = u2[0]
    v1[-1] = v2[0]
    u2[-1] = u1[0]
    v2[-1] = v1[0]

    uv = 0.5 * (u1 + u2) + 1j * 0.5 * (v1 + v2)
    return uv


def datenum2datetime(matlab_datenum):
    """Convert Matlab datenum to python datetime object
        at this point, only size-1 skalar can be converted.
    """
    from datetime import datetime, timedelta
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(
        days=matlab_datenum % 1) - timedelta(days=366)


def mem_usage(pandas_obj):
    ''' Output memory usage of pandas object
    '''
    import pandas as pd
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024**2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def optimize_df(df):
    '''
    decrease dataframe filesize by optimizing float format
    '''
    import pandas as pd
    import numpy as np
    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    optimized_df = df.copy()
    optimized_df[converted_float.columns] = converted_float
    optimized_df.drop(optimized_df[~np.isfinite(optimized_df.lat)].index,
                      inplace=True)
    return optimized_df
