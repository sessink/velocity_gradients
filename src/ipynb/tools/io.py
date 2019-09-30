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

def read_feather(infile):
    import pandas as pd
    dat = pd.read_feather(infile)
    dat = dat.rename(index=str,columns={'index':'length'})
    dat.set_index('length',inplace=True)
    return dat
