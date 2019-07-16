include: 'utils.smk'

rule all:
    input:
        expand("viz/{graph}.{fmt}", graph=["rulegraph", "dag"], fmt=["pdf", "png"]),
        'data/drifters/posveldata_3h.pkl',
        'data/psom/sensitivity_length.feather'

rule read_data:
    input:
        'data/drifters/asiri_RR1513_data.mat'
    output:
        'data/drifters/posveldata_3h.pkl'
    script:
        'src/data/read_filter_bin_pandas_data.py'

rule sensitivity:
    input:
        'data/psom/zgrid.out',
        'data/psom/full_08325.cdf',

    output:
        'data/psom/sensitivity_length.feather'
        # 'sensitivity_aspect.feather'
        # 'sensitivity_number.feather'
    script:
        'src/model/sensitivity.py'

rule plot_sensitivity:
    input:
        'figures/sensitivity.nc'
    output:
        'figures/sensitivity.pdf'
    script:
        'src/model/plot_sensitivity.py'
