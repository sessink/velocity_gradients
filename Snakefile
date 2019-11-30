include: 'utils.smk'

rule all:
    input:
        expand("viz/{graph}.{fmt}", graph=["rulegraph", "dag"], fmt=["pdf", "png"]),
        # 'data/drifters/posveldata_3h.pkl',
        # 'data/psom/sensitivity_length.feather',
        # 'figures/sensitivity.pdf',
        expand('data/clusters/combinations_lt_10_1h_{split}.nc', split=['00','01','02','03','04','05'])

rule read_data:
    input:
        'data/drifters/asiri_RR1513_data.mat'
    output:
        'data/drifters/posveldata_3h.pkl'
    script:
        'src/data/read_filter_bin_pandas_data.py'

rule convert_to_xr:
    input:
        'data/drifters/posveldata_3h.pkl'
    output:
        'data/drifters/posveldata_xr.nc'
    script:
        'src/data/convert_to_xr.py'

rule initial_length:
    input:
        'data/drifters/posveldata_xr.nc'
    output:
        'data/clusters/initial_lengths.npy'
    script:
        'src/data/compute_initial_lengths.py'

rule compute_grad_u:
    input:
        'data/drifters/posveldata_xr.nc',
        'data/clusters/initial_lengths.npy'
    output:
        'data/clusters/combinations_lt_10_1h_{split}.nc'
    script:
        'src/data/gradu_calculation.py'

# rule sensitivity:
#     input:
#         'data/psom/zgrid.out',
#         'data/psom/full_08325.cdf'
#     output:
#         'data/psom/sensitivity_length.feather',
#         'data/psom/sensitivity_number.feather',
#         'data/psom/sensitivity_aspect.feather'
#     script:
#         'src/model/sensitivity.py'
#
# rule plot_sensitivity:
#     input:
#         'data/psom/sensitivity_length.feather',
#         'data/psom/sensitivity_number.feather',
#         'data/psom/sensitivity_aspect.feather'
#     output:
#         'figures/sensitivity.pdf'
#     script:
#         'src/model/plot_sensitivity.py'
