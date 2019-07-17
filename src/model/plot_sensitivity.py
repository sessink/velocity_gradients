# Scientific Computing
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set(style='ticks', context='paper')
mpl.rc('figure', dpi=100, figsize=[7.8, 3.5])
mpl.rc('savefig', dpi=500, bbox='tight')
mpl.rc('legend', frameon=False)

sns.set(style='ticks', context='paper')

def read_feather(infile):
    dat = pd.read_feather(infile)
    dat = dat.rename(index=str,columns={'index':'length'})
    dat.set_index('length',inplace=True)
    return dat

#%%
lengthfile = 'data/psom/sensitivity_length.feather'
numberfile = 'data/psom/sensitivity_number.feather'
aspectfile = 'data/psom/sensitivity_aspect.feather'

f,ax = plt.subplots(1,3,sharey=True)

# dat = read_feather(snakemake.input[0])
dat = read_feather(lengthfile)
for filt in dat['filter'].unique():
    temp = dat[dat['filter']==filt]
    ax[0].fill_between(temp.index, temp.ci_low, temp.ci_high, color='grey', alpha=.3)
    ax[0].plot(temp.index, temp.error, label=f'model {filt:02d}km')
ax[0].legend()
ax[0].set_xlabel('length scale [km]')
ax[0].set_ylabel(r'R squared')

# dat = read_feather(snakemake.input[1])
dat = read_feather(numberfile)
for filt in dat['filter'].unique():
    temp = dat[dat['filter']==filt]
    ax[1].fill_between(temp.index, temp.ci_low, temp.ci_high, color='grey', alpha=.3)
    ax[1].plot(temp.index, temp.error, label=f'model {filt:02d}km')
ax[1].legend()
ax[1].set_xlabel('number of drifters')

# dat = read_feather(snakemake.input[2])
dat = read_feather(aspectfile)
for filt in dat['filter'].unique():
    temp = dat[dat['filter']==filt]
    ax[2].fill_between(temp.index, temp.ci_low, temp.ci_high, color='grey', alpha=.3)
    ax[2].plot(temp.index, temp.error, label=f'model {filt:02d}km')
ax[2].legend()
ax[2].set_xlabel(r'aspect ratio $\alpha$')

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVQXYZ'
for j,axx in enumerate(ax):
    axx.annotate(alphabet[j],(0,1.02),xycoords='axes fraction',weight='bold')

# plt.savefig('figures/sensitivity.pdf')
plt.tight_layout()
plt.show()
