# Scientific Computing
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks', context='paper')
plt.style.use('sebstyle')

dat = pd.read_feather('data/psom/sensitivity_length.feather')

dat = dat.rename(index=str,columns={'index':'length'})
dat.set_index('length',inplace=True)

dat = dat.pivot(columns='filter')
#%%
plt.figure(figsize=(8, 8))
plt.fill_between(dat.index, dat.ci_low, dat.ci_high, color='grey', alpha=.3)
plt.plot(dat.index, dat.error, '.', label='model')

plt.legend()
plt.xlabel('length scale [km]')
plt.ylabel(r'R squared')
#plt.savefig('error.png', bbox_inches='tight',dpi=400)
plt.show()
