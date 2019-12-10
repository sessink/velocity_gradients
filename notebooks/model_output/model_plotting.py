import matplotlib.pyplot as plt
import numpy as np
import gsw
import pandas as pd
from scipy.io import netcdf

def plot_snapshot(cdffile):
    #cdffile = 'full_17400.cdf'
    path = './output_2013asiri_05/'
    zgrid = pd.read_csv(path+'zgrid.txt', skipinitialspace=True, sep=' ', header=None)
    zgrid = zgrid[1].values[1:33]
    time = cdffile.split('.')[0].split('_')[1]
    
    swapaxesint = int( cdffile.split('_')[1].split('.')[0] )
    with netcdf.netcdf_file(path+cdffile,'r') as f:
        # print(f.variables)
        temptop = f.variables['temp'][:][32, 1:321, 1:193].copy()
        tempface = f.variables['temp'][:][1:33, 1:321, 1].copy()
        saltop = f.variables['s'][:][32, 1:321, 1:193].copy()
        salface = f.variables['s'][:][1:33, 1:321, 1].copy()
        rhotop = f.variables['rho'][:][32, 1:321, 1:193].copy()
        rhoface = f.variables['rho'][:][1:33, 1:321, 1].copy()
        vortop = f.variables['vor'][:][32, 1:321, 1:193].copy()/ gsw.f(17)
        vorface = f.variables['vor'][:][1:33, 1:321, 1].copy()/ gsw.f(17)
        x = f.variables['xc'][1:193].copy()
        y = f.variables['yc'][1:321] .copy()

        fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(16, 10))

        if  swapaxesint > 10000:
            v = np.linspace(17, 29.2, 101)
        else:
            v = np.linspace(17, 29.2, 101)
        ax1.contourf(y, x, temptop.T, v, cmap='viridis',extend='both')
        ax1.set(title='Temperature %s' % time, ylabel='x', xticks=[])
        h1 = ax2.contourf(y, zgrid, tempface, v, cmap='viridis', extend='both')
        CS = ax2.contour(y, zgrid, rhoface-1000)
        plt.clabel(CS, inline=1, fontsize=10,fmt='%2.0f')
        ax2.set(xlabel='y', ylabel='z', ylim=(-150, 0))
        cbar_ax = fig.add_axes([0.36, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(h1, cax=cbar_ax,ticks=np.arange(16,30,2))
        cbar.ax.set_xticklabels(np.arange(16,30,2))

        v = np.linspace(31.5, 35, 101)
        ax3.contourf(y, x, saltop.T, v, cmap='viridis',extend='both')
        ax3.set(title='Salinity %s' % time, yticks=[], xticks=[])
        h2 = ax4.contourf(y, zgrid, salface, v, cmap='viridis',extend='both')
        CS = ax4.contour(y, zgrid, rhoface-1000)
        plt.clabel(CS, inline=1, fontsize=10,fmt='%2.0f')
        ax4.set(xlabel='y', ylim=(-150, 0), yticks=[])

        cbar_ax = fig.add_axes([0.635, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(h2, cax=cbar_ax,ticks=np.arange(30,36,1))
        cbar.ax.set_xticklabels(np.arange(30,36,1))

        v = np.linspace(-4, 4, 11)
        ax5.contourf(y, x, vortop.T, v, cmap='RdBu_r', extend='both')
        ax5.set(title='Vorticity %s' % time, yticks=[], xticks=[])
        h3 = ax6.contourf(y, zgrid, vorface, v, cmap='RdBu_r', extend='both')
        CS = ax6.contour(y, zgrid, rhoface-1000)
        plt.clabel(CS, inline=1, fontsize=10,fmt='%2.0f')
        ax6.set(xlabel='y', ylim=(-150, 0), yticks=[])

        cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(h3, cax=cbar_ax,ticks=np.arange(-4,5,2))
        cbar.ax.set_xticklabels(np.arange(-4,5,2))

        plt.subplots_adjust(wspace=0.2, hspace=0.10)

        filename='update_s_t_vor_%s.png' % time
        plt.savefig(path+'/figures/'+filename, bbox_inches='tight')
        plt.clf()
        #print('plotted')
    return filename

def plot_single_snapshot(cdffile):
    #cdffile = 'full_17400.cdf'
    path = './output_2013asiri_05/'
    zgrid = pd.read_csv(path+'zgrid.txt', skipinitialspace=True, sep=' ', header=None)
    zgrid = zgrid[1].values[1:33]
    time = cdffile.split('.')[0].split('_')[1]
    
    swapaxesint = int( cdffile.split('_')[1].split('.')[0] )
    with netcdf.netcdf_file(path+cdffile,'r') as f:
        # print(f.variables)
        temptop = f.variables['temp'][:][32, 1:321, 1:193]
        tempface = f.variables['temp'][:][1:33, 1:321, 1]
        saltop = f.variables['s'][:][32, 1:321, 1:193]
        salface = f.variables['s'][:][1:33, 1:321, 1]
        rhotop = f.variables['rho'][:][32, 1:321, 1:193]
        rhoface = f.variables['rho'][:][1:33, 1:321, 1]
        vortop = f.variables['vor'][:][32, 1:321, 1:193] / gsw.f(17)
        vorface = f.variables['vor'][:][1:33, 1:321, 1] / gsw.f(17)
        x = f.variables['xc'][1:193]
        y = f.variables['yc'][1:321] 

        fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(16, 10))

        if  swapaxesint > 10000:
            v = np.linspace(17, 24.2, 101)
        else:
            v = np.linspace(17, 29.2, 101)
        ax1.contourf(y, x, temptop.T, v, cmap='viridis',extend='both')
        ax1.set(title='Temperature %s' % time, ylabel='x', xticks=[])
        h1 = ax2.contourf(y, zgrid, tempface, v, cmap='viridis', extend='both')
        CS = ax2.contour(y, zgrid, rhoface-1000)
        plt.clabel(CS, inline=1, fontsize=10,fmt='%2.0f')
        ax2.set(xlabel='y', ylabel='z', ylim=(-150, 0))
        cbar_ax = fig.add_axes([0.36, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(h1, cax=cbar_ax,ticks=np.arange(16,30,2))
        cbar.ax.set_xticklabels(np.arange(16,30,2))

        v = np.linspace(31.5, 35, 101)
        ax3.contourf(y, x, saltop.T, v, cmap='viridis',extend='both')
        ax3.set(title='Salinity %s' % time, yticks=[], xticks=[])
        h2 = ax4.contourf(y, zgrid, salface, v, cmap='viridis',extend='both')
        CS = ax4.contour(y, zgrid, rhoface-1000)
        plt.clabel(CS, inline=1, fontsize=10,fmt='%2.0f')
        ax4.set(xlabel='y', ylim=(-150, 0), yticks=[])

        cbar_ax = fig.add_axes([0.635, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(h2, cax=cbar_ax,ticks=np.arange(30,36,1))
        cbar.ax.set_xticklabels(np.arange(30,36,1))

        v = np.linspace(-4, 4, 11)
        ax5.contourf(y, x, vortop.T, v, cmap='RdBu_r', extend='both')
        ax5.set(title='Vorticity %s' % time, yticks=[], xticks=[])
        h3 = ax6.contourf(y, zgrid, vorface, v, cmap='RdBu_r', extend='both')
        CS = ax6.contour(y, zgrid, rhoface-1000)
        plt.clabel(CS, inline=1, fontsize=10,fmt='%2.0f')
        ax6.set(xlabel='y', ylim=(-150, 0), yticks=[])

        cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(h3, cax=cbar_ax,ticks=np.arange(-4,5,2))
        cbar.ax.set_xticklabels(np.arange(-4,5,2))

        plt.subplots_adjust(wspace=0.2, hspace=0.10)

        filename='update_s_t_vor_%s.png' % time
        plt.savefig(path+'/figures'+filename, bbox_inches='tight')
        print('plotted')
    return filename