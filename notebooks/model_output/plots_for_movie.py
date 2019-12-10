#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make plots for movie.

Created on Sat Jan 13 23:37:19 2018
@author: seb
"""

import seaborn as sns
sns.set(style='whitegrid',context='poster')
import glob
import multiprocessing
from model_plotting import plot_snapshot

pool = multiprocessing.Pool()
ndir = glob.glob('./output_2013asiri_04/full_*.cdf')
results = [pool.apply_async( plot_snapshot, args= (ndir[t].split('/')[2], ) ) for t in range(len(ndir))]
pool.close()
pool.join()
