import sys
sys.path.append('../scripts/')

import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import gammaln
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr

from deformtools.haversine import haversine
import scipy.linalg as la
from scipy.spatial import ConvexHull
import bottleneck as bn
import gsw
from scipy.special import comb
from itertools import combinations

import warnings
warnings.simplefilter("ignore",category=FutureWarning)
warnings.simplefilter("ignore",category=RuntimeWarning)
