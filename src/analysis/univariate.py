# =========================================================================== #
#                                 ANALYSIS                                    #
# =========================================================================== #
'''Analysis and inference functions'''

# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import os
import sys
import inspect
sys.path.append("./src")

import itertools
from itertools import combinations
import math
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew
import seaborn as sns
from sklearn import preprocessing
import statistics as stat
import textwrap
import warnings
warnings.filterwarnings('ignore')

from visualization import visual
# %%
# ---------------------------------------------------------------------------- #
#                                    DESCRIBE                                  #
# ---------------------------------------------------------------------------- #


def describe_qual_df(df):
    stats = pd.DataFrame()
    cols = df.columns
    for col in cols:
        d = pd.DataFrame(df[col].describe())
        d = d.T
        d['missing'] = df[col].isna().sum()
        stats = stats.append(d)
    return stats


def describe_quant_df(df):

    stats = pd.DataFrame()
    cols = df.columns
    for col in cols:
        d = pd.DataFrame(df[col].describe())
        d = d.T
        d['skew'] = skew(df[col].notnull())
        d['kurtosis'] = kurtosis(df[col].notnull())
        d['missing'] = df[col].isna().sum()
        c = ['count', 'missing', 'min', '25%', 'mean', '50%',
             '75%', 'max', 'skew', 'kurtosis']
        stats = stats.append(d[c])
    return stats


def describe_qual_x(x):
    d = pd.DataFrame(x.describe())
    d = d.T
    d['missing'] = x.isna().sum()
    return d


def describe_quant_x(x):
    d = pd.DataFrame(x.describe())
    d = d.T
    d['skew'] = skew(x.notnull())
    d['kurtosis'] = kurtosis(x.notnull())
    d['missing'] = x.isna().sum()
    c = ['count', 'missing', 'min', '25%', 'mean', '50%',
         '75%', 'max', 'skew', 'kurtosis']

    return d[c]

# ---------------------------------------------------------------------------- #
#                             UNIVARIATE ANALYSIS                              #
# ---------------------------------------------------------------------------- #
# %%


def univariate(df):
    cols = df.columns
    for col in cols:
        if (df[col].dtype == np.dtype('int64') or
                df[col].dtype == np.dtype('float64')):
            d = describe_quant_x(df[col])
            v = visual.boxplot(df, xvar=col)
        else:
            d = describe_qual_x(df[col])
            v = visual.countplot(xvar=col, df=df)
    return d, v
