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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import partial_corr
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import textwrap

sys.path.append("../../src")
from shared import directories
from shared import filenames
from shared import variables
sys.path.append(directories.ANALYSIS_DIR)

import independence
import description
import visual


# %%
# ---------------------------------------------------------------------------- #
#                                 ANALYSIS                                     #
# ---------------------------------------------------------------------------- #


def analysis(df, x, y, z):

    k = independence.Kruskal()
    a = independence.Anova()

    if ((df[x].dtype == np.dtype('int64') or df[x].dtype == np.dtype('float64')) and
            (df[y].dtype == np.dtype('int64') or df[y].dtype == np.dtype('float64'))):
        dx = description.group_describe(df, x, z)
        dy = description.group_describe(df, y, z)
        desc = pd.concat([dx, dy], axis=0)
        ind = independence.corr_table(df, x=[x], y=[y], target=[z])
        plots = visual.condition(df=df, x=x, y=y, z=z)
    elif ((df[x].dtype == np.dtype('object')) and (df[y].dtype == np.dtype('object'))):
        ind = independence.association(df, x, y, z)
        desc = pd.crosstab(df[z], [df[x], df[y]], rownames=[z],
                           colnames=[x, y], margins=False, normalize='columns')
        plots = mosaic(df, [x, y, z])
        plots = plots[0]
    elif ((df[x].dtype == np.dtype('int64') or df[x].dtype == np.dtype('float64')) and
            (df[y].dtype == np.dtype('object'))):
        ind = independence.factorial_anova(df, y, x, z)
        desc = description.group_describe(df, y, x, z)
        plots = visual.boxplot(df, x=x, y=y, hue=z)
    else:
        ind = independence.factorial_anova(df, x, y, z)
        desc = description.group_describe(df, x, y, z)
        plots = visual.boxplot(df, x=y, y=x, hue=z)
    return ind, desc, plots


def race_match(df):
    races = df['race'].unique()
    rm = pd.DataFrame()
    for r in races:
        df_r = df[df['race'] == r]
        ct = pd.crosstab(df_r['dec'], [df_r['gender'], df_r['race_o']], rownames=['dec'],
                         colnames=['gender', 'race_o'], normalize='columns')
        y = ct.loc['Yes'].rename(r)
        rm = rm.append(y)
    fix, ax = plt.subplots()
    sns.set(style="whitegrid", font_scale=1)
    sns.heatmap(rm, annot=True, ax=ax)
    return(rm, ax)


# %%
# df = pd.read_csv(os.path.join('../', directories.INTERIM_DATA_DIR,
#                               filenames.INTERIM_FILENAME),
#                  encoding="Latin-1", low_memory=False)
# race_match(df)
