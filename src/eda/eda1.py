# =========================================================================== #
#                                     EDA                                     #
# =========================================================================== #
'''
This module contains the functions that produce the univariate exploratory data
analysis for the Predictive Dating project.
'''
# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import os
import sys
import inspect
home = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "src")
sys.path.append(home)
current = os.path.join(home, "eda")
sys.path.append(current)

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import seaborn as sns

import analysis
from shared import directories
from shared import filenames
from shared import variables
from visualization import visual

# --------------------------------------------------------------------------- #
#                          QUALITATIVE ANALYSIS                               #
# --------------------------------------------------------------------------- #


def qualitative(df):
    qual = df.select_dtypes(include=['object'])
    visual.multi_countplot(qual, ncols=3, width=10, title="Categorical Variable "
                           "Frequency Analysis ")
    return(qual.describe().T)


def quantitative(df):
    quant = df.select_dtypes(include=['int', 'float64'])
    visual.multi_histogram(quant, ncols=3, width=10, height=20, title="Quantitative "
                           "Variable Histogram Analysis ")
    visual.multi_boxplot(quant, ncols=3, width=10, height=20, title="Quantitative "
                         "Variable Boxplot Analysis ")
    return(quant.describe().T)


# ============================================================================ #
#                                CORRELATION                                   #
# ============================================================================ #

def corrplot(df):
    df = df.select_dtypes(include=['float64', 'int64'])
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlation between Variables")
    return(ax)


# ============================================================================ #
#                                ASSOCIATION                                   #
# ============================================================================ #

def assocplot(df):
    df = df.select_dtypes(include=['object'])
    cols = list(df.columns.values)
    corrM = np.zeros((len(cols), len(cols)))

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = analysis.cramers_corrected_stat(
            pd.crosstab(df[col1], df[col2]))
        corrM[idx2, idx1] = corrM[idx1, idx2]

    assoc = pd.DataFrame(corrM, index=cols, columns=cols)

    mask = np.zeros_like(assoc, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(assoc, mask=mask, ax=ax, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=True, yticklabels=True)
    ax.set_title("Cramer's V Association between Variables")
    return(ax)


# --------------------------------------------------------------------------- #
#                                 MAIN                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(directories.INTERIM_DATA_DIR, filenames.TRAIN_FILENAME),
                     encoding="Latin-1", low_memory=False)
    qual = qualitative(df)
    quant = quantitative(df)
    rt = analysis.corrtable(df, threshold=0.4)
    visual.print_df(rt)
    '''r = corrplot(df)
    a = assocplot(df)'''
