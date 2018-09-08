
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import analysis
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import settings

#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
def read(file_name):    
    # Imports training data into a pandas DataFrame.   
    df = pd.read_csv(os.path.join(settings.INTERIM_DATA_DIR, file_name), 
    encoding = "Latin-1", low_memory = False)
    return(df)

#%%
# ============================================================================ #
#                                CORRELATION                                   #
# ============================================================================ #

def correlation(df):    
    df = df.select_dtypes(include=['float64', 'int64'])
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, ax = ax, mask = mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            xticklabels=True, yticklabels=True)
    ax.set_title("Correlation between Variables")
    return(ax)

#%%
# ============================================================================ #
#                                ASSOCIATION                                   #
# ============================================================================ #
def association(df):
    df = df.select_dtypes(include=['object'])
    cols = list(df.columns.values)
    corrM = np.zeros((len(cols),len(cols)))

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = analysis.cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        corrM[idx2, idx1] = corrM[idx1, idx2]

    assoc = pd.DataFrame(corrM, index=cols, columns=cols)
        
        
    mask = np.zeros_like(assoc, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(assoc, mask=mask, ax = ax, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            xticklabels=True, yticklabels=True)
    ax.set_title("Cramer's V Association between Variables")
    return(ax)


#%%
# =============================================================================
if __name__ == "__main__":
    df = read("train.csv")
    corrPlot = correlation(df)
    assocPlot = association(df)


