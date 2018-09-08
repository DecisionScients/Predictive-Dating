
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import math
import matplotlib.pyplot as plt
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
#                               QUANT TABLE                                    #
# ============================================================================ #
def quant_table(df):
    # Produces a table with summary statistics for quantitative variables
    df = df.select_dtypes(include=['float64', 'int64'])
    qt = df.describe().T
    return(qt)

#%%
# ============================================================================ #
#                               QUANT PLOT                                     #
# ============================================================================ #
def quant_plot(df):
    # Produces a table with summary statistics for quantitative variables
    df = df.select_dtypes(include=['float64', 'int64'])
    cols = 3
    cols = math.ceil(len(df.columns) / cols)
    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(cols, rows, figsize=(12,36), sharey= False, 
    tight_layout = True)
    ax = ax.ravel()
    for i in range(len(df.columns)):    
        sns.boxplot(x = df.iloc[:,i],ax = ax[i])
    return(ax)
#%%
# ============================================================================ #
#                               QUAL TABLE                                     #
# ============================================================================ #
def qual_table(df):
    # Produces a table with summary statistics for qualitative variables
    df = df.select_dtypes(include=['object'])
    qt = df.describe().T
    return(qt)

#%%
# ============================================================================ #
#                               QUAL PLOT                                      #
# ============================================================================ #
def qual_plot(df):
    # Produces a table with summary statistics for quantitative variables
    df = df.select_dtypes(include=['object'])
    cols = 3
    cols = math.ceil(len(df.columns) / cols)
    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(cols, rows, figsize=(12,36), sharey= False, 
    tight_layout = True)
    ax = ax.ravel()
    for i in range(len(df.columns)):    
        sns.countplot(x = df.iloc[:,i],ax = ax[i])
    return(ax)

#%%
# =============================================================================
if __name__ == "__main__":
    df = read("train.csv")    
    quant_tbl = quant_table(df)
    quant_plt = quant_plot(df)
    qual_tbl = qual_table(df)
    qual_plt = qual_plot(df)
    


