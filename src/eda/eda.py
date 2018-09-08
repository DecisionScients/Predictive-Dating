# =========================================================================== #
#                                     EDA                                     #
# =========================================================================== #

#%%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import settings

# --------------------------------------------------------------------------- #
#                                    READ                                     #
# --------------------------------------------------------------------------- #
def read(file_name):    
    # Imports training data into a pandas DataFrame.   
    df = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, file_name), 
    encoding = "Latin-1", low_memory = False)
    df = df[['attractive_partner','sincere_partner', 'intelligence_partner',
             'funny_partner', 'ambition_partner', 'interests_correlate', 
             'decision']]
    return(df)

# --------------------------------------------------------------------------- #
#                                  PLOT                                       #
# --------------------------------------------------------------------------- #
def eda_plot(df):
    # Plotting Parameters
    mpl.rc('xtick', labelsize=20) 
    mpl.rc('ytick', labelsize=20) 

    fig, ax = plt.subplots(2,3)

    sns.set(style="whitegrid", font_scale=1.5)
    ax[0,0] = sns.boxplot(x='decision', y = 'attractive_partner', data = df, ax = ax[0,0])
    ax[0,0].set_title("Attractive",fontsize=20)
    ax[0,0].set_xlabel("Decision", fontsize=20)
    ax[0,0].set_ylabel("Attractive", fontsize=20)

    ax[0,1] = sns.boxplot(x='decision', y = 'intelligence_partner', data = df, ax = ax[0,1])
    ax[0,1].set_title("Intelligence", fontsize=20)
    ax[0,1].set_xlabel("Decision", fontsize=20)
    ax[0,1].set_ylabel("Intelligence", fontsize=20)

    ax[0,2] = sns.boxplot(x='decision', y = 'ambition_partner', data = df, ax = ax[0,2])
    ax[0,2].set_title("Ambition", fontsize=20)
    ax[0,2].set_xlabel("Decision", fontsize=20)
    ax[0,2].set_ylabel("Ambition", fontsize=20)

    ax[1,0] = sns.boxplot(x='decision', y = 'sincere_partner', data = df, ax = ax[1,0])
    ax[1,0].set_title("Sincerity", fontsize=20)
    ax[1,0].set_xlabel("Decision", fontsize=20)
    ax[1,0].set_ylabel("Sincerity", fontsize=20)

    ax[1,1] = sns.boxplot(x='decision', y = 'funny_partner', data = df, ax = ax[1,1])
    ax[1,1].set_title("Funny", fontsize=20)
    ax[1,1].set_xlabel("Decision", fontsize=20)
    ax[1,1].set_ylabel("Funny", fontsize=20)

    ax[1,2] = sns.boxplot(x='decision', y = 'interests_correlate', data = df, ax = ax[1,2])
    ax[1,2].set_title("Interests", fontsize=20)
    ax[1,2].set_xlabel("Decision", fontsize=20)
    ax[1,2].set_ylabel("Interests", fontsize=20)
    return fig, ax

# --------------------------------------------------------------------------- #
#                                 MAIN                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df = read("train.csv")    
    eda_plot(df)
