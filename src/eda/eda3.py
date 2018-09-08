
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import analysis
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns
import settings
import visual

#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
df = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, 'train.csv'), 
encoding = "Latin-1", low_memory = False)
 
#%%
# ============================================================================ #
# Gender Selectivity                                                           #
# ============================================================================ #
# Data
g = df[['gender', 'decision']]
g = g.groupby(['gender']).agg({'decision':'mean'}).reset_index()

# Plot
visual.bar_plot(g, xval='gender', yval='decision', title= 'Positive Decision Proportion by Gender')

# Independence
x2 = analysis.Independence()
x2.test(x = df['gender'], y = df['decision'])
x2.summary()

#%%0
# ============================================================================ #
# Gender and Race Selectivity                                                  #
# ============================================================================ #
# Data
g = df[['gender', 'race', 'decision']]
g = g.groupby(['gender', 'race']).agg({'decision':'mean'}).reset_index()

# Plot
sns.set_palette("GnBu_d")
ax = sns.catplot(x='gender', y='decision', hue='race', 
                         data=g, kind='bar')
ax.fig.suptitle("Positive Decision Proportion by Gender and Race")

# Independence
x2.test(x = df['race'], y = df['decision'])
x2.summary()

#%%
# ---------------------------------------------------------------------------- #
# Gender and Race Selectivity                                                  #
# ---------------------------------------------------------------------------- #
# Chi Square test of significant differences in yes rates among females
female = df.loc[df['gender'] == 'female', ['race', 'decision']]
x2.test(x = female['race'], y = female['decision'])
x2.summary()
x2.post_hoc()

# Chi Square test of significant differences in yes rates among males
male = df.loc[df['gender'] == 'male', ['race', 'decision']]
x2.test(x = male['race'], y = male['decision'])
x2.summary()

#%%



#%%
# ============================================================================ #
# Model: Gender and Race Selectivity by Race: Asian                            #
# ============================================================================ #
# Plot Data 
g = df.loc[df['race'] == 'asian', ['gender', 'race_o', 'decision']]
g = g.groupby(['gender', 'race_o']).agg({'decision':'mean'}).reset_index()
sns.set_palette("GnBu_d")
ax = sns.catplot(x='gender', y='decision', hue='race_o', 
                         data=g, kind='bar')
ax.fig.suptitle("Asian Positive Decision Proportion by Race and Gender")

#%%
# ============================================================================ #
# Model: Gender and Race Selectivity by Race: Black                            #
# ============================================================================ #
# Plot Data 
g = df.loc[df['race'] == 'black', ['gender', 'race_o', 'decision']]
g = g.groupby(['gender', 'race_o']).agg({'decision':'mean'}).reset_index()
sns.set_palette("GnBu_d")
ax = sns.catplot(x='race_o', y='decision', hue='gender', 
                         data=g, kind='bar')
ax.fig.suptitle("Black Positive Decision Proportion by Race and Gender")
