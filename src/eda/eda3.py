# =========================================================================== #
#                                     EDA                                     #
# =========================================================================== #
'''
This module performs a bivariate analysis of examines measures of age, 
preference, self-assessments, and partner assessments by gender, race and
decision.
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
#                                  DATA                                       #
# --------------------------------------------------------------------------- #
df = pd.read_csv(os.path.join(
    directories.INTERIM_DATA_DIR, filenames.TRAIN_FILENAME))

# --------------------------------------------------------------------------- #
#                         PREFERENCE SCATTERPLOTS                             #
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=[12, 4])
fig.suptitle('Preferences of Partner')
sns.scatterplot(x='pref_o_attractive', y='pref_o_sincere',
                palette='blues', data=df, ax=axes[0])
sns.scatterplot(x='pref_o_attractive', y='pref_o_ambitious',
                palette='blues', data=df, ax=axes[1])
sns.scatterplot(x='pref_o_attractive', y='pref_o_shared_interests',
                palette='blues', data=df, ax=axes[2])

# --------------------------------------------------------------------------- #
#                     PREFERENCE CORRELATION BARPLOTS                         #
# --------------------------------------------------------------------------- #
vars = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
        "pref_o_funny",	"pref_o_ambitious",	"pref_o_shared_interests"]
rt = analysis.corrtable(df[vars])
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
visual.bar_plot(rt, xvar='Correlation', yvar='Pair',
                title='Preference Correlations')

# --------------------------------------------------------------------------- #
#                       QUALITY CORRELATION BARPLOTS                          #
# --------------------------------------------------------------------------- #
vars = ["attractive_o",	"sincere_o",	"intelligence_o",	"funny_o",
        "ambitous_o",	"shared_interests_o"]
rt = analysis.corrtable(df[vars])
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
visual.bar_plot(rt, xvar='Correlation', yvar='Pair',
                title='Quality Correlations')


# --------------------------------------------------------------------------- #
#                    CORRELATION with DECISION BARPLOTS                       #
# --------------------------------------------------------------------------- #
rt = analysis.corrtable(df, target='decision_o', threshold=0.2)
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
visual.bar_plot(rt, xvar='Correlation', yvar='Pair',
                title='Correlations with Decision')
