# =========================================================================== #
#                                     EDA                                     #
# =========================================================================== #
'''
This module contains the functions that produce the univariate exploratory data
analysis for the Predictive Dating project.
'''
# %%
# =========================================================================== #
#                                 LIBRARIES                                   #
# =========================================================================== #
import os
import sys
import inspect
home = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "src")
sys.path.append(home)
current = os.path.join(home, "eda")
sys.path.append(current)

import pandas as pd
import seaborn as sns

from shared import directories
from shared import filenames
from shared import variables
from visualization import visual
# ============================================================================ #
#                                    DATA                                      #
# ============================================================================ #
df = pd.read_csv(os.path.join(directories.INTERIM_DATA_DIR,
                              filenames.TRAIN_FILENAME),
                 encoding="Latin=1", low_memory=False)
# %%
# ============================================================================ #
#                                CORRELATION                                   #
# ============================================================================ #
visual.correlation(df)
visual.corrtable(df, threshold=0.4)

# %%
# --------------------------------------------------------------------------- #
#                      STRONG PREFERENCE CORRELATIONS                         #
# --------------------------------------------------------------------------- #
vars = ["attractive_important",	"sincere_important",
        "ambition_important", "shared_interests_important"]
sc = df[vars]
sns.pairplot(sc, kind="scatter")

# %%
# --------------------------------------------------------------------------- #
#                     PREFERENCE CORRELATION BARPLOTS                         #
# --------------------------------------------------------------------------- #
vars = ["attractive_important",	"sincere_important",
        "ambition_important", "shared_interests_important"]
rt = visual.corrtable(df[vars])
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
sns.barplot(x='Correlation', y='Pair', data=rt, hue='Strength',
            dodge=False).set_title('Preference Correlations')
# %%
# --------------------------------------------------------------------------- #
#                 SELF-ASSESSMENT CORRELATION BARPLOTS                        #
# --------------------------------------------------------------------------- #
vars = ["attractive",	"sincere",	"intelligence",	"funny", "ambition"]
rt = visual.corrtable(df[vars])
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
sns.barplot(x='Correlation', y='Pair', data=rt, hue='Strength',
            dodge=False).set_title('Self-Assessment Correlations')


# %%
# --------------------------------------------------------------------------- #
#                STRONG PARTNER-ASSESSMENT CORRELATIONS                       #
# --------------------------------------------------------------------------- #
vars = ["attractive_partner", "sincere_partner",
        "intelligence_partner", "funny_partner", "ambition_partner",
        "shared_interests_partner"]
rt = visual.corrtable(df[vars])
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
sns.barplot(x='Correlation', y='Pair', data=rt, hue='Strength',
            dodge=False).set_title('Partner Assessment Correlations')

# %%
# --------------------------------------------------------------------------- #
#                 PARTNER/SELF ASSESSMENT CORRELATIONS                        #
# --------------------------------------------------------------------------- #
p = ["attractive_partner", "sincere_partner",
     "intelligence_partner", "funny_partner", "ambition_partner"]
s = ["attractive",	"sincere",	"intelligence",	"funny", "ambition"]
rt = visual.corrtable(df=df, x=p, y=s)
rt['Pair'] = rt['x'].map(str) + ' & ' + rt['y']
rt['absr'] = rt['AbsCorr']
rt = rt.sort_values(by='absr', ascending=False)
sns.barplot(x='Correlation', y='Pair', data=rt, hue='Strength',
            dodge=False).set_title('Partner Assessment Correlations')

# %%
# ============================================================================ #
#                                ASSOCIATION                                   #
# ============================================================================ #
visual.association(df)
visual.assoctable(df)
