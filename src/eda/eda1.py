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

from shared import directories
from shared import filenames
from shared import variables
from visualization import visual

# --------------------------------------------------------------------------- #
#                                  DATA                                       #
# --------------------------------------------------------------------------- #
df = pd.read_csv(os.path.join(directories.INTERIM_DATA_DIR,
                              filenames.TRAIN_FILENAME),
                 encoding="Latin-1", low_memory=False)

# --------------------------------------------------------------------------- #
#                                OVERVIEW                                     #
# --------------------------------------------------------------------------- #
#data_set = df.info()

# --------------------------------------------------------------------------- #
#                                 GENDER                                      #
# --------------------------------------------------------------------------- #
visual.count_plot(df, x='gender', title='Gender Counts')

# %%
# --------------------------------------------------------------------------- #
#                              AGE ANALYSIS                                   #
# --------------------------------------------------------------------------- #
vars = ['age', 'age_o', 'd_age']
quant = visual.describe_quant(df[vars])
visual.print_df(quant)
visual.multiplot(df[vars], title='Age')

# %%
# --------------------------------------------------------------------------- #
#                                 DECISIONS                                   #
# --------------------------------------------------------------------------- #
visual.count_plot(df, x='decision', title='Decision Counts')
# %%
# --------------------------------------------------------------------------- #
#                             PREFERENCE ANALYSIS                             #
# --------------------------------------------------------------------------- #
vars = ["attractive_important",
        "sincere_important", "intelligence_important", "funny_important",
        "ambition_important", "shared_interests_important"]

quant = visual.describe_quant(df[vars])
visual.multiplot(df[vars], title='Preferences')
# %%
# --------------------------------------------------------------------------- #
#                         SELF-ASSESSMENT ANALYSIS                            #
# --------------------------------------------------------------------------- #
vars = ["attractive", "sincere", "intelligence", "funny", "ambition"]

quant = visual.describe_quant(df[vars])
visual.print_df(quant)
visual.multiplot(df[vars], title='Self-Assessment')

# %%
# --------------------------------------------------------------------------- #
#                       PARTNER-ASSESSMENT ANALYSIS                           #
# --------------------------------------------------------------------------- #
vars = ["attractive_partner", "sincere_partner", "intelligence_partner",
        "funny_partner", "ambition_partner", "shared_interests_partner",
        "interests_correlate"]
quant = visual.describe_quant(df[vars])
visual.print_df(quant)
visual.multiplot(df[vars], title='Partner Assessment')
