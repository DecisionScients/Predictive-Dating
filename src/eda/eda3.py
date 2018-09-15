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
#                             AGE BY GENDER                                   #
# --------------------------------------------------------------------------- #
vars = ["gender", "age",	"age_o", "d_age"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='gender', title='Age by Gender')


# --------------------------------------------------------------------------- #
#                       PREFERENCES BY GENDER                                 #
# --------------------------------------------------------------------------- #
vars = ["gender", "attractive_important",	"sincere_important",
        "ambition_important", "shared_interests_important"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='gender', title='Preferences by Gender')


# --------------------------------------------------------------------------- #
#                  SELF-ASSESSMENTS BY GENDER                                 #
# --------------------------------------------------------------------------- #
vars = ["gender", "attractive",	"sincere",	"intelligence",	"funny", "ambition"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='gender', title='Self-Assessments by Gender')


# --------------------------------------------------------------------------- #
#                  PARTNER ASSESSMENTS BY GENDER                              #
# --------------------------------------------------------------------------- #
vars = ["gender", "attractive_partner",	"sincere_partner",
        "intelligence_partner",	"funny_partner", "ambition_partner"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='gender',
                     title='Partner Assessments by Gender')


# --------------------------------------------------------------------------- #
#                       PREFERENCES BY DECISION                               #
# --------------------------------------------------------------------------- #
vars = ["decision", "age",	"age_o", "d_age"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='decision', title='Age by Decision')


# --------------------------------------------------------------------------- #
#                       PREFERENCES BY DECISION                               #
# --------------------------------------------------------------------------- #
vars = ["decision", "attractive_important",	"sincere_important",
        "ambition_important", "shared_interests_important"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='decision', title='Preferences by Decision')


# --------------------------------------------------------------------------- #
#                  SELF-ASSESSMENTS BY DECISION                               #
# --------------------------------------------------------------------------- #
vars = ["decision", "attractive",	"sincere",
        "intelligence",	"funny", "ambition"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='decision',
                     title='Self-Assessments by Decision')


# --------------------------------------------------------------------------- #
#                  PARTNER ASSESSMENTS BY DECISION                            #
# --------------------------------------------------------------------------- #
vars = ["decision", "attractive_partner",	"sincere_partner",
        "intelligence_partner",	"funny_partner", "ambition_partner"]
pd = df[vars]
visual.multi_boxplot(pd, groupby='decision',
                     title='Partner Assessments by Decision')
