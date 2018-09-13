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
# visual.correlation(df)
visual.corrtable(df, threshold=0.25)
# ============================================================================ #
#                                ASSOCIATION                                   #
# ============================================================================ #
# visual.association(df)
# visual.assoctable(df)

# %%
# =========================================================================== #
#                      PREFERENCES BY GENDER STATISTICS                       #
# =========================================================================== #

vars = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
        "pref_o_funny",	"pref_o_ambitious",	"pref_o_shared_interests"]

male = df[df['gender'] == 'male']
female = df[df['gender'] == 'female']
stats = pd.DataFrame()
for v in vars:
    m = analysis.describe_quant(pd.DataFrame(male[v]))
    m.insert(0, 'term', v)
    m.insert(0, 'gender', 'male')
    f = analysis.describe_quant(pd.DataFrame(female[v]))
    f.insert(0, 'term', v)
    f.insert(0, 'gender', 'female')
    stats = stats.append(m, ignore_index=True)
    stats = stats.append(f, ignore_index=True)
print(stats)
