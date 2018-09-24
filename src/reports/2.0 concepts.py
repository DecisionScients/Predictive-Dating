
# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import os
import sys
import inspect
home = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "src")
sys.path.append(home)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from shared import directories
# %%
# ============================================================================ #
#                                    DATA                                      #
# ============================================================================ #
df = pd.read_csv(os.path.join(directories.EXTERNAL_DATA_DIR, 'ames.csv'),
                 encoding="Latin-1", low_memory=False)
# %%
# ============================================================================ #
#                             LINEAR REGRESSION                                #
# ============================================================================ #
sns.lmplot(x='area', y='price', data=df.head(100), ci=None)
