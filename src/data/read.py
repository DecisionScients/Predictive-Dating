# =========================================================================== #
#                                     READ                                    #
# =========================================================================== #
'''
Module that reads the data from the interim folder.  Created to remove 
clutter from notebooks that require the data.
'''
#%%
# --------------------------------------------------------------------------- #
#                                   IMPORTS                                   #
# --------------------------------------------------------------------------- #
import os
import sys
import inspect

import numpy as np
import pandas as pd

sys.path.append("./src")
from shared import directories
from shared import filenames
from shared import variables

# --------------------------------------------------------------------------- #
#                                READ MODULE                                  #
# --------------------------------------------------------------------------- #
def read():
    data = {}
    df = pd.read_csv(os.path.join(directories.INTERIM_DATA_DIR,
                                filenames.INTERIM_FILENAME),
                    encoding="Latin-1", low_memory=False)
    df_male = df[df['gender']=='Male']
    df_female = df[df['gender']=='Female']
    data['all'] = df
    data['male'] = df_male
    data['female'] = df_female
    return(data)

