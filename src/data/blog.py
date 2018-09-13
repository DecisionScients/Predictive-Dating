# =========================================================================== #
#                                   BLOG                                      #
# =========================================================================== #
''' This module imports the data for the blog project. It corrects the 
a few misspellings in the variable names and saves the data in the 
interim data directory.'''
# %%
import os
import sys
import inspect
current = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "src")
sys.path.append(current)

import pandas as pd
import numpy as np
import shutil
from sklearn import model_selection

from decorators import check_types
from shared import directories
from shared import filenames
from shared import variables


# --------------------------------------------------------------------------- #
#                                   DATA                                      #
# --------------------------------------------------------------------------- #

def read(directory, filename, vars):
    df = pd.read_csv(os.path.join(directory, filename),
                     encoding="Latin-1", low_memory=False)

    # Correct spelling
    df.rename({'sinsere_o': 'sincere_o',
               'intellicence_important': 'intelligence_important',
               'ambtition_important': 'ambition_important'},
              inplace=True,
              axis='columns')

    # Recode race levels
    df['race'] = df['race'].replace({
        'asian/pacific islander/asian-american': 'asian',
        'european/caucasian-american': 'caucasian',
        'black/african american': 'black',
        'latino/hispanic american': 'latino',
        'other': 'other'})
    df['race_o'] = df['race_o'].replace({
        'asian/pacific islander/asian-american': 'asian',
        'european/caucasian-american': 'caucasian',
        'black/african american': 'black',
        'latino/hispanic american': 'latino',
        'other': 'other'})

    # Obtain variables of interest
    if vars:
        df = df[vars]
    # Obtain complete cases
    df = df.dropna()

    return(df)

# --------------------------------------------------------------------------- #
#                                SPLIT                                        #
# --------------------------------------------------------------------------- #


def split(df):
    idx = np.random.rand(len(df)) < 0.8
    train = df[idx]
    test = df[~idx]
    return train, test

# --------------------------------------------------------------------------- #
#                                WRITE                                        #
# --------------------------------------------------------------------------- #


def write(df, directory, filename):
    if isinstance(df, pd.DataFrame):
        if isinstance(filename, str):
            if not os.path.isdir(directory):
                os.mkdir(directory)
            df.to_csv(os.path.join(directory, filename),
                      index=False, index_label=False)
            return(True)
        else:
            return(False)
    else:
        return(False)


# --------------------------------------------------------------------------- #
#                                 Main                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df = read(directories.RAW_DATA_DIR,
              filenames.RAW_FILENAME, variables.BLOG)
    train, test = split(df)
    write(train, directories.INTERIM_DATA_DIR, filenames.TRAIN_FILENAME)
    write(test, directories.INTERIM_DATA_DIR, filenames.TEST_FILENAME)
