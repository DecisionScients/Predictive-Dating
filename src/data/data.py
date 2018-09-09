
# %%
import os
import sys
import inspect
current = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "src")
sys.path.append(current)

import pandas as pd

from decorators import check_types
import settings
# --------------------------------------------------------------------------- #
#                                   DATA                                      #
# --------------------------------------------------------------------------- #


@check_types
def read(directory: str, filename: str, vars: list = None)-> "Data frame " \
        "with complete cases including requested variables":
    df = pd.read_csv(os.path.join(directory, filename),
                     encoding="Latin-1", low_memory=False)

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

    # Correct spelling
    df.rename({'sinsere_o': 'sincere_o',
               'intellicence_important': 'intelligence_important',
               'ambtition_important': 'ambition_important'})

    # Obtain variables of interest
    if vars:
        df = df[vars]
    # Obtain complete cases
    df = df.dropna()
    return(df)

# --------------------------------------------------------------------------- #
#                                Write                                        #
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
    df = read(settings.RAW_DATA_DIR, settings.RAW_DATA_FILENAME, settings.VARS)
    write(df, settings.INTERIM_DATA_DIR, 'speed_dating.csv')
