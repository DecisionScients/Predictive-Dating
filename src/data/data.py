
# %%
import os
import sys
import inspect
current = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd

import decorators
import settings
# --------------------------------------------------------------------------- #
#                                   DATA                                      #
# --------------------------------------------------------------------------- #


@decorators.check_types
def read(directory: str, filename: str, vars: list = None)-> "Data frame " \
        "with complete cases including requested variables":
    df = pd.read_csv(os.path.join(directory, filename),
                     encoding="Latin-1", low_memory=False)

    # Obtain variables of interest
    if vars:
        df = df[vars]
    # Obtain complete cases
    df = df.dropna()
    print(df.info())
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
