#%%
# --------------------------------------------------------------------------- #
#                                   DATA                                      #
# --------------------------------------------------------------------------- #


def read(directory, filename, vars):
    '''
    This function reads the requested variables from the designated filename
    and directory, then returns only the complete cases.

    Args:
        directory (str): The path to the file designated in the filename
                         argument.
        filename (str): The name of the file to be read, relative to the
                        raw data directory.
        vars (str): The names of the variables to be returned.

    Returns:
        pandas.DataFrame containing the complete cases
    '''
    if isinstance(directory, str):
        if isinstance(filename, str):
            df = pd.read_csv(os.path.join(directory, filename),
                             encoding="Latin-1", low_memory=False)

            # Obtain variables of interest
            df = df[vars]
            # Obtain complete cases
            df = df.dropna()
            return(df)
        else:
            return False
    else:
        return False

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
    import os
    import imp
    import pandas as pd
    settings = imp.load_source('settings',
                               r"c:\Users\John\Documents\Data Science\Projects\Predictive Dating\src\settings.py")
    
    df = read(settings.RAW_DATA_DIR, settings.RAW_DATA_FILENAME, settings.VARS)
    print(df.info())
    write(df, settings.INTERIM_DATA_DIR, 'speed_dating.csv')
