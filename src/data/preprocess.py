
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import analysis
import os
import pandas as pd
import settings

# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
def read(file_name):    
    # Imports training data into a pandas DataFrame.   
    df = pd.read_csv(os.path.join(settings.INTERIM_DATA_DIR, file_name), 
    encoding = "Latin-1", low_memory = False)
    return(df)

# ============================================================================ #
#                                  PREPROCESS                                  #
# ============================================================================ #
def preprocess(df):    

# --------------------------------------------------------------------------- #
#                               PREPROCESS                                    #
# --------------------------------------------------------------------------- #    
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

       

        # Compute subject perception of relative differences
        df['rd_attractive'] = (df['attractive_partner'] - df['attractive']) / df['attractive']
        df['rd_sincere'] = (df['sincere_partner'] - df['sincere']) / df['sincere']
        df['rd_intelligence'] = (df['intelligence_partner'] - df['intelligence']) / df['intelligence']
        df['rd_funny'] = (df['funny_partner'] - df['funny']) / df['funny']
        df['rd_ambition'] = (df['ambition_partner'] - df['ambition']) / df['ambition']

        # Compute relative difference in subject and partner impressions
        df['rd_attractive_o'] = (df['attractive_o'] - df['attractive']) / df['attractive']
        df['rd_sincere_o'] = (df['sinsere_o'] - df['sincere']) / df['sincere']
        df['rd_intelligence_o'] = (df['intelligence_o'] - df['intelligence']) / df['intelligence']
        df['rd_funny_o'] = (df['funny_o'] - df['funny']) / df['funny']
        df['rd_ambition_o'] = (df['ambitous_o'] - df['ambition']) / df['ambition']    
        return(df) 

    return(df)
# ============================================================================ #
#                                 Write                                        #
# ============================================================================ #
def write(df, file_name):
    if (os.path.isdir(settings.PROCESSED_DATA_DIR) == False):
        os.mkdir(settings.PROCESSED_DATA_DIR)
    df.to_csv(os.path.join(settings.PROCESSED_DATA_DIR, file_name),
    index = False, index_label = False)

# =============================================================================
if __name__ == "__main__":
    train = read("train.csv")
    test = read("test.csv")
    train = preprocess(train)
    test = preprocess(test)
    write(train, "train.csv")
    write(test, "test.csv")


