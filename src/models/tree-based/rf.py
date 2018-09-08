
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import analysis
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import settings

import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from time import time
import visual

#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
def read(file_name):    
    # Imports training data into a pandas DataFrame.   
    df = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, file_name), 
    encoding = "Latin-1", low_memory = False)
    return(df)

#%%
# ============================================================================ #
#                                    SPLIT                                     #
# ============================================================================ #
def split(df):    
    # Create training and test sets
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    # Save to disk
    train.to_csv(os.path.join(settings.MODEL_DATA_DIR, "train.csv"),
    index = False, index_label = False)
    test.to_csv(os.path.join(settings.MODEL_DATA_DIR, "test.csv"),
    index = False, index_label = False)    

    # Split Train into training and validation sets
    X = train.drop("match", axis=1)
    y = train['match']
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=0)
    return(X_train, X_validation, y_train, y_validation)
#%%
# ============================================================================ #
#                                  REPORT                                      #
# ============================================================================ #
def report(results, n_top=3):
    # Utility function to report cross-validation results 
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean AUC score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#%%
# ============================================================================ #
#                              TUNE ESTIMATORS                                 #
# ============================================================================ #
def estimators(X_train, X_validation, y_train, y_validation, n_estimators):    
    train = []
    validation = []
    for estimators in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimators, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train.append(roc_auc)

        y_pred = rf.predict(X_validation)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation.append(roc_auc)
    visual.plot_AUC(x=n_estimators, y1 = train, y2 = validation, xlab = "Estimators",
    y1lab = "Train AUC", y2lab = "Validation AUC") 
#%%
# ============================================================================ #
#                              TUNE MAX DEPTH                                  #
# ============================================================================ #
def max_depth(X_train, X_validation, y_train, y_validation, depths):    
    train = []
    validation = []
    for max_depth in depths:
        rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train.append(roc_auc)

        y_pred = rf.predict(X_validation)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation.append(roc_auc)
    visual.plot_AUC(x=depths, y1 = train, y2 = validation, xlab = "Max Depth",
    y1lab = "Train AUC", y2lab = "Validation AUC") 

#%%
# ============================================================================ #
#                            TUNE MIN SAMPLES PER SPLIT                        #
# ============================================================================ #
def min_samples_split(X_train, X_validation, y_train, y_validation, min_samples_splits):    
    train = []
    validation = []
    for min_samples in min_samples_splits:
        rf = RandomForestClassifier(min_samples_split=min_samples, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train.append(roc_auc)

        y_pred = rf.predict(X_validation)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation.append(roc_auc)
    visual.plot_AUC(x=min_samples_splits, y1 = train, y2 = validation, xlab = "Min Samples per Split",
    y1lab = "Train AUC", y2lab = "Validation AUC") 

#%%
# ============================================================================ #
#                          TUNE MIN SAMPLES PER LEAF                           #
# ============================================================================ #
def min_samples_leaf(X_train, X_validation, y_train, y_validation, min_samples):    
    train = []
    validation = []
    for min_samples_leaf in min_samples:
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train.append(roc_auc)

        y_pred = rf.predict(X_validation)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation.append(roc_auc)
    visual.plot_AUC(x=min_samples, y1 = train, y2 = validation, xlab = "Min Samples per Leaf",
    y1lab = "Train AUC", y2lab = "Validation AUC") 

#%%
# ============================================================================ #
#                            TUNE MAX FEATURES                                 #
# ============================================================================ #
def max_features(X_train, X_validation, y_train, y_validation, features):    
    train = []
    validation = []
    for max_features in features:
        rf = RandomForestClassifier(max_features=max_features, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train.append(roc_auc)

        y_pred = rf.predict(X_validation)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation.append(roc_auc)
    visual.plot_AUC(x=features, y1 = train, y2 = validation, xlab = "Max Features",
    y1lab = "Train AUC", y2lab = "Validation AUC") 

#%%
# ============================================================================ #
#                                 GRIDSEARCH                                   #
# ============================================================================ #
def gridsearch(X_train, X_validation, y_train, y_validation):
    # Build classifier
    rf = RandomForestClassifier()

    # Set parameter grid
    param_grid = {"n_estimators": range(200,300,20),
                "max_depth": range(10,20,1),
                "max_features": ['auto','sqrt'],
                "min_samples_split": np.linspace(0.001, .01, 10, endpoint=True),
                "min_samples_leaf": np.linspace(0.001, .01, 10, endpoint=True),
                "bootstrap": [True]}

    # run grid search
    model_rf = GridSearchCV(rf, param_grid=param_grid, n_jobs=-1, cv=3, scoring="roc_auc",
                            random_state = 5)
    start = time()
    model_rf.fit(X_train, y_train)
    return(model_rf)    

#%%
# =============================================================================
if __name__ == "__main__":
    df = read("speed_dating.csv")
    X_train, X_validation, y_train, y_validation = split(df)
    '''

    n_estimators = range(10,1000, 50)
    estimators(X_train, X_validation, y_train, y_validation, n_estimators)

    depths = range(1,40, 1)
    max_depth(X_train, X_validation, y_train, y_validation, depths)

    min_samples_splits = np.linspace(0.001, .075, 100, endpoint=True)
    min_samples_split(X_train, X_validation, y_train, y_validation, min_samples_splits)

    min_samples = np.linspace(0.001, .02, 100, endpoint=True)
    min_samples_leaf(X_train, X_validation, y_train, y_validation, min_samples)

    features = list(range(1,X_train.shape[1]))
    max_features(X_train, X_validation, y_train, y_validation, features)
    '''
    rf = gridsearch(X_train, X_validation, y_train, y_validation)
    report(rf.cv_results_)
    print ('%s pipeline AUC score: %.3f' % ("Random Forests", 
                                            roc_auc_score(y_validation, 
                                            rf.predict_proba(X_validation)[:,1])))


