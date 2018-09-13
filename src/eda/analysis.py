# ============================================================================ #
#                                  ANALYSIS                                    #
# ============================================================================ #
# %%
# ---------------------------------------------------------------------------- #
#                                 LIBRARIES                                    #
# ---------------------------------------------------------------------------- #
import os
import sys

from collections import OrderedDict
from itertools import combinations
from itertools import product
import matplotlib as mp
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn import preprocessing
import textwrap

# %%
# ---------------------------------------------------------------------------- #
#                                    DESCRIBE                                  #
# ---------------------------------------------------------------------------- #


def describe_quant(df):

    stats = pd.DataFrame()
    cols = df.columns
    for col in cols:
        d = pd.DataFrame(df[col].describe())
        d = d.T
        d['skew'] = skew(df[col])
        d['kurtosis'] = kurtosis(df[col])
        c = ['count', 'min', '25%', 'mean', '50%',
             '75%', 'max', 'skew', 'kurtosis']
        stats = stats.append(d[c])
    return stats

# %%
# ---------------------------------------------------------------------------- #
#                                  CRAMER'S V                                  #
# ---------------------------------------------------------------------------- #


def cramers_corrected_stat(contingency_table):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328

        Args:
            contingency_table (pd.DataFrame): Contingency table containing
                                              counts for the two variables
                                              being analyzed
        Returns:
            float: Corrected Cramer's V measure of Association                                    
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# ---------------------------------------------------------------------------- #
#                               INDEPENDENCE                                   #
# ---------------------------------------------------------------------------- #


class Independence:
    "Class that performs a test of independence"

    def __init__(self):
        self._sig = 0.05
        self._x2 = 0
        self._p = 0
        self._df = 0
        self._obs = []
        self._exp = []

    def summary(self):
        print("\n*", "=" * 78, "*")
        print('{:^80}'.format("Pearson's Chi-squared Test of Independence"))
        print('{:^80}'.format('Data'))
        print('{:^80}'.format("x = " + self._xvar + " y = " + self._yvar + "\n"))
        print('{:^80}'.format('Observed Frequencies'))
        visual.print_df(self._obs)
        print("\n", '{:^80}'.format('Expected Frequencies'))
        visual.print_df(self._exp)
        results = ("Pearson's chi-squared statistic = " + str(round(self._x2, 3)) + ", Df = " +
                   str(self._df) + ", p-value = " + '{0:1.2e}'.format(round(self._p, 3)))
        print("\n", '{:^80}'.format(results))
        print("\n*", "=" * 78, "*")

    def post_hoc(self, rowwise=True, verbose=False):

        dfs = []
        if rowwise:
            rows = range(0, len(self._obs))
            for pair in list(combinations(rows, 2)):
                ct = self._obs.iloc[[pair[0], pair[1]], ]
                levels = ct.index.values
                x2, p, dof, exp = stats.chi2_contingency(ct)
                df = pd.DataFrame({'level_1': levels[0],
                                   'level_2': levels[1],
                                   'x2': x2,
                                   'N': ct.values.sum(),
                                   'p_value': p}, index=[0])
                dfs.append(df)
            self._post_hoc_tests = pd.concat(dfs)
        else:
            cols = range(0, len(self._obs.columns.values))
            for pair in list(combinations(cols, 2)):
                ct = self._obs.iloc[:, [pair[0], pair[1]]]
                levels = ct.columns.values
                x2, p, dof, exp = stats.chi2_contingency(ct)
                df = pd.DataFrame({'level_1': levels[0],
                                   'level_2': levels[1],
                                   'x2': x2,
                                   'N': ct.values.sum(),
                                   'p_value': p}, index=[0])
                dfs.append(df)
            self._post_hoc_tests = pd.concat(dfs)
        if (verbose):
            visual.print_df(self._post_hoc_tests)

        return(self._post_hoc_tests)

    def test(self, x, y, sig=0.05):
        self._x = x
        self._y = y
        self._xvar = x.name
        self._yvar = y.name
        self._n = x.shape[0]
        self._sig = sig

        ct = pd.crosstab(x, y)
        x2, p, dof, exp = stats.chi2_contingency(ct)

        self._x2 = x2
        self._p = p
        self._df = dof
        self._obs = ct
        self._exp = pd.DataFrame(exp).set_index(ct.index)
        self._exp.columns = ct.columns

        if p < sig:
            self._result = 'significant'
            self._hypothesis = 'reject'
        else:
            self._result = 'not significant'
            self._hypothesis = 'fail to reject'

        return x2, p, dof, exp

    def report(self):
        "Prints results in APA format"
        tup = ("A Chi-square test of independence was conducted to "
               "examine the relation between " + self._xvar + " and " + self._yvar + ". "
               "The relation between the variables was " + self._result + ", "
               "X2(" + str(self._df) + ", N = ", str(self._n) + ") = " +
               str(round(self._x2, 2)) + ", p = " + '{0:1.2e}'.format(round(self._p, 3)))

        self._report = ''.join(tup)

        wrapper = textwrap.TextWrapper(width=80)
        lines = wrapper.wrap(text=self._report)
        for line in lines:
            print(line)

# %%
# ---------------------------------------------------------------------------- #
#                                ASSOCTABLE                                    #
# ---------------------------------------------------------------------------- #


def assoctable(df):
    '''For a dataframe containing categorical variables, this function 
    computes a series of association tests for each pair of categorical
    variables. It returns the adjusted Cramer's V measure of 
    association between the pairs of categorical variables.  Note, this 
    is NOT  a hypothesis test. 

    Args:
        df (pd.DataFrame): Data frame containing categorical variables

    Returns:
        Data frame containing the results of the pairwise association measures.
    '''
    df = df.select_dtypes(include='object')
    terms = df.columns

    tests = []
    for pair in list(combinations(terms, 2)):
        x = df[pair[0]]
        y = df[pair[1]]
        ct = pd.crosstab(x, y)
        ct = pd.crosstab(x, y)
        cv = cramers_corrected_stat(ct)
        tests.append(OrderedDict(
            {'x': pair[0], 'y': pair[1], "Cramer's V": cv}))
    tests = pd.DataFrame(tests)
    return(tests)


# %%
# ---------------------------------------------------------------------------- #
#                                CORRTABLE                                     #
# ---------------------------------------------------------------------------- #
def corrtable(df, target=None, threshold=0):
    '''For a dataframe containing numeric variables, this function 
    computes pairwise pearson's R tests of correlation correlation.

    Args:
        df (pd.DataFrame): Data frame containing numeric variables
        threshold (float): Threshold above which correlations should be
                           reported.

    Returns:
        Data frame containing the results of the pairwise tests of correlation.
    '''
    df2 = df.select_dtypes(include=['int', 'float64'])
    terms = df2.columns

    if target:
        if target in df2.columns:
            pass
        else:
            df2 = df2.join(df[target])

    tests = []

    if target:
        for term in terms:
            x = df2[term]
            y = df2[target]
            r = stats.pearsonr(x, y)
            tests.append(OrderedDict(
                {'x': term, 'y': target, "Correlation": r[0], "p-value": r[1]}))
        tests = pd.DataFrame(tests)
        tests['AbsCorr'] = tests['Correlation'].abs()
        top = tests.loc[tests['AbsCorr'] > threshold]
        return top
    else:
        for pair in list(combinations(terms, 2)):
            x = df2[pair[0]]
            y = df2[pair[1]]
            r = stats.pearsonr(x, y)
            tests.append(OrderedDict(
                {'x': pair[0], 'y': pair[1], "Correlation": r[0], "p-value": r[1]}))
        tests = pd.DataFrame(tests)
        tests['AbsCorr'] = tests['Correlation'].abs()
        top = tests.loc[tests['AbsCorr'] > threshold]
        return top
