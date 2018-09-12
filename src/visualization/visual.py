# =========================================================================== #
#                                 VISUAL                                      #
# =========================================================================== #
'''Modules for creating scatterplots, histograms, barplots, and other 
visualizations.'''

# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.pylab import rc, rcParams
import seaborn as sns
import statistics as stat
import tabulate
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
#                           PRINT DATAFRAME                                   #
# --------------------------------------------------------------------------- #
def print_df(df, centered = True):
    '''Pretty prints a data frame'''    
    print(tabulate.tabulate(df, headers='keys', tablefmt='psql'))

# --------------------------------------------------------------------------- #
#                              MULTI-COUNTPLOT                                #
# --------------------------------------------------------------------------- #
def multi_countplot(df, nrows=None, ncols=None, width=None, height=None,
                    title=None): 
    '''Prints count plots for the categorical variables in a data frame.
    The plots can be customized using the number rows, the number columns
    of columns or both. Users may also designate the height and width
    of the figure containing the individual count plots.

    Args:
        ncols (int): The number of plots to render per row
        nrows (int): The number of rows of plots to render 
        height (int): The height of the figure in inches
        width (int): The width of the figure in inches  
    '''

    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
 
    # Sets number of rows and columns
    if all(v is None for v in [nrows, ncols]):
        nrows = len(df.columns)
        ncols = 1
    elif not nrows:
        nrows = -(-len(df.columns) // ncols)
    else:
        ncols = -(-len(df.columns) // nrows)  

    # Sets height and width 
    if not width:
        width = plt.rcParams.get('figure.figsize')[0]
    if not height:
        height = plt.rcParams.get('figure.figsize')[1] 
    figsize = [width, height]       

    # Designates sub plots
    fig, axes = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)
    cols = df.columns
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=1)

    # Renders count plots for each subplot 
    for ax, cols in zip(axes.flat, cols):
        sns.countplot(x = df[cols], ax=ax)
    plt.tight_layout()

# --------------------------------------------------------------------------- #
#                              MULTI-HISTOGRAM                                #
# --------------------------------------------------------------------------- #
def multi_histogram(df: pd.DataFrame, nrows: int=None, ncols: int=None,
                    width: [int, float]=None, height: [int, float]=None,
                    title : str=None) -> 'figure containing multiple histograms':  
    import pandas as pd

    warnings.filterwarnings('ignore')

    sns.set(style="whitegrid", font_scale=1)
    sns.set_color_codes("dark")
    
    # Sets number of rows and columns
    if all(v is None for v in [nrows, ncols]):
        nrows = len(df.columns)
        ncols = 1
    elif not nrows:
        nrows = -(-len(df.columns) // ncols)
    else:
        ncols = -(-len(df.columns) // nrows)        

    if not width:
        width = plt.rcParams.get('figure.figsize')[0]
    if not height:
        height = plt.rcParams.get('figure.figsize')[1] 
    figsize = [width, height]       

    fig, axes = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)    
    cols = df.columns

    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=1)

    for ax, cols in zip(axes.flat, cols):
        sns.distplot(a = df[cols], kde=False, ax=ax, color='b')
    plt.tight_layout()

# --------------------------------------------------------------------------- #
#                               MULTI-BOXPLOT                                 #
# --------------------------------------------------------------------------- #
def multi_boxplot(df: pd.DataFrame, nrows: int=None, ncols: int=None,
                    width: [int, float]=None, height: [int, float]=None,
                    title : str=None) -> 'figure containing multiple boxplots':  

    sns.set(style="whitegrid", font_scale=1)
    sns.set_color_codes("dark")
    
    # Sets number of rows and columns
    if all(v is None for v in [nrows, ncols]):
        nrows = len(df.columns)
        ncols = 1
    elif not nrows:
        nrows = -(-len(df.columns) // ncols)
    else:
        ncols = -(-len(df.columns) // nrows)        

    if not width:
        width = plt.rcParams.get('figure.figsize')[0]
    if not height:
        height = plt.rcParams.get('figure.figsize')[1] 
    figsize = [width, height]       

    fig, axes = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)
    cols = df.columns

    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=1)
        
    for ax, cols in zip(axes.flat, cols):
        sns.boxplot(x = df[cols], ax=ax)
    plt.tight_layout()


# --------------------------------------------------------------------------- #
#                                 BARPLOT                                     #
# --------------------------------------------------------------------------- #
def bar_plot(df, xvar, yvar, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statistics as stat
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    sns.barplot(x=xvar, y=yvar, data=df, ax=ax, color='b').set_title(title)       
    plt.tight_layout()
# --------------------------------------------------------------------------- #
#                                 HISTOGRAM                                   #
# --------------------------------------------------------------------------- #
def histogram(values, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    hist = sns.distplot(values,bins=40, ax=ax, kde=False).set_title(title)    
    return(hist)
# --------------------------------------------------------------------------- #
#                            CORRELATION PLOT                                 #
# --------------------------------------------------------------------------- #
def corrplot(df):

    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
# --------------------------------------------------------------------------- #
#                              AUC PLOT                                       #
# --------------------------------------------------------------------------- #
def plot_AUC(x, y1, y2, xlab, y1lab, y2lab):
   
    line1, = plt.plot(x, y1, 'b', label=y1lab)
    line2, = plt.plot(x, y2, 'r', label=y2lab)

    x1max = x[np.argmax(y1)]
    x2max = x[np.argmax(y2)]
    y1max = y1[np.argmax(y1)]
    y2max = y2[np.argmax(y2)]
    text1= "x={:.3f}, y={:.3f}".format(x1max, y1max)
    text2= "x={:.3f}, y={:.3f}".format(x2max, y2max)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC")
    plt.xlabel(xlab)    
    plt.annotate(text1, xy=(x1max, y1max), xytext=(.94,.70), **kw)
    plt.annotate(text2, xy=(x2max, y2max), xytext=(.94,.40), **kw)
    plt.show()
    
# --------------------------------------------------------------------------- #
#                                LINE PLOT                                    #
# --------------------------------------------------------------------------- #    
def plot_line(x,y, xlab, ylab):

    line = plt.plot(x, y, 'b')
    plt.ylabel(ylab)
    plt.xlabel(xlab)    
    plt.title(ylab + " by " + xlab)    
    xmax = x[np.argmax(y)]
    ymax = y[np.argmax(y)]
    text = "x={:.0f}, y={:.3f}".format(xmax, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data',textcoords="axes fraction",
            arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    plt.annotate(text, xy=(xmax, ymax), xytext=(.94,.20), **kw)
    plt.show()
# --------------------------------------------------------------------------- #
#                    RANDOM FORESTS HYPERPARAMETER PLOT                       #
# --------------------------------------------------------------------------- #
def plot_rf_hyperparameter(df, param):

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best AUC', color=color)
    ax1.plot(df['Iteration'], df['Best AUC'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel(param, color=color)  # we already handled the x-label with ax1
    ax2.plot(df['Iteration'], df[param], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(param + " Analysis")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def ezrc(fontSize=22., lineWidth=2., labelSize=None, tickmajorsize=10,
         tickminorsize=5, figsize=(6, 8)):    
  
    if labelSize is None:
        labelSize = fontSize + 5
    rc('figure', figsize=figsize)
    rc('lines', linewidth=lineWidth)
    rcParams['grid.linewidth'] = lineWidth
    rcParams['font.sans-serif'] = ['Open Sans']
    rcParams['font.serif'] = ['Open Sans']
    rcParams['font.family'] = ['Open Sans']
    rc('font', size=fontSize, family='sans-serif', weight='bold')
    rc('axes', linewidth=lineWidth, labelsize=labelSize)
    rc('legend', borderpad=0.1, markerscale=1., fancybox=False)
    rc('text', usetex=True)
    rc('image', aspect='auto')
    rc('ps', useafm=True, fonttype=3)
    rcParams['xtick.major.size'] = tickmajorsize
    rcParams['xtick.minor.size'] = tickminorsize
    rcParams['ytick.major.size'] = tickmajorsize
    rcParams['ytick.minor.size'] = tickminorsize
    rcParams['text.latex.preamble'] = ["\\usepackage{amsmath}"]