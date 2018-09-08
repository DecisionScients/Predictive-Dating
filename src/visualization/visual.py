def print_df(df, centered = True):
    # This function pretty prints a pandas dataframe
    import tabulate
    print(tabulate.tabulate(df, headers='keys', tablefmt='psql'))


def freq_dist(counts, title = None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    sns.set(style="whitegrid", font_scale=1)
    sns.set_color_codes("dark")
    fdp = sns.distplot(counts, bins=40, ax=ax, kde=False,
                            color='royalblue').set_title(title)    
    return(fdp)

def bar_plot(df, xval, yval, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statistics as stat
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    bp = sns.barplot(x=xval, y=yval, data=df, ax=ax).set_title(title)    
    if (stat.mean(df[[xval]].apply(len)) > 5):
        plt.xticks(rotation=30)
    return(bp)

def histogram(values, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    hist = sns.distplot(values,bins=40, ax=ax, kde=False).set_title(title)    
    return(hist)

def eda_uni_quant_plot(df, var):
    import analysis
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Get dataframe without outliers
    df2 = analysis.outliers(df, var)

    # Render boxplots
    fig, ax = plt.subplots(3,2)
    fig.suptitle('Univariate Analysis: ' + var)

    sns.set(style="whitegrid")
    ax[0,0] = sns.boxplot(x=df[var], ax=ax[0,0]).set_title("With Outliers")
    ax[0,1] = sns.boxplot(x=df2[var], ax=ax[0,1]).set_title("Without Outliers")
    ax[1,0] = sns.distplot(df[var], ax=ax[1,0],bins=40,kde=False,
                color='royalblue').set_title("With Outliers")
    ax[1,1] = sns.distplot(df2[var], ax=ax[1,1],bins=40,kde=False,
                color='royalblue').set_title("Without Outliers")
    ax[2,0] = sns.distplot(np.sqrt(df[var]), ax=ax[2,0],bins=40,kde=False,
                color='royalblue').set_title("Sqrt Transformation")
    ax[2,1] = sns.distplot(np.log(df[var]+1), ax=ax[2,1],bins=40,kde=False,
                color='royalblue').set_title("Log Transformation")
    return(ax)


def corrplot(df):
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

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

def plot_AUC(x, y1, y2, xlab, y1lab, y2lab):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
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
    
def plot_line(x,y, xlab, ylab):
    import numpy as np
    import matplotlib.pyplot as plt
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

def plot_rf_hyperparameter(df, param):
    import matplotlib.pyplot as plt
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
    from matplotlib.pylab import rc, rcParams
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