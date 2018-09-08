import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def repros():
  # Create data
  df = pd.DataFrame({'x': random.choices([0,1], k = 100),
                     'y': random.choices([0,1], k = 100)})
  
  # Create Plot
  fig, ax = plt.subplots()
  bp = sns.barplot(x='x', y='y', data=df, ax=ax)  
  return(bp)
