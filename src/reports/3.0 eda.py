
# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import graphviz
from sklearn import tree
import pandas as pd
import pydotplus
import settings
from IPython.display import Image
import os

# %%
# ============================================================================ #
#                                    DATA                                      #
# ============================================================================ #
df = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, 'train.csv'),
                 encoding="Latin-1", low_memory=False)
df = df.loc[1:10, ]
X = df[['attractive_partner', 'interests_correlate']]
y = df['decision']

# %%
# ============================================================================ #
#                             DECISION TREE                                    #
# ============================================================================ #
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=[
                                    'attractive_partner', 'interests_correlate'],
                                class_names=['No', 'Yes'],
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
graph.set_size('"20,5!"')
# Show graph
Image(graph.create_png())
