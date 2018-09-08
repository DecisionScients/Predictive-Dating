
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import pandas as pd
import settings
import os
import visual

#%%
# ============================================================================ #
#                                    DATA                                      #
# ============================================================================ #
df = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, 'train.csv'), 
encoding = "Latin-1", low_memory = False)
print(df.info())

#%%
# ============================================================================ #
#                             LOGISIC REGRESSION                               #
# ============================================================================ #
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=['attractive_partner', 'interests_correlate'],  
                                      class_names=['No', 'Yes'],
                                      filled=True, rounded=True,  
                                      special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.set_size('"20,5!"')
# Show graph
Image(graph.create_png())