import pandas as pd
import numpy as np
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.datasets import load_boston
                 
dataset = load_boston()

inputdata = preprocessing.scale(dataset.data)
output = dataset.target

#print(inputdata)
#print(output)

foldsCount = 5
scores = list()
p_range = np.linspace(1.0, 10.0, num=200)
for p in p_range:
    print(p)
    kfold = KFold(len(output), n_folds=5, shuffle=True, random_state=42)
               
    knn = KNeighborsRegressor(n_neighbors=5,weights='distance',metric='minkowski',p=p)
    valScore = cross_val_score(knn, inputdata, output, cv=kfold, scoring='neg_mean_squared_error')
    scores.append(valScore)
    print(max(valScore))
           
vmax = pd.DataFrame(scores, p_range).max(axis=1).sort_values(ascending=False).head(1).index[0]
print(vmax)
print(44)

# 1.135678    -11.836924