import pandas as pd
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.datasets import load_boston
                 
dataset = load_boston()








inputCols = [col for col in df.columns if col not in ['Class']]
inputDf = df[inputCols].as_matrix()
inputDf = preprocessing.scale(inputDf)

outputDf = df['Class'].as_matrix()

print(inputDf)
print(outputDf)

foldsCount = 5
avgMeanTotal = 0
for k in range(1, 51):
    print(k)
    kfold = KFold(n_splits=foldsCount, shuffle=True, random_state=42)
    meanTotal = 0
    for train, test in kfold.split(inputDf):
        #print(train, test) 
               
        int_train = inputDf[train] 
        int_test = inputDf[test]
        out_train = outputDf[train]
        out_test = outputDf[test]
        
        #print(int_train, int_train)
                
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, int_train, out_train, cv=foldsCount)
        mean = scores.mean()
        meanTotal = meanTotal + mean
    avgMean = meanTotal/foldsCount
    print("MeanAvg: %0.5f" % avgMean)
    avgMeanTotal = avgMeanTotal + avgMean

avgMeanResult = avgMeanTotal / 50
print("avgMeanResult: %0.5f" % avgMeanResult)

# the best result without scaling = 0.74, k=1






