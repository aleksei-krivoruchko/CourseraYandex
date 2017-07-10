import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold, cross_val_score

df = pd.read_csv('d:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\PythonApplication1\Week5.1 abalone.csv')
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = np.array(df.iloc[:, 0:8])
y = np.array(df['Rings'])

#Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50 
#(не забудьте выставить "random_state=1" в конструкторе). 
#Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам. 
#Используйте параметры "random_state=1" и "shuffle=True" при создании генератора кросс-валидации sklearn.cross_validation.KFold. 
#В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
                                                                    
kfold = KFold(y.size, n_folds=5, random_state=1, shuffle=True)

def r2Scorer(estimator, X, y):
    predicted = estimator.predict(X)
    return r2_score(y, predicted)
            
for i in range(1, 50):     
    forest = RandomForestRegressor(n_estimators=i, random_state=1)
    forest.fit(X, y)
    score = cross_val_score(forest, X=X, y=y, cv=kfold, scoring=r2Scorer)
    print("%s: %s. Avg: %s" % (i, score, sum(score) / len(score)))
                                                                    