import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score




dfClass = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week3.4 classification.csv')

y = np.array(dfClass['true'])
yPred = np.array(dfClass['pred'])

#Заполните таблицу ошибок классификации:
#Actual Positive	Actual Negative
#Predicted Positive	TP	FP
#Predicted Negative	FN	TN
#
#Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. 
#Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1. 
#Ответ в данном вопросе — четыре числа через пробел.

TP = len(dfClass[(dfClass.true==1) & (dfClass.pred==1)])
FN = len(dfClass[(dfClass['true']==1) & (dfClass['pred']==0)])
FP = len(dfClass[(dfClass['true']==0) & (dfClass['pred']==1)])
TN = len(dfClass[(dfClass['true']==0) & (dfClass['pred']==0)])

print(TP)
print(FP)
print(FN)
print(TN)

#3. Посчитайте основные метрики качества классификатора:
#Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
#Precision (точность) — sklearn.metrics.precision_score
#Recall (полнота) — sklearn.metrics.recall_score
#F-мера — sklearn.metrics.f1_score

Accuracy = accuracy_score(y,yPred)
Precision = precision_score(y, yPred)
Recall = recall_score(y, yPred)
F1 = f1_score(y, yPred)







dfScores = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week3.4 scores.csv', header=None)
