import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

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

#4. Имеется четыре обученных классификатора. 
#В файле scores.csv записаны истинные классы и значения степени принадлежности 
#положительному классу для каждого классификатора на некоторой выборке:
#для логистической регрессии — вероятность положительного класса (колонка score_logreg),
#для SVM — отступ от разделяющей поверхности (колонка score_svm),
#для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
#для решающего дерева — доля положительных объектов в листе (колонка score_tree). 

dfScores = pd.read_csv('d:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\PythonApplication1\Week3.4 scores.csv')

#Посчитайте площадь под ROC-кривой для каждого классификатора. 
#Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? 
#Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
yScoreTrue = np.array(dfScores['true'])

rocLog = roc_auc_score(yScoreTrue, np.array(dfScores['score_logreg']))
rocSvm = roc_auc_score(yScoreTrue, np.array(dfScores['score_svm']))
rocKnn = roc_auc_score(yScoreTrue, np.array(dfScores['score_knn']))
rocTree = roc_auc_score(yScoreTrue, np.array(dfScores['score_tree']))

#6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
#Чтобы получить ответ на этот вопрос, 
#найдите все точки precision-recall-кривой с помощью функции sklearn.metrics.precision_recall_curve. 
#Она возвращает три массива: precision, recall, thresholds. 
#В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds. 
#Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

prcLog = precision_recall_curve(yScoreTrue, np.array(dfScores['score_logreg']))
prcSvm = precision_recall_curve(yScoreTrue, np.array(dfScores['score_svm']))
prcKnn = precision_recall_curve(yScoreTrue, np.array(dfScores['score_knn']))
prcTree = precision_recall_curve(yScoreTrue, np.array(dfScores['score_tree']))

results = {}
for clf in dfScores.columns[1:]:
    prc = precision_recall_curve(yScoreTrue, dfScores[clf])
    prcDf = pd.DataFrame({'precision': prc[0], 'recall': prc[1]})
    results[clf] = prcDf[prcDf['recall'] >= 0.7]['precision'].max()























