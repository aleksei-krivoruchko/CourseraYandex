import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import math
from sklearn.ensemble  import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

df = pd.read_csv('d:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\PythonApplication1\Week5.2 gbm-data.csv')
X = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def sigmoid(y_pred):
    # Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}),
    return 1.0 / (1.0 + math.exp(-y_pred))

def estimateQuality(model, X, y):
    # Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке на каждой итерации.
    results = []
    for pred in model.staged_decision_function(X):
        yTransformed = [sigmoid(y_pred) for y_pred in pred]
        
        # Метрика log-loss предназначена для классификаторов, выдающих оценку принадлежности классу, а не бинарные ответы. 
        # чтобы строить такие прогнозы — для этого нужно использовать метод predict_proba:
#        xProba= classifier.predict_proba(X)
        logLoss = log_loss(y_true=y, y_pred=yTransformed) 
                
        results.append(logLoss)
    return results

def drawChart(rate, trainLogLoss, testLogLoss):
    # Вычислите и постройте график значений log-loss (которую можно посчитать с помощью функции
    # sklearn.metrics.log_loss) на обучающей и тестовой выборках, 
    # а также найдите минимальное значение метрики и номер итерации, на которой оно достигается.
    plt.figure()
    plt.plot(trainLogLoss, 'r', linewidth=2)
    plt.plot(testLogLoss, 'g', linewidth=2)
    plt.legend(['train', 'test'])
    plt.show()

#for rate in [1, 0.5, 0.3, 0.2, 0.1]:
for rate in [0.2]:
    classifier = GradientBoostingClassifier(learning_rate=rate, n_estimators=250, verbose=True, random_state=241)
    classifier.fit(X_train, y_train)
    
    # Вычислите и постройте график значений log-loss (которую можно посчитать с помощью функции sklearn.metrics.log_loss) 
    # на обучающей и тестовой выборках, а также найдите минимальное значение метрики и номер итерации, на которой оно достигается.
    trainLogLoss =  estimateQuality(classifier, X_train, y_train)
    testLogLoss =  estimateQuality(classifier, X_test, y_test)

    drawChart(rate, trainLogLoss, testLogLoss)

    # 4. Приведите минимальное значение log-loss на тестовой выборке и номер итерации, 
    # на котором оно достигается, при learning_rate = 0.2.
    minTestLossValue = min(testLogLoss)
    minTestLossIndex = testLogLoss.index(minTestLossValue) + 1
    
    print("minValue %s, minIndex: %s" % (minTestLossValue, minTestLossIndex))
    
#5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций, на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и остальными параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса? (Не забывайте, что предсказания нужно получать с помощью функции predict_proba. В данном случае брать сигмоиду от оценки вероятности класса не нужно)

randomForest = RandomForestClassifier(n_estimators=minTestLossIndex, random_state=241)
randomForest.fit(X_train, y_train)
y_pred = randomForest.predict_proba(X_test)[:, 1]
forestTestLoss = log_loss(y_test, y_pred)
print(forestTestLoss)
       
    
    

                                                                    