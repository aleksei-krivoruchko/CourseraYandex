import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score

df = pd.read_csv('d:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\PythonApplication1\Week3.3-dataLogistic.csv', header=None)

y = np.array(df[0])
X = np.array(df.loc[:, 1:])


# Реализуйте градиентный спуск для обычной 
# и L2-регуляризованной (с коэффициентом регуляризации 10) логистической регрессии. 
# Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
def runGrad(x, y, k, c):
    
    w1 = 0
    w2 = 0
    i = 0
    error = 1
    
    xlen = x.__len__()
    
    while error > 0.00001 and i < 10000:
        w1temp = w1
        w2temp = w2
        
        w1sum = 0;
        w2sum = 0;

        for l in range(xlen):
            r = y[l] * (1 - 1 / (1 + math.exp(-y[l] * (w1 * x[l][0] + w2 * x[l][1]))))
            w1sum += x[l][0] * r
            w2sum += x[l][1] * r
        
        w1 = w1 + k / xlen * w1sum - k * c * w1
        w2 = w2 + k / xlen * w2sum - k * c * w2
        
        error = math.sqrt((w1 - w1temp) ** 2 + (w2 - w2temp) ** 2)
        i += 1

    print(i)
    scores = np.empty(xlen)        
        
    for l2 in range(xlen):
        scores[l2] = 1 / (1 + math.exp(-w1 * x[l2][0] - w2 * x[l2][1]))

    #Обратите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей, 
    #подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией: a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).
    return roc_auc_score(y_true=y, y_score=scores)

# Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов на соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций десятью тысячами.

# Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
k=0.1
gradNoReg = runGrad(X,y,k,0)
gradWithReg = runGrad(X,y,k,10)

print(round(gradNoReg, 3))
print(round(gradWithReg, 3))