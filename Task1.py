import numpy as np

X = np.random.normal(loc=1, scale=10, size=(1000, 50))
print(X)

m = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = ((X - m)  / std)

Z = np.array([[4, 5, 0], 
             [1, 9, 3],              
             [5, 1, 1],
             [3, 3, 3], 
             [9, 9, 9], 
             [4, 7, 1]])

r = np.sum(Z, axis=1)
nonZero = np.nonzero(r > 10)

A = np.eye(3)
B = np.eye(3)
stacked = np.vstack((A,B))


import pandas
data = pandas.read_csv('D:\\Work\\MachineLearning\\CourseraYandex\\Exercises\\titanic.csv', index_col='PassengerId')

# 1
maleCount = data[data['Sex'] == 'male']
femaleCount = data[data['Sex'] == 'female']

print(len(data))
print(len(maleCount))
print(len(femaleCount))

#2 Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. 
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
survivedCount = len(data[data['Survived'] == 1])
print(survivedCount)

totalCount = len(data);
survivedPercent = (survivedCount/totalCount)*100;
print(survivedPercent)                  
                  
#3 Какую долю пассажиры первого класса составляли среди всех пассажиров? 
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.

survivedPercent = (len(data[data['Pclass'] == 1])/totalCount)*100;
print(survivedPercent)  
 
#4 Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. 
# В качестве ответа приведите два числа через пробел.

ages = data['Age']
print(ages)
ageMean = np.mean(ages.dropna());
print(ageMean)
                         
ageMedian = np.median(ages.dropna());
print(ageMedian)


twoCol = data[['SibSp','Parch']]#5 Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
#Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
print(twoCol)
corr = twoCol.corr(method='pearson', min_periods=1)
print(corr)                         

#Какое самое популярное женское имя на корабле? 
#Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name). 
girlNames = data[data.Sex == 'female']['Name']
girlNames = girlNames.apply(lambda x: x.split(',')[1].
                            replace('Miss. ', '').
                            replace('Mrs. ', '').
                            replace('Ms. ', '').
                            replace('"', '').
                            replace('(', '').
                            replace(')', '')
                            )

print(girlNames)

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Counter(" ".join(girlNames).split()).most_common(100)
                                     
                                      
                                      