import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from numpy import corrcoef

df = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week4.2 close_prices.csv')

#На загруженных данных обучите преобразование PCA с числом компоненты равным 10. 
#Скольких компонент хватит, чтобы объяснить 90% дисперсии?

pca = PCA(n_components=10)
X = np.array(df.loc[:, 'AXP':])
pca.fit(X)
np.sum(pca.explained_variance_ratio_)

percentage = 0
i = 0
for r in pca.explained_variance_ratio_:
    i += 1
    percentage += r
    if percentage >= 0.9:
        break

print(i)
print(percentage)

#Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.!
dfTransformed = pd.DataFrame(pca.transform(X))
col1 = dfTransformed[0]

#Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv. 
#Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
dfIndexes = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week4.2 djia_index.csv')
dj = np.array(dfIndexes['^DJI'])
pirsonCorr = corrcoef(col1, dj)
print(pirsonCorr[1, 0])

#Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.

comp0 = pd.Series(pca.components_[0])
comp0Index = comp0_w.sort_values(ascending=False).head(1).index[0]
company = df.columns[comp0Index+1]
print(company)