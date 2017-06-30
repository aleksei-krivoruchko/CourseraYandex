import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

dfTrain = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week4.1 salary-train.csv')
dfTest = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week4.1 salary-test-mini.csv')

def preprocess(dataFrame):
    return dataFrame.map(lambda s: s.lower()).replace('[^a-zA-Z0-9]', ' ', regex=True)

# Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова,
# которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
tfidf = TfidfVectorizer(min_df=5)
xTrainFullDescr = tfidf.fit_transform(preprocess(dfTrain['FullDescription']))

#Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. Код для этого был приведен выше.
dfTrain['LocationNormalized'].fillna('nan', inplace=True)
dfTrain['ContractTime'].fillna('nan', inplace=True)

#Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
dictVectorizer = DictVectorizer()
xTrainProps = dictVectorizer.fit_transform(dfTrain[['LocationNormalized', 'ContractTime']].to_dict('records'))

#Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
xTrainAll = hstack([xTrainFullDescr, xTrainProps])

#3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. 
#Целевая переменная записана в столбце SalaryNormalized.
yTrain = dfTrain['SalaryNormalized']
model = Ridge(alpha=1, random_state=241)
model.fit(xTrainAll, yTrain)

#4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. 
#Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
xTestFullDescr = tfidf.transform(preprocess(dfTest['FullDescription']))
xTestProps = dictVectorizer.transform(dfTest[['LocationNormalized', 'ContractTime']].to_dict('records'))
xTestAll = hstack([xTestFullDescr, xTestProps])
yTest = model.predict(xTestAll)