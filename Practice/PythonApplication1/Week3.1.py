import pandas as pd
import numpy as np
 
from sklearn.svm import SVC

df = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week3.2.svmData.csv', header=None)

y = df.ix[:, 0]
X = df.ix[:,1:]

svm = SVC(C=100000, kernel='linear', random_state=241)

svm.fit(X,y)
support = svm.support_
support.sort()
# прибавить 1 к индексам при публикации ответа!!!
