import pandas as pd
import numpy as np
 
from sklearn.svm import SVC

df = pd.read_csv('D:\Work\Github\CourseraYandex\Practice\PythonApplication1\Week3.3-dataLogistic.csv', header=None)

y = df.ix[:, 0]
X = df.ix[:,1:]
