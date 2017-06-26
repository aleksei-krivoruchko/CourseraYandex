import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score

df = pd.read_csv('d:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\PythonApplication1\Week3.3-dataLogistic.csv', header=None)

y = np.array(df[0])
X = np.array(df.loc[:, 1:])
