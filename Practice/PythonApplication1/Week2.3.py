import pandas as pd
import numpy as np
 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

dataTrain = pd.read_csv('D:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\TasksCSharp\Data\perceptron-train.csv', sep=",", names = ["res","p1","p2"])
dataTest = pd.read_csv('D:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\TasksCSharp\Data\perceptron-test.csv', sep=",", names = ["res","p1","p2"])

scaler = StandardScaler()

trainCols = [col for col in dataTrain.columns if col not in ['res']]
X_train_unscaled = dataTrain[trainCols].as_matrix()
X_train_scaled = scaler.fit_transform(X_train_unscaled)
y_train = dataTrain['res'].as_matrix()

testCols = [col for col in dataTest.columns if col not in ['res']]
X_test_unscaled = dataTest[trainCols].as_matrix()
X_test_scaled = scaler.transform(X_test_unscaled)
y_test = dataTest['res'].as_matrix()

perceptron = Perceptron(random_state=241)
perceptron.fit(X_train_unscaled, y_train)
predictions = perceptron.predict(X_test_unscaled)

score = accuracy_score(y_test, predictions)
print(score)
#0.655

perceptron.fit(X_train_scaled, y_train)
predictions = perceptron.predict(X_test_scaled)
score = accuracy_score(y_test, predictions)
print(score)
#0.845

0.845 - 0.655
#0.18