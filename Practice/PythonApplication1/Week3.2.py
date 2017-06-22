# Programming Assignment: Анализ текстов
import numpy as np
from sklearn import datasets
from sklearn.grid_search import GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import pandas

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
newsgroups.data # массив текстов
newsgroups.target # номер класса 

tfidf = TfidfVectorizer()
tfidf_data = tfidf.fit_transform(newsgroups.data)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(newsgroups.target.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X=tfidf_data, y=newsgroups.target)

max = 0
C = 0
for a in gs.grid_scores_:
	if a.mean_validation_score > max:
		max = a.mean_validation_score
		C = a.parameters['C']

clf = SVC(kernel='linear', random_state=241, C=C)
fitRes = clf.fit(X=tfidf_data, y=newsgroups.target)

feature_mapping = tfidf.get_feature_names()

coefFrame = pandas.DataFrame(clf.coef_.data, clf.coef_.indices)
frameIndexes = coefFrame[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index
indArray = np.array(frameIndexes)
type(indArray)

list=[]

for row in np.nditer(indArray):
    list.append(feature_mapping[row])

list.sort()

print(','.join(list))
