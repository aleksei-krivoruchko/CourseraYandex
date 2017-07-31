# постановка задачи, описание данных, инструкции по выполнению
# https://d3c33hcgiwev3.cloudfront.net/_241f17ef513f517fb19931139a6be20a_final-statement.html?Expires=1501200000&Signature=kkQq-bu5wREZgeKwbx2-E2eUdRLJKTlDUjpitNoM0HxpKBZ2ZgHbFUKoY8oXNxeZZOqwlpGBmW~KTZjosLzPvGHBMc2YESVLsU5GWqwW7y8A8d~Te8l9AoxtJiPQH9TFvAJf7CNSL6G6PnTpcCHykXonquJLWBOTVTBp2GPqjmo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A

import pandas as pd
import numpy as np
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
import datetime
import matplotlib.pyplot as plt

#Подход 1: градиентный бустинг "в лоб"
#1 Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. 
#  Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).

scriptLocationFolder = 'd:\\Work\\TestProjects\\CourseraYandex\\Practice\\PythonApplication1\\'
                                               
dfTrain = pd.read_csv(scriptLocationFolder + 'Week7.1 features.csv', index_col='match_id')
dfTest = pd.read_csv(scriptLocationFolder + 'Week7.1 features_test.csv', index_col='match_id')
                                              
#2 Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число заполненных значений. 
#  Много ли пропусков в данных? 
# Запишите названия признаков, имеющих пропуски, и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.
trainCount = dfTrain.count()
testCount = dfTest.count()

trainWithMissings = trainCount[trainCount < trainCount.max()]
testWithMissings = testCount[testCount < testCount.max()]

#Обоснование пропущенных признаков:
# Признаки события "первая кровь" (first blood) - видимо событие попросту иногда не случается, число событий по группе (first_blood_time, first_blood_team,first_blood_player1) одинаковое 
# * first_blood_player2 - вероятно даже в случае first blood убийство происходит в одиночку
# * события покупки предметов (radiant_bottle_time, dire_bottle_time...)
#   аналогичным образом иногда не происходят (в первые 5 минут)
#   это кореллирует с конечным итогом матча и возможно является показателем уровня 
#   игроков - если никто не купил ward_sentry то вероятность проигрыша команды больше 
#   чем у соперника (победа в 0.45 случаях)

dfTrain.loc[dfTrain["radiant_first_ward_time"].isnull()]["radiant_win"].mean()
dfTrain.loc[dfTrain["dire_first_ward_time"].isnull()]["radiant_win"].mean()

#3 Замените пропуски на нули с помощью функции fillna(). 
#  На самом деле этот способ является предпочтительным для логистической регрессии, 
#  поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание. 
#  Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение — 
#  в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева. 
#  Также есть и другие подходы — например, замена пропуска на среднее значение признака. 
#  Мы не требуем этого в задании, но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.
dfTrain.fillna(0, inplace=True)
dfTest.fillna(0, inplace=True)

#4 Какой столбец содержит целевую переменную? Запишите его название.
# целевой столбец - 'radiant_win'

after5MinsColumns = [
'duration',
'radiant_win',
'tower_status_radiant',
'tower_status_dire',
'barracks_status_radiant',
'barracks_status_dire']

meaninglessColumns = ['start_time']

columnsToExclude = after5MinsColumns + meaninglessColumns

XTrain = np.array(dfTrain[[col for col in dfTrain.columns if col not in columnsToExclude]])
yTrain = np.array(dfTrain['radiant_win'])

#  Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не забудьте перемешать при этом выборку (shuffle=True), 
#  поскольку данные в таблице отсортированы по времени, и без перемешивания можно столкнуться с нежелательными эффектами 
#  при оценивании качества. 
#  Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при этом разное количество деревьев 
# (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). 
#  Долго ли настраивались классификаторы? 
#  Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего, продолжит расти при дальнейшем его увеличении?
               
kfold = KFold(yTrain.size, n_folds=5, random_state=1, shuffle=True)

iterationKeys = []
iterationScores = []

#for rate in [1,0.5,0.3,0.2,0.1]:
#    for treesCount in [10,20,30,40,50,75,100, 250]:
rate = 0.3
treesCount = 250
print("rate %s, treesCount: %s" % (rate, treesCount))
startTime = datetime.datetime.now()

classifier = GradientBoostingClassifier(learning_rate=rate, n_estimators=treesCount, verbose=True, random_state=241)

scores = cross_val_score(classifier, X=XTrain, y=yTrain, cv=kfold, scoring='roc_auc')
iterationKeys.append("{}-{}".format(str(rate),str(treesCount)))
iterationScores.append(np.mean(scores))
print("score: %s" % (np.mean(scores)))
print("time: %s" % (datetime.datetime.now() - startTime))
    
iterationKeysNums = np.arange(len(iterationKeys))
plt.bar(iterationKeysNums, iterationScores, align='center')
plt.xticks(iterationKeysNums, iterationKeys)
plt.show()

#rate 0.3, treesCount: 30
#score: 0.701294212647
#time: 0:02:54.546348

#rate 0.3, treesCount: 250
#score: 0.718800529795
#time: 0:19:32.840704

#Точность градиентного бустинга на 30 деревьях с learning_rate 0.3 составила 0.701294212647, время обучения 0:02:54
#При увеличении числа деревьев качество увеличивается, на 100 девевьях оценка составила 0.713445239132

#ОТЧЕТ
#Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
#Как называется столбец, содержащий целевую переменную?
#Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC.
#Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?




# Логистическая регрессия
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) 
#с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга. 
#Подберите при этом лучший параметр регуляризации (C). Какое наилучшее качество у вас получилось? 
#Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? 
#Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

standardScaler = StandardScaler()
XTrainLinear = standardScaler.fit_transform(XTrain)

def logisticRegressionTest(X, y):
    scores = []

    powersRange = range(-5, 5)
    powersRange = [-2]

    for cPower in powersRange:
        C = 10 ** cPower
        print("C: %s" % (str(C)))
        startTime = datetime.datetime.now()
        model = LogisticRegression(C=C, random_state=241)    
        modelScores = cross_val_score(model, X=X, y=y, cv=kfold, scoring='roc_auc')
        print(modelScores)
        print(np.mean(modelScores))   
        print("score: %s" % (np.mean(modelScores)))
        print("time: %s" % (datetime.datetime.now() - startTime))
        scores.append(np.mean(modelScores))
    
    plt.plot(powersRange, scores)
    plt.xlabel('C')
    plt.ylabel('score')
    plt.show()
    
    maxValue = max(scores)
    print("maxValue: %s" % (maxValue))
    print("maxIndex: %s" % (scores.index(maxValue)))

logisticRegressionTest(XTrainLinear, yTrain)

# лучший результат 0.71630004186733975 при C = 0.01
# это сопоставимо с результатом градиентного бустинга (250 деревьев, 0.71880) но значительно быстрее 
# 23 сек vs 19,5 минут тк это линейный метод и он быстрее набора деревьев в градиентном бустинге

#Среди признаков в выборке есть категориальные, которые мы использовали как числовые, 
#что вряд ли является хорошей идеей. Категориальных признаков в этой задаче одиннадцать: 
#lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero.
#Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке 
#с подбором лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?

heroColumns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

categoryColumns = ['lobby_type'] + heroColumns

columnsToExclude = after5MinsColumns + meaninglessColumns + categoryColumns

XTrain = np.array(dfTrain[[col for col in dfTrain.columns if col not in columnsToExclude]])
XTrainLinear = standardScaler.fit_transform(XTrain)
XTrain.shape
logisticRegressionTest(XTrainLinear, yTrain)


# избавление от категариальных признаков не дело существенного прироста в качестве, видимо прошлая модель посчитала их не значительными 
#score: 0.716329944695
#time: 0:00:23.500439



#Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных героев. 
#Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; 
#единице, если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire. 
#Ниже вы можете найти код, который выполняет данной преобразование. 
#Добавьте полученные признаки к числовым, которые вы использовали во втором пункте данного этапа.

def appendHeroes(dataSet):
    uniqueHeroes = np.unique(dataSet[heroColumns]).tolist()
    X_pick = np.zeros((dataSet.shape[0], len(uniqueHeroes)))
    
    heroNums = range(5)
    for i, match_id in enumerate(dataSet.index):
        for p in heroNums:
            rVal = dataSet.ix[match_id, 'r%d_hero' % (p+1)]
            rInd = uniqueHeroes.index(rVal)
            dVal = dataSet.ix[match_id, 'd%d_hero' % (p+1)]
            dInd = uniqueHeroes.index(dVal)
            
            X_pick[i, rInd] = 1
            X_pick[i, dInd] = -1
    
    for j in range(X_pick.shape[1]):
        colName = "hero_" + str(j)
        col = X_pick[:, j]
        col.shape
        dataSet[colName] = pd.Series(X_pick[:, j], index = dataSet.index)

appendHeroes(dfTrain)

columnsToExclude = after5MinsColumns + meaninglessColumns + categoryColumns

XTrain = np.array(dfTrain[[col for col in dfTrain.columns if col not in columnsToExclude]])
XTrainLinear = standardScaler.fit_transform(XTrain)

logisticRegressionTest(XTrainLinear, yTrain)

#после преобразования информации о героях в разреженную матрицу качество существенно повысилось до 0.751786273753

appendHeroes(dfTest)

XTest = np.array(dfTest[[col for col in dfTest.columns if col not in columnsToExclude]])
XTestLinear = standardScaler.fit_transform(XTest)

C = 0.001
model = LogisticRegression(C=C, random_state=241)    
model.fit(XTrainLinear, yTrain)
val = model.predict_proba(XTestLinear)


np.amin(val)
np.amax(val)












    