import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import pylab
import math

#Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241. 
#После выделения кластеров все пиксели, отнесенные в один кластер, попробуйте 
#заполнить двумя способами: медианным и средним цветом по кластеру.

def doClusterIteration(clustresCount, imagePixels, weight, height, depth):
    kmeans = KMeans(n_clusters=clustresCount, init='k-means++', random_state=241)
    imagePixels['cluster'] = kmeans.fit_predict(imagePixels)
    clusterGroups = pixels.groupby('cluster')    
    
    medianValues = clusterGroups.median().values
    medianPixels = [medianValues[c] for c in pixels['cluster'].values]
    medianImage = np.reshape(medianPixels, (weight, height, depth))
    
    meanValues = clusterGroups.mean().values
    meanPixels = [meanValues[c] for c in pixels['cluster'].values]
    meanImage = np.reshape(meanPixels, (weight, height, depth))
    
    return medianImage, meanImage

def psnr(originalImage, imageWithNoise):
    mse = np.mean((originalImage - imageWithNoise) ** 2)
    
    maxPixValue = np.max(originalImage)
    psnr = 10 * math.log10(maxPixValue / mse)
    return psnr

#Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1. 
#Для этого можно воспользоваться функцией img_as_float из модуля skimage. 
#Обратите внимание на этот шаг, так как при работе с исходным изображением вы получите некорректный результат.
image = imread('d:\Work\SmartInMedia\TestProjects\CourseraYandex\Practice\PythonApplication1\Week6.1 parrots.jpg')
pylab.imshow(image)

imagePoints = img_as_float(image)
#Создайте матрицу объекты-признаки: 
#характеризуйте каждый пиксель тремя координатами - значениями интенсивности в пространстве RGB.

weight, height, depth = imagePoints.shape
pixels = pd.DataFrame(np.reshape(imagePoints, (weight*height, depth)))

for i in range(1, 21):

    print('ClustersCount: ' + str(i))
    medianImage, meanImage = doClusterIteration(i, pixels, weight, height, depth)

    psnrMedian = psnr(imagePoints, medianImage)
    print('psnrMedian: %s' % psnrMedian)
    pylab.imshow(medianImage)
    pylab.show()
       
    psnrMean = psnr(imagePoints, meanImage)
    print('psnrMean: %s' % psnrMean)
    pylab.imshow(meanImage)
#    pylab.show()  
    
    
    
    
    