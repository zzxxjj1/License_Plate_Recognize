#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:21:17 2020

@author: victor
"""

from Plate import Plate
from Segment import SegLetter
from Predict import KNN
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    path = 'images/1.png'
    img1 = cv2.imread(path)
    plate = Plate()
    plate.getEdge(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plate.getContour()
    plate.getCandidate()
    plate.adjustCandidate()
    plate.getPlate()
    
    seg = SegLetter()
    img2 = cv2.imread('plate.png')
    seg.preProcess(img2)
    seg.crop()
    
    knn = KNN()
    path2 = 'data/Training'
    knn.loadData(path2)
    
    lst = os.listdir('characters')
    lst.sort()
    prediction = []
    for file in lst:
        pth = 'characters/' + file
        if pth.endswith('.png'):
            test = cv2.imread(pth)
            test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            ret,test = cv2.threshold(test, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            test = cv2.resize(test, (28,28), interpolation=cv2.INTER_NEAREST)
            # pca = PCA(n_components=20)
            # test = pca.fit_transform(test)
            test = test.reshape(-1,1).T
            predict = knn.predict(test, 5)
            prediction.append(predict)
    print(prediction)