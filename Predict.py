#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:04:06 2020

@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import operator

class KNN:
    
    def __init__(self):
        self.trainData = np.zeros([28,28,36,29])
        self.trainLabels = np.zeros([36,29])
        self.maps = {}
    
    def loadData(self,path):
        
        trainFile = os.listdir(path)
        idx = 0
        
        for files in trainFile:
            self.maps[idx] = files
            pth = path + '/'+ str(files)
            loc = 0
            
            for file in os.listdir(pth):
                im_path = pth + "/" + file
                image = cv2.imread(im_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret,image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # image = cv2.resize(image, (28,28),interpolation=cv2.INTER_NEAREST)
                image = cv2.resize(image, (28,28))
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if image[i,j] < 255:
                            image[i,j] = 0
                        
                self.trainData[:,:,idx,loc] = image
                self.trainLabels[idx,loc] = idx
                loc +=1
            idx += 1
        self.trainLabels = self.trainLabels.reshape(-1,1)
        self.trainData = self.trainData.reshape(28*28, 36*29).T
        
    def predict(self,testData, k = 3):
        
        size = self.trainData.shape[0]
        classCount = {}
        diff = np.abs(np.tile(testData, (size,1)) - self.trainData)
        
        distance = np.sum(diff, axis = 1)
        distIndex = np.argsort(distance)
        
        for i in range(k):
            label = self.trainLabels[distIndex[i]][0]
            classCount[label] = classCount.get(label,0) + 1
        classCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
        
        return self.maps[classCount[0][0]]
        # norms_train = np.apply_along_axis(np.linalg.norm, 1, self.trainData) + 1.0e-7
        # norms_test = np.apply_along_axis(np.linalg.norm, 1, testData) + 1.0e-7
        # train_x = self.trainData / np.expand_dims(norms_train, -1)
        # test_x = testData / np.expand_dims(norms_test, -1)
    
        # corr = np.dot(test_x, np.transpose(train_x))
        # argmax = np.argmax(corr, axis=1)
        # preds = self.trainLabels[argmax]
        # return self.maps[preds[0][0]]
        
