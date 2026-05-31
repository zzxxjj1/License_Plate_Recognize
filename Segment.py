#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:06:19 2020

@author: victor
"""

import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
from skimage import measure
import os

class SegLetter:
    
    def __init__(self):
        self.thresh = None
        self.gray = None
        self.flag = False
        self.flagList = []
        self.imList = []
        self.char = []
        self.thelta = 2
        
    def crop(self):
        labels = measure.label(self.thresh, connectivity=2, background=0)
        mask = np.zeros(self.thresh.shape, dtype='uint8')
        #print(labels)
        for y in range(len(labels)):
            for x in range(len(labels[y])):
                if labels[y][x] > 0:
                    mask[y][x] = 1

        for i in range(labels.shape[1]):
            count = np.count_nonzero(labels[:,i])
            if count > self.thelta:
                self.flag = True
            else:
                self.flag = False
            self.flagList.append(self.flag)
            
        for i in range(len(self.flagList)):
            if self.flagList[i]:
                self.char.append(mask[:,i])
            else:
                if self.char:
                    if len(self.char) > 2:
                        self.imList.append(self.char)
                        self.char = []
        if self.char:
            if len(self.char) > 2:
                self.imList.append(self.char)
                self.char = []
        
        num = 1
        if not os.path.exists('characters'):
            os.makedirs('characters')
            
        for ele in self.imList:
            root = 'characters/'
            if num > 9:
                root = 'characters/a'
            ele = np.transpose(ele)
            #print(ele)
            #plt.figure(),plt.imshow(ele,cmap='gray')
            plt.imsave(root + str(num) + '.png', ele,cmap='gray')
            num += 1
        # plt.imshow(mask, cmap='gray')
        
    
    def preProcess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        mainArea = img[y:y + h, x:x + w, :].copy()
        self.gray = cv2.cvtColor(mainArea, cv2.COLOR_BGR2GRAY)
        
        _, self.thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #plt.imshow(self.thresh, cmap='gray')
        # for i in range(self.thresh.shape[0]):
        #     for j in range(self.thresh.shape[1]):
        #         if self.thresh[i,j] < 255:
        #             self.thresh[i,j] = 0
        # plt.imshow(self.gray,cmap='gray')
    
