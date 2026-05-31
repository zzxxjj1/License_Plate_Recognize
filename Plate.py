#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:03:57 2020

@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from imutils import perspective

class Plate:
    
    def __init__(self):
        self.block = []
        self.ker1 = np.ones((5,19),np.uint8)
        self.ker2 = np.ones((11,5),np.uint8)
        self.maxWeight = 0
        self.maxIndex = -1
        self.edge = None
        self.contour = None
        self.img = None
        self.lower = np.array([0,0,150])
        self.upper = np.array([30,60,255])
        self.gray = None
        self.shift = True
        
    def getEdge(self,img):
        
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = cv2.GaussianBlur(self.img, (3,3), 0)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ret, img_thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.edge = cv2.Canny(img_thresh, 128, 200)

        
    def getContour(self):
        c1_img= cv2.morphologyEx(self.edge, cv2.MORPH_CLOSE,self.ker1)
        o1_img = cv2.morphologyEx(c1_img, cv2.MORPH_OPEN,self.ker1)
        o2_img = cv2.morphologyEx(o1_img, cv2.MORPH_OPEN, self.ker2)
        cnts,_ = cv2.findContours(o2_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contour = cnts
      
    def getCandidate(self):
        for cnts in self.contour:
            y,x = [], []
            for c in cnts:
                y.append(c[0][0])
                x.append(c[0][1])
          
            rect = cv2.minAreaRect(cnts)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            

            angle = rect[2]
            h,w = rect[1]
            if h > w:
                h,w = w, h
            r = w / h
            
            if r > 2 and r < 8.5:
                self.block.append([[[min(y), min(x)], [max(y), max(x)]],box,angle])

        # self.block=sorted(self.block,key=lambda b: b[3])
        
    def adjustCandidate(self):
        
        for i in range(len(self.block)):
            start = self.block[i][0][0]
            end = self.block[i][0][1]
            candidate = self.img[start[1]:end[1], start[0]:end[0]]
         
            hsv = cv2.cvtColor(candidate, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            
            w1 = 0
            for m in mask:
                w1 += m / 255.0
                
            w2 = 0
            for wi in w1:
                w2 += wi
        
            if w2 > self.maxWeight:
                self.maxIndex = i
                self.maxWeight = w2
        
            
            
    def getPlate(self):
        
        angle = self.block[self.maxIndex][2]
        if np.abs(angle) == 0 or (np.abs(angle) > 0 and np.abs(angle) < 1):
            self.shift = False
            
        if self.shift:
            box = self.block[self.maxIndex][1].astype('float32')
            # dst = np.array([[0, 0],   [200, 0],  [200, 50], [0, 50]], dtype=np.float32)
            # print('ok')
            # M = cv2.getPerspectiveTransform(box, dst)
            # plate = cv2.warpPerspective(self.img, M, (200,50))
            # plate = plate.T
            plate = perspective.four_point_transform(self.img, box)
        else:
            loc = self.block[self.maxIndex]
            start = (loc[0][0][0], loc[0][0][1])
            end = (loc[0][1][0], loc[0][1][1])
            plate = self.img[start[1]:end[1],start[0]:end[0],:]
            # draw = cv2.rectangle(self.img,start, end,(0,255,0), 2)
            # plt.imshow(draw)
            # plt.imshow(self.img, cmap='gray')
        # plate = cv2.flip(plate, -1)
        plt.imsave('plate.png',plate)
        # plt.figure(),plt.imshow(plate)
        # plt.figure(),plt.imshow(self.img)
        # return plate

    