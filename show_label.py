#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:14:39 2019

@author: sgiit
"""

import os
from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np

#label_all_file = 'label_all_name_file.txt'
label_all_together = open('label_all_file','a+')

image_file = "/home/sgiit/disk_1T/sgiit/Pengming_Feng/Dataset/Ship_Classification/data/19.tif"

image = cv.imread(image_file)

tree = ET.parse('/home/sgiit/disk_1T/sgiit/Pengming_Feng/Dataset/Ship_Classification/data/19.xml')
root = tree.getroot()
result = root.getchildren()[-1]
size = len(result)

label_all_name = []
label_all_pt = []

for item in range(0,size,2):
    #print(item)
    object_name = result[item]
    pixel = result[item + 1]
    temp_point = []
    for pt in pixel:
        x = float(pt.attrib['LeftTopX'])
        y = float(pt.attrib['LeftTopY'])
        temp_point.append([x,y])
    
    
#    print(object_name.text)
    label_all_name.append(object_name.text)
    label_all_pt.append(temp_point)

name_str = '\n'.join(label_all_name)    
label_all_together.write(name_str)
label_all_together.close()


for name, pt in zip(label_all_name, label_all_pt):
    pts = np.array(pt, np.int32)
    cv.polylines(image,[pts], True, (0,0,255), thickness = 5)
    


#cv.imshow('image',image)
#cv.waitKey(0)   

cv.imwrite('image_19_with_label.jpg', image)
 

