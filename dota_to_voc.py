#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 09:47:58 2019

@author: pengming
"""

 
import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA import DOTA
import dota_utils as util
import pylab
import sys
import torch
    
    
from ImgSplit import splitbase
from tqdm import tqdm

import json

out_path = '/home/pengming/Model_Data_2TB/DOTA/DOTA_800/train_800'



examplesplit = DOTA(out_path)

attrDict = dict()
#images = dict()
#images1 = list()
attrDict["categories"]=[
                    {"supercategory":"none","id":1,"name":"small-vehicle"},
                    {"supercategory":"none","id":2,"name":"ship"},
                    {"supercategory":"none","id":3,"name":"swimming-pool"},
                    {"supercategory":"none","id":4,"name":"harbor"},
                    {"supercategory":"none","id":5,"name":"large-vehicle"},
                    {"supercategory":"none","id":6,"name":"plane"},
                    {"supercategory":"none","id":7,"name":"storage-tank"},
                    {"supercategory":"none","id":8,"name":"bridge"},
                    {"supercategory":"none","id":9,"name":"soccer-ball-field"},
                    {"supercategory":"none","id":10,"name":"basketball-court"},
                    {"supercategory":"none","id":11,"name":"roundabout"},
                    {"supercategory":"none","id":12,"name":"ground-track-field"},
                    {"supercategory":"none","id":13,"name":"baseball-diamond"},
                    {"supercategory":"none","id":14,"name":"tennis-court"},
                    {"supercategory":"none","id":15,"name":"helicopter"},
                    {"supercategory":"none","id":16,"name":"container-crane"},
                    ]
images = list()
annotations = list()

imgids = examplesplit.getImgIds()
# anns = example.loadAnns(imgId=imgid)
image_id = 0


if len(imgids) > 0:
    for ids in imgids:
        image_id = image_id + 1
        anns = examplesplit.loadAnns(imgId=ids)
        img = examplesplit.loadImgs(ids)[0]
        sizes = img.shape
        image = dict()
        #keyList = list()
        #print doc['annotation']['filename']
        image['file_name'] = ids + '.png'
        #keyList.append("file_name")
        image['height'] = int(sizes[0])
        #keyList.append("height")
        image['width'] = int(sizes[1])
        #keyList.append("width")
    
        #image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
        image['id'] = image_id
        print("File Name: {} and image_id {}".format(ids, image_id))
        images.append(image)
        
        
        
        id1 = 1
        if len(anns) > 0:
            for obj in anns:
                for value in attrDict["categories"]:
                    annotation = dict()
                    #if str(obj['name']) in value["name"]:
                    if str(obj['name']) == value["name"]:
                        #print str(obj['name'])
                        annotation["segmentation"] = []
                        annotation["iscrowd"] = 0
                        #annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
                        annotation["image_id"] = image_id
                        x1 = obj["poly"][0][0]
                        y1 = obj["poly"][0][1]
                        x2 = obj["poly"][2][0]
                        y2 = obj["poly"][2][1]
                        annotation["bbox"] = [x1, y1, x2, y2]
                        annotation["area"] = float(x2 * y2)
                        annotation["category_id"] = value["id"]
                        annotation["ignore"] = 0
                        annotation["id"] = id1
                        annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
                        id1 +=1
        
                        annotations.append(annotation)

        else:
            print("File: {} doesn't have any object".format(ids))
                    #image_id = image_id + 1
                    
else:
    print("File: not found")

attrDict["images"] = images
attrDict["annotations"] = annotations
attrDict["type"] = "instances"

#print attrDict
jsonString = json.dumps(attrDict)
with open("train_800.json", "w") as f:
    f.write(jsonString)