# -*- coding: utf-8 -*-

import os
from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np
import copy
from lxml.etree import Element, SubElement, tostring, ElementTree
import glob

Annotation_dir = '/home/sgiit/disk_1T/sgiit/Pengming_Feng/Dataset/hrsc2016/HRSC2016/FullDataSet/Annotations/'
Target_dir = '/home/sgiit/disk_1T/sgiit/Pengming_Feng/Dataset/hrsc2016/HRSC2016/FullDataSet/Annotations_voc/'

files = glob.glob(Annotation_dir+'*.xml')

for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    HRSC_Objects = root.getchildren()[-1]
    
    if len(HRSC_Objects) > 0:
        
        file_name = root.getchildren()[0].text
        image_width = root.getchildren()[10].text
        image_height = root.getchildren()[11].text
        image_depth = root.getchildren()[12].text
        
        object_class = []
        object_pt = []
               
        for HRSC_Object in HRSC_Objects:
            obj_cls = HRSC_Object.getchildren()[1].text
            object_class.append(obj_cls)
            print(obj_cls)
            obj_pt = [HRSC_Object.getchildren()[5].text, 
                      HRSC_Object.getchildren()[6].text,
                      HRSC_Object.getchildren()[7].text,
                      HRSC_Object.getchildren()[8].text,
                      HRSC_Object.getchildren()[9].text,
                      HRSC_Object.getchildren()[10].text,
                      HRSC_Object.getchildren()[11].text,
                      HRSC_Object.getchildren()[12].text,
                      HRSC_Object.getchildren()[13].text]
            object_pt.append(obj_pt)
            
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'JPEGImages' # set correct folder name
        ET.SubElement(root, 'filename').text = file_name
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image_width)
        ET.SubElement(size, 'height').text = str(image_height)
        ET.SubElement(size, 'depth').text = str(image_depth)
        
        ET.SubElement(root, 'segmented').text = '0'
        
        for obj_cls, obj_pt in zip(object_class,object_pt):
            name = obj_cls
            xmin = obj_pt[0]
            ymin = obj_pt[1]
            xmax = obj_pt[2]
            ymax = obj_pt[3]
            center_x = obj_pt[4]
            center_y = obj_pt[5]
            box_width = obj_pt[6]
            box_height = obj_pt[7]
            box_ang = obj_pt[8]
            
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'occluded').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bx, 'xmin').text = str(xmin)
            ET.SubElement(bx, 'ymin').text = str(ymin)
            ET.SubElement(bx, 'xmax').text = str(xmax)
            ET.SubElement(bx, 'ymax').text = str(ymax)
            
            ET.SubElement(bx, 'center_x').text = str(center_x)
            ET.SubElement(bx, 'center_y').text = str(center_y)
            ET.SubElement(bx, 'box_width').text = str(box_width)
            ET.SubElement(bx, 'box_height').text = str(box_height)
            ET.SubElement(bx, 'box_ang').text = str(box_ang)
            
            
        xml_file = file.split('/')[-1]
        tree = ET.ElementTree(root)
        tree.write(Target_dir+xml_file)
