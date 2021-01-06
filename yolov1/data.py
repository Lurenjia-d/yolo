import cv2
import config
import xml.etree.ElementTree as ET
import os
import numpy as np
import math
lables=['diningtable', 'chair', 'horse', 'person',
        'tvmonitor', 'bird', 'cow', 'dog', 'bottle', 'pottedplant',
        'aeroplane', 'car', 'cat', 'sheep', 'bicycle', 'sofa', 'boat', 'train', 'motorbike', 'bus']
#输入448*448*3
#输出7*7*30
def GetTrainData(isl=True):
    imglist=[]
    li=[]
    xmlList = os.listdir(config.config.xmlPath)
    xmlList=xmlList[0:100]
    for p in xmlList:
        tree=ET.parse(config.config.xmlPath+p)
        root = tree.getroot()
        img=cv2.imread(config.config.imgPath+root.find('filename').text)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(448,448))
        img=img/255.0
        if isl:
            imglist.append(img)
        else:
            imglist.append(root.find('filename').text)

        n=np.zeros((7,7,30))
        size=root.find('size')
        width=float(size.find('width').text)
        height=float(size.find('height').text)
        objects=root.findall('object')
        for obj in objects:
            bndbox=obj.find('bndbox')
            xmin=float(bndbox.find('xmin').text)
            ymin=float(bndbox.find('ymin').text)
            xmax =float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            c_x=(xmax+xmin)/2
            c_y=(ymax+ymin)/2
            w=xmax-xmin
            h=ymax-ymin
            i=int(math.ceil(c_x/(width/7)))-1
            j=int(math.ceil(c_y/(height/7)))-1
            k=lables.index(obj.find('name').text)
            n[i,j,k+10]=1.0
            if isl:
                n[i,j,0:5]=c_x/width,c_y/height,w/width,h/height,1.0
            else:
                n[i, j, 0:5] = c_x, c_y, w, h, 1.0
            n[i,j,5:10]= n[i,j,0:5]
        li.append(n)
    return imglist,li
