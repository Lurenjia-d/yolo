import torch
import cv2
import config
import xml.etree.ElementTree as ET
import os
import numpy as np
import math
import yolov1.data as da
import yolov1.darkNet as mod
import yolov1.loss as yololoss
imgList,lables=da.GetTrainData()
mod=mod.GetNet()
lables=torch.from_numpy(np.array(lables))
imgList=torch.from_numpy(np.array(imgList))
opt=torch.optim.SGD(mod.parameters(),lr=0.001)
l=torch.zeros(0)
for i in range(10):
   opt.zero_grad()
   input=imgList.view(-1,3,448,448).float()
   output=mod(input)
   loss=yololoss.GetLoss(lables,output.float())
   print(loss)
   print(loss>l)
   loss.backward()
   #更新学习率
   opt.step()
   l = loss

