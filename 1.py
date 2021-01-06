import os
import config
import shutil
img=os.listdir(config.config.xmlPath)
for i in img:
    shutil.copy(config.config.xmlPath+i,'/Users/apple/Desktop/voc/')