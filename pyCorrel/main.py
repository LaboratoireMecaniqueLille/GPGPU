#coding: utf-8
from __future__ import division, print_function
from pyCorrel import gridCorrel,Resize
import cv2
import numpy as np
import sys

from time import time


def openImage(address):
  img = cv2.imread(address,0)
  if img is None:
    print("OpenCV failed to open",address)
    sys.exit(-1)
  return img.astype(np.float32)

ntx = 16
nty = 16
stages = 7
img = [openImage("../Images/ref2.png")]

# -- Class used to resize the images using hardware linear interpolation --
R = Resize()

# -- Resampling the original image stages times --
for i in range(stages):
  img.append(R.resize(img[i]))

correl = [0]*stages
maxRes=800

# -- creating the table of stages gridCorrel instances --
for stage in range(stages):
  correl[stage] = gridCorrel(img[stage],ntx,nty,verbose=1,maxRes=maxRes/(2**stage))
# Note: even if the residual is normalized (divided by the number of pixels), a resampled image will have a lower residual, so maxRes is not the same on every stages

# -- Starting the critical part (to be run at the fastest possible refresh rate)
# -- Resampling the second image --
img_d = [openImage("../Images/ref2_d.png")]
t1 = time()
for i in range(stages):
  img_d.append(R.resize(img_d[i]))
t2 = time()
print("Resizing:",1000*(t2-t1),"ms.")


df=np.zeros((ntx,nty,2),np.float32)
for stage in reversed(range(stages)):
  print("\n**** Stage",stage,"****\n\n")
  t1 = time()
  correl[stage].setOriginalDisplacement(df*2)
  df = correl[stage].getDisplacementField(img_d[stage])
  t2 = time()
  print("stage",stage,"duration:",1000*(t2-t1),"ms.")

correl[0].showDisplacement()
