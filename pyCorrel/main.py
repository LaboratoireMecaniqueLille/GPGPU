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

stages = 4
img = [openImage("../Images/ref2.png")]

R = Resize()
for i in range(stages):
  img.append(R.resize(img[i]))

img_d = [openImage("../Images/ref2_d.png")]
t1 = time()
for i in range(stages):
  img_d.append(R.resize(img_d[i]))
t2 = time()
print("Resizing:",1000*(t2-t1),"ms.")

correl = [0]*stages

for stage in range(stages):
  correl[stage] = gridCorrel(img[stage],16,16,2)

df=np.zeros((16,16,2),np.float32)
for stage in reversed(range(stages)):
  print("\n**** Stage",stage,"****\n\n")
  t1 = time()
  correl[stage].setOriginalDisplacement(df*2)
  df = correl[stage].getDisplacementField(img_d[stage])
  print(df[:5,:5,0])
  t2 = time()
  print("stage",stage,"duration:",1000*(t2-t1),"ms.")

