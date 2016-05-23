#coding: utf-8
from __future__ import division, print_function
from pyCorrel import gridCorrel
import cv2
import numpy as np

from time import time


def openImage(address):
  img = cv2.imread(address,0)
  if img is None:
    print("OpenCV failed to open",address)
    sys.exit(-1)
  return img.astype(np.float32)


img = openImage("../Images/lena.png")
img_d = openImage("../Images/lena_d.png")

correl = gridCorrel(img,16,16,3)
t1 = time()
df = correl.getDisplacementField(img_d)
t2 = time()
print("Elapsed",1000*(t2-t1),"ms.")
print(df[4:6,2:4,:])
