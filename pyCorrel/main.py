#coding: utf-8
from __future__ import division, print_function
from pyCorrel import PyramidalCorrel
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

img = openImage("../Images/ref2.png")

img_d = openImage("../Images/ref2_d.png")

correl = PyramidalCorrel(img,6,(32,32),factor=1.5,overlap=1.5,verbose=2)

correl.compute(img_d)
