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
img = [openImage("../Images/lena.png")]

R = Resize()
for i in range(stages):
  img.append(R.resize(img[i]))

img_d = [openImage("../Images/lena_d.png")]
t1 = time()
for i in range(stages):
  img_d.append(R.resize(img_d[i]))
t2 = time()
print("Resizing:",1000*(t2-t1),"ms.")

correl = [0]*stages

for stage in range(stages):
  correl[stage] = gridCorrel(img[stage],16,16,2)

for stage in reversed(range(stages)):
  t1 = time()
  df = correl[stage].getDisplacementField(img_d[stage])
  t2 = time()
  print("stage",stage,"duration:",1000*(t2-t1),"ms.")
#TODO: appliquer le champ trouvé à l'itération suivante (sinon, cela ne sert à rien >.<

print(df[:10,:10,0])
