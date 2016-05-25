#coding: utf-8
from __future__ import division, print_function
from pyCorrel import gridCorrel,Resize
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

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
maxRes=800

for stage in range(stages):
  correl[stage] = gridCorrel(img[stage],ntx,nty,verbose=1,maxRes=maxRes)

df=np.zeros((ntx,nty,2),np.float32)
for stage in reversed(range(stages)):
  print("\n**** Stage",stage,"****\n\n")
  t1 = time()
  correl[stage].setOriginalDisplacement(df*2)
  df = correl[stage].getDisplacementField(img_d[stage])
  t2 = time()
  print("stage",stage,"duration:",1000*(t2-t1),"ms.")

correl[0].showDisplacement()

"""
st = 0 # Stage to visualize
norm = 2
tx,ty=img[st].shape[0]/ntx,img[st].shape[1]/nty
plt.imshow(img[st],cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
ax = plt.axes()
scale = img[st].shape[0]/400
for i in range(1,ntx-1):
  for j in range(1,nty-1):
    if resGrid[i,j] < maxRes:
      ax.arrow((i+.5)*tx, (j+.5)*ty, df[i,j,0]*scale*norm, df[i,j,1]*scale*norm, width = scale, head_width=4*scale, head_length=8*scale, fc='red', ec='red')
    else:
      ax.arrow((i+.5)*tx, (j+.5)*ty, df[i,j,0]*scale*norm, df[i,j,1]*scale*norm, width = scale, head_width=4*scale, head_length=8*scale, fc='blue', ec='blue')
plt.show()
"""
