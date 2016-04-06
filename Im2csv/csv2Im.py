#coding utf-8

import cv2
import numpy as np


im_addr = 'out.png'
csv_addr = 'img.csv'

tab = []

with open(csv_addr) as f:
  for l in f:
    tab.append(l.split(',')[:-1])
a = np.array(tab,np.uint8)

cv2.imwrite('la.png',a)
