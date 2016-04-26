#coding utf-8

import cv2
import numpy as np
import sys

if(len(sys.argv) != 3 or '-h' in sys.argv):
  print("Usage: python[2|3] csv2Im.py in.csv out.[jpg|bmp|png]")
  sys.exit(-1)


im_addr = sys.argv[2]
csv_addr = sys.argv[1]

tab = []

with open(csv_addr) as f:
  for l in f:
    tab.append(l.split(',')[:-1])
a = np.array(tab,np.float)

"""
a[a<0] = 0
a[a>255] = 255
print a[0:5][0:5]
"""

#cv2.imwrite(im_addr,a.astype(np.uint8))
cv2.imwrite(im_addr,np.clip(a,0,255).astype(np.uint8))
