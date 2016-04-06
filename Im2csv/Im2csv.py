#coding: utf-8

import cv2
import sys

if(len(sys.argv) != 3 or '-h' in sys.argv):
  print("Usage: python[2|3] Im2csv.py in.[jpg|bmp|png] out.csv")
  sys.exit(-1)

im_addr = sys.argv[1]
csv_addr = sys.argv[2]

img = cv2.imread(im_addr,0) #0 = cv2.IMREAD_GRAYSCALE

if img == None:
  print("Imopssible d'ouvrir l'image",im_addr)

with open(csv_addr,'w') as f:
  for i in range(len(img)):
    for j in range(len(img[0])):
      f.write(str(img[i,j]))
      f.write(',')
    f.write('\n')
