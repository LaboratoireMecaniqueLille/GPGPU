#coding: utf-8

import cv2

im_addr = 'img.png'
csv_addr = 'out.csv'

img = cv2.imread(im_addr,0) #0 = cv2.IMREAD_GRAYSCALE

with open(csv_addr,'w') as f:
  for i in range(len(img)):
    for j in range(len(img[0])):
      f.write(str(img[i,j]))
      f.write(',')
    f.write('\n')
