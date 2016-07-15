#coding: utf-8

import numpy as np

def printField(fieldX,fieldY):
  import matplotlib.pyplot as plt
  y,x = fieldX.shape
  assert fieldY.shape == (y,x), "The fields don't have the same size >.<'"
  plt.imshow(np.zeros((y,x)),cmap=plt.get_cmap('gray'),vmin=-1,vmax=0)
  ax = plt.axes()
  scale = .6
  ascale = .1

  for i in range(x):
    for j in range(y):
      lx,ly = fieldX[j,i],fieldY[j,i]
      if lx != 0 or ly != 0:
        ax.arrow(i,j,lx*scale,ly*scale,width=ascale, head_width=2*ascale, head_length=4*ascale)
  plt.show()


sq = .5**.5
  
h,w=15,20
print((h,w))
ones = np.ones((h,w))
zeros = np.zeros((h,w))

x = (ones,zeros)
y = (zeros,ones)
z = np.meshgrid(np.arange(-sq,sq,2*sq/w),np.arange(-sq,sq,2*sq/h)) # Shear !
s = (z[1],z[0]) # Zoom
r = (z[1],-z[0]) # Rotation
exx = (np.concatenate((np.arange(-1,1,2./w)[np.newaxis,:],)*h,axis=0),zeros)
eyy = (zeros,np.concatenate((np.arange(-1,1,2./h)[:,np.newaxis],)*w,axis=1))
uxx = (np.concatenate(((np.arange(-1,1,2./w)**2)[np.newaxis,:],)*h,axis=0),zeros)
uyy = (np.concatenate(((np.arange(-1,1,2./h)**2)[:,np.newaxis],)*w,axis=1),zeros)
uxy = (np.array([[i*j for j in np.arange(-1,1,2./w)] for i in np.arange(-1,1,2./h)]),zeros)
vxx = (zeros,np.concatenate(((np.arange(-1,1,2./w)**2)[np.newaxis,:],)*h,axis=0))
vyy = (zeros,np.concatenate(((np.arange(-1,1,2./h)**2)[:,np.newaxis],)*w,axis=1))
vxy = (zeros,np.array([[i*j for j in np.arange(-1,1,2./w)] for i in np.arange(-1,1,2./h)]))
#^
#|
#ok


test = r
print test[0].shape
printField(*test)

i = 0
for f in [x,y,s,z,r,exx,eyy,uxx,uyy,uxy,vxx,vxy,vyy]:
  print i
  i+=1
  print f[0].shape == f[1].shape,("Ok" if f[0].shape == (h,w) else f[0].shape)
