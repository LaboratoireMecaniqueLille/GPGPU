#coding: utf-8
from __future__ import division, print_function
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
import sys
import cv2

import pycuda.autoinit


def debug(*args):
  return


class gridCorrel:
  def __init__(self,originalImage,numTilesX,numTilesY,verbose=0,kernelFile="kernels.cu"):
    # -- To print debug infos if asked to (3 different levels of verbosity) --
    if verbose:
      global debug
      def debug(l,*args):
        if verbose >= l:
          print(*args)

    # -- Assigning useful values --
    self.orig = originalImage
    assert type(self.orig)==np.ndarray,"The original image must be a numpy.ndarray"
    assert len(self.orig.shape) == 2,"The original image is not a 2D array"
    self.iteration = range(10)
    self.shape = originalImage.shape
    self.w,self.h = self.shape
    self.numTilesX,self.numTilesY = numTilesX,numTilesY
    self.t_w,self.t_h = self.w/numTilesX,self.h/numTilesY
    self.it_w,self.it_h = int(round(self.t_w)),int(round(self.t_h))
    self.t_shape = self.it_w,self.it_h
    self.normalizedTileWidth = self.t_w/self.w
    self.normalizedTileHeight = self.t_h/self.h
    self.grid = ((self.w+31)//32,(self.h+31)//32)
    self.block = (min(self.w,32),min(self.h,32),1)
    self.t_grid = (self.it_w//32,(self.it_h//32))
    self.t_block = (min(self.it_w,32),min(self.it_h,32),1)

    debug(3,"Dimensions:",self.w,self.h)
    debug(2,"Grid:",self.grid)
    debug(2,"Block:",self.block)
    debug(2,"Tile grid:",self.t_grid)
    debug(3,"Tile block",self.t_block)

    self.devOut = gpuarray.GPUArray(self.shape,np.float32)
    self.devTileOut = gpuarray.GPUArray(self.t_shape,np.float32)
    self.devGradX = gpuarray.GPUArray(self.shape,np.float32)
    self.devGradY = gpuarray.GPUArray(self.shape,np.float32)
    self.devTempX = gpuarray.GPUArray((self.it_w,self.it_h),np.float32)
    self.devTempY = gpuarray.GPUArray((self.it_w,self.it_h),np.float32)

    # -- Reading kernel file (note: there are #DEFINE directives to set the img size at compilation time) --
    with open(kernelFile,"r") as f:
      self.mod = SourceModule(f.read()%(self.shape+self.t_shape))

    # -- Creating and setting properties of the textures --
    self.tex = self.mod.get_texref('tex')
    self.tex_d = self.mod.get_texref('tex_d')
    self.texGradX = self.mod.get_texref('texGradX')
    self.texGradY = self.mod.get_texref('texGradY')

    self.tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    self.tex_d.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    self.texGradX.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    self.texGradY.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)

    self.tex.set_filter_mode(cuda.filter_mode.LINEAR)
    self.tex_d.set_filter_mode(cuda.filter_mode.LINEAR)
    self.texGradX.set_filter_mode(cuda.filter_mode.LINEAR)
    self.texGradY.set_filter_mode(cuda.filter_mode.LINEAR)

    self.tex.set_address_mode(0,cuda.address_mode.CLAMP)
    self.tex_d.set_address_mode(0,cuda.address_mode.CLAMP)
    self.texGradX.set_address_mode(0,cuda.address_mode.CLAMP)
    self.texGradY.set_address_mode(0,cuda.address_mode.CLAMP)

    self.tex.set_address_mode(1,cuda.address_mode.CLAMP)
    self.tex_d.set_address_mode(1,cuda.address_mode.CLAMP)
    self.texGradX.set_address_mode(1,cuda.address_mode.CLAMP)
    self.texGradY.set_address_mode(1,cuda.address_mode.CLAMP)

    # -- Assigning kernels to functions --
    self.__makeDiff = self.mod.get_function("makeDiff")
    self.__makeDiff.prepare('Pffff',texrefs=[self.tex,self.tex_d])
    self.__gradient = self.mod.get_function("gradient")
    self.__gradient.prepare('PP',texrefs=[self.tex])
    self.__gdProduct = self.mod.get_function("gdProduct")
    self.__gdProduct.prepare('PPPff',texrefs=[self.texGradX,self.texGradY])
    self.__squareResidual = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*x[i]",arguments="float *x")
    #self.__gdKernel = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*y[i]",arguments="float* x, float* y")

    # -- Assigning the first image --
    self.setOriginalImage(self.orig)


  def setOriginalImage(self,image):
    assert image.shape == self.shape,"New image has a different size"
    # -- Create texture of the original image --
    self.imgArray = cuda.matrix_to_array(self.orig,"C")
    self.tex.set_array(self.imgArray)
    # -- Computes the gradients of the image and create the associated textures --
    self.__gradient.prepared_call(self.grid,self.block,self.devGradX.gpudata,self.devGradY.gpudata)
    self.gradXArray = cuda.matrix_to_array(self.devGradX.get(),"C")
    self.texGradX.set_array(self.gradXArray)
    self.gradYArray = cuda.matrix_to_array(self.devGradY.get(),"C")
    self.texGradY.set_array(self.gradYArray)


  def __getTileDisplacement(self,tx,ty):
    assert tx < self.numTilesX and ty < self.numTilesY,"__getTileDisplacement call out of bounds"
    debug(2,"Tile",tx,",",ty)
    res = 1e38
    x=y=0
    for i in self.iteration:
      debug(3,"Iteration",i)
      debug(3,"x=",x,"y=",y)
      # -- Computes the difference between the original and deformed image
      self.__makeDiff.prepared_call(self.t_grid,self.t_block,\
      self.devTileOut.gpudata,\
      tx*self.normalizedTileWidth,ty*self.normalizedTileHeight,\
      x/self.w,y/self.h)

      #cv2.imwrite("t{}.png".format(i),self.devTileOut.get()+128)
      # -- Computes the residual --
      oldres = res
      res = self.__squareResidual(self.devTileOut).get()/self.t_w/self.t_h
      debug(3,"Residual:",res,"\n")
      if res > oldres:
        debug(3,"Residual rising ! Exiting the loop...")
        x+=vx
        y+=vy
        debug(2,"Residual:",oldres)
        return x,y
      # -- Computes the direction of research --
      self.__gdProduct.prepared_call(self.t_grid,self.t_block,\
      self.devTempX.gpudata,self.devTempY.gpudata,\
      self.devTileOut.gpudata,\
      tx*self.normalizedTileWidth,ty*self.normalizedTileHeight)


      vx = gpuarray.sum(self.devTempX).get()/(self.t_w*self.t_h)**2*256
      vy = gpuarray.sum(self.devTempY).get()/(self.t_w*self.t_h)**2*256
      debug(3,"Direction: ",vx,",",vy)
      x-=vx
      y-=vy

    debug(2,"Residual:",res)
    return x,y
    

  def getDisplacementField(self,img_d):
    assert img_d.shape == self.shape,"Displaced image has a different size"
    dispField = np.zeros((self.numTilesX,self.numTilesY,2),np.float32)
    self.img_dArray = cuda.matrix_to_array(img_d,"C")
    self.tex_d.set_array(self.img_dArray)


    #"""
    for i in range(self.numTilesX):
      for j in range(self.numTilesY):
        """
        i,j = 8,8
        if True:
          if True:
        """
        dispField[i,j,:] = self.__getTileDisplacement(i,j)


    return dispField
    
"""

imgArray = cuda.matrix_to_array(img,"C")
img_dArray = cuda.matrix_to_array(img_d,"C")

tex.set_array(imgArray)
tex_d.set_array(img_dArray)



makeDiff.prepared_call(grid,block,devOut.gpudata,0,0)

out = devOut.get()

sat = lambda x: max(0,min(x,255))
to_uint8 = np.vectorize(sat,otypes=[np.uint8]) # Crée une fonction pour passer de floats non bornés à du uint8 sans dépassement

squareResidual = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*x[i]",arguments="float *x")

gdKernel = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*y[i]",arguments="float* x, float* y")




x,y = 0,0
for i in range(20):
  debug(3,"Iteration",i)
  makeDiff.prepared_call(grid,block,devOut.gpudata,x,y)
  vx = gdKernel(devOut,devGradX).get()/4194304/128
  vy = gdKernel(devOut,devGradY).get()/4194304/128
  x+=vx
  y+=vy
  debug(3,"Direction:",vx,vy)
  debug(3,"Value",x,y)
  debug(3,"Residual:",squareResidual(devOut).get()/4194304,"\n")

out = devOut.get()+128

cv2.imwrite("out.png",to_uint8(out))

"""
