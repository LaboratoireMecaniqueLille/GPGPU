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
    self.dispField = np.zeros((self.numTilesX,self.numTilesY,2),np.float32)
    self.iterCoeffs = [1.1,1.5,3,5,10]
    self.t_w,self.t_h = self.w/numTilesX,self.h/numTilesY
    self.it_w,self.it_h = int(round(self.t_w)),int(round(self.t_h))
    self.t_shape = self.it_w,self.it_h
    self.normalizedTileWidth = self.t_w/self.w
    self.normalizedTileHeight = self.t_h/self.h
    self.grid = ((self.w+31)//32,(self.h+31)//32)
    self.block = (min(self.w,32),min(self.h,32),1)
    self.t_grid = ((self.it_w+31)//32,((self.it_h+31)//32))
    self.t_block = (min(self.it_w,32),min(self.it_h,32),1)

    debug(3,"Dimensions:",self.w,self.h)
    debug(2,"Grid:",self.grid)
    debug(2,"Block:",self.block)
    debug(2,"Tile grid:",self.t_grid)
    debug(2,"Tile block",self.t_block)

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

  def computeDiff(self,tx,ty,x,y):
      self.__makeDiff.prepared_call(self.t_grid,self.t_block,\
      self.devTileOut.gpudata,\
      tx*self.normalizedTileWidth,ty*self.normalizedTileHeight,\
      x/self.w,y/self.h)

  def residual(self):
      return self.__squareResidual(self.devTileOut).get()/self.t_w/self.t_h

  def gradientDescent(self,tx,ty):
      self.__gdProduct.prepared_call(self.t_grid,self.t_block,\
      self.devTempX.gpudata,self.devTempY.gpudata,\
      self.devTileOut.gpudata,\
      tx*self.normalizedTileWidth,ty*self.normalizedTileHeight)

      vx = gpuarray.sum(self.devTempX).get()/(self.t_w*self.t_h)/32768
      vy = gpuarray.sum(self.devTempY).get()/(self.t_w*self.t_h)/32768
      return vx,vy
    

  def __getTileDisplacement(self,tx,ty,ox=0,oy=0):
    assert tx < self.numTilesX and ty < self.numTilesY,"__getTileDisplacement call out of bounds"
    debug(2,"Tile",tx,",",ty)
    x,y=ox,oy
    self.computeDiff(tx,ty,x,y)
    res = self.residual()
    for i in self.iteration:
      debug(3,"Iteration",i)
      debug(3,"x=",x,"y=",y)
      vx,vy = self.gradientDescent(tx,ty)
      debug(3,"vx=",vx,"vy=",vy)
      for c in range(5):
        vx*=self.iterCoeffs[c]
        vy*=self.iterCoeffs[c]
        x-=vx
        y-=vy
        self.computeDiff(tx,ty,x,y)
        oldres=res
        res = self.residual()
        debug(3,"Add",c,":\nvx=",vx,"vy=",vy,", residual:",res)
        if res > oldres:
          debug(3,"Residual increasing, reverting")
          x+=vx
          y+=vy
          res = oldres
          break
      if c == 0:
        debug(3,"Cannot progress any further, returning.")
        debug(2,"Final residual:",res)
        return x,y
    debug(2,"Final residual:",res)
    return x,y

  def getDisplacementField(self,img_d):
    assert img_d.shape == self.shape,"Displaced image has a different size"
    debug(2,"Working on a",self.w,",",self.h,"image")
    self.img_dArray = cuda.matrix_to_array(img_d,"C")
    self.tex_d.set_array(self.img_dArray)
    #Uncomment the following line to work on a single tile
    #"""
    for i in range(self.numTilesX):
      for j in range(self.numTilesY):
        """
    i,j = 0,12
    if True:
      if True:
      #"""
        self.dispField[i,j,:] = self.__getTileDisplacement(i,j,*self.dispField[i,j,:])


    return self.dispField
    
  def setOriginalDisplacement(self,array):
    assert array.shape == (self.numTilesX,self.numTilesY,2),"Incorrect initialisation of the displacement field"
    self.dispField=array

class Resize:
  def __init__(self):
    mod = SourceModule("""
  texture<float, cudaTextureType2D, cudaReadModeElementType> tex;

  __global__ void resize(float* out, int w, int h)
  {
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int idy = threadIdx.y+blockIdx.y*blockDim.y;

    out[idx+w*idy] = tex2D(tex,(float)idx/w,(float)idy/h);
  }
    """)
    self.devResize = mod.get_function("resize")
    self.tex = mod.get_texref('tex')
    self.tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    self.tex.set_filter_mode(cuda.filter_mode.LINEAR)
    self.tex.set_address_mode(0,cuda.address_mode.CLAMP)
    self.tex.set_address_mode(1,cuda.address_mode.CLAMP)

  def resize(self,img):
    devImg = gpuarray.to_gpu(img)
    w,h = img.shape[0]//2,img.shape[1]//2
    array = cuda.matrix_to_array(img,"C")
    self.tex.set_array(array)

    devOut = gpuarray.GPUArray((w,h),np.float32)
    grid = ((w+31)//32,(h+31)//32)
    block = (min(w,32),min(h,32),1)
    self.devResize(devOut.gpudata,np.uint32(h),np.uint32(w),grid=grid,block=block)
    return devOut.get()
