#coding: utf-8
from __future__ import division, print_function
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from time import time
from scipy import signal

import pycuda.autoinit

def redStr(*s):
  return '\033[31m\033[1m'+str.join(" ",[str(i) for i in s])+'\033[0m'

def debug(*args):
  return

class gridCorrel:
  """
  Class meant to continuously compute the local displacement of an image compared to an original one
  """
  def __init__(self,originalImage,numTilesX,numTilesY,**kwargs):
    # -- To print debug infos if asked to (3 different levels of verbosity) --
    verbose=kwargs.get('verbose',0)
    if verbose:
      global debug
      def debug(l,*args):
        if verbose >= l:
          print(*args)

    # -- Assigning useful values --
    self.orig = originalImage
    assert type(self.orig)==np.ndarray,"The original image must be a numpy.ndarray"
    assert len(self.orig.shape) == 2,"The original image is not a 2D array"
    self.nIter = kwargs.get('iterations',3)
    self.iteration = range(self.nIter)
    self.nAdds = kwargs.get('adds',3) # Number of times a computed vector will be added unless residual increases (and each times multiplied by iterCoeff[i])
    self.shape = originalImage.shape
    self.w,self.h = self.shape
    self.numTilesX,self.numTilesY = numTilesX,numTilesY
    self.dispField = np.zeros((self.numTilesX,self.numTilesY,2),np.float32)
    self.lastX = 0
    self.lastY = 0
    self.iterCoeffs = kwargs.get('addCoeffs',[2,5,8]+[10]*max(0,self.nAdds-3))
    self.maxRes = kwargs.get('maxRes',800) # Max residual before considering convergence failed
    self.t_w,self.t_h = self.w/numTilesX,self.h/numTilesY
    self.it_w,self.it_h = int(round(self.t_w)),int(round(self.t_h))
    self.t_shape = self.it_w,self.it_h
    self.normalizedTileWidth = self.t_w/self.w
    self.normalizedTileHeight = self.t_h/self.h
    self.grid = ((self.w+31)//32,(self.h+31)//32)
    self.block = (min(self.w,32),min(self.h,32),1)
    self.t_grid = ((self.it_w+31)//32,((self.it_h+31)//32))
    self.t_block = (min(self.it_w,32),min(self.it_h,32),1)
    self.res= np.ones((self.numTilesX,self.numTilesY))*1e38
    cen = .3
    bor = (1-cen)/8.
    self.filterMatrix=np.array([[bor,bor,bor],[bor,cen,bor],[bor,bor,bor]])

    debug(3,"Dimensions:",self.w,self.h)
    debug(3,"Grid:",self.grid)
    debug(3,"Block:",self.block)
    debug(3,"Tile grid:",self.t_grid)
    debug(3,"Tile block",self.t_block)

    # -- Creating GPU arrays used to store images --
    self.devDiff = gpuarray.GPUArray(self.t_shape,np.float32)
    self.devGradX = gpuarray.GPUArray(self.shape,np.float32)
    self.devGradY = gpuarray.GPUArray(self.shape,np.float32)
    self.devTempX = gpuarray.GPUArray((self.it_w,self.it_h),np.float32)
    self.devTempY = gpuarray.GPUArray((self.it_w,self.it_h),np.float32)

    # -- Reading kernel file (note: there are #DEFINE directives to set the img and tile size at compilation time) --
    kernelFile = kwargs.get('kernelFile',"kernels.cu")
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
    # To create the table of difference between original and corrected displaced texture 
    self.__makeDiff = self.mod.get_function("makeDiff")
    self.__makeDiff.prepare('Pffff',texrefs=[self.tex,self.tex_d])
    # To compute the gradients of the original image (only called when setting the original image)
    self.__gradient = self.mod.get_function("gradient")
    self.__gradient.prepare('PP',texrefs=[self.tex])
    # To create the table to be reduced to get the direction to move the picture to
    self.__gdProduct = self.mod.get_function("gdProduct")
    self.__gdProduct.prepare('PPPff',texrefs=[self.texGradX,self.texGradY])
    # To compute the residual given the diff image
    self.__squareResidual = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*x[i]",arguments="float *x")

    # -- Assigning the first image --
    self.setOriginalImage(self.orig)

    # -- For profiling --
    self.t0 = 0
    self.t1 = 0
    self.t2 = 0

  def setOriginalImage(self,image):
    """
    To set the "original" image (the reference from which the displacement field will be computed)
    """
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

  def __computeDiff(self,tx,ty,x,y):
    """
    Wrapper to write in self.devDiff the difference between the two textures given the tile coordinates and the displacement in pixels
    tx,ty: integers refering to the tile's coordinates
    x,y: offset of the tile in pixels
    """
    self.__makeDiff.prepared_call(self.t_grid,self.t_block,\
    self.devDiff.gpudata,\
    tx*self.normalizedTileWidth,ty*self.normalizedTileHeight,\
    x/self.w,y/self.h)

  def __residual(self,tx,ty,x,y):
    """
    Wrapper to compute the residual once self.devDiff has been computed
    """
    if (x,y) != (self.lastX,self.lastY): # To compute the difference only if necessarry
      self.lastX=x
      self.lastT=y
      self.__computeDiff(tx,ty,x,y)
    return self.__squareResidual(self.devDiff).get()/self.t_w/self.t_h

  def __gradientDescent(self,tx,ty,x,y):
    """
    Wrapper to search for the direction of convergence of the tile (tx,ty) once self.devDiff has been computed
    tx,ty: integers refering to the tile's coordinates
    """
    if (x,y) != (self.lastX,self.lastY): # To compute the difference only if necessarry
      self.lastX=x
      self.lastT=y
      self.__computeDiff(tx,ty,x,y)
    self.__gdProduct.prepared_call(self.t_grid,self.t_block,\
    self.devTempX.gpudata,self.devTempY.gpudata,\
    self.devDiff.gpudata,\
    tx*self.normalizedTileWidth,ty*self.normalizedTileHeight)

    vx = gpuarray.sum(self.devTempX).get()/(self.t_w*self.t_h)/32768
    vy = gpuarray.sum(self.devTempY).get()/(self.t_w*self.t_h)/32768
    return vx,vy
  
  def __getTileDisplacement(self,tx,ty,ox=0,oy=0):
    """
    Used to compute each vector of the dispacement field
    Iteratively moves the tile of the given image in order to place it on the original image
    tx,ty: integers refering to the tile's coordinates
    ox,oy: Initialization of the displacement in pixels
    """
    assert tx < self.numTilesX and ty < self.numTilesY,"__getTileDisplacement call out of bounds"
    debug(2,"Tile",tx,",",ty)
    x,y=ox,oy

    # -- Computes the residual before iterating (sometimes the direction is wrong and increases the residual) --
    self.res[tx,ty] = self.__residual(tx,ty,x,y)
    # -- Let's start iterating ! --
    for i in self.iteration:
      debug(3,"Iteration",i)
      debug(3,"x=",x,"y=",y)
      newX,newY = self.__iterate(tx,ty,x,y)
      if (newX,newY) == (x,y):
        debug(3,"Not moving anymore, stopping iterations")
        #debug(2,"Displacement:",x,",",y,"\nResidual:",self.res[tx,ty])
        debug(2,"Displacement:",x,",",y,"\nResidual:",self.res[tx,ty] if self.res[tx,ty]<self.maxRes else redStr(self.res[tx,ty]))
        return x,y
      x,y = newX,newY
    debug(2,"Displacement:",x,",",y,"\nResidual:",self.res[tx,ty] if self.res[tx,ty]<self.maxRes else redStr(self.res[tx,ty]))
    return x,y
  
  def __iterate(self,tx,ty,x,y):
    """
    Computes the research direction to converge towards the lowest residual and adds it multiple times to approach the solution
    """
    #TODO: profile and optimize (98% of execution time is this method)
    ## -- Get the research direction  --
    self.t0 += time()
    vx,vy = self.__gradientDescent(tx,ty,x,y)
    self.t1 += time()
    debug(3,"vx=",vx,"vy=",vy,"oldres:",self.res[tx,ty])
    for c in range(self.nAdds):
      # -- Still approaching the minimum ? Let's go faster (see self.iterCoeffs in __init__, the more iterations it takes, the faster we grow the direction)
      vx*=self.iterCoeffs[c]
      vy*=self.iterCoeffs[c]
      # -- Updating x and y --
      x-=vx
      y-=vy
      # -- Compute new diff --
      oldres=self.res[tx,ty]
      # -- Update residual --
      self.res[tx,ty] = self.__residual(tx,ty,x,y)
      debug(3,"Add",c,":\nvx=",vx,"vy=",vy,", residual:",self.res[tx,ty])
      if self.res[tx,ty] > oldres:
        # -- Oops, went to far ! Let's revert this iteration and start with a new direction --
        debug(3,"Residual increasing, reverting")
        x+=vx
        y+=vy
        self.res[tx,ty] = oldres
        break
    self.t2 += time()
    return x,y

  def getDisplacementField(self,img_d):
    assert img_d.shape == self.shape,"Displaced image has a different size"
    debug(1,"Working on a",self.w,"x",self.h,"image")
    self.img_dArray = cuda.matrix_to_array(img_d,"C")
    self.tex_d.set_array(self.img_dArray)
    self.converged = -np.ones((self.numTilesX,self.numTilesY),dtype=int)
    t0=t1=0
    for i in self.iteration:
      #print("Iteration",i)
      #    """
      for tx in range(self.numTilesX):
        for ty in range(self.numTilesY):
          if self.converged[tx,ty] >= 0:
            break
          if self.res[tx,ty] < self.maxRes/20:
            debug(3,"Residual low enough:",self.res[tx,ty],". Marking as converged")
            self.converged[tx,ty] = i
            break
          """
          tx,ty = 8,8
          #"""
          x,y = self.__iterate(tx,ty,*self.dispField[tx,ty,:]) # ~98% of execution time
          if np.array_equal((x,y),self.dispField[tx,ty,:]):
            debug(3,"Not moving anymore, marking tile as converged...")
            self.converged[tx,ty] = i
          else:
            self.dispField[tx,ty,:] = [x,y]
      self.__filterDispField()
    debug(3,self.converged)
    debug(1,"Average number of iterations:",np.mean(np.where(self.converged<0,i,self.converged)))
    debug(1,"Average residual:",self.res.mean())
    debug(1,(self.res<self.maxRes).sum(),"/",self.numTilesX*self.numTilesY,"below",self.maxRes)
    debug(1,"T01:",(self.t1-self.t0)*1000,"ms.")
    debug(1,"T12:",(self.t2-self.t1)*1000,"ms.")
    return self.dispField

  def __filterDispField(self):
    """
    Filters the direction after each iteration (and keeps the 'already good' ones as is)
    Great results !
    (but is it fast ?)
    """
    #TODO:  See the border issue with filtering (diverging because of an out of bound value)
    #       Allow changing of the filterfunction (simply by setting weight of central vector or advanced by passing directly a function
    self.dispField[:,:,0] = np.where(self.converged<0,signal.convolve2d(self.dispField[:,:,0],self.filterMatrix,mode='same',boundary='symm'),self.dispField[:,:,0])
    self.dispField[:,:,1] = np.where(self.converged<0,signal.convolve2d(self.dispField[:,:,1],self.filterMatrix,mode='same',boundary='symm'),self.dispField[:,:,1])
    
  def getLastResGrid(self):
    """
    Return the residuals of the tiles after calling getDisplacementField()
    Can be useful to remove incorrect data
    """
    return self.res

  def setOriginalDisplacement(self,array):
    """
    To set the original displacement field to something different than zeros
    """
    assert array.shape == (self.numTilesX,self.numTilesY,2),"Incorrect initialisation of the displacement field"
    self.dispField=array

  def showDisplacement(self,**kwargs):
    import matplotlib.pyplot as plt
    norm = kwargs.get('norm',10)
    scale = kwargs.get('scale',min(self.w,self.h)/400.)
    border=kwargs.get('border',False)
    plt.imshow(self.orig,cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
    ax=plt.axes()
    if border:
      rangeX=range(self.numTilesX)
      rangeY=range(self.numTilesY)
    else:
      rangeX = range(1,self.numTilesX-1)
      rangeY = range(1,self.numTilesY-1)

    for i in rangeX:
      for j in rangeY:
        if self.converged[i,j]>=0:
          color='green'
        elif self.res[i,j] < self.maxRes:
          color='red'
        else:
          color='blue'
        ax.arrow((i+.5)*self.t_w, (j+.5)*self.t_h, self.dispField[i,j,0]*norm, self.dispField[i,j,1]*norm, width = scale, head_width=4*scale, head_length=8*scale, fc=color, ec=color)
    plt.show()





class Resize:
  """
  Class meant to resize 2D images using linear interpolation, accelerated by GPU
  It takes advantage of the hardware interpolation of textures to resize efficiently
  """
  def __init__(self):
    """
    No parameters required, just to initialize and compile the kernel and configure the texture
    """
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
    """
    Simply takes a numpy array of an image (of type np.float32) and returns an image with half the size (new_x = floor(x/2), idem for y)
    """
    devImg = gpuarray.to_gpu(img)
    w,h = img.shape[0]//2,img.shape[1]//2
    array = cuda.matrix_to_array(img,"C")
    self.tex.set_array(array)

    devOut = gpuarray.GPUArray((w,h),np.float32)
    grid = ((w+31)//32,(h+31)//32)
    block = (min(w,32),min(h,32),1)
    self.devResize(devOut.gpudata,np.uint32(h),np.uint32(w),grid=grid,block=block)
    return devOut.get()
