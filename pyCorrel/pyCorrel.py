#coding: utf-8
from __future__ import division, print_function
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from scipy import signal

import pycuda.autoinit

def redStr(*s):
  return '\033[31m\033[1m'+str.join(" ",[str(i) for i in s])+'\033[0m'

def debug(*args):
  return

class gridCorrel:
  """
  Class meant to repeatedly compute the local displacement of an image compared to an original one
  """
  def __init__(self,originalImage,**kwargs):
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
    self.nIter = kwargs.get('iterations',2)
    self.iteration = range(self.nIter)
    self.nAdds = kwargs.get('adds',2) # Number of times a computed vector will be added unless residual increases (and each times multiplied by iterCoeff[i])
    self.shape = originalImage.shape
    self.w,self.h = self.shape
    self.numTilesX,self.numTilesY = kwargs.get("numTiles",(16,16))
    self.dispField = np.zeros((self.numTilesX,self.numTilesY,2),np.float32)
    self.lastX = 0
    self.lastY = 0
    self.iterCoeffs = kwargs.get('addCoeffs',[2,5,8]+[10]*max(0,self.nAdds-3))
    self.maxRes = kwargs.get('maxRes',800) # Max residual before considering convergence failed
    self.t_shape = kwargs.get('tileSize',(int(round(self.w/self.numTilesX)),int(round(self.h/self.numTilesY))))
    self.t_w,self.t_h = self.t_shape
    #self.it_w,self.it_h = int(round(self.t_w)),int(round(self.t_h))
    self.normalizedTileWidth = self.t_w/self.w
    self.normalizedTileHeight = self.t_h/self.h
    self.grid = ((self.w+31)//32,(self.h+31)//32)
    self.block = (min(self.w,32),min(self.h,32),1)
    self.t_grid = ((self.t_w+31)//32,((self.t_h+31)//32))
    self.t_block = (min(self.t_w,32),min(self.t_h,32),1)
    self.res= np.ones((self.numTilesX,self.numTilesY))*1e38
    cen = .7
    bor = (1-cen)/8.
    self.filterMatrix=np.array([[bor,bor,bor],[bor,cen,bor],[bor,bor,bor]])
    self.customFilter = kwargs.get('filter',False)

    debug(3,"Dimensions:",self.w,self.h)
    debug(3,"Grid:",self.grid)
    debug(3,"Block:",self.block)
    debug(3,"Tile grid:",self.t_grid)
    debug(3,"Tile block",self.t_block)

    # -- Creating GPU arrays used to store images --
    self.devDiff = gpuarray.GPUArray(self.t_shape,np.float32)
    self.devGradX = gpuarray.GPUArray(self.shape,np.float32)
    self.devGradY = gpuarray.GPUArray(self.shape,np.float32)
    self.devTempX = gpuarray.GPUArray((self.t_w,self.t_h),np.float32)
    self.devTempY = gpuarray.GPUArray((self.t_w,self.t_h),np.float32)

    # -- Reading kernel file (note: there are #DEFINE directives to set the img and tile size at compilation time) --
    kernelFile = kwargs.get('kernelFile',"kernels.cu")
    with open(kernelFile,"r") as f:
      self.mod = SourceModule(f.read()%(self.shape+self.t_shape))

    # -- Creating and setting properties of the textures --
    self.tex = self.mod.get_texref('tex')
    self.tex_d = self.mod.get_texref('tex_d')
    self.texGradX = self.mod.get_texref('texGradX')
    self.texGradY = self.mod.get_texref('texGradY')

    for tex in [self.tex,self.tex_d,self.texGradX,self.texGradY]:
      tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
      tex.set_filter_mode(cuda.filter_mode.LINEAR)
      tex.set_address_mode(0,cuda.address_mode.CLAMP)
      tex.set_address_mode(1,cuda.address_mode.CLAMP)

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
    tx/self.numTilesX,ty/self.numTilesY,\
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
    tx/self.numTilesX,ty/self.numTilesY)
    #tx*self.normalizedTileWidth,ty*self.normalizedTileHeight)

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

    # -- Computes the residual before iterating (sometimes the direction is somehow wrong and increases the residual in one iteration) --
    self.res[tx,ty] = self.__residual(tx,ty,x,y)
    # -- Let's start iterating ! --
    for i in self.iteration:
      debug(3,"Iteration",i)
      debug(3,"x=",x,"y=",y)
      newX,newY = self.__iterate(tx,ty,x,y)
      if (newX,newY) == (x,y):
        debug(3,"Not moving anymore, stopping iterations")
        debug(2,"Displacement:",x,",",y,"\nResidual:",self.res[tx,ty] if self.res[tx,ty]<self.maxRes else redStr(self.res[tx,ty]))
        return x,y
      x,y = newX,newY
    debug(2,"Displacement:",x,",",y,"\nResidual:",self.res[tx,ty] if self.res[tx,ty]<self.maxRes else redStr(self.res[tx,ty]))
    return x,y
  
  def __iterate(self,tx,ty,x,y):
    """
    Computes the research direction to converge towards the lowest residual and adds it multiple times to approach the solution
    """
    #TODO: profile and optimize
    ## -- Get the research direction  --
    vx,vy = self.__gradientDescent(tx,ty,x,y)
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
          if self.res[tx,ty] < self.maxRes/100:
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
    return self.dispField

  def __filterDispField(self):
    """
    Filters the direction after each iteration (and keeps the 'already good' ones as is)
    """
    if self.customFilter:
      self.dispField[:,:,0] = self.customFilter(self.dispField[:,:,0])
      self.dispField[:,:,1] = self.customFilter(self.dispField[:,:,1])
    else:
      self.dispField[:,:,0] = np.where(self.converged<0,signal.convolve2d(self.dispField[:,:,0],self.filterMatrix,mode='same',boundary='fill'),self.dispField[:,:,0])
      self.dispField[:,:,1] = np.where(self.converged<0,signal.convolve2d(self.dispField[:,:,1],self.filterMatrix,mode='same',boundary='fill'),self.dispField[:,:,1])

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
    import matplotlib.patches as patches
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
          color='red'
        elif self.res[i,j] < self.maxRes:
          color='orange'
        else:
          color='blue'
        ax.arrow((i+.5)/self.numTilesX*self.w, (j+.5)/self.numTilesX*self.h, self.dispField[i,j,0]*norm, self.dispField[i,j,1]*norm, width = scale, head_width=4*scale, head_length=8*scale, fc=color, ec=color)
    ax.add_patch(patches.Rectangle((1.5*self.w/self.numTilesX,.5*self.w/self.numTilesY),self.t_w,self.t_h,fill=False,color='green',linewidth=3))
    ax.add_patch(patches.Rectangle((.5*self.w/self.numTilesX,1.5*self.w/self.numTilesY),self.t_w,self.t_h,fill=False,color='green',linewidth=3))
    ax.add_patch(patches.Rectangle((.5*self.w/self.numTilesX,.5*self.w/self.numTilesY),self.t_w,self.t_h,fill=False,color='yellow',linewidth=2))
    plt.show()


class CorrelParam:
  def __repr__(self):
    try:
      return "Image: {}x{}, grid: {}x{}, tile: {}x{}, maxRes: {}".format(self.shape[0],self.shape[1],self.gridShape[0],self.gridShape[1],self.it_w,self.it_h,self.maxRes)
    except AttributeError:
      return "Parameters not defined yet"

  def __call__(self):
    return {"numTiles":self.gridShape ,"tileSize":(self.it_w,self.it_h),"maxRes":self.maxRes}


class PyramidalCorrel:
  def __init__(self, image, numOfStages, gridShape,**kwargs):
    self.verbose = kwargs.get("verbose",0)
    self.R = Resize()
    self.nStages = numOfStages
    self.resamplingFactor=kwargs.get("factor",2) # The size factor between each stage of the pyramid
    self.p = [CorrelParam() for i in range(self.nStages)]
    self.p[0].shape = image.shape
    self.p[0].gridShape = gridShape
    

    self.overlap = kwargs.get("overlap",1) # Tiles overlaping factor. 1 means tiles border are touching, 2 means tiles twice as big as overlap=1 (-> 4 times the surface)
    self.p[0].maxRes = kwargs.get('maxRes',800)

    self.img = [image]
    for i in range(1,self.nStages):
      self.img.append(self.R.resize(self.img[i-1],self.resamplingFactor))
      self.p[i].shape = self.img[i].shape
    self.prepareParameters()
    self.correl = []
    for i in range(self.nStages):
      kwargs.update(self.p[i]())
      self.correl.append(gridCorrel(self.img[i],**kwargs))


  def prepareParameters(self):
    for i in range(self.nStages):
      if i != 0:
        self.p[i].gridShape = int(round(self.p[0].gridShape[0]/self.resamplingFactor**i)),\
        int(round(self.p[0].gridShape[1]/self.resamplingFactor**i))

        self.p[i].maxRes = self.p[i-1].maxRes/(1+self.resamplingFactor)*2 # Completely empirical (to confirm)

      self.p[i].t_w = self.overlap*self.p[i].shape[0]/self.p[i].gridShape[0]
      self.p[i].t_h = self.overlap*self.p[i].shape[1]/self.p[i].gridShape[1]
      self.p[i].it_w = int(round(self.p[i].t_w))
      self.p[i].it_h = int(round(self.p[i].t_h))
      if self.verbose:
        print("Parameters for stage {}".format(i),self.p[i])

  def compute(self, img_d):
    img_d = [img_d]
    for i in range(self.nStages-1):
      img_d.append(self.R(img_d[i],self.resamplingFactor))
    for i in reversed(range(self.nStages)):
      self.correl[i].getDisplacementField(img_d[i])
      self.correl[i].showDisplacement()




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
    if(idx < w && idy < h)
      out[idx+w*idy] = tex2D(tex,(float)idx/w,(float)idy/h);
  }
    """)
    self.devResize = mod.get_function("resize")
    self.tex = mod.get_texref('tex')
    self.tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    self.tex.set_filter_mode(cuda.filter_mode.LINEAR)
    self.tex.set_address_mode(0,cuda.address_mode.CLAMP)
    self.tex.set_address_mode(1,cuda.address_mode.CLAMP)

  def resize(self,img,factor=2):
    """
    Simply takes a numpy array of an image (of type np.float32) and returns an image with half the size (new_x = floor(x/2), idem for y)
    """
    devImg = gpuarray.to_gpu(img)
    array = cuda.matrix_to_array(img,"C")
    self.tex.set_array(array)
    w,h = int(round(img.shape[0]/factor)),int(round(img.shape[1]/factor))

    devOut = gpuarray.GPUArray((w,h),np.float32)
    grid = ((w+31)//32,(h+31)//32)
    block = (min(w,32),min(h,32),1)
    self.devResize(devOut.gpudata,np.uint32(h),np.uint32(w),grid=grid,block=block)
    return devOut.get()

  def __call__(self,img,factor=2):
    return self.resize(img,factor)
