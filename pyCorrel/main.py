#coding: utf-8
from __future__ import division, print_function
import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
import sys

import time

import pycuda.autoinit


def openImage(address):
  img = cv2.imread(address,0)
  if img is None:
    print("OpenCV failed to open",address)
    sys.exit(-1)
  return img.astype(np.float32)


imgAddr = '../Images/lena.png'
img_dAddr = '../Images/lena_d.png'

img = openImage(imgAddr)
img_d = openImage(img_dAddr)
w,h = img.shape
assert (w,h) == img_d.shape,"Images don't have the same size !"

"""
with open("kernels.cu","r") as f:
  mod = SourceModule(f.read())
"""
mod = SourceModule("""
#include <cuda.h>
#define WIDTH %d
#define HEIGHT %d

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_d;

__global__ void makeDiff(float* out, float ox, float oy)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;

  out[idy*WIDTH+idx] = tex2D(tex_d,(idx-ox)/WIDTH,(idy-oy)/HEIGHT) - tex2D(tex,(float)idx/WIDTH,(float)idy/HEIGHT);
}


__global__ void gradient(float* gradX, float* gradY)
{
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  gradX[x+WIDTH*y] = (tex2D(tex,(x+1.5f)/WIDTH,(float)y/HEIGHT)+tex2D(tex,(x+1.5f)/WIDTH,(y+1.f)/HEIGHT)-tex2D(tex,(x-.5f)/WIDTH,(float)y/HEIGHT)-tex2D(tex,(x-.5f)/WIDTH,(y+1.f)/HEIGHT))*2;
  gradY[x+WIDTH*y] = (tex2D(tex,(float)x/WIDTH,(y+1.5f)/HEIGHT)+tex2D(tex,(x+1.f)/WIDTH,(y+1.5f)/HEIGHT)-tex2D(tex,(float)x/WIDTH,(y-.5f)/HEIGHT)-tex2D(tex,(x+1.f)/WIDTH,(y-.5f)/HEIGHT))*2;
}

""" % img.shape)

makeDiff = mod.get_function("makeDiff")
gradient = mod.get_function("gradient")

tex = mod.get_texref('tex')
tex_d = mod.get_texref('tex_d')

tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
tex_d.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
tex.set_filter_mode(cuda.filter_mode.LINEAR)
tex_d.set_filter_mode(cuda.filter_mode.LINEAR)
tex.set_address_mode(0,cuda.address_mode.CLAMP)
tex_d.set_address_mode(0,cuda.address_mode.CLAMP)
tex.set_address_mode(1,cuda.address_mode.CLAMP)
tex_d.set_address_mode(1,cuda.address_mode.CLAMP)

imgArray = cuda.matrix_to_array(img,"C")
img_dArray = cuda.matrix_to_array(img_d,"C")

tex.set_array(imgArray)
tex_d.set_array(img_dArray)

devOut = gpuarray.GPUArray(img.shape,np.float32)

grid = ((w+31)//32,(h+31)//32)
block = (min(w,32),min(h,32),1)
print("Grid:",grid)
print("Block:",block)

makeDiff.prepare('Pff',texrefs=[tex,tex_d])

makeDiff.prepared_call(grid,block,devOut.gpudata,0,0)

out = devOut.get()
print("OK!")

sat = lambda x: max(0,min(x,255))
to_uint8 = np.vectorize(sat,otypes=[np.uint8]) # Crée une fonction pour passer de floats non bornés à du uint8 sans dépassement

squareResidual = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*x[i]",arguments="float *x")

gdKernel = ReductionKernel(np.float32,neutral="0",reduce_expr="a+b",map_expr="x[i]*y[i]",arguments="float* x, float* y")

gradient.prepare('PP',texrefs=[tex])

devGradX = gpuarray.GPUArray(img.shape,np.float32)
devGradY = gpuarray.GPUArray(img.shape,np.float32)

gradient.prepared_call(grid,block,devGradX.gpudata,devGradY.gpudata) # Crée les gradients de l'image

x,y = 0,0
for i in range(20):
  makeDiff.prepared_call(grid,block,devOut.gpudata,x,y)
  vx = gdKernel(devOut,devGradX).get()/4194304/128
  vy = gdKernel(devOut,devGradY).get()/4194304/128
  x+=vx
  y+=vy
  print(vx,vy)
  print(x,y)
  print(squareResidual(devOut).get()/4194304,"\n")

out = devOut.get()+128

cv2.imwrite("out.png",to_uint8(out))
