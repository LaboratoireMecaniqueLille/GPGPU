#coding: utf-8
from __future__ import division, print_function
import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import sys

import pycuda.autoinit

img_addr = 'Images/lena.png'
out_addr = 'out.png'
offX = 10.58
offY = 35.4257

img = cv2.imread(img_addr,0)
if img is None:
  print("Erreur lors de l'ouverture de",img_addr)
  sys.exit(-1)

img = img.astype(np.float32)

src_mod = """
#include <cuda.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;

__global__ void interp(float* out, float ox, float oy, int w, int h)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;
  
  out[idy*w+idx] = tex2D(tex,(idx-ox)/w,(idy-oy)/h);
}
"""

mod = SourceModule(src_mod)
interp=mod.get_function("interp")

h,w = img.shape
print("w={}, h={}".format(w,h))

desc = cuda.ArrayDescriptor()
desc.width = w
desc.height = h
desc.format = cuda.dtype_to_array_format(np.float32)
desc.num_channels = 1

devArray = cuda.matrix_to_array(img,"C")


tex=mod.get_texref('tex')

tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
tex.set_filter_mode(cuda.filter_mode.LINEAR)
tex.set_address_mode(0,cuda.address_mode.CLAMP)
tex.set_address_mode(1,cuda.address_mode.CLAMP)


tex.set_array(devArray)


arg_types = "Pffii"
devOut = gpuarray.GPUArray(img.shape,np.float32)

grid = ((w+31)//32,(h+31)//32)
block = (min(w,32),min(h,32),1)
print("Grid:",grid)
print("Block:",block)

interp.prepare(arg_types,texrefs=[tex])
interp.prepared_call(grid,block,devOut.gpudata,offX,offY,*img.shape)

out = devOut.get()
#print(out[256,458])

cv2.imwrite(out_addr,out.astype(np.uint8))
print("Image décalée de {},{} écrite sous \"{}\"".format(offX,offY,out_addr))
