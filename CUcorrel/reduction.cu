#include <iostream>

#include "CUcorrel.h"

using namespace std;

__device__ void warpReduce(volatile float* sh_data, uint tid)
{
  sh_data[tid] += sh_data[tid+32];
  sh_data[tid] += sh_data[tid+16];
  sh_data[tid] += sh_data[tid+8];
  sh_data[tid] += sh_data[tid+4];
  sh_data[tid] += sh_data[tid+2];
  sh_data[tid] += sh_data[tid+1];
}

__global__ void reduce(float *array,uint size)
{
  uint tid = threadIdx.x;
  uint id = 2*blockDim.x*blockIdx.x;
  __shared__ float sh_data[BLOCKSIZE];
  sh_data[tid] = array[id+tid]+array[id+tid+blockDim.x];
  __syncthreads();
  for(uint s = blockDim.x/2; s >= 1; s >>=1)
  {
    if(tid < s)
    {sh_data[tid] += sh_data[tid+s];}
    __syncthreads();
  }
  if(tid == 0)
  {array[blockIdx.x] = sh_data[0];}


}

void sum(float *array, uint size)
{
  while(size > 1)
  {
    uint blocksize = (size+2*BLOCKSIZE-1)/2/BLOCKSIZE;
    reduce<<<blocksize,min(size/2,BLOCKSIZE)>>>(array,size/2);
    size = blocksize;
  }
}
