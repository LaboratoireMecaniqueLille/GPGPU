#include <iostream>
#include "main.h"

#define BLOCKSIZE 1024

using namespace std;

__device__ void warpReduce(volatile TYPE* sh_data, uint tid)
{
  sh_data[tid] += sh_data[tid+32];
  sh_data[tid] += sh_data[tid+16];
  sh_data[tid] += sh_data[tid+8];
  sh_data[tid] += sh_data[tid+4];
  sh_data[tid] += sh_data[tid+2];
  sh_data[tid] += sh_data[tid+1];
}

__global__ void reduce(TYPE *array,uint size)
{
  uint tid = threadIdx.x;
  uint id = 2*blockDim.x*blockIdx.x;
  __shared__ TYPE sh_data[BLOCKSIZE];
  sh_data[tid] = array[id+tid]+array[id+tid+blockDim.x];
  __syncthreads();
  //for(uint s = blockDim.x/2; s > 32; s >>=1)
  for(uint s = blockDim.x/2; s >= 1; s >>=1)
  {
    if(tid < s)
    {sh_data[tid] += sh_data[tid+s];}
    __syncthreads();
  }
  /*if(tid < 32) // Pour grapiller quelques millisecondes en réduisant le flux d'instruction sur les derniers warps car ils sont de toute façon synchrones. MAIS empêche le cas ou le nombre d'éléments n'est pas un multiple de 32 => Gênant.
  {warpReduce(sh_data,tid);}*/
  if(tid == 0)
  {array[blockIdx.x] = sh_data[0];}


}

void sum(TYPE *array, uint size)
{
  while(size > 1)
  {
    cout << "Reste à sommer: " << size << endl;
    uint blocksize = (size+2*BLOCKSIZE-1)/2/BLOCKSIZE;
    cout << "La somme se fera en " << blocksize << " bloc(s) de " << min(size/2,BLOCKSIZE) << endl;
    reduce<<<blocksize,min(size/2,BLOCKSIZE)>>>(array,size/2);
    size = blocksize;
  }
}
