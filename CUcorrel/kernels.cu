#include <iostream>
#include "CUcorrel.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
cudaArray* cuArray;
float *devSq;

using namespace std;

__global__ void deform2D(float *devOut, float2* devU, float *devParam)
{
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x < WIDTH && y < HEIGHT)
  {
    uint id = x+y*WIDTH;
    float2 u;
    u.x = 0;
    u.y = 0;
    for(int i = 0; i < PARAMETERS; i++)
    {
      u.x += devParam[i]*devU[i*WIDTH*HEIGHT+id].x;
      u.y += devParam[i]*devU[i*WIDTH*HEIGHT+id].y;
    }
    devOut[id] = tex2D(tex,(x+.5-u.x)/WIDTH,(y+.5-u.y)/HEIGHT);
  }
}

__global__ void lsq(float* out, float* devA, float* devB, int length)
{
  uint id = blockDim.x*blockIdx.x+threadIdx.x;
  if(id > length)
  {return;}
  out[id] = (devA[id]-devB[id])*(devA[id]-devB[id]);
}

__global__ void reduce(float* data, uint size)
{
  //Réduit efficacement (ou pas) en sommant tout un tableau et en écrivant les somme restantes de chaque bloc au début du tableau (à appeler plusieurs fois pour sommer plus de 1024 éléments).
  uint id = blockDim.x*blockIdx.x+threadIdx.x;
  if(id > size) //Si appelé plus que nécessaire, quitter
  {return;}
  __shared__ float array[BLOCKSIZE]; //Contient les éléments de chaque bloc, utilisé pour stocker les valeurs temporaires
  array[threadIdx.x] = data[id]; //Chaque thread copie une valeur
  __syncthreads(); // On attend tout le monde pour être sûr d'avoir toutes les valeurs écrites
  for(unsigned int s = blockDim.x/2;s>0;s/=2) //On divise par 2 le nombre de sommes à effectuer à chaque fois -> à optimiser car une partie des threads tourne dans le vide...
  {
    if(threadIdx.x < s)
    {array[threadIdx.x] += array[threadIdx.x+s];} //On effectue ladite somme...
  __syncthreads(); // On attend que tout le monde ai fini avant de réitérer
  }
  if(threadIdx.x == 0)
  {data[blockIdx.x] = array[0];} // Le thread de tête écrit le résultat de son bloc dans le tableau
}

__global__ void gradient(float* gradX, float* gradY)
{
  //Utilise l'algo le plus simple: les différences centrées.
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x >= WIDTH || y >= HEIGHT)
  {return;}
  gradX[x+y*WIDTH]=tex2D(tex,(x+1.)/WIDTH,(y+.5)/HEIGHT)-tex2D(tex,(float)x/WIDTH,(y+.5)/HEIGHT);
  gradY[x+y*WIDTH]=tex2D(tex,(x+.5)/WIDTH,(y+1.)/HEIGHT)-tex2D(tex,(x+.5)/WIDTH,(float)y/HEIGHT);
}



float residuals(float* devData1, float* devData2, uint size)
{
/*
TODO: Optimiser le kernel de réduction pour sommer tous les éléments !
voir le lien ci dessous:
http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
*/
  lsq<<<(HEIGHT*WIDTH+BLOCKSIZE-1)/BLOCKSIZE,min(HEIGHT*WIDTH,BLOCKSIZE)>>>(devSq, devData1, devData2, HEIGHT*WIDTH);
  while(size>1)
  {
    reduce<<<(size+BLOCKSIZE-1)/BLOCKSIZE,min(size,BLOCKSIZE)>>>(devSq,size);
    size = (size+BLOCKSIZE-1)/BLOCKSIZE;
  }
  float out;
  cudaMemcpy(&out,devSq,sizeof(float),cudaMemcpyDeviceToHost);
  return out;
}

void initCuda(float* data)
{
  cudaMalloc(&devSq,HEIGHT*WIDTH*sizeof(float));
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaMallocArray(&cuArray, &channelDesc,WIDTH,HEIGHT);
  tex.normalized = true;
  tex.filterMode = cudaFilterModeLinear;//cudaFilterModePoint;
  tex.addressMode[0] = cudaAddressModeClamp;//cudaAddressModeBorder;
  tex.addressMode[1] = cudaAddressModeClamp;//cudaAddressModeBorder;
  cudaMemcpyToArray(cuArray,0,0,data,WIDTH*HEIGHT*sizeof(float),cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex,cuArray,channelDesc);
}

void cleanCuda()
{
  cudaUnbindTexture(tex);
  cudaFree(cuArray);
  cudaFree(devSq);
}

__global__ void makeG(float* G, float2* U, float* gradX, float* gradY)
{
  uint id = threadIdx.x*WIDTH*HEIGHT;
  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      G[id+j*WIDTH+i] = gradX[j*WIDTH+i]*U[id+j*WIDTH+i].x+gradY[j*WIDTH+i]*U[id+j*WIDTH+i].y;
    }
  }

}

__global__ void makeMatrix(float* mat, float* G)
{
  uint x = threadIdx.x;
  uint y = threadIdx.y;
  if(x < y)
  {
    __syncthreads();
    mat[x+y*PARAMETERS] = mat[y+x*PARAMETERS];
  }
  else
  {
    /*func l_f[PARAMETERS];
    for(int i = 0;i < PARAMETERS; i++)
    {
      l_f[i] = f[i];
    }*/
    float val = 0;
    for(uint i = 0; i < WIDTH; i++)
    {
      for(uint j = 0; j < HEIGHT; j++)
      {
        //val += (Gx[i+j*WIDTH]*U[HEIGHT*WIDTH*x+j*WIDTH+i].x+Gy[i+j*WIDTH]*U[HEIGHT*WIDTH*x+j*WIDTH+i].y) * (Gx[i+j*WIDTH]*U[HEIGHT*WIDTH*y+j*WIDTH+i].x+Gy[i+j*WIDTH]*U[HEIGHT*WIDTH*y+j*WIDTH+i].y);
        val += G[x*WIDTH*HEIGHT+j*WIDTH+i]*G[y*WIDTH*HEIGHT+j*WIDTH+i];
      }
    }
    mat[x+y*PARAMETERS] = val/WIDTH/HEIGHT;
    __syncthreads();
  }
}

__global__ void gdSum(float* out, float* G, float* orig, float* def)
{
  uint id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id >= WIDTH * HEIGHT)
  {return;}
  float diff = orig[id] - def[id];
  out[id] = diff*G[id]/WIDTH/HEIGHT;
  //out[id] = 1;
}

void gradientDescent(float* devG, float* devOut, float* devDef, float* vect)
{
  float *devTemp;
  cudaMalloc(&devTemp,WIDTH*HEIGHT*sizeof(float));
  for(uint p = 0; p < PARAMETERS; p++)
  {
  gdSum<<<(HEIGHT*WIDTH+BLOCKSIZE-1)/BLOCKSIZE,min(HEIGHT*WIDTH,BLOCKSIZE)>>>(devTemp, devG+p, devOut, devDef);
  uint size;
    size = WIDTH*HEIGHT;
    while(size>1)
    {
      reduce<<<(size+BLOCKSIZE-1)/BLOCKSIZE,min(size,BLOCKSIZE)>>>(devTemp,size);
      size = (size+BLOCKSIZE-1)/BLOCKSIZE;
    }
    cudaMemcpy(vect+p,devTemp,sizeof(float),cudaMemcpyDeviceToHost);
  }
  cudaFree(devTemp);
}
