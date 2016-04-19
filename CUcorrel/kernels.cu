#include <iostream>
#include "CUcorrel.h"

float *devTemp;

using namespace std;

__global__ void deform2D(cudaTextureObject_t tex,float *devOut, float2* devU, float *devParam)
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
      u.x += devParam[i]*devU[i*IMG_SIZE+id].x;
      u.y += devParam[i]*devU[i*IMG_SIZE+id].y;
    }
    devOut[id] = tex2D<float>(tex,(x+.5f-u.x)/WIDTH,(y+.5f-u.y)/HEIGHT);
  }
}

__global__ void lsq(float* out, float* devA, float* devB, int length)
{
  uint id = blockDim.x*blockIdx.x+threadIdx.x;
  if(id > length)
  {return;}
  out[id] = (devA[id]-devB[id])*(devA[id]-devB[id]);
}

__device__ void warpReduce(volatile float* sh_data, uint tid)
{
  sh_data[tid] += sh_data[tid+32];
  sh_data[tid] += sh_data[tid+16];
  sh_data[tid] += sh_data[tid+8];
  sh_data[tid] += sh_data[tid+4];
  sh_data[tid] += sh_data[tid+2];
  sh_data[tid] += sh_data[tid+1];
}

__global__ void reduce(float* data, uint size)
{
  //Réduit efficacement (ou pas) en sommant tout un tableau et en écrivant les somme restantes de chaque bloc au début du tableau (à appeler plusieurs fois pour sommer plus de 1024 éléments).
/*
TODO: Optimiser la façon dont le kernel somme les éléments (et rendre possible le cas size != 2^k)
voir le lien ci dessous:
http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
*/
  uint id = 2*blockDim.x*blockIdx.x+threadIdx.x;
  uint tid = threadIdx.x;
  if(id > 2*size) //Si appelé plus que nécessaire, quitter
  {return;}
  __shared__ float array[BLOCKSIZE]; //Contient les éléments de chaque bloc, utilisé pour stocker les valeurs temporaires
  array[tid] = data[id] + data[BLOCKSIZE+id]; //Chaque thread copie une valeur
  __syncthreads(); // On attend tout le monde pour être sûr d'avoir toutes les valeurs écrites
  for(uint s = blockDim.x/2; s > 32; s >>= 1) //On divise par 2 le nombre de sommes à effectuer à chaque fois -> à optimiser car une partie des threads tourne dans le vide...
  {
    if(tid < s)
    {array[tid] += array[tid+s];} //On effectue ladite somme...
  __syncthreads(); // On attend que tout le monde ai fini avant de réitérer
  }
  if(tid < 32)
  {warpReduce(array,tid);}
  if(tid == 0)
  {data[blockIdx.x] = array[0];} // Le thread de tête écrit le résultat de son bloc dans le tableau
}

__global__ void gradient(cudaTextureObject_t tex, float* gradX, float* gradY)
{
  //Utilise l'algo le plus simple: les différences centrées.
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x >= WIDTH || y >= HEIGHT)
  {return;}
  gradX[x+y*WIDTH]=tex2D<float>(tex,(x+1.f)/WIDTH,(y+.5f)/HEIGHT)-tex2D<float>(tex,(float)x/WIDTH,(y+.5f)/HEIGHT);
  gradY[x+y*WIDTH]=tex2D<float>(tex,(x+.5f)/WIDTH,(y+1.f)/HEIGHT)-tex2D<float>(tex,(x+.5f)/WIDTH,(float)y/HEIGHT);
}

float residuals(float* devData1, float* devData2, uint size)
{
  lsq<<<(IMG_SIZE+BLOCKSIZE-1)/BLOCKSIZE,min(IMG_SIZE,BLOCKSIZE)>>>(devTemp, devData1, devData2, IMG_SIZE);
  while(size>1)
  {
    reduce<<<(size+2*BLOCKSIZE-1)/2/BLOCKSIZE,min(size,BLOCKSIZE)>>>(devTemp,size);
    size = (size+2*BLOCKSIZE-1)/2/BLOCKSIZE;
  }
  float out;
  cudaMemcpy(&out,devTemp,sizeof(float),cudaMemcpyDeviceToHost);
  return out;
}

void initCuda()
{
  cudaMalloc(&devTemp,HEIGHT*WIDTH*sizeof(float));
}

void cleanCuda()
{
  cudaFree(devTemp);
}

__global__ void makeG(float* G, float2* U, float* gradX, float* gradY)
{
  uint id = threadIdx.x*IMG_SIZE;
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
  if(x < y) // La matrice est symétrique: on ne calcule pas les coefficients connus mais on attend qu'ils soient calculés et on les copie
  {
    __syncthreads();
    mat[x+y*PARAMETERS] = mat[y+x*PARAMETERS];
  }
  else
  {
    float val = 0;
    for(uint i = 0; i < WIDTH; i++) // Possibilté de faire un kernel de réduction device ? (pas prioritaire car cette donction n'est lancée qu'une fois)
    {
      for(uint j = 0; j < HEIGHT; j++)
      {
        val += G[x*IMG_SIZE+j*WIDTH+i]*G[y*IMG_SIZE+j*WIDTH+i];
      }
    }
    mat[x+y*PARAMETERS] = val/IMG_SIZE;
    __syncthreads();
  }
}

__global__ void gdSum(float* out, float* G, float* orig, float* def)
{
  uint id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id >= WIDTH * HEIGHT)
  {return;}
  //float diff = orig[id] - def[id];
  out[id] = (orig[id]-def[id])*G[id]/IMG_SIZE;
}

void gradientDescent(float* devG, float* devOut, float* devDef, float* devVect)
{
  for(uint p = 0; p < PARAMETERS; p++)
  {
  gdSum<<<(IMG_SIZE+BLOCKSIZE-1)/BLOCKSIZE,min(IMG_SIZE,BLOCKSIZE)>>>(devTemp, devG+p*IMG_SIZE, devOut, devDef);
  uint size;
  uint blocksize;
    size = WIDTH*HEIGHT;
    while(size>1)
    {
      blocksize = (size+2*BLOCKSIZE-1)/2/BLOCKSIZE;
      reduce<<<blocksize,BLOCKSIZE>>>(devTemp,size);
      size = blocksize;
    }
    cudaMemcpy(devVect+p,devTemp,sizeof(float),cudaMemcpyDeviceToDevice);
  }
}

__global__ void myDot(float* A, float* b, float* out)
{
  int x = threadIdx.x; //Composante du vecteur (ou ligne de la matrice)
  float val = 0;
  extern __shared__ float sh_b[]; // Les mémoires partagées contenant les vecteurs du bloc
  sh_b[x] = b[x]; // On les place dans la mémoire partagées DU BLOC
  __syncthreads(); // Primordial: évite les "race condition": qu'un thread accède aux données avant qu'elles soient écrites
  for(int i = 0; i < PARAMETERS; i++)
  {
    val += A[x*PARAMETERS+i]*sh_b[i]; // On somme les produits sur la ligne
  }
  out[x] = val; // On écrit le résultat dans le vecteur sortie
}

__global__ void addVec(float* A, float* B)
{
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  A[x] += B[x];
}

__global__ void ewMul(float* A, float* B)
{
  int x = threadIdx.x;
  A[x] *= B[x];
}

__global__ void mipKernel(cudaTextureObject_t tex, float* out, const uint w, const uint h)
{
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  out[x+w*y/2] = tex2D<float>(tex,(2*x+1)/w,(2*y+1)/h);
}

void genMip(cudaTextureObject_t tex, cudaArray* array, uint w, uint h)
{
  dim3 gridsize((w+31)/32,(h+31)/32);
  dim3 blocksize(min(w,32),min(h,32));
  //cout << "Génération du mipmap de taille " << w << ", " << h << endl;
  mipKernel<<<gridsize,blocksize>>>(tex, devTemp, w, h);
  cudaMemcpyToArray(array,0,0,devTemp,w*h*sizeof(float),cudaMemcpyDeviceToDevice);
}
