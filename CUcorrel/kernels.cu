#include <iostream>
#include "CUcorrel.h"
#include "util.h"
#include <stdio.h>

#include "bicubic.cu"

float *devTemp;

using namespace std;

__global__ void deform2D(cudaTextureObject_t tex,float *devOut, float2* devU, float *devParam, uint w, uint h)
{
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x < w && y < h)
  {
    uint id = x+y*w;
    float2 u = make_float2(0.f,0.f);
    for(int i = 0; i < PARAMETERS; i++)
    {
      u.x += devParam[i]*devU[i*w*h+id].x;
      u.y += devParam[i]*devU[i*w*h+id].y;
    }
    devOut[id] = interpBicubic(tex,(x-u.x),(y-u.y),w,h); // Interpolation bicubique (~ 2.25x plus long)
  }
}

__global__ void deform2D_b(cudaTextureObject_t tex,float *devOut, float2* devU, float *devParam, uint w, uint h)
{
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x < w && y < h)
  {
    uint id = x+y*w;
    float2 u = make_float2(0.f,0.f);
    for(int i = 0; i < PARAMETERS; i++)
    {
      u.x += devParam[i]*devU[i*w*h+id].x;
      u.y += devParam[i]*devU[i*w*h+id].y;
    }
    devOut[id] = tex2D<float>(tex,(x+.5f-u.x)/w,(y+.5f-u.y)/h); // Interpolation bilinéaire (rapide mais altère plus l'image)
  }
}

__global__ void lsq(float* out, float* devA, float* devB, const uint length)
{
  uint id = blockDim.x*blockIdx.x+threadIdx.x;
  if(id > length)
  {return;}
  out[id] = (devA[id]-devB[id])*(devA[id]-devB[id]);
}

__device__ void warpReduce(volatile float* sh_data, const uint tid)
{
  sh_data[tid] += sh_data[tid+32];
  sh_data[tid] += sh_data[tid+16];
  sh_data[tid] += sh_data[tid+8];
  sh_data[tid] += sh_data[tid+4];
  sh_data[tid] += sh_data[tid+2];
  sh_data[tid] += sh_data[tid+1];
}

__global__ void reduce(float* data, const uint size)
{
  //Réduit relativement efficacement en sommant tout un tableau et en écrivant les somme restantes de chaque bloc au début du tableau (à appeler plusieurs fois pour sommer plus de 1024 éléments).
/*
TODO: Rendre possible le cas size != 2^k
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

__global__ void gradient(cudaTextureObject_t tex, float* gradX, float* gradY, uint w, uint h)
{
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  /*
  //Algo le plus simple: les différences centrées.
  gradX[x+y*w]=(tex2D<float>(tex,(x+1.f)/w,(y+.5f)/h)-tex2D<float>(tex,(float)x/w,(y+.5f)/h));//*w/WIDTH;
  gradY[x+y*w]=(tex2D<float>(tex,(x+.5f)/w,(y+1.f)/h)-tex2D<float>(tex,(x+.5f)/w,(float)y/h));//*h/HEIGHT;
  */
  //Sobel:
  gradX[x+w*y] = (tex2D<float>(tex,(x+1.5f)/w,(float)y/h)+tex2D<float>(tex,(x+1.5f)/w,(y+1.f)/h)-tex2D<float>(tex,(x-.5f)/w,(float)y/h)-tex2D<float>(tex,(x-.5f)/w,(y+1.f)/h))*2;
  gradY[x+w*y] = (tex2D<float>(tex,(float)x/w,(y+1.5f)/h)+tex2D<float>(tex,(x+1.f)/w,(y+1.5f)/h)-tex2D<float>(tex,(float)x/w,(y-.5f)/h)-tex2D<float>(tex,(x+1.f)/w,(y-.5f)/h))*2.f;
}

float residuals(float* devData1, float* devData2, uint size)
{
  lsq<<<(size+BLOCKSIZE-1)/BLOCKSIZE,min(size,BLOCKSIZE)>>>(devTemp, devData1, devData2, size);
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
  cudaMalloc(&devTemp,IMG_SIZE*sizeof(float));
}

void cleanCuda()
{
  cudaFree(devTemp);
}

__global__ void makeG(float* G, float2* U, float* gradX, float* gradY, uint w, uint h)
{
  uint idx = threadIdx.x+blockDim.x*blockIdx.x;
  uint idy = threadIdx.y+blockDim.y*blockIdx.y;
  G[idy*w+idx] = gradX[idy*w+idx]*U[idy*w+idx].x+gradY[idy*w+idx]*U[idy*w+idx].y;
}

void makeGArray(cudaArray* array, float2* U, float* gradX, float* gradY, uint w, uint h)
{
  dim3 blocksize(min(32,w),min(32,h));
  dim3 gridsize((w+31)/32,(h+31)/32);
  makeG<<<gridsize,blocksize>>>(devTemp, U, gradX, gradY, w, h);
  cudaMemcpyToArray(array, 0, 0, devTemp, w*h*sizeof(float), cudaMemcpyDeviceToDevice);
}

__global__ void gdSum(float* out, cudaTextureObject_t texG, float* orig, float* def, float* param, float2* field, uint w, uint h, uint p)
{
  uint idx = blockIdx.x*blockDim.x+threadIdx.x;
  uint idy = blockIdx.y*blockDim.y+threadIdx.y;
  uint id = w*idy+idx;
  //out[id] = (orig[id]-def[id])*tex2D<float>(texG,(idx-param[p]*field[id].x)/w,(idy-param[p]*field[id].y)/h)/(w*h);
  out[id] = max(-50.f,min(50.f,(orig[id]-def[id])))*tex2D<float>(texG,(idx-param[p]*field[id].x)/w,(idy-param[p]*field[id].y)/h)/(w*h); // Pour limiter l'impact des endroits où l'image est complètement à côté.
}

void gradientDescent(cudaTextureObject_t* texG, float* devOut, float* devDef, float* devVect, float* devParam, float2* devFields, uint w, uint h)
{
  for(uint p = 0; p < PARAMETERS; p++)
  {
  uint size = w*h;
  dim3 blocks(min(w,32),min(h,32));
  dim3 grid((w+31)/32,(h+31)/32);
  gdSum<<<grid,blocks>>>(devTemp, texG[p], devOut, devDef, devParam, devFields+w*h*p, w, h, p);
  uint blocksize;
    while(size>1)
    {
      blocksize = (size+2*BLOCKSIZE-1)/(2*BLOCKSIZE);
      reduce<<<blocksize,min(size,BLOCKSIZE)>>>(devTemp,size);
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
  out[x+w*y] = tex2D<float>(tex,(2.f*x+1.f)/2.f/w,(2.f*y+1.f)/2.f/h);
}

void genMip(cudaTextureObject_t tex, cudaArray* array, uint w, uint h)
{
  dim3 gridsize((w+31)/32,(h+31)/32);
  dim3 blocksize(min(w,32),min(h,32));
  //cout << "Génération du mipmap de taille " << w << ", " << h << endl;
  mipKernel<<<gridsize,blocksize>>>(tex, devTemp, w, h);
  cudaMemcpyToArray(array,0,0,devTemp,w*h*sizeof(float),cudaMemcpyDeviceToDevice);
  /* Vérif:
  float* test = (float*)malloc(w*h*sizeof(float));
  cudaMemcpy(test,devTemp,w*h*sizeof(float),cudaMemcpyDeviceToHost);
  char oAddr[20];
  sprintf(oAddr,"mip%d.csv",w);
  writeFile(oAddr,test,1,0,w,h);
  free(test);
  */
}

__global__ void resample(float* out, float* in, uint w)
{
  //à appeler avec les dimensions de la nouvelle image (cotés 2x plus petits), w est la largeur de la nouvelle image
  uint x = blockDim.x*blockIdx.x+threadIdx.x;
  uint y = blockDim.y*blockIdx.y+threadIdx.y;
  out[x+y*w] = (in[2*x+4*w*y]+in[2*x+1+4*w*y]+in[2*x+2*w*(2*y+1)]+in[2*x+1+2*w*(2*y+1)])/4.f;
}

__global__ void scalMul(float* vec, float scal)
{
  uint x = threadIdx.x;
  vec[x] *= scal;
}

__global__ void vecCpy(float* dest, float* source)
{
  uint id = threadIdx.x;
  dest[id] = source[id];
}

__global__ void evalProduct(float* out, cudaTextureObject_t t1, cudaTextureObject_t t2, uint w, uint h)
{
  uint idx = blockIdx.x*blockDim.x+threadIdx.x;
  uint idy = blockIdx.y*blockDim.y+threadIdx.y;
  out[w*idy+idx] = tex2D<float>(t1,(float)idx/w,(float)idy/h)*tex2D<float>(t2,(float)idx/w,(float)idy/h);
}

void makeHessian(float* devH, cudaTextureObject_t* texG)
{
  dim3 gridsize((WIDTH+31)/32,(HEIGHT+31)/32);
  dim3 blocksize(32,32);
  for(int i = 0; i < PARAMETERS; i++)
  {
    for(int j = 0; j <= i; j++)
    {
      evalProduct<<<gridsize,blocksize>>>(devTemp,texG[i],texG[j], WIDTH, HEIGHT);
      uint size = IMG_SIZE;
      uint blocksize;
      while(size>1)
      {
        blocksize = (size+2*BLOCKSIZE-1)/(2*BLOCKSIZE);
        reduce<<<blocksize,min(size,BLOCKSIZE)>>>(devTemp,size);
        size = blocksize;
      }
      scalMul<<<1,1>>>(devTemp,1.f/IMG_SIZE);
      cudaMemcpy(devH+PARAMETERS*i+j,devTemp,sizeof(float),cudaMemcpyDeviceToDevice);
      if(j != i)
        cudaMemcpy(devH+PARAMETERS*j+i,devTemp,sizeof(float),cudaMemcpyDeviceToDevice);
    }
  }
}
