#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

#define WIDTH 1024
#define HEIGHT 1024
#define SIZE (WIDTH*HEIGHT)

using namespace std;

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__device__ float w0(float a)
{
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);
}

__device__ float w1(float a)
{
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__device__ float w2(float a)
{
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__device__ float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}
__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

__device__ float interpBicubic(cudaTextureObject_t tex, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float r = g0(fy) * (g0x * tex2D<float>(tex, px + h0x, py + h0y)   +
                    g1x * tex2D<float>(tex, px + h1x, py + h0y)) +
          g1(fy) * (g0x * tex2D<float>(tex, px + h0x, py + h1y)   +
                    g1x * tex2D<float>(tex, px + h1x, py + h1y));
    return r;
}

__device__ float interpBilinear(cudaTextureObject_t tex,  float x, float y)
{
  return tex2D<float>(tex,x,y);
}

__global__ void bilinear(cudaTextureObject_t tex, float* out)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  float x = (float)idx/WIDTH;
  float y = (float)idy/HEIGHT;

  out[idx+WIDTH*idy] = interpBilinear(tex,x,y);

}

__global__ void bicubic(cudaTextureObject_t tex, float* out)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  float x = (float)idx/WIDTH;
  float y = (float)idy/HEIGHT;

  out[idx+WIDTH*idy] = interpBicubic(tex,x,y);
}


double timeDiff(struct timeval t1, struct timeval t2) //Retourne la différence en ms entre 2 mesures de temps avec gettimeofday(&t,NULL);
{
  return (t2.tv_sec-t1.tv_sec)*1000+(t2.tv_usec-t1.tv_usec)/1000.f;
}

int main(int argc, char** argv)
{

  struct timeval t1, t2;

  float tab[SIZE];
  for(int i = 0; i < SIZE;i++)
  {
    tab[i] = (float)(i%2)/SIZE;
  }

  cudaTextureObject_t tex=0;
  
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc,WIDTH,HEIGHT);
  cudaMemcpyToArray(cuArray,0,0,tab,SIZE*sizeof(float),cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = SIZE*sizeof(float);
  resDesc.res.linear.devPtr = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.addressMode[0] = cudaAddressModeBorder; //cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeBorder; //cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&tex,&resDesc,&texDesc,NULL);

  float *devOut;
  cudaMalloc(&devOut, SIZE*sizeof(float));

  dim3 gridsize(WIDTH/32,HEIGHT/32);
  dim3 blocksize(32,32);

  gettimeofday(&t1,NULL);
  for(int i = 0; i < 500; i++)
  bilinear<<<gridsize,blocksize>>>(tex,devOut);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  float d1 = timeDiff(t1, t2);
  cout << "Bilinéaire: " << d1 << endl;

  cudaMemcpy(tab,devOut,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
  cout << tab[0] << ", " << tab[10] << endl;
  
  gettimeofday(&t1,NULL);
  for(int i = 0; i < 500; i++)
  bicubic<<<gridsize,blocksize>>>(tex,devOut);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  float d2 = timeDiff(t1, t2);
  cout << "Bicubique: " << d2 << endl;

  cudaMemcpy(tab,devOut,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
  cout << tab[0] << ", " << tab[10] << endl;

  cout << "ratio: " << d2/d1 << endl;
  

}
