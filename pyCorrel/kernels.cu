#include <cuda.h>
#define WIDTH %d
#define HEIGHT %d
#define T_WIDTH %d
#define T_HEIGHT %d


texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_d;
texture<float, cudaTextureType2D, cudaReadModeElementType> texGradX;
texture<float, cudaTextureType2D, cudaReadModeElementType> texGradY;

__global__ void makeDiff(float* out, float t_x, float t_y, float ox, float oy)
{
/*
t_x, t_y: tile offset
ox, oy: variable offset
NOTE: Offsets are in NORMALIZED coordinates! (between 0 and 1)
*/
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int idy = threadIdx.y+blockIdx.y*blockDim.y;
  float x = (float)idx/WIDTH+t_x;
  float y = (float)idy/HEIGHT+t_y;

  out[idy*T_WIDTH+idx] = tex2D(tex_d,x+ox,y+oy) - tex2D(tex,x,y);
}


__global__ void gradient(float* gradX, float* gradY)
{
  //Sobel
  uint x = blockIdx.x*blockDim.x+threadIdx.x;
  uint y = blockIdx.y*blockDim.y+threadIdx.y;
  gradX[x+WIDTH*y] = (tex2D(tex,(x+1.5f)/WIDTH,(float)y/HEIGHT)+tex2D(tex,(x+1.5f)/WIDTH,(y+1.f)/HEIGHT)-tex2D(tex,(x-.5f)/WIDTH,(float)y/HEIGHT)-tex2D(tex,(x-.5f)/WIDTH,(y+1.f)/HEIGHT))*2;
  gradY[x+WIDTH*y] = (tex2D(tex,(float)x/WIDTH,(y+1.5f)/HEIGHT)+tex2D(tex,(x+1.f)/WIDTH,(y+1.5f)/HEIGHT)-tex2D(tex,(float)x/WIDTH,(y-.5f)/HEIGHT)-tex2D(tex,(x+1.f)/WIDTH,(y-.5f)/HEIGHT))*2;
}

__global__ void gdProduct(float* outX, float* outY, float* tab, float t_x, float t_y)
{
  uint idx = blockIdx.x*blockDim.x+threadIdx.x;
  uint idy = blockIdx.y*blockDim.y+threadIdx.y;
  uint id = idx+T_WIDTH*idy;
  float x = (float)idx/WIDTH+t_x;
  float y = (float)idy/HEIGHT+t_y;
  float val = tab[id];
  outX[id] = val*tex2D(texGradX,x,y);
  outY[id] = val*tex2D(texGradY,x,y);
  
}
