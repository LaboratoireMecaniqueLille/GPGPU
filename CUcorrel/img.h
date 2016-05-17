#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;


__global__ void interpolate(float* out, cudaTextureObject_t tex, float2* points, uint N);
__global__ void normalize(float2* points, float2 texMin, float2 texMax, uint w, uint h, uint N);
__global__ void makeZeroDisplacement(float2* disp, uint w, uint h);
__global__ void addDisplacement(float2* disp, float param, float2* field, uint w, uint h);
__global__ void ewMul(float* tab1, float* tab2);
__global__ void kMul2(float2* tab, float2 k);
__global__ void gradient(cudaTextureObject_t, float*, float*, uint, uint);
__global__ void devTrField(float2* disp, float2 mv, uint w, uint h);
__global__ void devGetDiff(float* diff, float* img, cudaTextureObject_t tex, float2 m, float2 p, uint w, uint h);
__global__ void devGradientDescent(float* outX, float* outY, float* tab, cudaTextureObject_t texGX, cudaTextureObject_t texGY, float2 m, float2 p, uint w, uint h);


void allocTemp();
void freeTemp();
void makeDisplacement(float2* devDisp, float param, float2* devField, uint w, uint h);
void makeTranslationField(float2* devDisp, float2 mv, uint w, uint h);

class Image
{
  public:
  Image();
  Image(uint w, uint h, float* pointer);
  Image(uint w, uint h, float* pointer, cudaTextureObject_t tex, uint stride, uint offset, float2 texMin, float2 texMax);
  void init(uint w, uint h, float* pointer);
  uint getW();
  uint getH();
  float getVal(uint x, uint y);
  void interpLinear(float* devOut, float2* devPoints, uint N);
  void interpCubic(float devOut, float2* devPoints, uint N);
  Image makeTile(uint x, uint y, uint w, uint h);
  void writeToFile(const char* address, float gain = 1.f, float offset = 0.f);
  float2 get_gtnc(float2 coord); // (get Global Texture Normalized Coordinates)
  float2 get_gtnc(float cX, float cY);
  void genTexture();
  void mip(float* devOut, uint w, uint h);
  void computeGradients(float* devGradX, float* devGradY);
  void getDiff(float* devImg, float* devDiff);

  //private:
  float* m_pointer;
  float* getAddr(uint x, uint y);
  float2 m_texMin;
  float2 m_texMax;
  cudaTextureObject_t m_tex;
  cudaArray *m_cuArray;
  uint m_w;
  uint m_h;
  uint m_stride;
  uint m_offset;
};

float2 gradientDescent(float* devDiff, Image gradX, Image gradY);
