#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define INTERP_NEAREST 1
#define INTERP_BILINEAR 2
#define INTERP_BICUBIC 3

typedef unsigned int uint;

class Image
{
  public:
  Image(uint w, uint h, float* pointer);
  Image(uint w, uint h, float* pointer, uint stride, uint offset, float2 texMin, float2 texMax);
  float getVal(uint x, uint y);
  float interpLinear(float x, float y);
  float interpCubic(float x, float y);
  Image makeTile(uint x, uint y, uint w, uint h);
  void writeToFile(const char* address);
  float2 get_gtnc(float2 coord); // (get Global Texture Normalized Coordinates)
  float2 get_gtnc(float cX, float cY);

  private:
  float* m_pointer;
  float* getAddr(uint x, uint y);
  float2 m_texMin;
  float2 m_texMax;
  cudaTextureObject_t m_tex;
  uint m_w;
  uint m_h;
  uint m_stride;
  uint m_offset;
};
