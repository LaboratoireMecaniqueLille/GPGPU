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
  Image(uint w, uint h, float* pointer, uint stridei, uint offset);
  float getVal(uint x, uint y);
  float interpLinear(float x, float y);
  float interpCubic(float x, float y);
  Image makeTile(uint x, uint y, uint w, uint h);
  void writeToFile(const char* address);

  private:
  cudaTextureObject_t tex;
  float* getAddr(uint x, uint y);
  uint m_w;
  uint m_h;
  uint m_stride;
  uint m_offset;
  float* m_pointer;
};
