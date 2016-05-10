#include "img.h"
#include "lodepng/lodepng.h"

using namespace std;

Image::Image(uint w, uint h, float* pointer) : 
m_w(w), m_h(h), m_stride(w), m_pointer(pointer), 
m_offset(0), m_tex(0), 
m_texMin(make_float2(0.f,0.f)), m_texMax(make_float2(1.f,1.f))
{
}

Image::Image(uint w, uint h, float* pointer, uint stride, uint offset,
 float2 texMin, float2 texMax) : 
m_w(w), m_h(h), m_stride(stride), 
m_pointer(pointer), m_offset(offset), 
m_texMin(texMin), m_texMax(texMax)
{
}


float Image::getVal(uint x, uint y)
{
  float val = 0;
  cudaMemcpy(&val, getAddr(x,y), sizeof(float), cudaMemcpyDeviceToHost);
  return val;
}

float* Image::getAddr(uint x, uint y)
{
  if(x >= m_w || y >= m_h)
  {
    cout << "getAddr impossible ! (" << x << ", " << y << ")." << endl;
    return NULL;
  }
  return m_pointer+x+m_stride*y+m_offset;
}

Image Image::makeTile(uint x, uint y, uint w, uint h)
{
  if(x+w >= m_w || y+h >= m_h)
  {
    cout << "makeTile impossible: sort de l'image d'origine !" << endl;
    exit(-1);
  }
  float2 texMin = get_gtnc((float)x/m_w,(float)y/m_h);
  float2 texMax = get_gtnc((float)(x+w)/m_w,(float)(y+h)/m_h);
  Image ret(w,h,m_pointer,m_stride,m_offset+x+y*m_stride, texMin, texMax);
  return ret;
}

void Image::writeToFile(const char* address)
{
  float *tab = new float [m_w*m_h];
  for(uint i = 0; i < m_h; i++)
  {
    cudaMemcpy(tab+i*m_w,getAddr(0,i),m_w*sizeof(float),cudaMemcpyDeviceToHost);
  }
  unsigned char *image = new unsigned char [4*m_w*m_h];
  unsigned char val = 0;
  for(uint i = 0; i < m_w*m_h; i++)
  {
    val = max(0.f,min(255.f,tab[i]));
    image[4*i] = val;
    image[4*i+1] = val;
    image[4*i+2] = val;
    image[4*i+3] = 255;
  }
  if(lodepng_encode32_file(address, image, m_w, m_h))
  {
    cout << "Erreur lors de l'écriture !" << endl;
    exit(-1);
  }
  delete image;
}

float2 Image::get_gtnc(float2 coord)
{
  return make_float2((m_texMax.x-m_texMin.y)*coord.x+m_texMin.x,(m_texMax.y-m_texMin.y)*coord.y+m_texMin.y);
}

float2 Image::get_gtnc(float cX, float cY)
{
  return make_float2((m_texMax.x-m_texMin.y)*cX+m_texMin.x,(m_texMax.y-m_texMin.y)*cY+m_texMin.y);
}