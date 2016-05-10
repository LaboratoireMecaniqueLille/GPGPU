#include "img.h"
#include "lodepng/lodepng.h"

using namespace std;

__global__ void interpolate(float* out, cudaTextureObject_t tex, float2* points, uint N)
{
  uint id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id < N)
  {
    out[id] = tex2D<float>(tex,points[id].x,points[id].y);
  }
}

__global__ void normalize(float2* points, float2 texMin, float2 texMax, uint w, uint h, uint N)
{
  uint id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id < N)
  {
    points[id] = make_float2(points[id].x/w*(texMax.x-texMin.x)+texMin.x,points[id].y/h*(texMax.y-texMin.y)+texMin.y);
    //points[id] = make_float2(points[id].x/w,points[id].y/h);
  } 
}

Image::Image(uint w, uint h, float* pointer) : 
m_w(w), m_h(h), m_stride(w), m_pointer(pointer), 
m_offset(0), m_tex(0), 
m_texMin(make_float2(0.f,0.f)), m_texMax(make_float2(1.f,1.f))
{
  genTexture();
}

Image::Image(uint w, uint h, float* pointer, cudaTextureObject_t tex, uint stride, uint offset,
 float2 texMin, float2 texMax) : 
m_w(w), m_h(h),
m_pointer(pointer), m_tex(tex), 
m_stride(stride), m_offset(offset),
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

void Image::interpLinear(float* devOut, float2* devPoints, uint N)
{
  // Les coordonées des points attendues sont en pixels
  int gridsize = (N+1023)/1024;
  int blocksize = min(N,1024);
  normalize<<<gridsize,blocksize>>>(devPoints, m_texMin, m_texMax, m_w, m_h, N); // Normalise les points et applique la transformation de la tuile vers la texture

//-----TEST-----
  float2 *test = new float2 [m_w*m_h];
  cudaMemcpy(test,devPoints,m_w*m_h*sizeof(float),cudaMemcpyDeviceToHost);
  cout << test[500*2048+500].x << endl;
//---------------

  delete test;

  interpolate<<<gridsize,blocksize>>>(devOut, m_tex, devPoints,N);
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
  Image ret(w,h,m_pointer,m_tex,m_stride,m_offset+x+y*m_stride, texMin, texMax);
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

void Image::genTexture()
{
  cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaMallocArray(&m_cuArray, &chanDesc, m_w, m_h);
  cudaMemcpyToArray(m_cuArray, 0, 0, m_pointer, m_w*m_h*sizeof(float),cudaMemcpyHostToDevice);
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = m_w*m_h*sizeof(float);
  resDesc.res.linear.devPtr = m_cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = 1;
  cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, NULL);
}

__global__ void devMakeDisplacement(float2* disp, float param, float2* field, uint w, uint h)
{
  uint idx = threadIdx.x+blockIdx.x*blockDim.x;
  uint idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
  {
    disp[idx+w*idy] = make_float2(idx-param*field[idx+w*idy].x,idy-param*field[idx+w*idy].y);
  }

}

void makeDisplacement(float2* devDisp, float param, float2* devField, uint w, uint h)
{
  dim3 grid((w+31)/32,(h+31)/32);
  dim3 block(min(w,32),min(h,32));
  devMakeDisplacement<<<grid,block>>>(devDisp, param, devField, w, h);
  
  float2 *test = new float2 [w*h];
  cudaMemcpy(test,devDisp,w*h*sizeof(float2),cudaMemcpyDeviceToHost);
}
