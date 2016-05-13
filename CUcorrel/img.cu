#include "img.h"
#include "CUcorrel.h"
#include "lodepng/lodepng.h"

using namespace std;

float* devTemp;
float2* devTemp2;

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
      return make_float2(a.x - b.x, a.y - b.y);
}

void allocTemp()
{
  cudaMalloc(&devTemp,IMG_SIZE*sizeof(float));
  cudaMalloc(&devTemp2,IMG_SIZE*sizeof(float2));
}

void freeTemp()
{
  cudaFree(devTemp);
  cudaFree(devTemp2);
}
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

Image::Image() : 
m_w(0), m_h(0), m_stride(0), m_pointer(NULL), m_offset(0), m_tex(0), m_texMin(make_float2(0.f,0.f)), m_texMax(make_float2(1.f,1.f))
{
}

Image::Image(uint w, uint h, float* pointer) 
{
  init(w,h,pointer);
}

void Image::init(uint w, uint h, float* pointer) 
{
  m_w=w;
  m_h=h;
  m_stride=w;
  m_pointer=pointer; 
  m_offset=0;
  m_tex=0; 
  m_texMin=make_float2(0.f,0.f);
  m_texMax=make_float2(1.f,1.f);
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

uint Image::getW()
{
  return m_w;
}

uint Image::getH()
{
  return m_h;
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
  
  interpolate<<<gridsize,blocksize>>>(devOut, m_tex, devPoints, N);
}

Image Image::makeTile(uint x, uint y, uint w, uint h)
{
  if(x+w > m_w || y+h > m_h)
  {
    cout << "makeTile impossible: sort de l'image d'origine !" << endl;
    exit(-1);
  }
  float2 texMin = get_gtnc((float)x/m_w,(float)y/m_h);
  float2 texMax = get_gtnc((float)(x+w)/m_w,(float)(y+h)/m_h);
  Image ret(w,h,m_pointer,m_tex,m_stride,m_offset+x+y*m_stride, texMin, texMax);
  return ret;
}

void Image::writeToFile(const char* address, float gain, float offset)
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
    val = max(0.f,min(255.f,gain*tab[i]+offset));
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

void Image::mip(float* devOut, uint w, uint h)
{
  if(w <= m_w && h <= m_h)
  {
    dim3 gridsize((w+31)/32,(h+31)/32);
    dim3 blocksize(min(32,w),min(32,h));

    makeZeroDisplacement<<<gridsize,blocksize>>>(devTemp2, w, h);
    ewMul<<<(w*h+1023)/1024,min(1024,w*h)>>>(devTemp2,make_float2((float)m_w/w,(float)m_h/h));
    interpLinear(devOut,devTemp2,w*h);
  }
  
}

void Image::computeGradients(float* devGradX, float* devGradY)
{
  if(m_w != m_stride)
  {
    cout << "Erreur: Image::computeGradients ne permet pas encore de calculer le gradient d'une sous-image !" << endl; // Cette implémentation n'est pas nécessaaire pour le moment (on a le gradient découpé de l'image globale)
    exit(-1);
  }
  dim3 grid((m_w+31)/32,(m_h+31)/32);
  dim3 block(min(32,m_w),min(m_h,32));
  gradient<<<grid,block>>>(m_tex,devGradX,devGradY,m_w,m_h);
}

void Image::getDiff(float* devImg, float* devDiff)
{
  dim3 grid((m_w+31)/32,(m_h+31));
  dim3 block(min(m_w,32),min(m_h,32));

  devGetDiff<<<grid,block>>>(devDiff, devImg, m_tex, m_texMax-m_texMin,m_texMin,m_w, m_h);

}

__global__ void makeZeroDisplacement(float2* disp, uint w, uint h)
{
  uint idx = threadIdx.x+blockIdx.x*blockDim.x;
  uint idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
  {
    disp[idx+w*idy] = make_float2(idx,idy);
  }
}
__global__ void addDisplacement(float2* disp, float param, float2* field, uint w, uint h)
{
  uint idx = threadIdx.x+blockIdx.x*blockDim.x;
  uint idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
  {
    float2 val = disp[idx+w*idy];
    disp[idx+w*idy] = make_float2(val.x+param*field[idx+w*idy].x,val.y-param*field[idx+w*idy].y);
  }

}

__global__ void ewMul(float2* tab, float2 k)
{
  uint id = threadIdx.x+blockIdx.x*blockDim.x;
  tab[id] = make_float2(k.x*tab[id].x,k.y*tab[id].y);
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

void makeDisplacement(float2* devDisp, float param, float2* devField, uint w, uint h)
{
  dim3 grid((w+31)/32,(h+31)/32);
  dim3 block(min(w,32),min(h,32));
  
  makeZeroDisplacement<<<grid,block>>>(devDisp, w, h);
  addDisplacement<<<grid,block>>>(devDisp, param, devField, w, h); // à boucler...
}

__global__ void devTrField(float2* disp, float mvX, float mvY, uint w, uint h)
{
  uint idx = threadIdx.x+blockIdx.x*blockDim.x;
  uint idy = threadIdx.y+blockIdx.y*blockDim.y;
  if(idx < w && idy < h)
  {
    disp[idx+idy*w] = make_float2(idx+.5f+mvX,idy+.5f+mvY);
  }
}
void makeTranslationField(float2* devDisp, float mvX, float mvY, uint w, uint h)
{
  dim3 grid((w+31)/32,(h+31)/32);
  dim3 block(min(w,32),min(h,32));
  devTrField<<<grid,block>>>(devDisp,mvX,mvY,w,h);
}

__global__ void devGetDiff(float* diff, float* img, cudaTextureObject_t tex, float2 m, float2 p, uint w, uint h)
{
  uint idx = blockIdx.x*blockDim.x+threadIdx.x;
  uint idy = blockIdx.y*blockDim.y+threadIdx.y;
  
  diff[idx+w*idy] = tex2D<float>(tex,m.x*((float)idx/w)+p.x,m.y*((float)idy/h)+p.y)-img[idx+w*idy];
}
