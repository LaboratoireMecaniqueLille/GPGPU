#include <iostream>
/*#include <cuda.h>
#include <cuda_runtime.h>*/

#include "lodepng/lodepng.h"
#include "img.h"


using namespace std;

void readFile(const char* address, float* data, float norm = 1)
{
  unsigned char *image;
  uint i_w, i_h;
  if(lodepng_decode32_file(&image,&i_w,&i_h,address))
  {
    cout << "Erreur lors de l'ouverture du fichier." << endl;
    exit(-1);
  }
  cout << "Dimensions de l'image: " << i_w << "x" << i_h << endl;
  for(int i = 0; i < i_w*i_h; i++)
  {
    data[i] = image[4*i]/3.f+image[4*i+1]/3.f+image[4*i+2]/3.f;
  }
  free(image);
}

typedef unsigned int uint;

int main(int argc, char** argv)
{
  uint w = 2048;
  uint h = 2048;
  float *tab = new float [w*h];
  readFile("lena.png", tab);
  srand(time(NULL));

  float *devTab;
  cudaMalloc(&devTab,w*h*sizeof(float));
  cudaMemcpy(devTab,tab,w*h*sizeof(float),cudaMemcpyHostToDevice);

  delete tab;

  Image img(w,h,devTab);

  Image tile = img.makeTile(10,10,1024,1024);
  Image ttile = tile.makeTile(10,10,512,512);

  uint x = (int)(rand()%500+20);
  uint y = (int)(rand()%500+20);

  cout << "Point: " << x << ", " << y << endl;
  cout << "Image: " << img.getVal(x,y) << endl;
  cout << "Tuile: " << tile.getVal(x-10,y-10) << endl;
  cout << "sous-tuile: " << ttile.getVal(x-20,y-20) << endl;
  //tile.writeToFile("tile.png");

  float *devOut;
  cudaMalloc(&devOut, w*h*sizeof(float));
  float2 *devU;
  cudaMalloc(&devU, w*h*sizeof(float2));

  uint tile_w = 1024;
  uint tile_h = 1024;

  float2 *U = new float2 [tile_w*tile_h];
  for(int i = 0; i < tile_w; i++)
  {
    for(int j = 0; j < tile_h; j++)
    {
      U[i+tile_w*j] = make_float2(2.f*i/tile_w-1.f,0.f);
    }
  }
  
  cudaMemcpy(devU,U,tile_w*tile_h*sizeof(float2),cudaMemcpyHostToDevice);
  float k = 15.f;
  
  float2 *devDisp;
  cudaMalloc(&devDisp, tile_w*tile_h*sizeof(float2));

  makeDisplacement(devDisp,k,devU,tile_w,tile_h);
  tile.interpLinear(devOut,devDisp,tile_w*tile_h);
  cudaDeviceSynchronize();
  Image out(tile_w,tile_h,devOut);
  out.writeToFile("out.png");


  cudaFree(devOut);
  cudaFree(devU);
  delete U;


  
  return EXIT_SUCCESS;
}
