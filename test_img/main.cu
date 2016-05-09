#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lodepng/lodepng.h"
#include "img.h"

#define WIDTH 2048
#define HEIGHT 2048
#define IMG_SIZE (WIDTH*HEIGHT)

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
  if(i_w != WIDTH || i_h != HEIGHT)
  {
    cout << "Taille de l'image incorecte: (" << i_w << ", " << i_h << ") au lieu de (" << WIDTH << ", " << HEIGHT << ")." << endl;
    exit(-1);
  }
  for(int i = 0; i < IMG_SIZE; i++)
  {
    data[i] = image[4*i]/3.f+image[4*i+1]/3.f+image[4*i+2]/3.f;
  }
  free(image);
}

typedef unsigned int uint;

int main(int argc, char** argv)
{
  uint w = WIDTH;
  uint h = HEIGHT;
  float *tab = new float [w*h];
  readFile("lena.png", tab);

  float *devTab;
  cudaMalloc(&devTab,w*h*sizeof(float));
  cudaMemcpy(devTab,tab,w*h*sizeof(float),cudaMemcpyHostToDevice);

  delete tab;

  Image img(w,h,devTab);

  Image tile = img.makeTile(10,10,1024,1024);
  Image ttile = tile.makeTile(10,10,512,512);

  cout << "Image: " << img.getVal(500,500) << endl;
  cout << "Tuile: " << tile.getVal(490,490) << endl;
  cout << "Tuile de tuile: " << ttile.getVal(480,480) << endl;

  tile.writeToFile("tile.png");
  ttile.writeToFile("ttile.png");
  return EXIT_SUCCESS;
}
