#include "CUcorrel.h"
//#include "util.h"
//#include <iostream>

//using namespace std;

//2 champs: 2 mvt de corps solide

void writeFields(float2* devFields, uint w, uint h)
{
//  cout << "W: " << w << "\nH: " << h << endl;
  //Assignation des champs
  size_t taille2 = w*h*sizeof(float2);
  float2 *field = (float2*)malloc(taille2);

  float valW = (float)w/WIDTH;
  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = valW; // Move X
      field[i+w*j].y = 0.f;
    }
  }
  //printMat2D(field,w,h,w/16);
  cudaMemcpy(devFields,field,taille2,cudaMemcpyHostToDevice);
  float valH = (float)h/HEIGHT;
  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 0.f; // Move Y
      field[i+w*j].y = valH;
    }
  }
  //printMat2D(field,w,h,w/16);
  cudaMemcpy(devFields+w*h,field,taille2,cudaMemcpyHostToDevice);
}
