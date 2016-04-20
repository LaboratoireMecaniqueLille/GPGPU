#include "CUcorrel.h"
//#include "util.h"

//6 champs: 3 mvt de corps solide, 2 déformations uniformes et 1 cisaillement

void writeFields(float2* devFields, uint w, uint h)
{
  //Assignation des champs
  size_t taille2 = w*h*sizeof(float2);
  float2 *field = (float2*)malloc(PARAMETERS*taille2);

  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 1.f; // Move X
      field[i+w*j].y = 0.f;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields,field,taille2,cudaMemcpyHostToDevice);
  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 0.f; // Move Y
      field[i+w*j].y = 1.f;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+w*h,field,taille2,cudaMemcpyHostToDevice);

  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 1.4142135624f*(j-h/2.)/h; // Rotation
      field[i+w*j].y = 1.4142135624f*(w/2.-i)/w;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+2*w*h,field,taille2,cudaMemcpyHostToDevice);
  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 2.f*i/w-1.f; // Stretch X
      field[i+w*j].y = 0.f;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+3*w*h,field,taille2,cudaMemcpyHostToDevice);
  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 0.f; // Stretch Y
      field[i+w*j].y = 2.f*j/h-1.f;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+4*w*h,field,taille2,cudaMemcpyHostToDevice);

  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
    {
      field[i+w*j].x = 1.4142135624f*(j/h-.5f); // Shear
      field[i+w*j].y = 1.4142135624f*(i/w-.5f); 
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+5*w*h,field,taille2,cudaMemcpyHostToDevice);
}
