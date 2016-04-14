#include "CUcorrel.h"
//#include "util.h"

//7 champs: 3 mvt de corps solide, 2 déformations uniformes et 2 cisaillements

void writeFields(float2* devFields)
{
  //Assignation des champs
  size_t taille2 = WIDTH*HEIGHT*sizeof(float2);
  float2 *field = (float2*)malloc(PARAMETERS*taille2);

  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 1; // Move X
      field[i+WIDTH*j].y = 0;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields,field,taille2,cudaMemcpyHostToDevice);
  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 0; // Move Y
      field[i+WIDTH*j].y = 1;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);

  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 1.4142135624*(j-HEIGHT/2.)/HEIGHT; // Rotation
      field[i+WIDTH*j].y = 1.4142135624*(WIDTH/2.-i)/WIDTH;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+2*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);
  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 2.*i/WIDTH-1.; // Stretch X
      field[i+WIDTH*j].y = 0;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+3*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);
  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 0; // Stretch Y
      field[i+WIDTH*j].y = 2.*j/HEIGHT-1.;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+4*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);

  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 1.4142135624*j/HEIGHT-.5; // Shear
      field[i+WIDTH*j].y = 1.4142135624*i/WIDTH-.5; 
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+5*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);
  free(field);
}
