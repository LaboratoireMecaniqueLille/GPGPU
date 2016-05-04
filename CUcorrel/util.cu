#include <iostream> //cout
//#include <fstream> //f.open
//#include <stdlib.h> //atof
#include <cusolverDn.h> // Pour invert
#include "kernels.cuh"//reduce
#include "CUcorrel.h"//BLOCKSIZE
#include "lodepng/lodepng.h"

#include "util.h"

using namespace std;


MemMap::MemMap(uint2 i_dImg, uint2 i_dTile, uint2 i_tOffset)
{
  //Constructeur: on récupère les données d'init et on calcule les coeffs utiles pour tileToImg
  tOffset = i_tOffset;
  dImg = i_dImg;
  dTile = i_dTile;
  a = tOffset.x+dImg.x*tOffset.y;
  b = dImg.y;
}

uint MemMap::tileToImg(uint x, uint y)
{
  //Retourne l'adresse de l'élément dans l'image globale à partir des coordonées x,y dans le sous-élément
  return a+x+b*y;
}

uint MemMap::imgToTile(uint x, uint y)
{
  //Pour récupérer l'adresse dans le sous élément à partir des coordonnées globales (revoie 0 si hors du sous-élément)
  if(x < tOffset.x || x >= dTile.x+tOffset.x)
  {
    cout << "Coordonnée X hors de la tuile !" << endl;
    return 0;
  }
  if(y < tOffset.y || x >= dTile.y+tOffset.y)
  {
    cout << "Coordonnée Y hors de la tuile !" << endl;
    return 0;
  }
  return x-tOffset.x+dTile.y*(y-tOffset.y);
}

void printMat(float* data,uint x,uint y, uint step)
{
  cout << "\n============" << endl;
  for(int j = 0; j < y/step; j++)
  {
    for(int i = 0; i < x/step; i++)
    {
      cout << ((i==0)?"":", ") << data[step*(x*j+i)];
    }
    cout << endl;
  }
  cout << "\n" << endl;
}

void printMat2D(float2* data,uint x,uint y, uint step)
{
  cout << "\n============" << endl;
  for(int j = 0; j < y/step; j++)
  {
    for(int i = 0; i < x/step; i++)
    {
      cout << ((i==0)?"":"; ") << data[step*(x*j+i)].x << "," << data[step*(x*j+i)].y;
    }
    cout << endl;
  }
}


double timeDiff(struct timeval t1, struct timeval t2) //Retourne la différence en ms entre 2 mesures de temps avec gettimeofday(&t,NULL);
{
  return (t2.tv_sec-t1.tv_sec)*1000+(t2.tv_usec-t1.tv_usec)/1000.f;
}


void readParam(char** argv,float* values, int len)
{
  for(int i = 0;i < len;i++)
  {
    values[i] = atof(argv[i+1]);
  }
}

void readFile(char* address, float* data, float norm)
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

void writeFile(char* address, float* data, float offset, uint w, uint h)
{
  unsigned char *image = (unsigned char*)malloc(4*w*h*sizeof(unsigned char));
  unsigned char val = 0;
  for(int i = 0; i < w*h; i++)
  {
    val = max(0.f,min(255.f,data[i]+offset));
    image[4*i] = val;
    image[4*i+1] = val;
    image[4*i+2] = val;
    image[4*i+3] = 255;
  }
  if(lodepng_encode32_file(address, image, w, h))
  {
    cout << "Erreur lors de l'écriture !" << endl;
    exit(-1);
  }
  free(image);
}

void writeDiffFile(char* address, float* data1, float* data2, float gain, uint w, uint h)
{
  // Attention: cette fonction est longue à exécuter, ne pas en abuser sous risque de voir la durée d'exécution exploser
  unsigned char *image = (unsigned char*)malloc(4*w*h*sizeof(unsigned char));
  unsigned char val = 0;
  float diff,r,g;
  for(int i = 0; i < w*h; i++)
  {
    // Niveau de gris (sombre là ou d2 < d1, clair si d2 > d1)
    /*
    val = 128.f+max(-128.f,min(127.f,gain*(data1[i]-data2[i])));
    image[4*i] = val;
    image[4*i+1] = val;
    image[4*i+2] = val;
    image[4*i+3] = 255;
    */

    // Un peu plus avancé: affiche l'image déformée avec du rouge là ou d2 > d1 et du vert si d2 < d1
    diff = (data1[i] - data2[i])/256.f;
    r = min(255.f/256.f,-min(0.f,diff*gain));
    g = min(255.f/256.f,-min(0.f,-diff*gain));
    val = max(0.f,min(255.f,(data1[i])));
    val *= (1-r)*(1-g);
    r*=256.f;
    g*=256.f;
    image[4*i] = val+r;
    image[4*i+1] = val+g;
    image[4*i+2] = val;
    image[4*i+3] = 255;
  }
  if(lodepng_encode32_file(address, image, w, h))
  {
    cout << "Erreur lors de l'écriture !" << endl;
    exit(-1);
  }
  free(image);
}

void checkError(cusolverStatus_t cuSolverStatus)
{
  if(cuSolverStatus == CUSOLVER_STATUS_SUCCESS)
  {return;}
  else
  {
    cout << "/!\\ Cusolver error: ";
    
    switch(cuSolverStatus)
    {
      case CUSOLVER_STATUS_NOT_INITIALIZED:
        cout << "CUSOLVER_STATUS_NOT_INITIALIZED";
        break;
      case CUSOLVER_STATUS_ALLOC_FAILED:
        cout << "CUSOLVER_STATUS_ALLOC_FAILED";
        break;
      case CUSOLVER_STATUS_INVALID_VALUE:
        cout << "CUSOLVER_STATUS_INVALID_VALUE";
        break;
      case CUSOLVER_STATUS_ARCH_MISMATCH:
        cout << "CUSOLVER_STATUS_ARCH_MISMATCH";
        break;
      case CUSOLVER_STATUS_EXECUTION_FAILED:
        cout << "CUSOLVER_STATUS_EXECUTION_FAILED";
        break;
      case CUSOLVER_STATUS_INTERNAL_ERROR:
        cout << "CUSOLVER_STATUS_INTERNAL_ERROR";
        break;
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        cout << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        break;
      default:
        cout << "Unknown error: " << cuSolverStatus;
    }
    cout << endl;
    exit(-1);
  }
}

void invert(float* devA, float* devInv)
{
  cusolverDnHandle_t handle = NULL;
  cusolverDnCreate(&handle);
  //cudaMalloc(&devInv,PARAMETERS*PARAMETERS*sizeof(double));// On suppose qu'il est déjà alloué...
  float B[PARAMETERS*PARAMETERS] = {0};
  for(int i = 0; i < PARAMETERS; i++)
  {B[(PARAMETERS+1)*i] = 1;}
  cudaMemcpy(devInv,B,PARAMETERS*PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);
  int bufferSize=0;
  cusolverDnSgetrf_bufferSize(handle,PARAMETERS,PARAMETERS,devA,PARAMETERS,&bufferSize);
  float* buffer;
  cudaMalloc(&buffer,bufferSize*sizeof(float));
  int* piv;
  cudaMalloc(&piv,PARAMETERS*sizeof(float));
  int *info;
  cudaMalloc(&info,sizeof(int));
  checkError(cusolverDnSgetrf(handle,PARAMETERS,PARAMETERS,devA,PARAMETERS,buffer,piv,info));
  checkError(cusolverDnSgetrs(handle,CUBLAS_OP_N,PARAMETERS,PARAMETERS,devA,PARAMETERS,piv,devInv,PARAMETERS,info));
  cudaFree(buffer);
  cudaFree(piv);
  cudaFree(info);
}
