#include <iostream> //cout
#include <fstream> //f.open
#include <stdlib.h> //atof
#include <cusolverDn.h> // Pour invert
#include "kernels.cuh"//reduce
#include "CUcorrel.h"//BLOCKSIZE

using namespace std;

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


float GPUsum(float* devData, uint size)
{

  while(size>1)
  {
    reduce<<<(size+BLOCKSIZE-1)/BLOCKSIZE,min(size,BLOCKSIZE)>>>(devData,size);
    size = (size+BLOCKSIZE-1)/BLOCKSIZE;
  }
  float out;
  cudaMemcpy(&out,devData,sizeof(float),cudaMemcpyDeviceToHost);
  return out;
}

void readParam(char** argv,float* values, int len)
{
  for(int i = 0;i < len;i++)
  {
    values[i] = atof(argv[i+1]);
  }
}


void getValues(string s, float* tab, int M)
{
  string sep = ",";
  for(int i=0;i < M; i++)
  {
    tab[i] = atoi(s.substr(0,s.find(sep)).c_str());
    s = s.erase(0,s.find(sep)+1);
    //cout << tab[i] << ", ";
  }
  //cout << endl;

}

void readFile(char* address, float* data, float norm)
{
  ifstream f;
  string line;
  f.open(address);
  if(!f.is_open()){cout << "Erreur lors de l'ouverture du fichier" << endl;exit(-1);}
  for(int j = 0; j < HEIGHT;j++)
  {
    getline(f,line);
    getValues(line,data+j*WIDTH,HEIGHT);
  }
  f.close();
}

void writeFile(char* address, float* data, float norm, int offset, uint w, uint h)
{
  ofstream f;
  f.open(address);
  if(!f.is_open()){cout << "Erreur lors de l'ouverture du fichier" << endl;exit(-1);}
  for(int j = 0;j < h;j++)
  {
    for(int i = 0; i < w;i++)
    {
      f << (int)(norm*data[i+j*w]-1.f)+offset << ",";
    }
    f << "\n";
  }
  f.close();
  
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
