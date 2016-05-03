#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "reduction.cuh"
#include "main.h"

using namespace std;

int main(int argc,char** argv)
{
  size_t taille = N*sizeof(TYPE);
  struct timeval t1, t2;
  TYPE *array = (TYPE*)malloc(taille);
  TYPE *devArray;
  cudaMalloc(&devArray,taille);
  srand(time(NULL));
  for(uint i = 0; i < N; i++)
  {
    //array[i] = (TYPE)rand()/RAND_MAX/N;
    //array[i] = ((float)i/N)*((float)i/N);
    array[i] = 1;
  }
  cudaMemcpy(devArray,array,taille,cudaMemcpyHostToDevice);
  gettimeofday(&t1,NULL);
  sum(devArray,N);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cudaMemcpy(array,devArray,sizeof(TYPE),cudaMemcpyDeviceToHost);
  cout << "temps de calcul: " << (t2.tv_sec-t1.tv_sec)*1000+1.*(t2.tv_usec-t1.tv_usec)/1000 << "ms." << endl;
  cout << "somme: " << array[0] << endl;
  cout << "N: " << N << endl;

  free(array);
  cudaFree(devArray);
  cudaError_t err;
  err = cudaGetLastError();
  cout << "Cuda status: " << ((err == 0)?"OK.":"ERREUR !!") << endl;
  //cout << err << endl;
  if(err != 0)
  {cout << cudaGetErrorName(err) << endl;}
  return 0;
}
