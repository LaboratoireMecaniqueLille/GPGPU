#include <iostream> //cout
#include <fstream> //f.open
#include <stdlib.h> //atof
#include <cusolverDn.h> // Pour invert
#include "kernels.h"//reduce
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
  return (t2.tv_sec-t1.tv_sec)*1000+1.*(t2.tv_usec-t1.tv_usec)/1000;
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

void writeFile(char* address, float* data, float norm)
{
  ofstream f;
  f.open(address);
  if(!f.is_open()){cout << "Erreur lors de l'ouverture du fichier" << endl;exit(-1);}
  for(int j = 0;j < HEIGHT;j++)
  {
    for(int i = 0; i < WIDTH;i++)
    {
      f << 128+(int)(norm*data[i+j*WIDTH]) << ",";
    }
    f << "\n";
  }
  f.close();
  
}

