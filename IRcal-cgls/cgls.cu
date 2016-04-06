#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <sys/time.h>
#include "util.h"
#include "kernels.h"

#define NOM_FICHIER "data.csv"


using namespace std;

int main(int argc, char** argv)
{
  int N = 6, M = 14;
  int nb_lines = 640*512;
  struct timeval t1,t2;
  string line;
  ifstream file;
  double *data = NULL;

  cout << "Ouverture du fichier " << NOM_FICHIER << "..." << flush;
  gettimeofday(&t1,NULL);
  file.open(NOM_FICHIER);
  if(!file.is_open()){exit(-1);}
  file.seekg(0,ios::beg);*/
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  
  cout << "Allocation et remlpissage du tableau..." << flush;
  gettimeofday(&t1, NULL);
  data = (double*)malloc(nb_lines*M*sizeof(double));
  int l = 0;
  while(getline(file, line))
  {
    getValues(line, data+l*M, M);
    l++;
  }
  file.close();
  gettimeofday(&t2, NULL);
  cout << " Ok. (" << timeDiff(t1,t2) << " ms)." << endl;

  cout << "Préparation..." << flush;
  gettimeofday(&t1,NULL);
  double *Y = (double*)malloc(M*sizeof(double));
  for(int i = 0; i < M; i++)
  {
    Y[i] = 17+i;
  }
  double *devY = NULL;
  cudaMalloc(&devY,M*sizeof(double));
  cudaMemcpy(devY,Y,M*sizeof(double),cudaMemcpyHostToDevice);

  double *A = (double*)malloc(nb_lines*N*N*sizeof(double));
  
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;

  cout << "Création des matrices et vecteurs des systèmes (GPU)..." << flush;
  gettimeofday(&t1,NULL);
  double *devData = NULL, *devPow = NULL;
  cudaMalloc(&devData, M*nb_lines*sizeof(double));
  cudaMalloc(&devPow, M*(2*N-1)*nb_lines*sizeof(double));
  cudaMemcpy(devData, data, M*nb_lines*sizeof(double),cudaMemcpyHostToDevice);

  double *devA = NULL; // Tableau 1D contenant toutes les matrices des systèmes à la suite (peu importe le mode row-major ou column-major: elles sont symétriques).
  double *devB = NULL; // Tableau contenant les vecteurs des membres de droite des systèmes à la suite
  cudaMalloc(&devA, nb_lines*N*N*sizeof(double));
  powArray<<<M*nb_lines/1024,1024>>>(devData,devPow, M,2*N-1); // On stocke les différentes puissances des valeurs (jusqu'à ^2N-1)
  sumCol<<<(2*N-1)*nb_lines/1024,1024>>>(devPow,devData,M); // On les somme pour donner les différents coefficients de la matrice (AT.A)
  dim3 taille(N,N,8);
  HankelFill<<<nb_lines/8,taille>>>(devData,devA,N);  // On remplit les matrices de Hankel des systèmes



  cudaMemcpy(devData, data, M*nb_lines*sizeof(double),cudaMemcpyHostToDevice);
  cudaMalloc(&devB, nb_lines*N*sizeof(double));
  powArray<<<M*nb_lines/1024,1024>>>(devData,devPow, M,N); // On stocke les différentes puissances des valeurs (jusqu'à ^N cette fois)
  ewMul<<<N*M*nb_lines/1024,1024>>>(devPow,devY,M);
  sumCol<<<N*nb_lines/1024,1024>>>(devPow,devB,M);


  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;

  cudaMemcpy(A,devPow,50*sizeof(double),cudaMemcpyDeviceToHost);
  /*printVec(A,50);
  exit(0);*/


  cudaFree(devData);
  cudaFree(devPow);


  cout << "Allocation et copie sur le device..." << flush;
  gettimeofday(&t1,NULL);
  double *devO, *devX;
  cudaMalloc(&devO, nb_lines*N*sizeof(double));
  cudaMalloc(&devX, nb_lines*N*sizeof(double));

  cudaMemset(devX,0,nb_lines*N*sizeof(double));

  double *devR, *devP, *devF, *devAp, *devAlpha, *devBuff; 
  cudaMalloc(&devR, nb_lines*N*sizeof(double));
  cudaMalloc(&devP, nb_lines*N*sizeof(double));
  cudaMalloc(&devF, nb_lines*sizeof(double));
  cudaMalloc(&devAp, nb_lines*N*sizeof(double));
  cudaMalloc(&devAlpha, nb_lines*sizeof(double));
  cudaMalloc(&devBuff, N*nb_lines*sizeof(double));

  //------------ Initialisation ------------
  //cudaMemcpy(devR,B,nb_lines*N*sizeof(double),cudaMemcpyHostToDevice); //On part de X = {0} donc R = B
  vecCpy<<<N*nb_lines/1024,1024>>>(devR,devB);
  //cudaMemcpy(devP,B,nb_lines*N*sizeof(double),cudaMemcpyHostToDevice);
  vecCpy<<<N*nb_lines/1024,1024>>>(devP,devB);
  norm2<<<nb_lines/1024,1024>>>(devR,devF,N);
  dim3 size(N,128);
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  //------------ Itérations ---------------
  cout << "Résolution..." << flush;
  gettimeofday(&t1,NULL);
  for(int i = 0; i < N; i++)
  {
    
    myDot<<<nb_lines/128,size,128*N*sizeof(double)>>>(devA,devP,devAp,N,N); //Ap = A.p
    vecMul<<<nb_lines/1024,1024>>>(devP,devAp,devBuff, N); //Buff = Pt.Ap
    ewDiv<<<nb_lines/1024,1024>>>(devF,devBuff,devAlpha); //alpha = F/Buff
    scalMul<<<nb_lines/1024,1024>>>(devP,devAlpha,N,devBuff); //Buff = alpha*p
    vecSum<<<(nb_lines*N)/1024,1024>>>(devX,devBuff);  //x += Buff
    scalMul<<<nb_lines/1024,1024>>>(devAp,devAlpha,N,devBuff); //Buff = alpha*Ap
    vecDiff<<<(nb_lines*N)/1024,1024>>>(devR,devBuff); //r -= Buff
    vecCpy<<<nb_lines/1024,1024>>>(devBuff,devF);   //Buff = F
    norm2<<<nb_lines/1024,1024>>>(devR,devF,N);  //F = ||r||²
    ewDiv<<<nb_lines/1024,1024>>>(devF,devBuff,devBuff); //Buff = F/Buff
    scalMul<<<nb_lines/1024,1024>>>(devP,devBuff,N); // p = buff*p
    vecSum<<<(nb_lines*N)/1024,1024>>>(devP,devR);  // p += r
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;

  cout << "Copie vers l'hôte..." << flush;
  gettimeofday(&t1,NULL);
  double *X = (double*)malloc(N*nb_lines*sizeof(double));
  cudaMemcpy(X,devX,N*nb_lines*sizeof(double),cudaMemcpyDeviceToHost);
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << "ms)."  << endl;

  // Pour la debug: 

  /*printVec(X,3*N);
  printVec(X+N*nb_lines/2,3*N);



  for(int i = nb_lines/2 - 10; i < nb_lines/2; i++) // Verif partielle
  {
    cout << "Pixel " << i / 512 << ", " << i % 512 << ":" << endl;
    for(int j = 0; j < M; j++)
    {
      cout << Y[j] << ", " << pol(data[i*M+j],X+i*N,N) << " (" << abs(Y[j] - pol(data[i*M+j],X+i*N,N))  << ")" << endl;
    }
  }

  cout << "Vérification..." << endl;
  int err = 0, loop = 0;   //Verif complète
  for(int i = 0; i < nb_lines; i++)
  {
    loop = 0;
    for(int j = 0; j < M; j++)
    {
      if(abs(Y[j]-pol(data[i*M+j],X+i*N,N)) > 0.05)
      {
      if(loop == 0){err++;loop=1;
      //cout << "Pixel " << i / 512 << ", " << i % 512 << ":" << endl;
      }
      //cout << Y[j] << ", " << pol(data[i*M+j],X+i*N,N) << " (" << abs(Y[j] - pol(data[i*M+j],X+i*N,N))  << ")" << endl;
      }
    }
  }
  cout << "Erreurs: " << err << "/" << nb_lines << "(" << 100.*err/nb_lines << "% des pixels)." << endl;*/



  
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devO);
  cudaFree(devX);
  cudaFree(devR);
  cudaFree(devP);
  cudaFree(devF);
  cudaFree(devAp);
  cudaFree(devAlpha);
  cudaFree(devBuff);
  return 0;
}
