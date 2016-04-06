#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.h"
#include "CUcorrel.h"
#include "util.h"


using namespace std;

int main(int argc, char** argv)
{

  struct timeval t1, t2;
  size_t taille = WIDTH*HEIGHT*sizeof(float);
  size_t taille2 = WIDTH*HEIGHT*sizeof(float2);
  int nbIter=20;
  char iAddr[10] = "img.csv";
  char oAddr[10] = "out0.csv";
  srand(time(NULL));
  float *orig = (float*)malloc(taille);
  for(int i = 0; i < HEIGHT*WIDTH; i++)
  {
    orig[i] = (float)i/WIDTH/HEIGHT+(float)rand()/RAND_MAX/1000;
    //orig[i] = (float)rand()/RAND_MAX;
  }
  float step = 1;
  float vecStep[PARAMETERS] = {0.1,0.1,0.1,1,1,1,1};

  readFile(iAddr,orig,256);
  cout << "Image d'origine" << endl;
  printMat(orig,WIDTH,HEIGHT,256);

  dim3 blocksize(min(32,WIDTH),min(32,HEIGHT));
  dim3 gridsize((WIDTH+31)/32,(HEIGHT+31)/32);

  float *devOrig; // Image originale
  float *devDef; //Image déformée à recaler (ici calculée à partir de l'image d'origine)
  float *devGradX; //Gradient de l'image d'origine par rapport à X
  float *devGradY; //.. à Y
  float2 *devFields; // Contient les PARAMETERS champs de déplacements élémentaires dont on cherche l'influence par autant de paramètres
  float *devG; //Les PARAMETERS matrices gradient*champ
  float *devOut; // L'image interpolée à chaque itération


  cudaMalloc(&devFields,PARAMETERS*taille2); // Les champs de déformations élémentaires correspondants aux différents modes, placés successivements dans un tableau

  //Assignation des champs
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
      field[i+WIDTH*j].x = (float)i/WIDTH; // Stretch X
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
      field[i+WIDTH*j].y = (float)j/HEIGHT;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+4*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);

  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = (float)j/HEIGHT; // Shear X
      field[i+WIDTH*j].y = 0; 
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+5*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);

  for(int i = 0; i < WIDTH; i++)
  {
    for(int j = 0; j < HEIGHT; j++)
    {
      field[i+WIDTH*j].x = 0; // Shear Y
      field[i+WIDTH*j].y = (float)i/WIDTH;
    }
  }
  //printMat2D(field,WIDTH,HEIGHT,256);
  cudaMemcpy(devFields+6*WIDTH*HEIGHT,field,taille2,cudaMemcpyHostToDevice);



  cudaMalloc(&devOrig,taille);
  cudaMalloc(&devDef,taille);
  cudaMalloc(&devGradX,taille);
  cudaMalloc(&devGradY,taille);


  cudaMemcpy(devOrig,orig,taille,cudaMemcpyHostToDevice);
  initCuda(devOrig);
  gettimeofday(&t1,NULL);
  gradient<<<gridsize,blocksize>>>(devGradX,devGradY);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << "\nCalcul des gradients: " << timeDiff(t1,t2) << " ms." << endl;
  
  //-------- Vérification des gradients -------
  cout << "Gradient Y:" << endl;
  cudaMemcpy(orig,devGradY,taille,cudaMemcpyDeviceToHost);
  printMat(orig,WIDTH,HEIGHT,256);
  //-------------------------------------------

  gettimeofday(&t1,NULL);
  cudaMalloc(&devG,PARAMETERS*taille);
  makeG<<<1,PARAMETERS>>>(devG,devFields,devGradX,devGradY);
  cudaDeviceSynchronize();
  if(cudaGetLastError() == cudaErrorMemoryAllocation)
  {cout << "Erreur d'allocation (manque de mémoire graphique ?)" << endl;exit(-1);}
  else if(cudaGetLastError() != cudaSuccess)
  {cout << "Erreur lors de l'allocation." << endl;exit(-1);}
  gettimeofday(&t2,NULL);
  cout << "Calcul des matrices G: " << timeDiff(t1,t2) << " ms." << endl;


  // ------- POUR VISUALISER G -----------

  /*for(int i = 0;i < PARAMETERS;i++)
  {
  cudaMemcpy(orig,devG+i*WIDTH*HEIGHT,taille,cudaMemcpyDeviceToHost);
  sprintf(oAddr,"%d",i);
  writeFile(oAddr, orig, 1);                             //
  }
  exit(0);*/
  

/* -- Code pour générer la Hessienne, plus utilisé ---

  cudaMalloc(&devMatrix,PARAMETERS*PARAMETERS*sizeof(float));
  dim3 tailleMat(PARAMETERS,PARAMETERS);
  gettimeofday(&t1,NULL);
  makeMatrix<<<1,tailleMat>>>(devMatrix,devG);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  cout << "Génération de la matrice: " << timeDiff(t1,t2) << " ms." << endl;


  float test[PARAMETERS*PARAMETERS];
  cudaMemcpy(test,devMatrix,PARAMETERS*PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
  cout << "\nMatrice:" << endl;
  printMat(test,PARAMETERS,PARAMETERS);
*/


  //float param[7] = {2.81,-.86,1.36,.145,4.037,.0036,-4.97};
  float param[7] = {0,0,0,.145,4.037,.0036,-4.97};
  //float param[7] = {7.81,-3.86,6.36,3.145,4.037,.0036,-4.97};
  //float param[7] = {0,0,2,0,0,0,0};
  cout << "Paramètres réels: ";
  for(int i = 0; i < PARAMETERS;i++){cout << param[i] << ", ";}
  cout << endl;
  float* devParam;
  cudaMalloc(&devParam,PARAMETERS*sizeof(float));
  cudaMemcpy(devParam, param, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);
  

  deform2D<<<gridsize,blocksize>>>(devDef,devFields,devParam); //Calcule l'image à recaler
  
  cudaMemcpy(orig,devDef,taille,cudaMemcpyDeviceToHost); // Pour récupérer l'image
  writeFile(oAddr, orig, 1);                             //

  
  cudaMalloc(&devOut,taille);

    //param[0] = 2.7;param[1] = -0.86;param[2] = 1.6;param[3] = .345;param[4] = 3.7;param[5] = .06;param[6] = -3.97;
  param[0] = 0;param[1] = 0;param[2] = 0;param[3] = 0;param[4] = 0;param[5] = 0;param[6] = 0;
  //readParam(argv,param); // Pour tester des valeurs sans recompiler
  cudaMemcpy(devParam, param, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);


  float res = 10000000000;

  float* devVec;
  float oldres=0;
  cudaMalloc(&devVec,PARAMETERS*sizeof(float));
  float vec[PARAMETERS];
  for(int i = 0;i < nbIter; i++)
  {
    cout << "Boucle n°" << i+1 << endl;
    cout << "Paramètres calculés: ";
    for(int i = 0; i < PARAMETERS;i++){cout << param[i] << ", ";}
    cout << endl;

    gettimeofday(&t1,NULL);
    deform2D<<<gridsize,blocksize>>>(devOut, devFields, devParam);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    cout << "\nInterpolation: " << timeDiff(t1,t2) << "ms." << endl;
    //cudaMemcpy(orig,devOut,taille,cudaMemcpyDeviceToHost);
    //cout << "Image modifiée:" << endl;
    //printMat(orig,WIDTH,HEIGHT,256);

    gettimeofday(&t1,NULL);
    gradientDescent(devG, devOut, devDef, vec);
    gettimeofday(&t2,NULL);
    cout << "Calcul des gradients des paramètres: " << timeDiff(t1,t2) << " ms." << endl;
    cout << "Valeurs:" << endl;
    printMat(vec,PARAMETERS,1);
    for(int i = 0;i < PARAMETERS; i++)
    {
      param[i] -= step*vecStep[i]*vec[i];
    }
    cudaMemcpy(devParam,param,PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);
    

    gettimeofday(&t1, NULL);
    oldres = res;
    res = residuals(devOut, devDef, HEIGHT*WIDTH)/HEIGHT/WIDTH;
    if(res <= oldres)
    {step *= 1.2;}
    else
    {step *= -.1;cout << "Reduction du pas !" << endl;}
    //cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    cout << "\nÉcart: "<< res << ", Calcul de l'écart: " << timeDiff(t1,t2) << "ms." << endl;
  }
  int err = 0;
  err = cudaGetLastError();
  cout << "Cuda status: " << ((err == 0)?"OK.":"ERREUR !!") << endl;
  cout << err << endl;
  cleanCuda();
  cudaFree(devOut);
  cudaFree(devG);
  cudaFree(devOrig);
  cudaFree(devDef);
  cudaFree(devGradX);
  cudaFree(devGradY);

  return 0;
}
