#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.cuh"
#include "CUcorrel.h"
#include "util.h"

using namespace std;


int main(int argc, char** argv)
{

  struct timeval t0, t1, t2; // Pour mesurer les durées d'exécution
  size_t taille = WIDTH*HEIGHT*sizeof(float); // Taille d'un tableau contenant une image
  size_t taille2 = WIDTH*HEIGHT*sizeof(float2); // idem à 2 dimensions (fields)
  int nbIter=20; // Le nombre d'itérations
  char iAddr[10] = "img.csv"; // Le nom du fichier à ouvrir
  float *orig = (float*)malloc(taille); // le tableau contenant l'image sur l'hôte
  dim3 blocksize(min(32,WIDTH),min(32,HEIGHT)); // Pour l'appel aux kernels sur toute l'image
  dim3 gridsize((WIDTH+31)/32,(HEIGHT+31)/32); // ...
  dim3 tailleMat(PARAMETERS,PARAMETERS); // La taille de la hessienne

  float *devOrig; // Image originale
  float *devGradX; // Gradient de l'image d'origine par rapport à X
  float *devGradY; // .. à Y
  float2 *devFields; // Contient les PARAMETERS champs de déplacements élémentaires à la suite dont on cherche l'influence par autant de paramètres
  float *devG; // Les PARAMETERS matrices gradient*champ
  float *devParam; // Contient la valeur actuelle calculée des paramètres
  float *devDef; // Image déformée à recaler (ici calculée à partir de l'image d'origine)
  float *devOut; // L'image interpolée à chaque itération
  float *devMatrix; // La hessienne utilisée pour la méthode de Newton
  float *devInv;  // L'inverse de la Hessienne
  float *devVec; // Vecteur pour stocker les PARAMETERS valeurs du gradient à chaque itération
  float *devVecStep; // Multiplie terme à terme la direction avant le l'ajouter aux paramètres

  srand(time(NULL)); // Seed pour générer le bruit avec rand()

  // ---------- Allocation tous les tableaux du device ---------
  cudaMalloc(&devOrig,taille);
  cudaMalloc(&devGradX,taille);
  cudaMalloc(&devGradY,taille);
  cudaMalloc(&devFields,PARAMETERS*taille2);
  cudaMalloc(&devG,PARAMETERS*taille);
  cudaMalloc(&devParam,PARAMETERS*sizeof(float));
  cudaMalloc(&devDef,taille);
  cudaMalloc(&devOut,taille);
  cudaMalloc(&devMatrix,PARAMETERS*PARAMETERS*sizeof(float));
  cudaMalloc(&devInv,PARAMETERS*PARAMETERS*sizeof(float));
  cudaMalloc(&devVec,PARAMETERS*sizeof(float));
  cudaMalloc(&devVecStep,PARAMETERS*sizeof(float));

  // ---------- Lecture du fichier et écriture sur le device ---------
  readFile(iAddr,orig,256);
  cudaMemcpy(devOrig,orig,taille,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cout << "Image d'origine" << endl;
  printMat(orig,WIDTH,HEIGHT,256);

  // --------- Initialisation de la texture et calcul des gradients ---------
  gettimeofday(&t1,NULL);
  initCuda(devOrig);
  gradient<<<gridsize,blocksize>>>(devGradX,devGradY);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << "\nCalcul des gradients: " << timeDiff(t1,t2) << " ms." << endl;

  //-------- Affichage des gradients -------
  cout << "Gradient X:" << endl;
  cudaMemcpy(orig,devGradX,taille,cudaMemcpyDeviceToHost);
  printMat(orig,WIDTH,HEIGHT,256);

  // --------- Écriture des fields définis dans fields.cu ----------
  writeFields(devFields);

  // --------- Calcul des matrices G ----------
  gettimeofday(&t1,NULL);
  makeG<<<1,PARAMETERS>>>(devG,devFields,devGradX,devGradY);
  cudaDeviceSynchronize();
  if(cudaGetLastError() == cudaErrorMemoryAllocation)
  {cout << "Erreur d'allocation (manque de mémoire graphique ?)" << endl;exit(-1);}
  else if(cudaGetLastError() != cudaSuccess)
  {cout << "Erreur lors de l'allocation." << endl;exit(-1);}
  gettimeofday(&t2,NULL);
  cout << "Calcul des matrices G: " << timeDiff(t1,t2) << " ms." << endl;

  // ------- [Facultatif] Écriture des G en .csv pour les visualiser -----------
  /*
  char oAddr[3];
  for(int i = 0;i < PARAMETERS;i++)
  {
  cudaMemcpy(orig,devG+i*WIDTH*HEIGHT,taille,cudaMemcpyDeviceToHost);
  sprintf(oAddr,"%d",i);
  writeFile(oAddr, orig, 1);
  }
  */
  
  // --------- Allocation et assignation des paramètres de déformation de devDef ----------
  float param[PARAMETERS] = {-.2,-2.318,3.22,-1.145,1.37,2.3,0};
  cout << "Paramètres réels: ";
  for(int i = 0; i < PARAMETERS;i++){cout << param[i] << ", ";}
  cout << endl;
  cudaMemcpy(devParam, param, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);

  // ---------- Calcul de l'image à recaler ----------
  deform2D<<<gridsize,blocksize>>>(devDef,devFields,devParam);

  // ---------- Bruitage de l'image déformée ---------
  for(int i = 0; i < WIDTH*HEIGHT ; i++)
  { 
    orig[i] = (float)rand()/RAND_MAX*4-2;
  }
  cudaMemcpy(devOut,orig,taille,cudaMemcpyHostToDevice);// Pour ajouter le bruit
  addVec<<<WIDTH*HEIGHT/1024,1024>>>(devDef,devOut);

  // ---------- Calcul de la Hessienne ----------
  gettimeofday(&t1,NULL);
  makeMatrix<<<1,tailleMat>>>(devMatrix,devG);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  cout << "Génération de la matrice: " << timeDiff(t1,t2) << " ms." << endl;

  // ---------- [Facultatif] Affichage de la Hessienne ----------
  float test[PARAMETERS*PARAMETERS];
  cudaMemcpy(test,devMatrix,PARAMETERS*PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
  cout << "\nHessienne:" << endl;
  printMat(test,PARAMETERS,PARAMETERS);

  // ---------- Inversion de la hessienne ----------
  gettimeofday(&t1,NULL);
  invert(devMatrix,devInv);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << "Inversion de la matrice: " << timeDiff(t1,t2) << " ms." << endl;

  // ---------- [Facultatif] Affichage de l'inverse ----------
  cudaMemcpy(test,devInv,PARAMETERS*PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
  cout << "\nMatrice inversée:" << endl;
  printMat(test,PARAMETERS,PARAMETERS);

  // --------- [Facultatif] Écriture de l'image déformée en .csv pour la visualiser ----------
  /*
  char oAddr[10] = "out.csv";
  cudaMemcpy(orig,devDef,taille,cudaMemcpyDeviceToHost); // Pour récupérer l'image
  writeFile(oAddr, orig, 1);
  */

  // ---------- Écriture des paramètres initiaux ----------
  for(int i = 0; i < PARAMETERS; i++)
  {param[i] = 0;}
  //readParam(argv,param); // Pour tester des valeurs de paramètres par défaut sans recompiler
  cudaMemcpy(devParam, param, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);

  // ---------- Écriture du pas des paramètres ----------
  float vecStep[PARAMETERS] = {2,2,2,2,2,2,2};
  cudaMemcpy(devVecStep,vecStep,PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);

  float res = 10000000000;
  float oldres=0;
  float vec[PARAMETERS];

  for(int i = 0;i < nbIter; i++)
  {
    gettimeofday(&t0,NULL);
    cout << "Boucle n°" << i+1 << endl;
    cudaMemcpy(param,devParam,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
    cout << "Paramètres calculés: ";
    for(int i = 0; i < PARAMETERS;i++){cout << param[i] << ", ";}
    cout << endl;

    gettimeofday(&t1,NULL);
    deform2D<<<gridsize,blocksize>>>(devOut, devFields, devParam);//--
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    cout << "\nInterpolation: " << timeDiff(t1,t2) << "ms." << endl;

    gettimeofday(&t1,NULL);
    gradientDescent(devG, devOut, devDef, devVec);//--
    cudaDeviceSynchronize();
    gettimeofday(&t2,NULL);
    cout << "Calcul des gradients des paramètres: " << timeDiff(t1,t2) << " ms." << endl;

    cudaMemcpy(vec,devVec,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
    cout << "Gradient des paramètres:" << endl;
    printMat(vec,PARAMETERS,1);
    
    gettimeofday(&t1,NULL);
    myDot<<<1,PARAMETERS,PARAMETERS*sizeof(float)>>>(devInv,devVec,devVec);//--
    ewMul<<<1,PARAMETERS>>>(devVec,devVecStep);//--
    addVec<<<1,PARAMETERS>>>(devParam,devVec);//--
    cudaDeviceSynchronize();
    gettimeofday(&t2,NULL);
    cout << "Mise à jour des valeurs: " << timeDiff(t1,t2) << " ms." << endl;

    cudaMemcpy(vec,devVec,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
    cout << "Direction:" << endl;
    printMat(vec,PARAMETERS,1);

    gettimeofday(&t1, NULL);
    oldres = res;//--
    res = residuals(devOut, devDef, HEIGHT*WIDTH)/HEIGHT/WIDTH;//--
    if(oldres - res < 0)//--
    {cout << "Augmentation de la fonctionnelle !!" << endl;}//--
    gettimeofday(&t2, NULL);
    cout << "\nÉcart: "<< res << ", Calcul de l'écart: " << timeDiff(t1,t2) << "ms." << endl;
    cout << "\nExécution de toute la boucle: " << timeDiff(t0,t2) << "ms.\n**********************\n\n\n" << endl;

  }

  //Vérification d'erreur éventuelle
  cudaError_t err;
  err = cudaGetLastError();
  cout << "Cuda status: " << ((err == 0)?"OK.":"ERREUR !!") << endl;
  cout << err << endl;
  if(err != 0)
  {cout << cudaGetErrorName(err) << endl;}

  //Pour libérer ce qui a été alloué avec initCuda
  cleanCuda();

  //On libère toute la mémoire GPU
  cudaFree(devOrig);
  cudaFree(devGradX);
  cudaFree(devGradY);
  cudaFree(devFields);
  cudaFree(devG);
  cudaFree(devParam);
  cudaFree(devDef);
  cudaFree(devOut);
  cudaFree(devMatrix);
  cudaFree(devInv);
  cudaFree(devVec);
  cudaFree(devVecStep);
  return 0;
}
