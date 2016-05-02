#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "kernels.cuh"
#include "CUcorrel.h"
#include "util.h"

using namespace std;


int main(int argc, char** argv)
{
  struct timeval t0, t1, t2, t00; // Pour mesurer les durées d'exécution
  cudaError_t err; // Pour récupérer les erreurs éventuelles
  size_t taille = IMG_SIZE*sizeof(float); // Taille d'un tableau contenant une image
  size_t taille2 = IMG_SIZE*sizeof(float2); // idem à 2 dimensions (fields)
  int nbIter=10; // Le nombre d'itérations
  char iAddr[10] = "img.png"; // Le nom du fichier à ouvrir
  float *orig = (float*)malloc(taille); // le tableau contenant l'image sur l'hôte
  dim3 blocksize[LVL]; // Pour l'appel aux kernels sur toute l'image (une pour chaque étage)
  dim3 gridsize[LVL];
  char oAddr[25]; // pour écrire les noms des fichiers de sortie
  float param[PARAMETERS] = {0}; // Stocke les paramètres calculés
  float res; // Le résidu 
  float oldres; // Pour stocker le résidu de l'itération précédente et comparer
  float vec[PARAMETERS]; // Pour stocker sur l'hôte les paramètres calculés
  int c = 0; // Pour compter les boucles et quitter si on ajoute trop
  uint div = 1; // Pour diviser la taille dans les boucles
  for(int i = 0; i < LVL; i++)
  {
    blocksize[i].x = min(32,WIDTH/div);
    blocksize[i].y = min(32,HEIGHT/div);
    blocksize[i].z = 1;
    gridsize[i].x = (WIDTH/div+31)/32;
    gridsize[i].y = (HEIGHT/div+31)/32;
    gridsize[i].z = 1;
    div *= 2;
  }
  dim3 tailleMat(PARAMETERS,PARAMETERS); // La taille de la hessienne

  float *devOrig; // Image originale
  float *devGradX; // Gradient de l'image d'origine par rapport à X
  float *devGradY; // .. à Y
  float2 *devFields[LVL]; // Contient les PARAMETERS champs de déplacements élémentaires à la suite dont on cherche l'influence par autant de paramètres
  float *devParam; // Contient la valeur actuelle calculée des paramètres
  float *devDef[LVL]; // Image déformée à recaler (ici calculée à partir de l'image d'origine)
  float *devOut; // L'image interpolée à chaque itération
  float *devMatrix; // La hessienne utilisée pour la méthode de Newton
  float *devInv;  // L'inverse de la Hessienne
  float *devVec; // Vecteur pour stocker les PARAMETERS valeurs du gradient à chaque itération
  float *devVecStep; // Multiplie terme à terme la direction avant le l'ajouter aux paramètres
  float *devVecOld; // Pour stocker le vecteur précédent et le restaurer si nécessaire

  srand(time(NULL)); // Seed pour générer le bruit avec rand()

  // ---------- Allocation de tous les tableaux du device ---------
  cudaMalloc(&devOrig,taille);
  cudaMalloc(&devGradX,taille);
  cudaMalloc(&devGradY,taille);
  div = 1;
  for(int i = 0; i < LVL; i++)
  {
    cudaMalloc(&devFields[i],PARAMETERS*taille2/div);
    cudaMalloc(&devDef[i],taille/div);
    div*=4;
  }
  cudaMalloc(&devParam,PARAMETERS*sizeof(float));
  cudaMemcpy(devParam,param,PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(&devOut,taille);
  cudaMalloc(&devMatrix,PARAMETERS*PARAMETERS*sizeof(float));
  cudaMalloc(&devInv,PARAMETERS*PARAMETERS*sizeof(float));
  cudaMalloc(&devVec,PARAMETERS*sizeof(float));
  cudaMalloc(&devVecStep,PARAMETERS*sizeof(float));
  cudaMalloc(&devVecOld,PARAMETERS*sizeof(float));
  initCuda();
  if(cudaGetLastError() == cudaErrorMemoryAllocation)
  {cout << "Erreur d'allocation (manque de mémoire graphique ?)" << endl;exit(-1);}
  else if(cudaGetLastError() != cudaSuccess)
  {cout << "Erreur lors de l'allocation." << endl;exit(-1);}

  // ---------- Écriture des fields définis dans fields.cu ----------
  div = 1;
  for(uint i = 0; i < LVL;i++)
  {
    writeFields(devFields[i],WIDTH/div,HEIGHT/div);
    div *= 2;
  }

  // ---------- Lecture du fichier et écriture sur le device ---------
  readFile(iAddr,orig,256);
  cudaMemcpy(devOrig,orig,taille,cudaMemcpyHostToDevice);

  // ---------- [Facultatif] Affichage de l'image fixe ----------
  cout << "Image d'origine" << endl;
  printMat(orig,WIDTH,HEIGHT,256);

  // ---------- Allocation des bindless textures et copie des données ----------
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaArray *cuArray[LVL];
  cudaMallocArray(&cuArray[0], &channelDesc,WIDTH,HEIGHT);
  cudaMemcpyToArray(cuArray[0],0,0,orig,IMG_SIZE*sizeof(float),cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = IMG_SIZE*sizeof(float);
  resDesc.res.linear.devPtr = cuArray[0];

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.addressMode[0] = cudaAddressModeBorder; //cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeBorder; //cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t tex[LVL]={0};
  cudaCreateTextureObject(&tex[0],&resDesc,&texDesc,NULL);

  div = 2;
  for(int i = 1; i < LVL; i++)
  {
    cudaMallocArray(&cuArray[i], &channelDesc,WIDTH/div,HEIGHT/div);
    resDesc.res.linear.sizeInBytes = IMG_SIZE/div/div*sizeof(float);
    resDesc.res.linear.devPtr = cuArray[i];
    genMip(tex[i-1],cuArray[i],WIDTH/div,HEIGHT/div);
    cudaCreateTextureObject(&tex[i],&resDesc,&texDesc,NULL);
    div *= 2;
  }

  // --------- [Facultatif] Ecriture de l'image originale aux différentes échelles -------
  /*
  div = 1;
  for(int i=0; i < LVL;i++)
  {
    deform2D<<<gridsize[i],blocksize[i]>>>(tex[i], devOut, devFields[i], devParam, WIDTH/div, HEIGHT/div);
    cudaDeviceSynchronize();
    cudaMemcpy(orig,devOut,IMG_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    sprintf(oAddr,"out/orig%d.png",i);
    writeFile(oAddr, orig, 0, WIDTH/div, HEIGHT/div);
    div *= 2;
  }
  cout << "OK" << endl;
  */

  // --------- Calcul des matrices G ----------
  gettimeofday(&t1,NULL);
  cudaTextureObject_t texG[LVL][PARAMETERS]={{0}};
  cudaArray *Garray[LVL][PARAMETERS];
  div = 1;
  for(int i = 0; i < LVL; i++)
  {
    gradient<<<gridsize[i],blocksize[i]>>>(tex[i],devGradX,devGradY, WIDTH/div, HEIGHT/div);
    resDesc.res.linear.sizeInBytes = IMG_SIZE/div/div*sizeof(float);
    for(int j = 0; j < PARAMETERS; j++)
    {
      cudaMallocArray(&Garray[i][j], &channelDesc, WIDTH/div, HEIGHT/div);
      makeGArray(Garray[i][j],devFields[i]+j*IMG_SIZE/div/div, devGradX, devGradY, WIDTH/div, HEIGHT/div);
      resDesc.res.linear.devPtr = Garray[i][j];
      cudaCreateTextureObject(&texG[i][j],&resDesc,&texDesc,NULL);
    }
    div *= 2;
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << "Calcul des matrices G: " << timeDiff(t1,t2) << " ms." << endl;

  // --------- [Facultatif] Ecriture des G en .png -----------
/*
  cudaMemcpy(devParam,param,PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);
  div = 1;
  for(int l = 0; l < LVL; l++)
  {
    for(int p = 0; p < PARAMETERS; p++)
    {
      deform2D_b<<<gridsize[l],blocksize[l]>>>(texG[l][p],devOut,devFields[l],devParam,WIDTH/div,HEIGHT/div);
      cudaMemcpy(orig,devOut,IMG_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
      sprintf(oAddr,"out/Gl%d-p%d.png",l,p);
      writeFile(oAddr,orig,128,WIDTH/div,HEIGHT/div);
    }
    div*=2;
  }
*/

  // --------- Allocation et assignation des paramètres de déformation de devDef ----------
  float paramI[PARAMETERS] = {-12,30}; // Commenter la boucle pour tester les réglages sur un même jeu de paramètres
  for(int i = 0; i < PARAMETERS; i++)
  paramI[i] = 80.f*rand()/RAND_MAX-40.f;

  cout << "Paramètres réels: ";
  for(int i = 0; i < PARAMETERS;i++){cout << paramI[i] << ", ";}
  cout << endl;
  cudaMemcpy(devParam, paramI, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);

  // ---------- Calcul de l'image à recaler ----------
  deform2D<<<gridsize[0],blocksize[0]>>>(tex[0], devDef[0],devFields[0],devParam,WIDTH,HEIGHT);

  // ---------- Bruitage de l'image déformée ---------
  for(int i = 0; i < WIDTH*HEIGHT ; i++)
  { 
    orig[i] = (float)rand()/RAND_MAX*10.f-5.f;
  }
  cudaMemcpy(devOut,orig,taille,cudaMemcpyHostToDevice); // Pour ajouter le bruit...
  addVec<<<WIDTH*HEIGHT/1024,1024>>>(devDef[0],devOut); // ..directement sur le device

  // ---------- Pour lire l'image déformée plutôt que la générer -----------
  /*
  readFile("img_d.png",orig, 256);
  cudaMemcpy(devDef[0],orig,IMG_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  */

  // ---------- Rééchantillonage de l'image pour les différents étages ----------
  gettimeofday(&t1, NULL);
  div = 2;
  for(int i = 1; i < LVL; i++)
  {
    resample<<<gridsize[i],blocksize[i]>>>(devDef[i],devDef[i-1],WIDTH/div);
    div *= 2;
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  cout << "Rééchantillonage de l'image déformée: " << timeDiff(t1,t2) << " ms." << endl;

  // ---------- [Facultatif] Affichage de l'image déformée ----------
  cudaMemcpy(orig,devDef[0],IMG_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
  cout << "Image déformée:\n" << endl;
  printMat(orig,WIDTH,HEIGHT,256);

/*
  // ---------- [Facultatif] ecriture en .png des images déformées mippées ----------
  div = 1;
  for(int i = 0; i < LVL; i++)
  {
    cudaMemcpy(orig,devDef[i],IMG_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
    sprintf(oAddr,"out/mip_%d.png",i);
    writeFile(oAddr, orig, 0, WIDTH/div,HEIGHT/div);
    div *= 2;
  }
*/

  // ---------- Calcul de la Hessienne ----------
  gettimeofday(&t1,NULL);
  //makeMatrix<<<1,tailleMat>>>(devMatrix,texG[0]); // On ne peut pas passer un tableau d'array au device !
  makeHessian(devMatrix,texG[0]);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  cout << "Génération de la hessienne: " << timeDiff(t1,t2) << " ms." << endl;

  // ---------- [Facultatif] Affichage de la Hessienne ----------
  //*
  float test[PARAMETERS*PARAMETERS];
  cudaMemcpy(test,devMatrix,PARAMETERS*PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
  cout << "\nHessienne:" << endl;
  printMat(test,PARAMETERS,PARAMETERS);
  //*/

  // ---------- Inversion de la hessienne ----------
  gettimeofday(&t1,NULL);
  invert(devMatrix,devInv);
  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << "Inversion de la hessienne: " << timeDiff(t1,t2) << " ms." << endl;

  // ---------- [Facultatif] Affichage de l'inverse ----------
  //*
  cudaMemcpy(test,devInv,PARAMETERS*PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
  cout << "\nMatrice inversée:" << endl;
  printMat(test,PARAMETERS,PARAMETERS);
  //*/

  // ---------- Écriture des paramètres initiaux ----------
  for(int i = 0; i < PARAMETERS; i++)
  param[i] = 0;
  //readParam(argv,param); // Pour tester des valeurs de paramètres par défaut sans recompiler
  cudaMemcpy(devParam, param, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);

  
  div /= 2; // On se sert de la dernière valeur de div (cela equivaut à div = pow(LVL-1,2) )
  // ---------- La boucle principale ---------
  for(int l = LVL-1; l >= 0; l--) // Boucler sur les étages de la pyramide
  {
    cout << " ###  Niveau n°" << l << " ###\n" << endl;
    cout << " Taille de l'image: " << WIDTH/div << "x" << HEIGHT/div << endl;
    gettimeofday(&t00,NULL);
    res = 10000000000; // -- On remet une valeur hénaurme pour être sûr d'avoir une décroissante à la première itération

    for(int i = 0;i < nbIter; i++) // Itérer sur cet étage (en pratique, on fait rarement toutes les itérations)
    {
      // ---------- Infos -------
      gettimeofday(&t0,NULL);
      cout << "Boucle n°" << i+1 << endl;
      cudaMemcpy(param,devParam,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
      cout << "Paramètres réels: ";
      for(int j = 0; j < PARAMETERS;j++){cout << paramI[j] << ", ";}
      cout << endl;
      cout << "Paramètres calculés: ";
      for(int j = 0; j < PARAMETERS;j++){cout << param[j] << ", ";}
      cout << endl;
      cout << "Différence: ";
      for(int j = 0; j < PARAMETERS;j++){cout << param[j]-paramI[j] << ", ";}
      cout << endl;

      // --------- Interpolation ----------
      gettimeofday(&t1,NULL);
      deform2D<<<gridsize[l],blocksize[l]>>>(tex[l], devOut, devFields[l], devParam,WIDTH/div,HEIGHT/div); //--
      cudaDeviceSynchronize();
      gettimeofday(&t2, NULL);
      cout << "\nInterpolation: " << timeDiff(t1,t2) << "ms." << endl;

/*
      // --------- [Facultatif] Pour enregistrer en .png l'image à chaque itération ----------
      cudaMemcpy(orig,devOut,IMG_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
      sprintf(oAddr,"out/devOut%d-%d.png",LVL-l,i);
      writeFile(oAddr,orig,0,WIDTH/div,HEIGHT/div);
*/

/*
      // --------- [Facultatif] Pour enregistrer en .png la différence de l'image ----------
      float def[WIDTH*HEIGHT];
      if(i == 0)
      {
        cudaMemcpy(def,devDef[l],IMG_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
      }
      cudaMemcpy(orig,devOut,IMG_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
      sprintf(oAddr,"out/diffDevOut%d-%d.png",LVL-l,i);
      writeDiffFile(oAddr,orig,def,4.f,WIDTH/div,HEIGHT/div);
*/

      // ------------ Calcul de la direction de recherche ------------
      gettimeofday(&t1,NULL);
      gradientDescent(texG[l], devOut, devDef[l], devVec, devParam, devFields[l], WIDTH/div, HEIGHT/div); //--
      cudaDeviceSynchronize();
      gettimeofday(&t2,NULL);

      // ----------- Affiche le gradient et le temps de calcul -------------
      cout << "Calcul des gradients des paramètres: " << timeDiff(t1,t2) << " ms." << endl;
      cudaMemcpy(vec,devVec,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
      cout << "Gradient des paramètres:" << endl;
      printMat(vec,PARAMETERS,1);
      
      // ---------- Methode de Newton (la matrice est déjà inversée) -------------
      gettimeofday(&t1,NULL);
      myDot<<<1,PARAMETERS,PARAMETERS*sizeof(float)>>>(devInv,devVec,devVec); //--
      //scalMul<<<1,PARAMETERS>>>(devVec,2.f); // Pour un pas fixe
      cudaDeviceSynchronize();
      gettimeofday(&t2,NULL);

      // ----------- Affiche le vecteur que l'on va ajouter aux paramètres -----------
      cout << "Mise à jour des valeurs: " << timeDiff(t1,t2) << " ms." << endl;
      cudaMemcpy(vec,devVec,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
      cout << "Direction:" << endl;
      printMat(vec,PARAMETERS,1);

      // ------------ Ajouter tant que la fonctionnelle diminue ---------
      c = 0; // --
      gettimeofday(&t1,NULL);
      while(c<10)
      {
        vecCpy<<<1,PARAMETERS>>>(devVecOld,devParam); //--
        scalMul<<<1,PARAMETERS>>>(devVec,1.f+.15f*c); // -- En augmentant sa taille à chaque fois pour accélérer la convergence
        addVec<<<1,PARAMETERS>>>(devParam,devVec); // --
        deform2D<<<gridsize[l],blocksize[l]>>>(tex[l], devOut, devFields[l], devParam,WIDTH/div,HEIGHT/div); //--
        oldres = res; // --
        res = residuals(devOut, devDef[l], IMG_SIZE/div/div)/IMG_SIZE*div*div; //--
        c++; // -- (quelle ironie...)
        cout << "Ajout: " << c << endl;
        cout << "Résidu: "<< res <<  endl << endl;
        if(res >= oldres) // --
        {
          gettimeofday(&t2,NULL);
          cout << res << " >= " << oldres << "! On annule" << endl;
          vecCpy<<<1,PARAMETERS>>>(devParam,devVecOld); // --
          res = oldres; // --
          cout << c << " ajouts successifs: " << timeDiff(t1,t2) << " ms." << endl;
          cout << "Exécution de toute la boucle: " << timeDiff(t0,t2) << " ms." << endl;
          break; // --
        }
      }
      err = cudaGetLastError(); // --
      if(err != cudaSuccess) // --
      {cout << "ERREUR !!\n" << cudaGetErrorName(err) << endl;exit(-1);} // --
      if(c<=1) // --
      {
        cout << "On n'avance plus... Étage suivant !" << endl;
        gettimeofday(&t2,NULL);
        break; // --
      }
    }
    cout << "Exécution de tout l'étage: " << timeDiff(t00,t2) << " ms." << endl;
    div /= 2; // --
  }

  // ---------- Vérification d'erreur éventuelle ----------
  err = cudaGetLastError();
  cout << "Cuda status: " << ((err == 0)?"OK.":"ERREUR !!") << endl;
  cout << err << endl;
  if(err != 0)
  {cout << cudaGetErrorName(err) << endl;}

  // ---------- Libération de ce qui a été alloué avec initCuda ----------
  cleanCuda();

  // ---------- Libération des arrays dans la mémoire du device ----------
  cudaFree(devOrig);
  cudaFree(devGradX);
  cudaFree(devGradY);
  cudaFree(devParam);
  for(uint i = 0; i < LVL; i++)
  {
    cudaFree(devDef[i]);
    cudaFree(devFields[i]);
    //cudaFree(devG[i]);
  }
  cudaFree(devOut);
  cudaFree(devMatrix);
  cudaFree(devInv);
  cudaFree(devVec);
  cudaFree(devVecStep);
  cudaFree(devVecOld);

  // ---------- Libération des arrays de l'hôte ----------
  free(orig);
  return 0;
}
