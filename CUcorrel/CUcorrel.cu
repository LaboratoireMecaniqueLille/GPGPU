#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "CUcorrel.h"
#include "util.h"
#include "img.h"

using namespace std;


int main(int argc, char** argv)
{
  //struct timeval t0, t1, t2, t00; // Pour mesurer les durées d'exécution
  //cudaError_t err; // Pour récupérer les erreurs éventuelles
  size_t taille = IMG_SIZE*sizeof(float); // Taille d'un tableau contenant une image
  size_t taille2 = IMG_SIZE*sizeof(float2); // idem à 2 dimensions (fields)
  int nbIter=10; // Le nombre d'itérations
  char iAddr[10] = "img.png"; // Le nom du fichier à ouvrir
  char iAddr_d[10] = "img_d.png"; // Le nom du fichier déformé à ouvrir
  float orig[IMG_SIZE]; // le tableau contenant l'image sur l'hôte
  //char oAddr[25]; // pour écrire les noms des fichiers de sortie
  float param[PARAMETERS] = {0}; // Stocke les paramètres calculés
  //float res; // Le résidu 
  //float oldres; // Pour stocker le résidu de l'itération précédente et comparer
  //float vec[PARAMETERS]; // Pour stocker sur l'hôte les paramètres calculés
  //int c = 0; // Pour compter les boucles et quitter si on ajoute trop
  uint div = 1; // Pour diviser la taille dans les boucles

  float *devOrig[LVL]; // Image originale
  float *devGradX[LVL]; // Gradient de l'image d'origine par rapport à X
  float *devGradY[LVL]; // .. à Y
  float2 *devFields[LVL]; // Contient les PARAMETERS champs de déplacements élémentaires à la suite dont on cherche l'influence par autant de paramètres
  float *devParam; // Contient la valeur actuelle calculée des paramètres
  float *devDef[LVL]; // Image déformée à recaler
  float *devOut; // L'image interpolée à chaque itération
  float *devMatrix; // La hessienne utilisée pour la méthode de Newton
  float *devInv;  // L'inverse de la Hessienne
  float *devVec; // Vecteur pour stocker les PARAMETERS valeurs du gradient à chaque itération
  float *devVecStep; // Multiplie terme à terme la direction avant le l'ajouter aux paramètres
  float *devVecOld; // Pour stocker le vecteur précédent et le restaurer si nécessaire
  float *devTileDef; // Pour stocker la tuile de l'image déformée à chaque fois

  srand(time(NULL)); // Seed pour générer le bruit avec rand()

  // ---------- Allocation de tous les tableaux du device ---------
  allocTemp();
  div = 1;
  for(int i = 0; i < LVL; i++)
  {
    cudaMalloc(&devFields[i],PARAMETERS*T_WIDTH*T_HEIGHT*sizeof(float)/div);
    cudaMalloc(&devOrig[i],taille/div);
    cudaMalloc(&devDef[i],taille/div);
    cudaMalloc(&devGradX[i],taille/div);
    cudaMalloc(&devGradY[i],taille/div);
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
  cudaMalloc(&devTileDef,T_SIZE*sizeof(float));
  if(cudaGetLastError() == cudaErrorMemoryAllocation)
  {cout << "Erreur d'allocation (manque de mémoire graphique ?)" << endl;exit(-1);}
  else if(cudaGetLastError() != cudaSuccess)
  {cout << "Erreur lors de l'allocation." << endl;exit(-1);}

  // ---------- Écriture des fields définis dans fields.cu ----------
  div = 1;
  for(uint i = 0; i < LVL;i++)
  {
    writeFields(devFields[i],T_WIDTH/div,T_HEIGHT/div);
    div *= 2;
  }

  // ---------- Lecture du fichier et écriture sur le device ---------
  readFile(iAddr,orig,256);
  cudaMemcpy(devOrig[0],orig,taille,cudaMemcpyHostToDevice);
  Image imgOrig[LVL];
  imgOrig[0].init(WIDTH,HEIGHT,devOrig[0]);

  // ---------- Lecture de l'image déformée  -----------
  readFile(iAddr_d,orig, 256);
  cudaMemcpy(devDef[0],orig,IMG_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  Image imgDef[LVL];
  imgDef[0].init(WIDTH,HEIGHT,devDef[0]);

  // ---------- Rééchantillonge et création des tuiles ---------
  Image t_imgOrig[LVL][NTILES*NTILES]; // Création du tableau d'images
  Image t_imgDef[LVL][NTILES*NTILES]; // idem pour l'image déformée
  div = 1;
  for(uint l = 0;l < LVL; l++) // Itérer sur les niveau de la pyramide
  {
    if(l != 0) // Calculer l'image en fonction de l'étage précédent (sauf pour le premier)
    {
      imgOrig[l-1].mip(devOrig[l],WIDTH/div,HEIGHT/div);
      imgOrig[l].init(WIDTH/div,HEIGHT/div,devOrig[l]);
      imgDef[l-1].mip(devDef[l],WIDTH/div,HEIGHT/div);
      imgDef[l].init(WIDTH/div,HEIGHT/div,devDef[l]);
    }
    for(uint x = 0; x < NTILES; x++) // Itérer sur les tuiles pour créer les objets correspondants
    {
      for(uint y = 0; y < NTILES; y++)
      {
        t_imgOrig[l][x+NTILES*y] = imgOrig[l].makeTile(T_WIDTH*x/div,T_HEIGHT*y/div,T_WIDTH/div,T_HEIGHT/div);
        t_imgDef[l][x+NTILES*y] = imgDef[l].makeTile(T_WIDTH*x/div,T_HEIGHT*y/div,T_WIDTH/div,T_HEIGHT/div);
      }
    }
    div *= 2;
  }


  // ---------- Calcul des gradients pour chaque niveau -------------
  Image imgGradX[LVL];
  Image imgGradY[LVL];
  Image t_imgGradX[LVL][NTILES*NTILES];
  Image t_imgGradY[LVL][NTILES*NTILES];
  div = 1;
  for(int l = 0; l < LVL; l++)
  {
    imgOrig[l].computeGradients(devGradX[l],devGradY[l]);
    imgGradX[l].init(WIDTH/div,HEIGHT/div,devGradX[l]);
    imgGradY[l].init(WIDTH/div,HEIGHT/div,devGradY[l]);
    for(uint x = 0; x < NTILES; x++) // Itérer sur les tuiles pour créer les objets correspondants
    {
      for(uint y = 0; y < NTILES; y++)
      {
        t_imgGradX[l][x+NTILES*y] = imgGradX[l].makeTile(T_WIDTH*x/div,T_HEIGHT*y/div,T_WIDTH/div,T_HEIGHT/div);
        t_imgGradY[l][x+NTILES*y] = imgGradY[l].makeTile(T_WIDTH*x/div,T_HEIGHT*y/div,T_WIDTH/div,T_HEIGHT/div);
      }
    }
    div *= 2;
  }

  //t_imgGradX[1][140].writeToFile("out/test.png",.2f,128.f);


  // ---------- Boucle principale ------------
  float2* devField;
  cudaMalloc(&devField,taille2);
  float* devDiff;
  cudaMalloc(&devDiff,taille);
  float mvX = 0;
  float mvY = 0;
  //float2 dir;
  for(int l = LVL-1; l>=0; l--) // On boucle sur les étages
  {
    div /= 2;
    for(uint i=0; i < nbIter; i++)
    {
      mvX=0;
      mvY=0;
      for(uint t = 0; t < NTILES*NTILES; t++)
      {
        makeTranslationField(devField,mvX,mvY,T_WIDTH/div,T_HEIGHT/div);
        t_imgOrig[l][t].interpLinear(devOut,devField,T_SIZE/div/div);
        t_imgDef[l][t].getDiff(devOut,devDiff);
        if(l == 0)
        {
          t_imgDef[l][t].writeToFile("out/def.png");
          Image out(T_WIDTH/div,T_HEIGHT/div,devOut);
          out.writeToFile("out/out.png");
          Image diff(T_WIDTH/div,T_HEIGHT/div,devDiff);
          diff.writeToFile("out/diff.png",.5f,128.f);
          exit(0);
        }
        /*dir = t_imgOrig[l][t].gradientDescent(devOut,t_imgGradX[l][t],t_imgGradY[l][t]);
        cout << dir.x << ", " << dir.y << endl;*/
      }
    }
  }
  cudaFree(devField);

/*
  uint divinit = div/2; // On se sert de la dernière valeur de div (cela equivaut à divinit = pow(LVL-1,2) )

int2 tile = make_int2(9,13);;
for(tile.x = 1; tile.x < NTILES-1; tile.x++) // Double boucle sur les tuiles (Ne parcoure pas les bordures !)
{
  for(tile.y = 1; tile.y < NTILES-1; tile.y++)
  {
  div = divinit;

  // ---------- Écriture des paramètres initiaux ----------
  for(int i = 0; i < PARAMETERS; i++)
  param[i] = 0;
  //readParam(argv,param); // Pour tester des valeurs de paramètres par défaut sans recompiler
  //param[0] = 100;param[1] = -100;
  cudaMemcpy(devParam, param, PARAMETERS*sizeof(float),cudaMemcpyHostToDevice);
    cout << "\n\n### Tile " << tile.x+1 << "/" << NTILES << ", " << tile.y+1 << "/" << NTILES  << " ###\n" << endl;

  // ---------- La boucle principale ---------
  for(int l = LVL-1; l >= 0; l--) // Boucler sur les étages de la pyramide
  {
    cout << " #  Niveau n°" << LVL-l << " #\n" << endl;
    cout << "Taille de l'image: " << WIDTH/div << "x" << HEIGHT/div << endl;
    cout << "Taille de la tuile: " << T_WIDTH/div << "x" << T_HEIGHT/div << endl;
    gettimeofday(&t00,NULL);

    // ------------ Copie dans un tableau de la taille de la tuile -------------
    MemMap map(make_uint2(WIDTH/div,HEIGHT/div),make_uint2(T_WIDTH/div,T_HEIGHT/div),make_uint2(T_WIDTH/div*tile.x,T_HEIGHT/div*tile.y));
    for(int k = 0; k < T_HEIGHT/div; k++)
    {
      cudaMemcpy(devTileDef+k*T_WIDTH/div,devDef[l]+map.tileToImg(0,k),T_WIDTH/div*sizeof(float),cudaMemcpyDeviceToDevice);
    }
    deform2D_t<<<gridsize_t[l],blocksize_t[l]>>>(tex[l], devOut, devFields[l], devParam,div,tile); //--
    res = residuals(devOut, devTileDef, T_SIZE/div/div)/T_SIZE*div*div; //--

    for(int i = 0;i < nbIter; i++) // Itérer sur cet étage (en pratique, on fait rarement toutes les itérations)
    {
      // ---------- Infos -------
      gettimeofday(&t0,NULL);
      cout << "Boucle n°" << i+1 << endl;
      cudaMemcpy(param,devParam,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
      cout << "Paramètres calculés: ";
      for(int j = 0; j < PARAMETERS;j++){cout << param[j] << ", ";}
      cout << endl;

      // --------- Interpolation ----------
      gettimeofday(&t1,NULL);
      deform2D_t<<<gridsize_t[l],blocksize_t[l]>>>(tex[l], devOut, devFields[l], devParam,div,tile); //--
      cudaDeviceSynchronize();
      gettimeofday(&t2, NULL);
      cout << "\nInterpolation: " << timeDiff(t1,t2) << "ms." << endl;

      // ------------ Calcul de la direction de recherche ------------
      gettimeofday(&t1,NULL);
      gradientDescent(texG[l], devOut, devTileDef, devVec, T_WIDTH/div, T_HEIGHT/div, tile); // --
      cudaDeviceSynchronize();
      gettimeofday(&t2,NULL);

      // ----------- [Facultatif] Affiche le gradient et le temps de calcul -------------
      cout << "Calcul des gradients des paramètres: " << timeDiff(t1,t2) << " ms." << endl;
      cudaMemcpy(vec,devVec,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
      cout << "Gradient des paramètres:" << endl;
      printMat(vec,PARAMETERS,1);
      
      // ---------- Methode de Newton (la matrice est déjà inversée) -------------
      gettimeofday(&t1,NULL);
      myDot<<<1,PARAMETERS,PARAMETERS*sizeof(float)>>>(devInv,devVec,devVec); //--
      //scalMul<<<1,PARAMETERS>>>(devVec,2.f); // Pour un pas fixe (peut accélérer !)
      cudaDeviceSynchronize();
      gettimeofday(&t2,NULL);

      // ----------- [Facultatif] Affiche le vecteur que l'on va ajouter aux paramètres -----------
      cout << "Mise à jour des valeurs: " << timeDiff(t1,t2) << " ms." << endl;
      cudaMemcpy(vec,devVec,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
      cout << "Direction:" << endl;
      printMat(vec,PARAMETERS,1);

      // ----------- Si la direction est incohérente, pas la peine d'itérer... ---------
      c = 0; // --
      if(abs(vec[0]) > 20)
      {
        cout << "WTF ?" << endl;
        c = 10;
        break;
      }

      // ------------ Ajouter tant que la fonctionnelle diminue ---------
      gettimeofday(&t1,NULL);
      while(c<10)
      {
        vecCpy<<<1,PARAMETERS>>>(devVecOld,devParam); //--
        scalMul<<<1,PARAMETERS>>>(devVec,1.f+.15f*c); // -- En augmentant sa taille à chaque fois pour accélérer la convergence
        addVec<<<1,PARAMETERS>>>(devParam,devVec); // --
        deform2D_t<<<gridsize_t[l],blocksize_t[l]>>>(tex[l], devOut, devFields[l], devParam,div,tile); //--
        oldres = res; // --
        cudaDeviceSynchronize();
        res = residuals(devOut, devTileDef, T_SIZE/div/div)/T_SIZE*div*div; //--
        cudaDeviceSynchronize();
        c++; // -- (quelle ironie...)
        cout << "Ajout: " << c << endl;
        cout << "Résidu: "<< res <<  endl << endl;



        if(res >= oldres) // --
        {
          cout << res << " >= " << oldres << "! On annule" << endl;
          vecCpy<<<1,PARAMETERS>>>(devParam,devVecOld); // --
          res = oldres; // --
          gettimeofday(&t2,NULL);
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
        cout << "Résidu: " << res << endl;
        cudaMemcpy(param,devParam,PARAMETERS*sizeof(float),cudaMemcpyDeviceToHost);
        cout << "Paramètres calculés: ";
        for(int j = 0; j < PARAMETERS;j++){cout << param[j] << ", ";}
        cout << endl;
        
      // --------- [Facultatif] Pour enregistrer en .png la différence de l'image (à la fin des itérations) ----------
      float def[T_WIDTH*T_HEIGHT];
      cudaMemcpy(orig,devOut,T_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(def,devTileDef,T_SIZE/div/div*sizeof(float),cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      sprintf(oAddr,"out/diffDevOutT%d-%d.png",tile.y,tile.x);
      //writeFile(oAddr,orig,0,T_WIDTH/div,T_HEIGHT/div);
      writeDiffFile(oAddr,orig,def,4.f,T_WIDTH/div,T_HEIGHT/div);
        break; // --
      }
    }
    gettimeofday(&t2,NULL);
    cout << "Exécution de tout l'étage: " << timeDiff(t00,t2) << " ms." << endl;
    div /= 2; // --
  }


  }} // Fin de la double boucle

  // ---------- Vérification d'erreur éventuelle ----------
  err = cudaGetLastError();
  cout << "Cuda status: " << ((err == 0)?"OK.":"ERREUR !!") << endl;
  cout << err << endl;
  if(err != 0)
  {cout << cudaGetErrorName(err) << endl;}
*/
  // ---------- Libération des arrays dans la mémoire du device ----------
  freeTemp();
  cudaFree(devOrig);
  cudaFree(devGradX);
  cudaFree(devGradY);
  cudaFree(devParam);
  for(uint i = 0; i < LVL; i++)
  {
    cudaFree(devDef[i]);
    cudaFree(devFields[i]);
  }
  cudaFree(devOut);
  cudaFree(devMatrix);
  cudaFree(devInv);
  cudaFree(devVec);
  cudaFree(devVecStep);
  cudaFree(devVecOld);

  return 0;
}
