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
  char oAddr[25]; // pour écrire les noms des fichiers de sortie
  float2 param = make_float2(0,0); // Stocke les paramètres calculés
  //float res; // Le résidu 
  //float oldres; // Pour stocker le résidu de l'itération précédente et comparer
  //int c = 0; // Pour compter les boucles et quitter si on ajoute trop
  uint div = 1; // Pour diviser la taille dans les boucles

  float *devOrig[LVL]; // Image originale
  float *devGradX[LVL]; // Gradient de l'image d'origine par rapport à X
  float *devGradY[LVL]; // .. à Y
  float2 *devParam; // Contient la valeur actuelle calculée des paramètres
  float *devDef[LVL]; // Image déformée à recaler
  float *devOut; // L'image interpolée à chaque itération
  float *devMatrix; // La hessienne utilisée pour la méthode de Newton
  float *devInv;  // L'inverse de la Hessienne
  float *devVec; // Vecteur pour stocker les 2 valeurs du gradient à chaque itération
  float *devVecStep; // Multiplie terme à terme la direction avant le l'ajouter aux paramètres
  float *devVecOld; // Pour stocker le vecteur précédent et le restaurer si nécessaire
  float *devTileDef; // Pour stocker la tuile de l'image déformée à chaque fois

  srand(time(NULL)); // Seed pour générer le bruit avec rand()

  // ---------- Allocation de tous les tableaux du device ---------
  allocTemp();
  div = 1;
  for(int i = 0; i < LVL; i++)
  {
    cudaMalloc(&devOrig[i],taille/div);
    cudaMalloc(&devDef[i],taille/div);
    cudaMalloc(&devGradX[i],taille/div);
    cudaMalloc(&devGradY[i],taille/div);
    div*=4;
  }
  cudaMalloc(&devParam,sizeof(float2));
  cudaMemcpy(devParam,&param,sizeof(float2),cudaMemcpyHostToDevice);
  cudaMalloc(&devOut,taille);
  cudaMalloc(&devMatrix,4*sizeof(float)); // Matrice 2x2
  cudaMalloc(&devInv,4*sizeof(float));
  cudaMalloc(&devVec,sizeof(float2));
  cudaMalloc(&devVecStep,sizeof(float2));
  cudaMalloc(&devVecOld,sizeof(float2));
  cudaMalloc(&devTileDef,T_SIZE*sizeof(float));
  if(cudaGetLastError() == cudaErrorMemoryAllocation)
  {cout << "Erreur d'allocation (manque de mémoire graphique ?)" << endl;exit(-1);}
  else if(cudaGetLastError() != cudaSuccess)
  {cout << "Erreur lors de l'allocation." << endl;exit(-1);}

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

  // ---------- Boucle principale ------------
  float2* devField;
  cudaMalloc(&devField,taille2);
  float* devDiff;
  cudaMalloc(&devDiff,taille);
  float2 dir;
  float2 vect[NTILES*NTILES] = {make_float2(0,0)};
  float res;
  for(int l = LVL-1; l>=0; l--) // On boucle sur les étages
  {
    DEBUG("\n\n##### Niveau " << l << " #####");
    div /= 2;
    for(uint t = 0; t < NTILES*NTILES; t++)
    {
      DEBUG("\nTuile " << t/NTILES << ", " << t%NTILES);
      for(uint i=0; i < nbIter; i++)
      {
        makeTranslationField(devField,vect[t],T_WIDTH/div,T_HEIGHT/div);
        t_imgOrig[l][t].interpLinear(devOut,devField,T_SIZE/div/div);
        t_imgDef[l][t].getDiff(devOut,devDiff);
        dir = gradientDescent(devDiff,t_imgGradX[l][t],t_imgGradY[l][t]);
          if(i==9)
          {
            sprintf(oAddr,"out/L%dt%db.png",LVL-l,t);
            Image diff(T_WIDTH/div,T_HEIGHT/div,devDiff);
            diff.writeToFile(oAddr,1.f,128.f);
          }
          if(i==0)
          {
            t_imgOrig[l][t].writeToFile("out/orig.png");
            t_imgDef[l][t].writeToFile("out/def.png");
            sprintf(oAddr,"out/L%dt%da.png",LVL-l,t);
            Image diff(T_WIDTH/div,T_HEIGHT/div,devDiff);
            diff.writeToFile(oAddr,1.f,128.f);
          }
        res = squareSum(devDiff,T_WIDTH*T_HEIGHT/div/div);
        vect[t] += dir;


          DEBUG("\nItération " << i);
          DEBUG("Direction: " << str(dir));
          DEBUG("Vect: " << str(vect[t]));
          DEBUG("Résidu: " << res);

      }
    }
  }
  cudaFree(devField);


  for(uint i = 0; i < NTILES; i++)
  {
    for(uint j = 0; j < NTILES; j++)
    {
      cout << str(vect[i*NTILES+j]) << " ;  ";
    }
    cout << endl;
  }


  // ---------- Libération des arrays dans la mémoire du device ----------
  freeTemp();
  cudaFree(devParam);
  for(uint i = 0; i < LVL; i++)
  {
    cudaFree(devDef[i]);
    cudaFree(devOrig[i]);
    cudaFree(devGradX[i]);
    cudaFree(devGradY[i]);
  }
  cudaFree(devOut);
  cudaFree(devMatrix);
  cudaFree(devInv);
  cudaFree(devVec);
  cudaFree(devVecStep);
  cudaFree(devVecOld);

  return 0;
}
