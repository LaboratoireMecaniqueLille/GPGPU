#include <string>
void printMat(float*,uint,uint, uint=1);
void printMat2D(float2*,uint,uint, uint=1);
double timeDiff(struct timeval, struct timeval);
void readParam(char**,float*, int=6);
void readFile(char*, float*, float);
void writeFile(char*, float*, float, uint, uint, float=1.f);
void invert(float*,float*);
void writeDiffFile(char*, float*, float*, float, uint, uint);

class MemMap
{
  /*
    Classe pour obtenir l'addresse d'un élément dans un tableau 1D en précisant ses coordonnées dans un sous élément
    dImg: les dimensions de l'image globale
    dTile: Les dimensions du sous-élément
    tOffset: Les coordonées de l'origine du sous élément dans l'image globale.
  */
  public:
  MemMap(uint2, uint2, uint2);
  uint tileToImg(uint, uint);
  uint imgToTile(uint, uint);
  private:
  uint2 dImg, dTile, tOffset;
  uint a,b;
};
