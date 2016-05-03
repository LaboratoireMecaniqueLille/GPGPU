#include <string>
void printMat(float*,uint,uint, uint=1);
void printMat2D(float2*,uint,uint, uint=1);
double timeDiff(struct timeval, struct timeval);
void readParam(char**,float*, int=6);
void readFile(char*, float*, float);
void writeFile(char*, float*, float, uint, uint);
void invert(float*,float*);
void writeDiffFile(char*, float*, float*, float, uint, uint);

class MemMap
{
  public:
  MemMap(uint2, uint2, uint2);
  uint tileToImg(uint, uint);
  uint imgToTile(uint, uint);
  private:
  uint2 dImg, dTile, tOffset;
  uint a,b;
};
