#include <string>
void printMat(float*,uint,uint, uint=1);
void printMat2D(float2*,uint,uint, uint=1);
double timeDiff(struct timeval, struct timeval);
float GPUsum(float*, unsigned int);
void readParam(char**,float*, int=6);
void readFile(char*, float*, float);
void writeFile(char*, float*, float, uint, uint);
void invert(float*,float*);
