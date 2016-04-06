#include <string>

using namespace std;

double pol(double, double*,int);
void getValues(std::string, double*, int);
double timeDiff(struct timeval, struct timeval);
void printVec(double*,int);
void genMat(double*, double*, int, int);
void genVect(double*, double*, double*, int, int);

typedef struct Matrix Matrix;
struct Matrix
{
  int height;
  int width;
  double* val;
};
void printMat(Matrix);
