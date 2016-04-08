__global__ void deform2D(float*, float2*, float*);
__global__ void lsq(float*, float*, float*, int);
__global__ void reduce(float*, uint);
__global__ void gradient(float*, float*);
__global__ void makeG(float*, float2*, float*, float*);
__global__ void makeMatrix(float*, float2*, float*, float*);
__global__ void makeMatrix(float*, float*);
__global__ void gdSum(float*, float*, float*, float*);
__global__ void myDot(float*, float*, float*);
__global__ void addVec(float*, float*);
__global__ void ewMul(float*,float*);
float residuals(float*, float*, uint);
void initCuda(float*);
void cleanCuda();
void gradientDescent(float*, float*, float*, float*);
