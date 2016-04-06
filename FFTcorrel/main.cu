#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <sys/time.h>

#define IMG_X 2048
#define TILE_X 16
#define GRID_X (IMG_X / TILE_X)

using namespace std;
typedef double2 Complex;

double timeDiff(struct timeval, struct timeval);
void cufftCheckError(int);
void cudaCheckError(int);
__global__ void conjugate_data(Complex *, int);
__global__ void ewCpxMul(Complex *, Complex *, int);
__global__ void ewCpxConjMul(Complex *, Complex *, int);
int maxTile(double *);

int main(int argc, char **argv)
{
  cufftHandle planD2Z, planZ2D;
  
  struct timeval t0,t1,t2;
  
  srand(time(NULL));
  cout << "Génération des données aléatoires: image de " << IMG_X << " par " << IMG_X << "..." << flush;
  gettimeofday(&t0,NULL);
  double *data = (double *)malloc(IMG_X*IMG_X*sizeof(double));
  for(int i = 0; i < IMG_X*IMG_X; i++)
  {
    data[i] = (float)rand()/RAND_MAX;
  }
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t0,t2) << " ms)." << endl;
  cout << "Allocation de la mémoire de l'hôte et du device..." << flush;
  gettimeofday(&t1,NULL);
  double **tile = (double **)malloc(GRID_X*GRID_X*sizeof(double*));   //Tableau des "tuiles" (ou imagettes) correspondant au découpage de l'image
  double **otile = (double **)malloc(GRID_X*GRID_X*sizeof(double*));  // La version que l'on compare (ici décalée de 1 pixel)
  
  double *devTile; // La version du device
  double *devOtile;

  Complex *devFFT1;
  Complex *devFFT2;

  cudaMalloc(&devTile,IMG_X*IMG_X*sizeof(double));
  cudaMalloc(&devOtile,IMG_X*IMG_X*sizeof(double));
  cudaMalloc(&devFFT1,IMG_X*IMG_X*sizeof(Complex));
  cudaMalloc(&devFFT2,IMG_X*IMG_X*sizeof(Complex));
  
  for(int i = 0; i < GRID_X*GRID_X; i++)
  {
    tile[i] = (double *)malloc(TILE_X*TILE_X*sizeof(double));
    otile[i] = (double *)malloc(TILE_X*TILE_X*sizeof(double));
  }
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  cout << "Divison en " << GRID_X*GRID_X << " tuiles de " << TILE_X << " de côté..." << flush;
  gettimeofday(&t1,NULL);
  for(int i = 0; i < GRID_X; i++)
  {
    for(int j = 0; j < GRID_X; j++)
    {
      for(int k = 0; k < TILE_X*TILE_X; k++)
      {
        tile[i+GRID_X*j][k] = data[i*TILE_X+k%TILE_X+IMG_X*(j*TILE_X+k/TILE_X)];
        //otile[i+GRID_X*j][k] = tile[i+GRID_X*j][k];
        //if(k+TILE_X+1 < TILE_X*TILE_X)
        //{otile[i+GRID_X*j][k] = tile[i+GRID_X*j][k+TILE_X+1];}
        if(k-TILE_X-1 >= 0)     //Opère le décalage de otile par rapport à tile
        {otile[i+GRID_X*j][k-TILE_X-1] = tile[i+GRID_X*j][k];}
        else
        {otile[i+GRID_X*j][k] = rand()*1./RAND_MAX;}  //Si on est hors de l'image, rajouter du bruit
      }
    }
  }
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  

/*
  for(int i = 0; i < IMG_X;i++) // Pour vérifier
  {
    for(int j = 0; i < IMG_X; i++)
    {
      if(data[i+IMG_X*j] != tile[i/TILE_X+j/TILE_X*GRID_X][i-TILE_X*(i/TILE_X)+(j-TILE_X*(j/TILE_X))*TILE_X])
      {cout << "FAIL!" <<endl;}
    }
  }
*/
  cout << "Copie sur le device..." << flush;
  gettimeofday(&t1,NULL);
  for(int i = 0; i < GRID_X*GRID_X;i++)
  {
    cudaMemcpy(devTile+TILE_X*TILE_X*i,tile[i],TILE_X*TILE_X*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(devOtile+TILE_X*TILE_X*i,otile[i],TILE_X*TILE_X*sizeof(double),cudaMemcpyHostToDevice);
  }
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  
  cout << "Création des plans de calcul de la fft..." << flush;
  gettimeofday(&t1,NULL);
  int n[2] = {TILE_X,TILE_X};
  //cufftCheckError(cufftPlan2d(&planD2Z, TILE_X, TILE_X, CUFFT_D2Z));
  //cufftCheckError(cufftPlan2d(&planZ2D, TILE_X, TILE_X, CUFFT_Z2D));
  cufftCheckError(cufftPlanMany(&planD2Z, 2, n, NULL, 1, IMG_X*IMG_X, NULL, 1, IMG_X*IMG_X, CUFFT_D2Z,GRID_X*GRID_X));
  cufftCheckError(cufftPlanMany(&planZ2D, 2, n, NULL, 1, IMG_X*IMG_X, NULL, 1, IMG_X*IMG_X, CUFFT_Z2D,GRID_X*GRID_X));
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  cout << "Calcul de la corrélation des tuiles..." << flush;
  gettimeofday(&t1,NULL);


  //struct timeval ta, tb;
  //gettimeofday(&ta,NULL);
  cufftCheckError(cufftExecD2Z(planD2Z,devTile,devFFT1));
  cufftCheckError(cufftExecD2Z(planD2Z,devOtile,devFFT2));
  //cudaDeviceSynchronize();
  //gettimeofday(&tb,NULL);
  //cout << "--2 fw fft: " << timeDiff(ta,tb) << endl;
  /*conjugate_data<<<(GRID_X*GRID_X+1023)/1024,1024>>>(devFFT2[i],TILE_X*TILE_X);
  ewCpxMul<<<(GRID_X*GRID_X+1023)/1024,1024>>>(devFFT1[i],devFFT2[i],TILE_X*TILE_X);*/
  //gettimeofday(&ta,NULL);
  ewCpxConjMul<<<(IMG_X*IMG_X+1023)/1024,1024>>>(devFFT1,devFFT2,IMG_X*IMG_X);
  //cudaDeviceSynchronize();
  //gettimeofday(&tb,NULL);
  //cout << "--Conj cpxMul: " << timeDiff(ta,tb) << endl;
  //gettimeofday(&ta,NULL);
  cufftCheckError(cufftExecZ2D(planZ2D,devFFT1,devTile));
  //cudaDeviceSynchronize();
  //gettimeofday(&tb,NULL);
  //cout << "--1 bw fft: " << timeDiff(ta,tb) << endl;
  //gettimeofday(&tb,NULL);

  cudaDeviceSynchronize();
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  
  cout << "Copie sur l'hôte..." << flush;
  gettimeofday(&t1,NULL);
  for(int i = 0; i < GRID_X*GRID_X; i++) // Copie les tuiles
  {
    cudaCheckError(cudaMemcpy(tile[i],devTile+i*TILE_X*TILE_X,TILE_X*TILE_X*sizeof(double),cudaMemcpyDeviceToHost));
  }
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  

  /*for(int i = 0; i < TILE_X; i++) // Affiche la tuile 0
  {
    for(int j = 0; j < TILE_X; j++)
    {
      cout << int(tile[0][i+j*TILE_X]) << ",";
    }
    cout << endl;
  }*/
  cout << "Vérification des points maxi..." << flush;
  gettimeofday(&t1,NULL);
  int loopMax;
  for(int i = 0; i < GRID_X*GRID_X;i++)
  {
    loopMax = maxTile(tile[i]);
    if(loopMax != TILE_X+1)
    {cout << "Tuile " << i << ": Max au point (" << loopMax%TILE_X << ", " << loopMax/TILE_X << ")." << endl;}
  }
  gettimeofday(&t2,NULL);
  cout << " Ok (" << timeDiff(t1,t2) << " ms)." << endl;
  cout << "Total: " << timeDiff(t0,t2) << " ms." << endl;
  
  return 0;
}

double timeDiff(struct timeval t1, struct timeval t2) //Retourne la différence en ms entre 2 mesures de temps avec gettimeofday(&t,NULL);
{
  return (t2.tv_sec-t1.tv_sec)*1000+1.*(t2.tv_usec-t1.tv_usec)/1000;
}

void cufftCheckError(int val)
{
  if(val == CUFFT_SUCCESS){return;}
  switch(val)
  {
    case CUFFT_ALLOC_FAILED:
      cout << "Erreur CUFFT_ALLOC_FAILED" << endl;
      exit(-1);
    case CUFFT_INVALID_PLAN:
      cout << "Erreur CUFFT_INALID_PLAN" << endl;
      exit(-1);
    case CUFFT_INVALID_VALUE:
      cout << "Erreur CUFFT_INVALID_VALUE" << endl;
      exit(-1);
    case CUFFT_INTERNAL_ERROR:
      cout << "Erreur CUFFT_INTERNAL_ERROR" << endl;
      exit(-1);
    case CUFFT_SETUP_FAILED:
      cout << "Erreur CUFFTSETUP_FAILED" << endl;
      exit(-1);
    case CUFFT_INVALID_SIZE:
      cout << "Erreur CUFFT_INVALID_SIZE" << endl;
      exit(-1);
    default:
      cout << "Erreur inconnue:\n" << val << endl;
      exit(-1);
  }
}

void cudaCheckError(int val)
{
  if(val == CUDA_SUCCESS){return;}
  cout << "Erreur n°" << val << endl;
  exit(-1);
}

__global__ void conjugate_data(Complex *data, int l)
{
  int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id >= l){return;}
  data[id].y = -data[id].y;
}

__global__ void ewCpxMul(Complex* v1, Complex* v2, int l)
{
  /*
    Réalise un produit éléments par éléments de deux vecteurs complexes.
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id >=l){return;}
  double buff = v1[id].x*v2[id].x-v1[id].y*v2[id].y;
  v1[id].y = v1[id].x*v2[id].y+v1[id].y*v2[id].x;
  v1[id].x = buff;
  
}

__global__ void ewCpxConjMul(Complex* v1, Complex* v2, int l)
{
  /*
    Réalise un produit éléments par éléments de v1 par le conjugué de v2 (résultat dans v1).
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id >=l){return;}
  double buff = v1[id].x*v2[id].x+v1[id].y*v2[id].y;
  v1[id].y = v1[id].y*v2[id].x-v1[id].x*v2[id].y;
  v1[id].x = buff;
  
}

int maxTile(double* tile)
{
  int x = 0;
  for(int i = 0; i < TILE_X*TILE_X; i++)
  {
    if(tile[i] > tile[x])
    {x = i;}
  }
  return x;
}
