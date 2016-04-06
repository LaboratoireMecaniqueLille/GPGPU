__global__ void myDot(double* A, double* b, double* out, int w, int h)
{
  /*
  Pour realiser des produits matrice-vecteur de "petites" (N*M< 1024) matrices le plus efficacement possible <<<dimGrid,dimBlock>>>: Dimblock de type Dim3 (x,y) ou x=h est la dimension du vecteur de sortie et y le nombre de matrice par bloc (x*y < 1024)
  dimGrid: nombre de blocs
  Ex:
   On appelle le kernel avec <<<n,(h,k), k*w*sizeof(double)>>>(A,b,o,w,h): 
   A est de dimension n*k*w*h( n*k matrices de taille w*h, en donnant les lignes successives)
   b est de dimension n*k*w ( n*k vecteurs de longueur w)
   o est de longueur n*k*h ( contiendra les n*k vecteurs de taille h)

   Realise n*k produits matrice-vecteur avec n*k matrices de taille w*h et autant de vecteurs de longueur w

  ATTENTION: il faut que h >= w !!!
  TODO: Pour se passer de cette limitation, il faut repartir la tache de copie entre les threads
  (boucle de longueur w/h pour attribuer la valeur de sh_b a la place de if(x<w))
  */
  int x = threadIdx.x; //Composante du vecteur (ou ligne de la matrice)
  int y = threadIdx.y; //Numero du couple matrice-vecteur du bloc
  int offset = blockIdx.x*blockDim.y+y; //Decalage du au bloc et au numero du vecteur
  double val = 0;
  extern __shared__ double sh_b[]; // La memoire partagees contenant les vecteurs du bloc
  if(x < w)  // On les copie si on est dans le vecteur: c'est pour ca qu'il faut que h>=w
  {sh_b[w*y+x] = b[w*offset+x];} // On les place dans la memoire partagees DU BLOC
  __syncthreads(); // Primordial: evite les "race condition": qu'un thread accede aux donnees avant qu'elles soient ecrites
  for(int i = 0; i < w; i++)
  {
    val += A[h*w*offset+x*w+i]*sh_b[w*y+i]; // On somme les produits sur la ligne
  }
  out[h*offset+x] = val; // On ecrit le resultat dans le vecteur sortie
}

__global__ void vecSum(double* v1, double* v2)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  v1[id] += v2[id];
}

__global__ void vecDiff(double* v1, double* v2)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  v1[id] -= v2[id];
}


__global__ void norm2(double* v, double* norm, int N)
{
  /*
    Calcule les normes au carre de k*N vecteurs, les ecrit dans norm
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double val = 0;
  for(int i = 0; i < N; i++)
  {
    val += v[N*id+i]*v[N*id+i];
  }
  norm[id] = val;
}

__global__ void vecMul(double* v1, double* v2, double* out, int N)
{
  /*
  Calcule v1.T*v2 pour des vecteurs de taille N, l'ecrit dans out
  ex:
    vecMul<<<4,1024>>>(v1,v2,o,5);
    v1 et v2: taille 4*1024*5 (soit 4096 vecteurs de taille 5)
    o: taille 4096
    o[i] vaut le produit de v1+N*i par v2+N*i
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double val = 0;
  for(int i = 0; i < N; i++)
  {
    val += v1[N*id+i]*v2[N*id+i];
  }
  out[id] = val;
}

__global__ void ewDiv(double* v1, double* v2, double* out = NULL)
{
  /*
    Realise une division elements par elements de deux vecteurs.
    Si un 3eme parametre est specifie, le resultat sera ecrit dedans, sinon il remplacera v1
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(out == NULL)
  {v1[id] /= v2[id];}
  else
  {out[id] = v1[id]/v2[id];}
}

__global__ void scalMul(double* v, double* k, int N, double* out = NULL)
{
  /*
    Multiplie les vecteurs de taille N par des scalaires. v est de taille a*N, k sont les scalaires (tableau de taille a) si out est specifie, le resultat sera ecrit dedans, sinon dans v.
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double coeff = k[id];
  if(out == NULL)
  {
    for(int i = 0; i < N; i++)
    {
      v[N*id+i] *= coeff;
    }
  }
  else
  {
    for(int i = 0; i < N; i++)
    {
      out[id*N+i] = v[N*id+i] * coeff;
    }
  }
}

__global__ void vecCpy(double* v1, double* v2)
{
  /*Copie v2 dans v1*/
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  v1[id] = v2[id];
}


__global__ void powArray(double* data, double* mat, int M, int N)
{
  /*
  Calcule les piussances de 0 a N des M elements de data et les ecrits (row major) dans mat (a appeler avec <<<k,m>>> ou M = k*m)
  */
  double buff = 1;
  int id = (blockIdx.x * blockDim.x + threadIdx.x)%M;
  int offset = (blockIdx.x * blockDim.x + threadIdx.x)/M;
  double d = data[offset*M+id];
  for(int i = 0;i < N-1; i++)
  {
    mat[N*M*offset+id+M*i] = buff;
    buff *= d;
  }
  mat[N*M*offset+id+M*(N-1)] = buff;
  //mat[blockIdx.x * blockDim.x+threadIdx.x] = N; //(blockIdx.x * blockDim.x+threadIdx.x-1)*100 ;
}

__global__ void sumCol(double* mat, double* tab, int M)
{
  /*
  Ecrit dans tab(taille k*n) les somme des colonnes de mat (taille (k*n)xM) (a appeler avec <<<k,n>>>)
  */
  double val = 0;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 0; i < M; i++)
  {
    val += mat[i+id*M];
  }
  tab[id] = val;
}

__global__ void HankelFill(double* val, double* mat, int N)
{
  /*
  Remplis k*p matrices Hankeliennes (i+j=cte -> Ci,j = cte) de taille N*N a partir du tableau de leurs k*p*2*N-1 valeurs (de 0,0 jusqu'a N,N)
  Utiliser avec <<<k,(N,N,p)>>>
  */
  //mat[threadIdx.y * N + threadIdx.x] = val[threadIdx.x+threadIdx.y];
  int offset = threadIdx.z + blockIdx.x*blockDim.z;

  mat[offset*N*N + threadIdx.y*N+threadIdx.x] = val[offset*(2*N-1) + threadIdx.x+threadIdx.y];
}

__global__ void ewMul(double* v1, double* v2, int l)
{
  /*
    Realise un produit elements par elements de deux vecteurs.
    Peut boucler sur v2: preciser sa longueur l
  */
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  v1[id] *= v2[id%l];
}
