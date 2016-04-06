#include <iostream>
#include "util.h"

using namespace std;

double pol(double x, double* coeffs, int deg)
{
  double out = 0;
  for(int i = 0; i < deg; i++)
  {out += coeffs[i]*pow(x,i);}
  return out;
}

void getValues(string s, double* tab, int M)
{
  string sep = ",";
  for(int i=0;i < M; i++)
  {
    tab[i] = atoi(s.substr(0,s.find(sep)).c_str())/16384.;
    s = s.erase(0,s.find(sep)+1);
    //cout << tab[i] << ", ";
  }
  //cout << endl;

}

double timeDiff(struct timeval t1, struct timeval t2) //Retourne la différence en ms entre 2 mesures de temps avec gettimeofday(&t,NULL);
{
  return (t2.tv_sec-t1.tv_sec)*1000+1.*(t2.tv_usec-t1.tv_usec)/1000;
}


void printVec(double* vec,int l)
{
  cout << "-----" << endl;
  for(int i = 0; i < l; i++)
  {
    cout << i << ": " << vec[i] << endl;
  }
  cout << "-----" << endl;
}

void printMat(Matrix M)
{
  for(int i = 0; i < M.height; i++)
  {
    for(int j = 0; j < M.width; j++)
    {
      cout << (j == 0?"":", ") << M.val[i*M.width+j];
    }
    cout << endl;
  }
}

void genMat(double* matrix, double* tab, int M, int N)
{
  /*
    Permet de générer directement la matrice (NxN) du système à résoudre en réalisant la somme pour les moindres carrés et les puissances qui vont avec.
  */

  //double *Mvalues = (double*)malloc((2*N-1)*sizeof(double)); // Pour l'ancienne méthode de remplissage
  double Mval;
  double *buff = (double*)malloc(M*sizeof(double));
  for(int i=0;i<M;i++){buff[i]=1;}
  for(int i=0;i < 2*N-1;i++)  //Parcoure les 2N-i éléments de la matrice (Hankelienne)
  {
    Mval = 0;
    for(int k=0; k < M;k++)  //Multiplie chaque élément par sa valeur précédente (pour accélérer le alcul de la puissance)
    {
      Mval += buff[k];  // Et on somme tous ces éléments pour calculer l'index
      buff[k] *= tab[k];
    }

    //Pour un remplissage direct de la matrice
    if(i < N){for(int j=0; j<=i;j++)    // On parcoure la diagonale inverse (cas où on est avant la grande diagonale)
      {
        //cout << "1) M[" << j << "," << i-j << "](" << (N-1)*j+i << ") = " << Mval << flush;
        matrix[(N-1)*j+i] = Mval;  //M(j,i-j)
        //cout << " Ok." << endl;
      }}
    else{for(int j=0; j<2*N-i-1;j++)  //idem, si on a passé la grande diagonale
      {
        //cout << "2) M[" << i-N+j+1 << "," << N-j-1 << "](" << N*(i+j-N+2)-j-1 << ") =" << Mval << flush;
        matrix[N*(i+j-N+2)-j-1] = Mval; //M(+j+1-N,N-j-1)
        //cout << " Ok." << endl;
      }}
  }
  
}

void genVect(double* vect, double* x, double* y, int M, int N)
{
  /*
  Pour générer le vecteur du membre de droite pour un système
  */
  double *buff = (double*)malloc(M*sizeof(double));
  for(int i=0;i<M;i++){buff[i]=1;}
  for(int i=0;i<N;i++)
  {
    vect[i] = 0;
    for(int j=0;j<M;j++)
    {
      //vect[i] += y[j]*pow(x[j],i); // Appel à pow() -> Peut être accéléré (Ok)
      vect[i] += y[j]*buff[j];
      buff[j] *= x[j];
    }
  }
}
