# coding: utf-8

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
 

class Fit:
  """
  Permet de calculer rapidement les coefficients de polynômes passant au mieux (selon les moindres carrés) par des points d'ordonnées identiques.
  Prend 3 arguments nommés:
  - data:
    Un np.array de taille (M,nb) où M est le nombre de points et nb le nombre de polynômes.
    Il contient pour chaque polynôme les abcisses des points.
  - y:
    Un vecteur de taille M qui contient les ordonnées. Il est le même pour tous les systèmes
  - deg: le degré des polynômes (valeur par défaut: 5)
  Les 2 premiers sont impératifs.

  Utilisation: placer le dossier Cudafit dans le projet, importer la classe avec "from Cudafit import cgls"
  créer une instance avec les paramètres: cfit = cgls.Fit(data=...,y=...,deg=...)
  Résoudre: cfit.compute()
  Les résultats sont dans cfit.X
  Après appel à la méthode compute(), le membre X contiendra un tableau avec tous les coefficients mis à la suite des polynômes passant au mieux par les points [x00,x01,x02,...,x0M][y1,y2,y3,..,yM] puis [x10,x11,...,x1M][y0,y1,y2,...,yM] etc... jusqu'à [x(nb)0,x(nb)1,...,x(nb)M][y0,y1,y2,...,yM] 

  Par exemple: en faisant data=numpy.array([[0.,1,2,3,4],[0,10,20,30,40]]).T,y=[0.,1,4,9,16],deg=2), on récupère les coefficients [0,0,1,0,0,0.001] car les valeurs correspondent à y = x² et 0.01x²

  Note: ce programme a été conçu pour de grandes valeurs de nb (ordre 1e5~1e6). Il les résoud par groupe de 1024 donc les performances sont optimales pour un nombre de système multiple de 1024. Pour une vitesse optimale, le mieux est d'essayer différentes valeurs jusqu'à être limité par la mémoire graphique. (Avec seulement 1Go, il est possible de monter à plus de 327680 systèmes à la fois, attention cette valeur dépend aussi de l'utilisation de la mémoire graphique par les autres programmes)
  """
  # ----- Configuration de base -----
  def __init__(self, **kwargs):
    self.kernel_file = "Cudafit/"+kwargs.get("kernel_file","kernels.cu")
    self.y = kwargs.get("y",[])
    self.data = kwargs.get("data",[])
    if self.y == [] or self.data == []:
      print "cgls.Fit a besoin au moins des arguments y et data"
      raise ValueError
    self.nb = self.data.shape[1]
    self.M = len(self.y) # On en déduit le nombre de points à interpoler
    self.N = kwargs.get("deg",5)+1 # Nombre d'inconnues dans le polynome (= degré du polynôme + 1)
    if self.M < self.N:
      print "Le nombre d'équation ne peut pas être inférieur au nombre d'inconnues"
      print "Il faut deg < len(y). Ici, len(y)=",self.M,"et deg=",self.N-1
      raise ValueError
    if self.data.shape[1] % 1024 != 0:
      print "Attention: Pour des performances optimales, le nombre de systèmes doit être un multiple de 1024 (ici,",self.data.shape[1],"n'en est pas un)."
      #Attention: le nombre de système doit être un multiple de 1024. Si ce n'est pas le cas on rajoute ce qui manque (avec des 0) et on recoupe le resultat à la fin
      self.nb = self.data.shape[1]
      self.data = np.concatenate((self.data,np.zeros((self.M,1024-self.nb%1024))),axis=1)
    
    self.nbp = self.data.shape[1]
    self.sizeofdouble = 8 # à priori, n'aura pas besoin d'être modifié
  # -------------------------- 

    with open(self.kernel_file,"r") as f: #Ouverture du ficher contenant les kernels
      mod = SourceModule(f.read())  #On utilise le texte lu comme source

    self.MyDot = mod.get_function("myDot") # On charge les kernels définis dans le fichier
    self.VecSum = mod.get_function("vecSum")
    self.VecDiff = mod.get_function("vecDiff")
    self.Norm2 = mod.get_function("norm2")
    self.VecMul = mod.get_function("vecMul")
    self.EwDiv = mod.get_function("ewDiv")
    self.ScalMul = mod.get_function("scalMul")
    self.VecCpy = mod.get_function("vecCpy")
    self.PowArray = mod.get_function("powArray")
    self.SumCol = mod.get_function("sumCol")
    self.HankelFill = mod.get_function("HankelFill")
    self.EwMul = mod.get_function("ewMul")


  def compute(self):
    N = self.N
    M = self.M
    iN = np.int32(N)
    iM = np.int32(M)
    sizeofdouble = self.sizeofdouble
    nbp = self.nbp
    devY = cuda.mem_alloc(M*sizeofdouble) # On alloue sur la carte la mémoire pour stocker le vecteur
    Y = np.array(self.y,np.float64)
    cuda.memcpy_htod(devY,Y) # On copie Y sur le device

    devA = cuda.mem_alloc(N*N*nbp*sizeofdouble) # On crée devA (contiendra toutes les matrices)
    devB = cuda.mem_alloc(N*nbp*sizeofdouble) # devB contiendra tous les membres de droite
    devData = cuda.mem_alloc(M*nbp*sizeofdouble) # Pour stocker et traiter les valeurs des points
    devPow = cuda.mem_alloc(M*(2*N-1)*nbp*sizeofdouble) # Tableau temporaire pour stocker les puissances
    """
Ici, on crée les matrices telles que c(i,j)=sum(x[a]**(i+j),a=0..M) (c(i,j) est le coefficient d'une matrice)
Pour cela on calcule d'abord toutes les puissances jusqu'à 2*N-1 (jusqu'à la valeur du coin inférieur gauche)
On somme les points des colonnes (ceux à la même puissance)
On les place selon des anti diagonales dans les matrices (dite de Hankel car c(i,j) ne dépend que de i+j
"""
    cuda.memcpy_htod(devData,self.data.flatten('F')) # On copie les points dans devData
    self.PowArray(devData,devPow,iM,np.int32(2*N-1),block=(1024,1,1),grid=(M*nbp/1024,1)) # On remplis devPow avec les puissances successives de devData
    self.SumCol(devPow,devData,iM,grid=((2*N-1)*nbp/1024,1),block=(1024,1,1)) # On somme les colonnes pour avoir les éléments de chaque matrice
    self.HankelFill(devData,devA,iN,block=(N,N,8),grid=(nbp/8,1)) # On remplis devA avec ces valeurs
    devPow.free() # On a plus besoin d'aller jusqu'à 2*N-1 donc on libère...


    """
Ici on cherche à remplir les vecteurs du membre de droite. Il sont définis par Ya=sum(Xi**a*yi,i=0..M)
"""
    devPow = cuda.mem_alloc(M*N*nbp*sizeofdouble) # ... et on réalloue jusqu'à N seulement
    cuda.memcpy_htod(devData,self.data.flatten('F')) # On réécrit les données dans devData (qui a servi de tampon précédemment)
    self.PowArray(devData, devPow, iM, iN, block=(1024,1,1), grid=(M*nbp/1024,1)) # On refait les calculs de puissance (nécessaire car l'agencement n'est pas le même: on ne va plus que jsuqu'à N)
    self.EwMul(devPow,devY,iM,block=(1024,1,1),grid=(M*N*nbp/1024,1)) # On Multiplie ces éléments par les valeurs de Y
    self.SumCol(devPow,devB, iM, block=(1024,1,1),grid=(N*nbp/1024,1)) # Et on fait la somme de colonnes

    devData.free() # On libère ce qui ne sert plus
    devPow.free()

    devX = cuda.mem_alloc(N*nbp*sizeofdouble) # Et on alloue ce dont on a besoin pour les gradients conjugués
    devR = cuda.mem_alloc(N*nbp*sizeofdouble)
    devP = cuda.mem_alloc(N*nbp*sizeofdouble)
    devF = cuda.mem_alloc(nbp*sizeofdouble)
    devAp = cuda.mem_alloc(N*nbp*sizeofdouble)
    devAlpha = cuda.mem_alloc(nbp*sizeofdouble)
    devBuff = cuda.mem_alloc(N*nbp*sizeofdouble)
    self.X = np.zeros(N*nbp,np.float64)

    #---- Initialisation -------
    cuda.memcpy_htod(devX,self.X) # On part de [0,0,...,0], peut importe car on converge nécessairement en N itérations
    self.VecCpy(devR,devB,block=(1024,1,1),grid=(N*nbp/1024,1)) # Donc R = B-A*[0] = B
    self.VecCpy(devP,devB,block=(1024,1,1),grid=(N*nbp/1024,1)) # Pareil pour P
    self.Norm2(devR,devF,iN,block=(1024,1,1),grid=(nbp/1024,1)) # On calcule les résidus

    # -------- Itérations ----------
    for i in range(max(4,N-1)): # Par expérience si N est petit, la précision est encore augmentée en faisant des itérations supplémentaires, sinon, N-1 suffisent
      self.MyDot(devA,devP,devAp,iN,iN,block=(N,128,1),grid=(nbp/128,1),shared=128*N*sizeofdouble)#Ap = A.p
      self.VecMul(devP,devAp,devBuff,iN,block=(1024,1,1),grid=(nbp/1024,1)) # Buff = Pt.Ap
      self.EwDiv(devF,devBuff,devAlpha,block=(1024,1,1),grid=(nbp/1024,1)) # alpha = F/Buff
      self.ScalMul(devP,devAlpha,iN,devBuff,block=(1024,1,1),grid=(nbp/1024,1)) # Buff = alpha*p
      self.VecSum(devX,devBuff,block=(1024,1,1),grid=(N*nbp/1024,1)) # X += Buff
      self.ScalMul(devAp,devAlpha,iN,devBuff,block=(1024,1,1),grid=(nbp/1024,1)) # Buff = alpha*Ap
      self.VecDiff(devR,devBuff,block=(1024,1,1),grid=(N*nbp/1024,1)) # r -= Buff
      self.VecCpy(devBuff,devF,block=(1024,1,1),grid=(nbp/1024,1)) # Buff = F
      self.Norm2(devR,devF,iN,block=(1024,1,1),grid=(nbp/1024,1)) # F = ||r||²
      self.EwDiv(devF,devBuff,devBuff,block=(1024,1,1),grid=(nbp/1024,1)) # Buff = F/Buff
      self.ScalMul(devP,devBuff,iN,devP,block=(1024,1,1),grid=(nbp/1024,1)) # p = Buff*p
      self.VecSum(devP,devR,block=(1024,1,1),grid=(N*nbp/1024,1)) # p += r

    cuda.memcpy_dtoh(self.X,devX) # On récupère X après N itérations
    self.X = self.X[:N*self.nb]
    print "Calcul terminé!"


  def check(self, delta = 0.05): # Pour vérifier la cohérence des valeurs obtenues, ne pas exécuter systématiquement: n'est pas optimisé !
    print "Vérification des valeurs... (peut prendre du temps)"
    err = 0
    for p in range(self.nb):
      pol = Pol(self.X[p*self.N:(p+1)*self.N])
      for m in range(self.M):
        if abs(self.y[m]-pol(self.data[m,p])) > delta:
          print "Point",p,", valeur",m
          print "Delta:", self.y[m]-pol(self.data[m,p])
          err += 1
          break
    print "Erreurs de plus de",delta,":",err,"/",self.nb,"(",100.*err/self.nb,"%)"



class Pol(): # Pour représenter simlpement un polynôme
  def __init__(self,coeffs):
    self.coeffs = coeffs
    self.f = lambda x: sum([coeffs[i]*(x**i) for i in range(len(self.coeffs))])

  def __call__(self,x):
    return self.f(x)

  def __repr__(self):
    s = ''
    for i in range(len(self.coeffs)):
      d = "" if i==0 else " + "
      s+=d+str(self.coeffs[i])+"*x^"+str(i)
    return s
