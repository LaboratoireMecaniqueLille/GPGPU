#coding: utf-8

from Cudafit import cgls
import numpy as np



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



# -------------- Préparation ---------------
nb = 102400 # Nombre de systèmes à résoudres 
N = 8 # Degré - 1
M = 25 # Nombre de points sur lesquels on échantillonne

pt = [1.*i/M for i in range(M)] # On prend M points entre 0 et 1 (par exemple)
coeffs = (np.random.rand(N)-.5) # On crée des coefficients aléatoires
print "Coefficients:",coeffs

data = []
p = Pol(coeffs)  # On crée un polynôme avec les coefficients aléatoires
y = [p(i) for i in pt] # On calcule l'image des points par ce coeff

for i in range(nb):
  data.append(pt+(np.random.rand(M)-.5)*.01)  # On bruite les points en ajoutant des valeurs aléatoires nb fois pour avoir nb systèmes différents
data = np.array(data).T # Les données sont contenues dans un np.array de dimension (M,nb)
print data.shape




# --------------- Résolution ---------------
cfit = cgls.Fit(y=y,data=data,deg=N-1) # On crée un solveur en donnant les nb systèmes (on ne passe qu'un seul membre de droite: il est identique pour tous les systèmes)
cfit.compute() # On exécute le calcul. Le résultat est dans cfit.X (un tableau avec les nb vecteurs solutions à la suite)





# --------------- Vérification -------------
cfit.check(0.01) # Petite fonction qui calcule les polynomes et vérifie si ils collent aux données. Ce test peut être lancé une seule fois sur des données de test mais ne pas l'utiliser en fonctionnement normal: il est BEAUCOUP plus lent que le calcul lui même

# Pour tracer quelques résultats et vérifier visuellement la précision
import matplotlib.pyplot as plt
x = cfit.X
for i in range(3):
  pick = np.random.randint(0,nb) # On choisit 3 points au hasard
  print "Système",pick,":"
  p = Pol(x[pick*N:(pick+1)*N])
  print p
  for j in range(M):
    print "p(",data[j,pick],")=",p(data[j,pick]),", valeur cible:",y[j],", delta:",abs(y[j]-p(data[j,pick]))
  plt.plot(data[:,pick],[p(data[j,pick]) for j in range(M)]) # Le polynome aux abcisses bruitées
  plt.scatter(data[:,pick],y,marker="x") # la fonction que l'on cherche à interpoler
  plt.show()

