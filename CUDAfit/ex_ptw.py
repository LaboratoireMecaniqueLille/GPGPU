#coding: utf-8

from Cudafit import cgls #cgls = Conjugate gradient least square. C'est la méthode utilisée pour résoudre les systèmes
import numpy as np
import img_calib.GetPTW as ptw
#ptw = __import__("img_calib.GetPTW")

y = [i for i in range(17,31)]
M = len(y)
N=6
folder = "img_calib"
prefix_img = "calib_"
suffix_img = ".ptm"

#----------- Lecture des images------------
img = []
for i in y:   # On importe toutes les images d'un coup dans un tableau img(n,pix)
  tmp = ptw.GetPTW(folder+"/"+prefix_img+str(i)+suffix_img)
  tmp.get_frame(0)
  img.append(1.*tmp.frame_data.flatten())
del(tmp)
img = np.array(img,np.float64)/16384. # On normalise pour avoir des valeurs entre 0 et 1
print "Importation réussie de",M,"images de",img.shape[1],"pixels."

#----------- Découpe en blocs --------------
chunkSize = 102400 # Idéalement, faire des multiples de 1024 les plus grands possibles
tab_img = []
for i in range(img.shape[1]//chunkSize):
  tab_img.append(img[:,i*chunkSize:(i+1)*chunkSize]) # On découpe en blocs: on peut jouer sur la taille des blocs selon la mémoire graphique à disposition

if img.shape[1]%chunkSize != 0: 
  tab_img.append(img[:,(i+1)*chunkSize:]) # On oublie pas le reste (si la taille du bloc n'est pas un multiple de la taille totale)

# tab_img est un tableau contenant des blocs de l'image (ici de 163840 pixels), tab_img[i] est un np.array de taille (M,163840) 

x=[]
l = 0
#-------------- Résolution -------------
for im in tab_img: # On parcourt les blocs
  l+=1
  print "Boucle",l,":",im.shape[1],"systèmes à résoudre."
  cfit = cgls.Fit(y=y,data=im,deg=N-1) # On initialise avec le vecteur de droite (les températures de la calibration)  et les valeurs qui correspondent aux points des abcisses (niveaux mesurés aux températures fixées)
  cfit.compute() # On résout
  cfit.check() # Utile seulement pour vérifier que le calcul s'est bien passé (prend du temps!)
  x.extend(cfit.X) # On récupère les valeurs

print len(x)/N,"==",img.shape[1],": On a bien récupéré tous les coefficients" if len(x) == N*img.shape[1] else "Problème !"
print x[:3*len(y)] # On affiche les premières valeurs
