#coding: utf-8

import numpy as np
from Cudafit import cgls

data = np.array([[0,1,2,3,4],[0,10,20,30,40]],np.float64).T # Ne pas oublier de transposer si nécessaire: l'input est de dimension (M,nb) (ici (5,2)) et non (nb,M)
y = np.array([0,1,4,9,16],np.float64)

"""
 On a p1(x) = x² et p2(x) = .01*x², en effet:
0² = 0
1² = 1
2² = 4
3² = 9
4² = 16

et

.01*0² = 0
.01*10² = 1
.01*20² = 4
.01*30² = 9
.01*40² = 16

donc P1(x) = 0+0x+1x² et P2(x) = 0+0x+0.01x²
On attend donc le résultat: [0,0,1,0,0,.01] (avec deg = 2)
nb: le membre de droite doit être le même pour tous les sytèmes à résoudre
"""

cfit = cgls.Fit(data=data,y=y,deg=2)
cfit.compute()
cfit.check(0.001)
print cfit.X
print "Ok !" if np.allclose(cfit.X,np.array([0.,0,1,0,0,.01])) else "Erreur ! Cuda est il correctement installé ?" # Si le résultat est incorrect, il peut y avoir plusieurs raisons: CUDA n'est pas installé, le périphérique ne supporte pas la double précision (nécessite une compute capability d'au moins 1.3), le format des entrées est-il correct (float64) ?
