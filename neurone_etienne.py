import numpy as np


class Layer:
    def __init__(self,vecteur_biais,nb_entrees,f_activation):
        nb_neurones=len(vecteur_biais)
        self.vecteur_biais=np.transpose(np.array(vecteur_biais))
        self.mat_poids=np.random.standard_normal((nb_neurones,nb_entrees))
        self.f_activation=f_activation
        
    def calcul_vecteur_z(self,vecteur_x):
        vecteur_x=np.transpose(np.array(vecteur_x))
        self.vecteur_z=np.transpose(self.mat_poids)@vecteur_x+self.vecteur_biais
    
    def forward(self):
        self.vecteur_a=self.f_activation(self.vecteur_z)
        return self.vecteur_a
    
    
class Reseau_Neurones:
    def __init__(self,liste_taille_couches,liste_f_activation,liste_derivee_f_activation,f_erreur,eta):
        self.eta=eta
        tmp=liste_taille_couches.copy()
        tmp.append(1)
        self.f_erreur=f_erreur
        self.liste_derivee_f_activation=liste_derivee_f_activation
        self.liste_couches=[Layer(np.random.standard_normal(taille),tmp[i-1],liste_f_activation[i]) for i,taille in enumerate(liste_taille_couches)]
    def forward(self,x):
        tmp=x
        for couche in self.liste_couches:
            couche.calcul_vecteur_z(tmp)
            tmp=couche.forward()
        return tmp
    
    def backward(self,x,y):
        eta=self.eta
        L=len(self.liste_couches)
        y_pred=self.forward(x)
        derivee_f_activation_L=self.liste_derivee_f_activation[L-1]
        couche_L=self.liste_couches[L-1]
        delta_L=(couche_L.vecteur_a-y)*derivee_f_activation_L(couche_L.vecteur_z)
        liste_delta=[delta_L]
        for l in range(L-2,-1,-1):
            couche_l=self.liste_couches[l]
            couche_l_plus_1=self.liste_couches[l+1]
            derivee_f_activation_l=self.liste_derivee_f_activation[l]
            delta_l=(np.transpose(couche_l_plus_1.mat_poids)@liste_delta[L-2-l])*derivee_f_activation_l(couche_l.vecteur_z)
            liste_delta.append(delta_l)
        liste_deltas=[liste_delta[L-1-i] for i in range(L)]
        vecteur_x=np.array(x)
        liste_gradient_poids=[liste_deltas[0]@np.transpose(vecteur_x)]
        liste_gradient_biais=[np.transpose(liste_deltas[0])]
        couche_1=self.liste_couches[0]
        couche_1.mat_poids=couche_1.mat_poids-eta*liste_gradient_poids[0]
        couche_1.vecteur_biais=couche_1.vecteur_biais-eta*liste_gradient_biais[0]
        for l in range(1,L-1):
            couche_l=self.liste_couches[l]
            couche_l_moins_1=self.liste_couches[l-1]
            liste_gradient_poids.append(liste_deltas[l]@np.transpose(couche_l_moins_1.vecteur_a))
            liste_gradient_biais.append(np.transpose(liste_deltas[l]))
            couche_l.mat_poids=couche_l.mat_poids-eta*liste_gradient_poids[l]
            couche_l.vecteur_biais=couche_l.vecteur_biais-eta*liste_gradient_biais[l]


def mse(y_chapeau,y):
    y_chapeau=np.array(y_chapeau)
    y=np.array(y)
    return 0.5*(y_chapeau-y)@(y_chapeau-y)

def g(x):
    return 1-np.tanh(x)**2


def main():
    liste_taille_couches=[1,16,16,16,16,1]
    liste_f_activation=[np.tanh,np.tanh,np.tanh,np.tanh,np.tanh,np.tanh]
    liste_derivee_f_activation=[g,g,g,g,g,g]
    eta=5e-2
    reseau=Reseau_Neurones(liste_taille_couches,liste_f_activation,liste_derivee_f_activation,mse,eta)
    epoch=2000
    x=np.random.uniform(low=0,high=np.pi*2,size=500)
    y=np.sin(x)
    for i in range(epoch):
        reseau.backward(x,y)
main()