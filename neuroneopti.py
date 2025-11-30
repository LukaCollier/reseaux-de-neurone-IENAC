import numpy as np
from random import randint
from math import tanh
def Relu(x):
    return max(0,x)

def LeakyRelu(x):
    return 0.1*x if x<0 else x

def Tanh(x):
    return tanh(x)
def dTanh(x):
    return 1 - tanh(x)**2

class Layer:
    '''
    n_input : taille de l'entrée
    n_neurone : nombre de neurone dans le layer
    biais: vecteur de bais avec |biais|=n_neurone
    w: matrice de poids
    x : entrée
    z : résultat de Wx+b
    f : fonction d'activation sur des vecteur
    '''
    def __init__(self,n_input,n_neurone,activ):
        self.n_input=n_input
        self.n_neurone=n_neurone
        self.biais=np.random.randn(n_neurone)
        self.w=np.array([np.random.randn(n_input) for _ in range(n_neurone)])
        self.x=np.zeros(n_input)
        self.z=np.zeros(n_neurone)
        self.f=np.zeros(n_neurone)
        self.activ=np.vectorize(activ) #np.vectorize tf f scalaire -> f vecteur equivalent a map en ocaml
    def forward(self,x):
        self.x=np.array(x)
        self.z=self.w @ x + self.biais
        self.f=self.activ(self.z)
        return self.f
        
        
        
class Neural_Network:
    '''
    l:tableau de layer
    a:tableau des activation avec a[0] qui est l'entrée
    nbl:nombre de layer
    '''
    def __init__(self,n_input_init,nb_n_l,activ):
        self.l=[]
        self.a=[]
        self.nbl=len(nb_n_l)
        n_input=n_input_init
        for nb_n in nb_n_l:
            self.l.append(Layer(n_input,nb_n,activ))
            n_input=nb_n
    def forward(self,x):
        self.a=[x]
        for neu in self.l:
            x=neu.forward(x)
            self.a.append(x)
        return x
    def MSE(self,y_pred,y):
        return 0.5*np.sum((y_pred-y)**2)
    def backward(self,x,y,lr,dactiv):
        '''
        x : donnée d'entrée étant dans l'apprentissage
        y : résultat attendu possiblement |y|> 1 si plusieur neurones sur la couche la plus haute
        lr : pas de déplacement des poids et biais
        dactiv : dérivé de la fonction d'acitvation
        Principe d'abord calcul delta pour la dernière couche car ne peut être mis dans une boucle récursive puis itéré jusqu'a la première couche
        Memo:
        np.outer : réalise un produit externe de deux vecteur 1D  a,b de taille respective n et m. Renvoie une matrice (n,m) avec
        c_ij=a_i *b_j
        permet de faire l'équivalent du produit d'une matrice (n,1) et d'une matrice (1,m)
        '''
        vect_dactiv=np.vectorize(dactiv)
        y_pred = self.forward(x)
        y = np.array(y, dtype=float)
        L = self.nbl - 1
        delta=[None]*(L+1)
        #couche la plus haute
        neu=self.l[-1]
        delt=(y_pred-y)*vect_dactiv(neu.z)
        delta[-1]=delt
        for i in range(L-1,-1,-1):
            neu=self.l[i]
            next_neu=self.l[i+1]
            delt=(next_neu.w.T @ delta[i+1])*vect_dactiv(neu.z)
            delta[i]=delt
        for (i,neu) in enumerate(self.l):
            neu.w-=lr*np.outer(delta[i],self.a[i]) #on réalise un produit exterieur car delta[i] à pour dimension (h^(i),) et self.a[i] à pour dimension (h^(i-1),)
            neu.biais-=lr*delta[i]
        return y_pred
    


         