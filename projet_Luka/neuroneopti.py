import numpy as np
import Activation

class Layer:
    '''
    n_input : taille de l'entrée
    n_neurone : nombre de neurone dans le layer
    biais: vecteur de biais avec |biais|=n_neurone
    w: matrice de poids
    x : entrée
    z : résultat de Wx+b
    f : fonction d'activation sur des vecteurs
    '''
    def __init__(self,n_input,n_neurone,activ):
        self.n_input=n_input
        self.n_neurone=n_neurone
        self.biais=np.random.randn(n_neurone)
        self.w = np.random.randn(n_neurone, n_input)
        self.x=np.zeros(n_input)
        self.z=np.zeros(n_neurone)
        self.f=np.zeros(n_neurone)
        self.activ=activ
    def forward(self,x):
        self.x=np.array(x)
        self.z=self.w @ x + self.biais
        self.f=self.activ.function(self.z)
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
        for i,nb_n in enumerate(nb_n_l):
            self.l.append(Layer(n_input,nb_n,activ[i]))
            n_input=nb_n
    def forward(self,x):
        self.a = [np.array(x)] #self.a=[x]
        for neu in self.l:
            x=neu.forward(x)
            self.a.append(x)
        return x
    def MSE(self,y_pred,y):
        return 0.5*np.sum((y_pred-y)**2)
    def backward(self,y,lr):
        '''
        y : résultat attendu possiblement |y|> 1 si plusieurs neurones sur la couche la plus haute
        lr : pas de déplacement des poids et biais
        Principe d'abord calcul delta pour la dernière couche car ne peut être mis dans une boucle récursive puis itéré jusqu'à la première couche
        Memo:
        np.outer : réalise un produit externe de deux vecteur 1D  a,b de taille respective n et m. Renvoie une matrice (n,m) avec
        c_ij=a_i *b_j
        permet de faire l'équivalent du produit d'une matrice (n,1) et d'une matrice (1,m)
        '''
        y_pred=self.a[-1]
        y = np.array(y, dtype=float)
        L = self.nbl - 1
        delta=[None]*(L+1)
        #couche la plus haute
        neu=self.l[-1]
        delt=(y_pred-y)*neu.activ.derivative(neu.z)
        delta[-1]=delt
        for i in range(L-1,-1,-1):
            neu=self.l[i]
            next_neu=self.l[i+1]
            delt=(next_neu.w.T @ delta[i+1])*neu.activ.derivative(neu.z)
            delta[i]=delt
        for (i,neu) in enumerate(self.l):
            neu.w-=lr*np.outer(delta[i],self.a[i]) #on réalise un produit exterieur car delta[i] à pour dimension (h^(i),) et self.a[i] à pour dimension (h^(i-1),)
            neu.biais-=lr*delta[i]
        return y_pred
    



         
