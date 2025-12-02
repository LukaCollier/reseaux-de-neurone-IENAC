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
    
    def forward(self, x):
        # x est de shape: (n_input,) ou (n_input, batch_size)
        self.x = np.array(x)
        
        # Si x est (n_input,)
        if self.x.ndim == 1:
            self.x = self.x.reshape(1, -1)
        
        # z = W @ x + b (broadcast le biais)
        self.z = self.w @ self.x + self.biais.reshape(-1, 1)
        self.f = self.activ.function(self.z)
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
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.a = [x]
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
        '''
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        y_pred=self.a[-1]
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
            # Gradient: delta[i] est de shape (n_neurons, batch_size) self.a[i] est de shape (n_inputs, batch_size)
            # Produit matriciel: (n_neurons, batch_size) @ (batch_size, n_inputs) = (n_neurons, n_inputs)
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1) #permet d'éviter les problèmes et de perdre 1H X)
            
            neu.w -= lr * grad_w
            neu.biais -= lr * grad_b
        return y_pred