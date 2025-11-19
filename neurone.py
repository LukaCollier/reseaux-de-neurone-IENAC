#Projet réseau de neurones
import numpy as np
from random import randint
from math import tanh
def Relu(x):
    return max(0,x)

def LeakyRelu(x):
    return 0.1*x if x<0 else x

def Tanh(x):
    return tanh(x)
class Neurone:
    '''
    n : |w|
    b : |biais|
    w : liste de poids associée au neurone
    x : entrée
    z = (w)Tx+b

    '''
    def __init__(self,n,b):
        self.n=n
        self.w=np.random.randn(n)
        self.b=b
        self.x=np.zeros(n)
        self.z=0
        self.a=0
    def activation(self):
        self.a =Relu(self.z)
        return self.a
    def zupdate(self,x):
        self.x = np.array(x, dtype=float)
        self.z=np.dot(self.w,self.x)+self.b
    
    def forward(self,x):
        self.zupdate(x)
        return self.activation()
    def bupdate(self,b):
        self.b=b
    def __repr__(self):
        return f" Poids du neurone {self.w} \n Entrée: {self.x} \n biais :{self.b} \n z={self.z}"
   
class Layer:
    def __init__(self,n_inputs,b):
        '''
        n_inputs =|x|
        n = nb neurones sur le layer
        b : tableaux de n biais
        l : paramètre temporaire pour signifier
            la couche représenté dans le réseaux
            plus tard matrice
        '''
        #self.l=l
        self.n_inputs=n_inputs
        self.n_neurone=len(b)
        self.tab=[Neurone(n_inputs,i) for i in b]
        self.f=[]
        
    def forward(self,x):
        self.f=[i.forward(x) for i in self.tab]
        return self.f
    
    def __repr__(self):
        txt = ""
        for i, neurone in enumerate(self.tab):
            txt += f"\nNeurone {i} :\n{neurone}\n"
        return txt
    

class Reseaux_Neurone:
    def __init__(self,y,yf,dy,n_inputs_init):
        '''
        x  : nb layer du réseaux
        y  : nb neurones sur la couche la plus basse
        yf : nb neurones sur la couche finale
        dy : y-n*dy = nb neurones sur la n-ième couche
        n_inputs_init : inputs initiaux
        a : liste de liste de l'activation de chaque couche
        l : liste de Layer 
        '''
        self.x=0
        self.y=y
        self.yf=yf
        self.dy=dy
        #TODO
        #réalisation des Layers
        self.l=[Layer(n_inputs_init,np.random.randn(y))]
        self.a=[self.l[0].f]
        
    def forward(self,x):
        #TODO
        #mis à jour de chaque couche pour faciliter la modification des biais et poids
        pass
            
        