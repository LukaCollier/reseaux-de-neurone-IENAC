#Projet réseau de neurones
import numpy as np
from random import randint
class Neurone:
    def __init__(self,n,b):
        self.n=n
        self.w=np.random.randn(n)
        self.b=b
        self.x=np.zeros(n)
        self.z=0
    def activation(self):
        pass
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
    def __init__(self,l,n_inputs,b):
        '''
        n_inputs =|x|
        n = nb neurones sur le layer
        b : tableaux de n biais
        l : paramètre temporaire pour signifier
            la couche représenté dans le réseaux
            plus tard matrice
        '''
        self.l=l
        self.n_inputs=n_inputs
        self.n_neurone=len(b)
        self.tab=[Neurone(n_inputs,i) for i in b]
        self.f=[]
        
    def forward(self,x):
        self.f=[i.forward(x) for i in self.tab]
        return self.f
    
    def __repr__(self):
        txt = f"Layer {self.l} avec {self.n_neurone} neurones.\n"
        for i, neurone in enumerate(self.tab):
            txt += f"\nNeurone {i} :\n{neurone}\n"
        return txt
        