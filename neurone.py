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
    b : biais
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
    def activation(self,f):
        self.a =f(self.z)
        return self.a
    def zupdate(self,x):
        self.x = np.array(x, dtype=float)
        self.z=np.dot(self.w,self.x)+self.b
    
    def forward(self,x,f):
        self.zupdate(x)
        return self.activation(f)
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
        
    def forward(self,x,f):
        self.f=[i.forward(x,f) for i in self.tab]
        return np.array(self.f)
    
    def __repr__(self):
        txt = ""
        for i, neurone in enumerate(self.tab):
            txt += f"\nNeurone {i} :\n{neurone}\n"
        return txt
    

class Reseaux_Neurone:
    def __init__(self,n_inputs_init,nb_n_l):
        '''
        nbl  : nb layer du réseaux
        y  : nb neurones sur la couche la plus basse
        yf : nb neurones sur la couche finale
        dy : y-n*dy = nb neurones sur la n-ième couche
        n_inputs_init : inputs initiaux
        a : liste de liste de l'activation de chaque couche
        l : liste de Layer
        nb_n_l: liste du nombre de neuronne sur chaque couche
        
        NOTA : y ,yf,dy obsolète mais nécessite la création de nb_n_l avant
        '''
        self.l=[]
        self.a=[]
        self.nbl=len(nb_n_l)
        n_input=n_inputs_init
        for nb_n in nb_n_l:
            biais=np.random.randn(nb_n)
            self.l.append(Layer(n_input,biais))
            n_input=nb_n
        
    def forward(self,x):
        self.a=[]
        entry=x
        for lay in self.l:
            lay.forward(entry,LeakyRelu)
            entry=lay.f
        return entry
        
