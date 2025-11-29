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
        self.db=0
        self.dw=np.zeros(n)
        self.delta=0
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
    def wupdate(self,nw):
        self.w=nw
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
    def weight_matrix(self):
        return np.array([neu.w for neu in self.tab])
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
        self.delta=None
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
    def MSE(self,y_pred,y):
        return 0.5*np.sum((y_pred-y)**2)
    def backward(self,x,y,lr,dactiv):
        '''
        x : donnée d'entrée étant dans l'apprentissage
        y : résultat attendu possiblement |y|> 1 si plusieur neurones sur la couche la plus haute
        lr : pas de déplacement des poids et biais
        dactiv : dérivé de la fonction d'acitvation
        Principe d'abord calcul delta pour la dernière couche car ne peut être mis dans une boucle récursive puis itéré jusqu'a la première couche
        '''
    y_pred = self.forward(x)
    y = np.array(y, dtype=float)
    L = self.nbl - 1
    for j, neu in enumerate(self.l[L].tab):
        neu.delta = (neu.a - y[j]) * dactiv(neu.z)
    for l in range(L - 1, -1, -1):
        current_layer = self.l[l]
        next_layer = self.l[l + 1]

        for j, neu in enumerate(current_layer.tab):
            s = 0
            for next_neu in next_layer.tab:
                s += next_neu.w[j] * next_neu.delta

            neu.delta = s * dactiv(neu.z)
    for layer in self.l:
        for neu in layer.tab:
            neu.dw = neu.delta * neu.x
            neu.db = neu.delta

            neu.w -= lr * neu.dw
            neu.b -= lr * neu.db

    return y_pred   