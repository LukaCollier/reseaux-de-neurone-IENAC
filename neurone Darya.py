import numpy as np
import random 


"""fonction d'initialisation du réseau de neurone"""
def initialisation(dimensions):
    parametres={}
    L=len(dimensions)

    for l in range(1,L):
        parametres['w' + str(l)]= np.random.randn(dimensions[l], dimensions[l-1])
        parametres['b' + str(l)]= np.random.randn(dimensions[l], 1)

        return parametres

"""fonctions d'activation à mettre dans la Classe Neurone"""

def relu(x):
    return [max(0, xi) for xi in x]

def relu_derivative(x):
    return [1 if xi > 0 else 0 for xi in x]

def softmax(x):
    exps = [np.exp(xi) for xi in x]
    s = sum(exps)
    return [e / s for e in exps]

"classes / réseau avec séparation (pré/post)-activation "

class Neurone:
    def __init__(self,xsize):
        self.w=np.random.randn(xsize) ##vecteur poids taille entrée
        self.b=np.random.randn()   ##initialise biai du neurone, scalaire
        self.last_input = None   ##stocker les entrées reçues lors de la dernière propagation avant (forward).
                                ##neurone en a besoin pendant le backward pour calculer les gradients
        self.z = None                     # pré-activation
        self.az = None                     # post-activation (sortie)
    
    def linear(self,x):
        self.last_input=x   #stocke entrée neurone
        self.z=np.dot(self.w,x)+self.b ## z=wT.x+b
        return self.z   #retourne sortie avant activation
   
   
    def forward(self,x):
        z = self.linear(x) 
        az = self.activate(z)  #fonction activation appliquée au neurone fa(z)
        return az  ## renvoie sortie neurone

    def activate(self, x):
        self.az= [max(0, xi) for xi in x]
        return self.az  # ReLU

    def activate_deriv(self, x):
        return [1 if xi > 0 else 0 for xi in x] #pour avoir dérivée fonction d'activation
    
    def backward(self, grad_output, lr):
        grad_z= grad_output * self.activate_deriv(self.z)
        grad_input=grad_z*self.w

        dw= grad_z*self.last_input
        db=grad_z

        self.w-=lr*dw  #réajuste cf poly
        self.b-=lr*db
    
        return grad_input
    

class Layer:
    def __init__(self,xsize,zsize):
        self.neurones=[Neurone(xsize) for k in range (zsize)] 
        #crée un neurone avec x entrées pour chaque sortie de la couche 

    def forward(self, x):
        output= np.array([neuron.forward(x) for neuron in self.neurones])
        return output
    ##sortie de la couche, qui servira d’entrée pour la couche suivante dans le réseau.
    
    def backward(self, grad_outputs, lr):
        grad_inputs = np.zeros_like(self.neurones[0].w)
        #crée vecteur nul même taille pour accumuler le gradient total sera transmis à la couche précédente
        #gradient de la perte par rapport à l'entrée de la couche

        for neuron, grad_out in zip(self.neurones, grad_outputs): 
            """grad_outputs: le vecteur de gradients reçu de la couche suivante
            Fait correspondre chaque neurone avec son gradient spécifique.
            zip permet de parcourir simultanément la liste des neurones et les gradients correspondants."""
            grad_inputs += neuron.backward(grad_out, lr)


        return grad_inputs


class Réseauneurone:
    def __init__(self, layer_sizes): 
        """layer_sizes contient nb neurones par couche"""
        self.layers = [] ##va contenir couches du réseau
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x    ##on applique le forward sur toutes les couches
    
    def MSE(self,y_pred,y):
        return 0.5*np.sum((y_pred-y)**2)  ##mettre la fonction de perte par rapport à laquelle corriger
    
    def backward(self, grad_output, lr):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, lr)
        return grad_output
    
    ##renvoie le gradient de la couche d'entrée

