import numpy as np
from random import randint
from math import tanh
def Relu(x):
    return max(0,x)

def LeakyRelu(x):
    return 0.1*x if x<0 else x

def Tanh(x):
    return tanh(x)


class Layer:
    
    def __init__(self,n_input,n_neurone):
        self.n_input=n_input
        self.n_neurone=n_neurone
        self.biais=np.random.randn(n_neurone)
        self.w=np.array([np.random.randn(n_input) for _ in range(n_neurone)])
        self.x=np.zeros(n_input)
        self.z=np.zeros(n_neurone)
        self.f=np.zeros(n_neurone)
        
    def forward(self,x,f):
        self.x=np.array(x)
        self.z=w.t*x+b
        self.f=f(self.z)
        
        
        
class Neural_Network:
    def 
    