import numpy as np
#cls permet d'écrire cls que ActivationF
#ajout d'Étienne : nouvel attribut name étant un str contenant le nom de la fonction utile pour la désérialisation
class ActivationF:
    """
    Classe des fonctions d'activation
    """
    def __init__(self, function, derivative,name): 
        self.function = function
        self.derivative = derivative
        self.name=name #ajout d'Étienne : name = str contenant le nom de la fonction (utile pour la désérialisation)
        
    @classmethod
    def creation_with_name(cls,name): #ajout d'Étienne
        if name=="sigmoid":
            return cls.sigmoid()
        if name=="relu":
            return cls.relu()
        if name=="leaky_relu":
            return cls.leaky_relu()
        if name=="tanh":
            return cls.tanh()
        
    def __call__(self, x):
        """permet l'appel de l'object comme fonction"""
        return self.function(x)
    
    @classmethod
    def sigmoid(cls):
        def sig(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def sig_deriv(x):
            s = sig(x)
            return s * (1 - s)
        
        return cls(function=sig, derivative=sig_deriv,name="sigmoid") #ajout d'Étienne pour name 
    
    @classmethod
    def relu(cls):
        return cls(
            function=lambda x: np.maximum(0, x),
            derivative=lambda x: (x > 0).astype(float),
            name="relu" #ajout d'Étienne
        )
    
    @classmethod
    def leaky_relu(cls, alpha=0.1):
        return cls(
            function=lambda x: np.where(x > 0, x, alpha * x),
            derivative=lambda x: np.where(x > 0, 1.0, alpha),
            name="leaky_relu" #ajout d'Étienne 
        )
    
    @classmethod
    def tanh(cls):
        def tanh_func(x):
            return np.tanh(x)
        
        def tanh_deriv(x):
            return 1 - np.tanh(x) ** 2
        
        return cls(function=tanh_func, derivative=tanh_deriv,name="tanh") #ajout d'Étienne pour name 

    @classmethod
    def identity(cls):
        def identity_func(x):
            return x
        
        def identity_deriv(x):
            return np.ones_like(x)
        
        return cls(function=identity_func, derivative=identity_deriv,name="identity") #ajout de Luka Etienne tu l'as oublié
    
    @classmethod
    def softmax(cls):
        def softmax_func(x):
            x_shifted = x - np.max(x, axis=0, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)

        def softmax_deriv(x):
            s = softmax_func(x)
            return s * (1 - s)  # dérivée approximative (diagonale uniquement)

        return cls(function=softmax_func, derivative=softmax_deriv, name="softmax")
