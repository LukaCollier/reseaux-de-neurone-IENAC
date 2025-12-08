import numpy as np
#cls permet d'écrire cls que ActivationF
class ActivationF:
    """
    Classe des fonction d'activation
    """
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
    
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
        
        return cls(function=sig, derivative=sig_deriv)
    
    @classmethod
    def relu(cls):
        return cls(
            function=lambda x: np.maximum(0, x),
            derivative=lambda x: (x > 0).astype(float)
        )
    
    @classmethod
    def leaky_relu(cls, alpha=0.1):
        return cls(
            function=lambda x: np.where(x > 0, x, alpha * x),
            derivative=lambda x: np.where(x > 0, 1.0, alpha)
        )
    
    @classmethod
    def tanh(cls):
        def tanh_func(x):
            return np.tanh(x)
        
        def tanh_deriv(x):
            return 1 - np.tanh(x) ** 2
        
        return cls(function=tanh_func, derivative=tanh_deriv)
    
    @classmethod
    def identity(cls):
        def identity_func(x):
            return x
        
        def identity_deriv(x):
            return np.ones_like(x)
        
        return cls(function=identity_func, derivative=identity_deriv)