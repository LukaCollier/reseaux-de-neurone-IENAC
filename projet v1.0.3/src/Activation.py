import numpy as np

class ActivationF:
    """
    Classe des fonctions d'activation
    """
    def __init__(self, function, derivative, name): 
        self.function = function
        self.derivative = derivative
        self.name = name
        
    @classmethod
    def creation_with_name(cls, name):
        """CORRECTION : Gestion complète de tous les cas avec exception"""
        mapping = {
            "sigmoid": cls.sigmoid,
            "relu": cls.relu,
            "leaky_relu": cls.leaky_relu,
            "tanh": cls.tanh,
            "identity": cls.identity,
            "softmax": cls.softmax
        }
        if name in mapping:
            return mapping[name]()
        raise ValueError(f"Fonction d'activation inconnue: {name}")
        
    def __call__(self, x):
        """Permet l'appel de l'object comme fonction"""
        return self.function(x)
    
    @classmethod
    def sigmoid(cls):
        def sig(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def sig_deriv(x):
            s = sig(x)
            return s * (1 - s)
        
        return cls(function=sig, derivative=sig_deriv, name="sigmoid")
    
    @classmethod
    def relu(cls):
        return cls(
            function=lambda x: np.maximum(0, x),
            derivative=lambda x: (x > 0).astype(float),
            name="relu"
        )
    
    @classmethod
    def leaky_relu(cls, alpha=0.1):
        return cls(
            function=lambda x: np.where(x > 0, x, alpha * x),
            derivative=lambda x: np.where(x > 0, 1.0, alpha),
            name="leaky_relu"
        )
    
    @classmethod
    def tanh(cls):
        def tanh_func(x):
            return np.tanh(x)
        
        def tanh_deriv(x):
            return 1 - np.tanh(x) ** 2
        
        return cls(function=tanh_func, derivative=tanh_deriv, name="tanh")

    @classmethod
    def identity(cls):
        def identity_func(x):
            return x
        
        def identity_deriv(x):
            return np.ones_like(x)
        
        return cls(function=identity_func, derivative=identity_deriv, name="identity")
    
    @classmethod
    def softmax(cls):
        def softmax_func(x):
            x_shifted = x - np.max(x, axis=0, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)

        def softmax_deriv(x):
            """
            Note: Cette dérivée est approximative (diagonale uniquement).
            En pratique, elle n'est jamais utilisée pour la dernière couche
            avec cross-entropy car on utilise le raccourci delta = y_pred - y.
            """
            s = softmax_func(x)
            return s * (1 - s)

        return cls(function=softmax_func, derivative=softmax_deriv, name="softmax")