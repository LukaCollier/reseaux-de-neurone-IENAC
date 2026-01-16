import numpy as np

class RegularisationF:
    
    def __init__(self,function,name,lambda_regularisation):
        self.function=function
        self.name=name
        self.lambda_regularisation=lambda_regularisation
    
    @classmethod
    def creation_with_name(cls, name, lambda_regularisation):
        """CORRECTION : Gestion complète de tous les cas avec exception"""
        mapping = {
            "L0": cls.L0,
            "L1": cls.L1,
            "L2": cls.L2
        }
        if name in mapping:
            return mapping[name](lambda_regularisation=lambda_regularisation)
        raise ValueError(f"Fonction d'activation inconnue: {name}, veuillez choisir parmi {[name for name in mapping]}")
    
    #different regularisation functions
    @classmethod
    def L0(cls,lambda_regularisation):
        def apply_l0(w,grad_w):
            return grad_w
        return cls(function=apply_l0,name='L0',lambda_regularisation=0)
    
    @classmethod
    def L1(cls,lambda_regularisation):
        def apply_l1(w, grad_w):
        #Apply L1 regularization to the weights only

            if lambda_regularisation > 0.0:
                grad_w = grad_w + lambda_regularisation * np.abs(w)
            return grad_w
    
        return cls(function=apply_l1,name='L1',lambda_regularisation=lambda_regularisation)
    
    
    @classmethod
    def L2(cls,lambda_regularisation):
        def apply_l2(w,grad_w):
        #Apply L1 regularization to the weights only

            if lambda_regularisation > 0.0:
                grad_w = grad_w + lambda_regularisation * w**2
            return grad_w
    
        return cls(function=apply_l2,name='L2',lambda_regularisation=lambda_regularisation)
    