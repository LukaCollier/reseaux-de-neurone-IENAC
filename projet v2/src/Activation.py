import numpy as np

class ActivationF:
    """
    Activation function class for neural networks.
    
    This class encapsulates activation functions along with their derivatives,
    providing a unified interface for various activation functions commonly
    used in neural networks.
    
    Attributes:
        function (callable): The activation function.
        derivative (callable): The derivative of the activation function.
        name (str): The name of the activation function.
    """
    
    def __init__(self, function, derivative, name): 
        """
        Initialize an activation function.
        
        Args:
            function (callable): The activation function to apply.
            derivative (callable): The derivative of the activation function.
            name (str): The name of the activation function.
        """
        self.function = function
        self.derivative = derivative
        self.name = name
        
    @classmethod
    def creation_with_name(cls, name):
        """
        Create an activation function instance by name.
        
        Args:
            name (str): Name of the activation function. Valid options are:
                       "sigmoid", "relu", "leaky_relu", "tanh", "identity", "softmax".
        
        Returns:
            ActivationF: An instance of the requested activation function.
            
        Raises:
            ValueError: If the activation function name is not recognized.
        """
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
        raise ValueError(f"Unknown activation function: {name}")
        
    def __call__(self, x):
        """
        Allow the object to be called as a function.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output after applying the activation function.
        """
        return self.function(x)
    
    @classmethod
    def sigmoid(cls):
        """
        Create a sigmoid activation function.
        
        The sigmoid function maps input values to the range (0, 1).
        Formula: σ(x) = 1 / (1 + e^(-x))
        
        Returns:
            ActivationF: Sigmoid activation function instance.
        """
        def sig(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def sig_deriv(x):
            s = sig(x)
            return s * (1 - s)
        
        return cls(function=sig, derivative=sig_deriv, name="sigmoid")
    
    @classmethod
    def relu(cls):
        """
        Create a ReLU (Rectified Linear Unit) activation function.
        
        ReLU outputs the input directly if positive, otherwise outputs zero.
        Formula: f(x) = max(0, x)
        
        Returns:
            ActivationF: ReLU activation function instance.
        """
        return cls(
            function=lambda x: np.maximum(0, x),
            derivative=lambda x: (x > 0).astype(float),
            name="relu"
        )
    
    @classmethod
    def leaky_relu(cls, alpha=0.1):
        """
        Create a Leaky ReLU activation function.
        
        Leaky ReLU allows a small gradient when the input is negative.
        Formula: f(x) = x if x > 0, else alpha * x
        
        Args:
            alpha (float, optional): Slope for negative values. Defaults to 0.1.
            
        Returns:
            ActivationF: Leaky ReLU activation function instance.
        """
        return cls(
            function=lambda x: np.where(x > 0, x, alpha * x),
            derivative=lambda x: np.where(x > 0, 1.0, alpha),
            name="leaky_relu"
        )
    
    @classmethod
    def tanh(cls):
        """
        Create a hyperbolic tangent (tanh) activation function.
        
        Tanh maps input values to the range (-1, 1).
        Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Returns:
            ActivationF: Tanh activation function instance.
        """
        def tanh_func(x):
            return np.tanh(x)
        
        def tanh_deriv(x):
            return 1 - np.tanh(x) ** 2
        
        return cls(function=tanh_func, derivative=tanh_deriv, name="tanh")

    @classmethod
    def identity(cls):
        """
        Create an identity activation function.
        
        The identity function returns the input unchanged.
        Formula: f(x) = x
        
        Returns:
            ActivationF: Identity activation function instance.
        """
        def identity_func(x):
            return x
        
        def identity_deriv(x):
            return np.ones_like(x)
        
        return cls(function=identity_func, derivative=identity_deriv, name="identity")
    
    @classmethod
    def softmax(cls):
        """
        Create a softmax activation function.
        
        Softmax converts a vector of values into a probability distribution.
        Commonly used in the output layer for multi-class classification.
        Formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
        
        Returns:
            ActivationF: Softmax activation function instance.
            
        Note:
            The derivative provided is an approximation (diagonal only).
            In practice, it's never used for the last layer with cross-entropy
            loss because we use the shortcut: delta = y_pred - y.
        """
        def softmax_func(x):
            x_shifted = x - np.max(x, axis=0, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)

        def softmax_deriv(x):
            """
            Compute the derivative of softmax (diagonal approximation).
            
            Note: This derivative is approximate (diagonal only).
            In practice, it's never used for the last layer with cross-entropy
            because we use the shortcut delta = y_pred - y.
            
            Args:
                x (np.ndarray): Input array.
                
            Returns:
                np.ndarray: Approximated derivative.
            """
            s = softmax_func(x)
            return s * (1 - s)

        return cls(function=softmax_func, derivative=softmax_deriv, name="softmax")