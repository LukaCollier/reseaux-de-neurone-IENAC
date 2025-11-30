import numpy as np


class Layer :

    def __init__(self,input_size, output_size, activation)    :
        self.input_size=input_size
        self.output_size=output_size
        self.activation=activation
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases=np.zeros((output_size, 1))
        self.x=None
        self.z=None
        self.a=None

    def forward(self,x):
        self.z=np.dot(self.weights,x) + self.biases
        self.a=self.activation(self.z)
        self.x=x
        return self.a

    def backward(self,delta_next,learning_rate):
        dz = delta_next * self.activation.derivative(self.z)
        dW = np.dot(dz, self.x.T)
        db=dz
        delta_prev = np.dot(self.weights.T, dz)
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        return delta_prev

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y, learning_rate, loss_fn):
        # delta de la dernière couche
        delta = loss_fn.derivative(self.layers[-1].a, y)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)


class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def __call__(self, x):
        return self.function(x)
            
    @classmethod
    def sigmoid(cls):
        return cls(
            function=lambda x: 1 / (1 + np.exp(-x)),
            derivative=lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))
        )

    @classmethod
    def ReLU(cls):
        return cls(
            function=lambda x: np.maximum(0,x),
            derivative=lambda x: (x > 0).astype(float)
        )
    
    @classmethod
    def Tanh(cls):
        return cls(
            function=lambda x: np.tanh(x),
            derivative= lambda x: 1-np.tanh(x)**2

        )
class ErrorFunction:

    def __init__(self,function,derivative):
        self.function=function
        self.derivative=derivative

    def __call__(self,ypred,ytrue):
        return self.function(ypred,ytrue) 
    
    @classmethod
    def MSE(cls):
        return cls(
            function= lambda  ypred,ytrue:((ypred-ytrue)**2)/2,
            derivative= lambda   ypred,ytrue:ypred-ytrue
        )
    
#mon exemple
tanh = ActivationFunction.Tanh()
layer1=Layer(3,4,tanh)
layer2=Layer(4,3,tanh)    
loss_fn = ErrorFunction.MSE()

NN=NeuralNetwork([layer1,layer2])
x = np.random.randn(3, 1)
y = np.array([[1],[0],[1]])
print("Sortie du reseau avant entrainement :", NN.forward(x))
learning_rate = 0.01
for i in range(100000):
    NN.forward(x)
    NN.backward(y, learning_rate, loss_fn)
print("Sortie du reseau apres entrainement :", NN.forward(x) )