import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir(r"c:\Users\camil\Documents\NN")
print("Répertoire courant :", os.getcwd())

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

    def forward(self,x):# x shape : (input_size, batch_size)
        self.z=np.dot(self.weights,x) + self.biases
        self.a=self.activation(self.z)
        self.x=x
        return self.a

    def backward(self,delta_next,learning_rate):
        batch_size = delta_next.shape[1]
        dz = delta_next * self.activation.derivative(self.z)
        dW = np.dot(dz, self.x.T)/batch_size
        db = np.mean(dz, axis=1, keepdims=True)
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

    def train(self, X, Y, epochs, batch_size, learning_rate, loss_fn,X_train,X_val,X_test,Y_train,Y_val,Y_test):
        train_errors = []
        val_errors = []
        test_errors = []
        n_samples = X.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(Y_batch, learning_rate, loss_fn)

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)
            test_pred = self.forward(X_test)

            train_error = np.mean(loss_fn(train_pred, Y_train))
            val_error = np.mean(loss_fn(val_pred, Y_val))
            test_error = np.mean(loss_fn(test_pred, Y_test))

            train_errors.append(train_error)
            val_errors.append(val_error)
            test_errors.append(test_error)

        print("Training complete.")        
        return train_errors, val_errors, test_errors


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
    
def plot_errors(train_errors, val_errors, test_errors):
    epochs = range(1, len(train_errors) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_errors, marker='o', label='Train Error')
    plt.plot(epochs, val_errors, marker='s', label='Validation Error')
    plt.plot(epochs, test_errors, marker='^', label='Test Error')
    plt.title("Évolution des erreurs pendant l'entraînement")
    plt.xlabel("Epoch")
    plt.ylabel("Erreur (MSE)")
    plt.grid(True)
    plt.legend()
    plt.show()

#my example 

data = pd.read_csv("boston.csv")

X = data.drop(columns=['MEDV']).values.T       #on enlève le prix à prédire = Y
Y = data['MEDV'].values.reshape(1, -1)        

# Normalisation des features
X = X / X.max(axis=1, keepdims=True)

# 2️ Séparer train / validation / test
from sklearn.model_selection import train_test_split

# Séparer train+val / test
X_temp, X_test, Y_temp, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)

# Séparer train / validation
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)  # 0.25*0.8=0.2

# Transposer pour correspondre au format attendu par NN
X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
Y_train = Y_train.T
Y_val = Y_val.T
Y_test = Y_test.T


identity = ActivationFunction(
    function=lambda x: x,
    derivative=lambda x: np.ones_like(x)
)


nn = NeuralNetwork([
    Layer(input_size=X_train.shape[0], output_size=10, activation=ActivationFunction.ReLU()),
    Layer(input_size=10, output_size=1, activation=identity)
])


loss_fn = ErrorFunction.MSE()

# Entraînement
train_err, val_err, test_err = nn.train(
    X_train, Y_train, epochs=100, batch_size=16, learning_rate=0.01,
    loss_fn=loss_fn,
    X_train=X_train, X_val=X_val, X_test=X_test,
    Y_train=Y_train, Y_val=Y_val, Y_test=Y_test
)

plot_errors(train_err, val_err, test_err)