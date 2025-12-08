import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


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

    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs, batch_size, learning_rate, loss_fn):
        train_errors = []
        val_errors = []
        test_errors = []
        n_samples = X_train.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]

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
    
def plot_errors(train_errors, val_errors):
    epochs = range(1, len(train_errors) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_errors, marker='+', label='Train Error')
    plt.plot(epochs, val_errors, marker='+', label='Validation Error')
    plt.title("Évolution des erreurs pendant l'entraînement")
    plt.xlabel("Epoch")
    plt.ylabel("Erreur (MSE)")
    plt.grid(True)
    plt.legend()
    plt.show()
def plot_predictions(nn, X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_mean, Y_std):
    """Affiche prédictions vs vraies valeurs pour train, val et test"""
    
    # Prédictions normalisées
    train_pred_norm = nn.forward(X_train)
    val_pred_norm = nn.forward(X_val)
    test_pred_norm = nn.forward(X_test)
    
    # Dénormalisation
    train_pred = train_pred_norm * Y_std + Y_mean
    train_true = Y_train * Y_std + Y_mean
    
    val_pred = val_pred_norm * Y_std + Y_mean
    val_true = Y_val * Y_std + Y_mean
    
    test_pred = test_pred_norm * Y_std + Y_mean
    test_true = Y_test * Y_std + Y_mean
    
    # Plot
    plt.figure(figsize=(15, 4))
    
    # Train
    plt.subplot(1, 3, 1)
    plt.scatter(train_true.flatten(), train_pred.flatten(), alpha=0.6)
    plt.plot([train_true.min(), train_true.max()], 
             [train_true.min(), train_true.max()], 'r--', lw=2)
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Prédictions")
    plt.title("Train - Prédictions vs Réalité")
    plt.grid(True, alpha=0.3)
    
    # Validation
    plt.subplot(1, 3, 2)
    plt.scatter(val_true.flatten(), val_pred.flatten(), alpha=0.6, color='orange')
    plt.plot([val_true.min(), val_true.max()], 
             [val_true.min(), val_true.max()], 'r--', lw=2)
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Prédictions")
    plt.title("Validation - Prédictions vs Réalité")
    plt.grid(True, alpha=0.3)
    
    # Test
    plt.subplot(1, 3, 3)
    plt.scatter(test_true.flatten(), test_pred.flatten(), alpha=0.6, color='green')
    plt.plot([test_true.min(), test_true.max()], 
             [test_true.min(), test_true.max()], 'r--', lw=2)
    plt.xlabel("Vraies valeurs")
    plt.ylabel("Prédictions")
    plt.title("Test - Prédictions vs Réalité")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
#my example 

data = pd.read_csv("boston.csv")

X = data.drop(columns=['MEDV']).values.T       #on enlève le prix à prédire = Y
Y = data['MEDV'].values.reshape(1, -1)        

# Normalisation des features
X_temp, X_test, Y_temp, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
Y_train = Y_train.T
Y_val = Y_val.T
Y_test = Y_test.T

X_mean = X_train.mean(axis=1, keepdims=True)
X_std = X_train.std(axis=1, keepdims=True)
X_std[X_std == 0] = 1
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
Y_mean = Y_train.mean()
Y_std = Y_train.std()
Y_train = (Y_train - Y_mean) / Y_std
Y_val = (Y_val - Y_mean) / Y_std
Y_test = (Y_test - Y_mean) / Y_std






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
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    epochs=100, batch_size=32, learning_rate=0.01,
    loss_fn=loss_fn
)

plot_errors(train_err, val_err)
plot_predictions(nn, X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_mean, Y_std)