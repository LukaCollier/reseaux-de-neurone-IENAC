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





### exemple/ modèle à améliorer/modifier pour tracé graphe d'erreurs 
def train_regression(reseau, X, y, epochs=100, batch_size=32, lr=0.001, validation_split=0.2):
    # séparer apprentissage / validation
    """X = les entrées du réseau (features)
        y = la valeur à prédire (target)"""
    
    n = len(X)
    n_val = int(n * validation_split)  ##combien d’échantillons iront dans la validation.

    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    loss_history = []  ##va stocker la valeur de la loss sur le jeu d’entraînement à chaque epoch.
    val_loss_history = [] ##Même principe, pour le jeu de validation.

    for epoch in range(epochs):

        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices] ##On réordonne les échantillons d’après la permutation indices.
        y_train = y_train[indices] 
        ##mélange X et y de manière cohérente pour que chaque entrée corresponde toujours à sa cible.

        # Boucle batchs
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]  ##les entrées du batch
            yb = y_train[i:i+batch_size]  ##les sorties correspondantes

            for x_i, y_i in zip(xb, yb): 
                ##parcourir simultanément chaque entrée x_i et sa cible y_i dans le batch.
                y_pred = reseau.forward(x_i) 
                ##résultat final y_pred = prédiction du réseau pour cet échantillon

                # Gradient MSE
                grad = (y_pred - y_i)

                reseau.backward(grad, lr)

        # Perte globale entraînement
        preds_train = np.array([reseau.forward(x) for x in X_train])
        loss_train = np.mean(0.5 * (preds_train - y_train)**2)
        loss_history.append(loss_train)

        # Perte validation
        preds_val = np.array([reseau.forward(x) for x in X_val])
        loss_val = np.mean(0.5 * (preds_val - y_val)**2)
        val_loss_history.append(loss_val)

        print(f"Epoch {epoch+1}/{epochs} - loss={loss_train:.4f} - val_loss={loss_val:.4f}")

    return loss_history, val_loss_history



loss, val_loss = train_regression(reseau, X, y, 100, 32, 0.001,0.2)

plt.plot(range(1, len(loss)+1), loss, label="Apprentissage")
plt.plot(range(1, len(val_loss)+1), val_loss, label="Validation")
plt.axhline(y=min(val_loss), color="k", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

print("Meilleur epoch :", np.argmin(val_loss) + 1)

