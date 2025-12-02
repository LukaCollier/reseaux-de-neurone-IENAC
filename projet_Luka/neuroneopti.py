import numpy as np
import Activation

class Layer:
    '''
    n_input : taille de l'entrée
    n_neurone : nombre de neurone dans le layer
    biais: vecteur de biais avec |biais|=n_neurone
    w: matrice de poids
    x : entrée
    z : résultat de Wx+b
    f : fonction d'activation sur des vecteurs
    '''

    
    def __init__(self,n_input,n_neurone,activ):
        self.n_input=n_input
        self.n_neurone=n_neurone
        self.biais=np.random.randn(n_neurone)
        self.w = np.random.randn(n_neurone, n_input)
        self.x=np.zeros(n_input)
        self.z=np.zeros(n_neurone)
        self.f=np.zeros(n_neurone)
        self.activ=activ



    def forward(self, x):
        # x est de shape: (n_input,) ou (n_input, batch_size)
        self.x = np.array(x)
        
        # Si x est (n_input,)
        if self.x.ndim == 1:
            self.x = self.x.reshape(1, -1)
        
        # z = W @ x + b (broadcast le biais)
        self.z = self.w @ self.x + self.biais.reshape(-1, 1)
        self.f = self.activ.function(self.z)
        return self.f



    def update(self,lr,g_w,g_b):
        self.w -= lr*g_w
        self.biais -= lr*g_b
        
        
        
class Neural_Network:
    '''
    l:tableau de layer
    a:tableau des activation avec a[0] qui est l'entrée
    nbl:nombre de layer
    activ: tableau des fonctions d'activations correspondant à chaque layer (à changer plus tard pour faire plus propre)
    ''' 


    def __init__(self,n_input_init,nb_n_l,activ):
        '''
        Mémo isinstance vérifie que activ est bien du même type que l'objet
        '''
        if isinstance(activ, Activation.ActivationF): #Permet d'éviter de devoir répéter à chaque fois l'activation pour un réseaux avec une unique activation
            activ = [activ] * len(nb_n_l)
        else:
            assert len(activ) == len(nb_n_l)         #Vérifie qu'il y a suffisament d'activation que de layers
        self.l=[]
        self.a=[]
        self.nbl=len(nb_n_l)
        n_input=n_input_init
        self.train_losses = []
        for i,nb_n in enumerate(nb_n_l):
            self.l.append(Layer(n_input,nb_n,activ[i]))
            n_input=nb_n



    def forward(self,x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.a = [x]
        for neu in self.l:
            x=neu.forward(x)
            self.a.append(x)
        return x



    def MSE(self,y_pred,y):
        return 0.5*np.sum((y_pred-y)**2)



    def backward(self,y,lr):
        '''
        y : résultat attendu possiblement |y|> 1 si plusieurs neurones sur la couche la plus haute
        lr : pas de déplacement des poids et biais
        Principe d'abord calcul delta pour la dernière couche car ne peut être mis dans une boucle récursive puis itéré jusqu'à la première couche
        '''
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        y_pred=self.a[-1]
        L = self.nbl - 1
        delta=[None]*(L+1)
        
        #couche la plus haute
        neu=self.l[-1]
        delt=(y_pred-y)*neu.activ.derivative(neu.z)
        delta[-1]=delt
        
        for i in range(L-1,-1,-1):
            neu=self.l[i]
            next_neu=self.l[i+1]
            delt=(next_neu.w.T @ delta[i+1])*neu.activ.derivative(neu.z)
            delta[i]=delt
            
        for (i,neu) in enumerate(self.l):
            # Gradient: delta[i] est de shape (n_neurons, batch_size) self.a[i] est de shape (n_inputs, batch_size)
            # Produit matriciel: (n_neurons, batch_size) @ (batch_size, n_inputs) = (n_neurons, n_inputs)
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1) #permet d'éviter les problèmes et de perdre 1H X)
            neu.update(lr,grad_w,grad_b)
        return y_pred
    
    
    def train_loss(self,epoch_loss, num_batches):
        train_loss = epoch_loss / num_batches
        self.train_losses.append(train_loss)
    
    def train(self, x_train, y_train, epochs, lr, batch_size):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if k % 100 == 0:
                print(f"Epoch {k}/{epochs}")
            
            # Mélanger les données à chaque epoch
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            # Parcourir par mini-batches
            for i in range(0, Nb_v_entr, batch_size):
                # Extraire le batch
                #calcul de l'indice de fin du batch
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # Forward et backward sur le batch
                self.forward(x_batch)
                self.backward(y_batch, lr)
                # Calculer la perte pour ce batch
                epoch_loss += self.MSE(self.a[-1].reshape(1,-1), y_batch.reshape(1, -1))
                num_batches += 1
            
            # Perte moyenne d'entraînement pour cette epoch
            self.train_loss(epoch_loss, num_batches)