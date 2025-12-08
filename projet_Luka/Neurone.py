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
        #pour ADAM
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.biais)
        self.v_b = np.zeros_like(self.biais)
        #pour RMSProp
        self.s_w = np.zeros_like(self.w)
        self.s_b = np.zeros_like(self.biais)


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

    def RMS_update(self,grad_w, grad_b, lr, beta=0.9, epsilon=1e-8):
        # Met à jour les moyennes mobiles des carrés des gradients
        self.s_w = beta * self.s_w + (1 - beta) * (grad_w ** 2)
        self.s_b = beta * self.s_b + (1 - beta) * (grad_b ** 2)

        # Mise à jour des poids et biais
        self.w -= lr * grad_w / (np.sqrt(self.s_w) + epsilon)
        self.biais -= lr * grad_b / (np.sqrt(self.s_b) + epsilon)

    def Adam_update(self,grad_w, grad_b, lr, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Met à jour les moments
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)

        # Correction des biais
        m_w_hat = self.m_w / (1 - beta1 ** t)
        v_w_hat = self.v_w / (1 - beta2 ** t)
        m_b_hat = self.m_b / (1 - beta1 ** t)
        v_b_hat = self.v_b / (1 - beta2 ** t)

        # Mise à jour des poids et biais
        self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.biais -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        
        
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

        #pour ADAM
        self.t = 0


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



    def SGD(self,y,lr):
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
    

    def RMS(self, y, lr):
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        # dernière couche
        neu = self.l[-1]
        delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)

        # propagation arrière
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        # update RMSProp
        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1)
            neu.RMS_update(grad_w, grad_b, lr)

        return y_pred

    

    def ADAM(self,y,lr):
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        y_pred=self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        # couche la plus haute
        neu = self.l[-1]
        delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)
        
        # pour les autres couches
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)
        
        #Ajout par ADAM comparé à la méthode backward
        self.t += 1 
        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1)
            neu.Adam_update(grad_w, grad_b, lr, self.t)
        
        return y_pred



    def train_loss(self,epoch_loss, num_batches):
        train_loss = epoch_loss / num_batches
        self.train_losses.append(train_loss)
    
    def evaluate(self, x_test, y_test):
        y_pred = self.forward(x_test)
        loss = self.MSE(y_pred, y_test)
        return loss



    def train_SGD(self, x_train, y_train, epochs, lr, batch_size):
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
                self.SGD(y_batch, lr)
                # Calculer la perte pour ce batch
                epoch_loss += self.MSE(self.a[-1], y_batch.T)
                num_batches += 1
            
            # Perte moyenne d'entraînement pour cette epoch
            self.train_loss(epoch_loss, num_batches)

    def train_ADAM(self, x_train, y_train, epochs, lr, batch_size):
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
                self.ADAM(y_batch, lr)
                # Calculer la perte pour ce batch
                epoch_loss += self.MSE(self.a[-1], y_batch.T)
                num_batches += 1
            
            # Perte moyenne d'entraînement pour cette epoch
            self.train_loss(epoch_loss, num_batches)