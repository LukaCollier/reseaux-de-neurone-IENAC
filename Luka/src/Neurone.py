import numpy as np
from . import Activation
from . import serialisation

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
        self.biais=np.zeros(n_neurone)
        self.w = np.random.randn(n_neurone, n_input) * np.sqrt(2. / n_input)  # Initialisation de He
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
        #pour Momentum
        self.vw_momentum = np.zeros_like(self.w)
        self.vb_momentum = np.zeros_like(self.biais)


    
    def to_json(self):
        return {
            "n_input": self.n_input,
            "n_neurone": self.n_neurone,
            "biais": serialisation.encode_numpy(self.biais),
            "w": serialisation.encode_numpy(self.w),
            "x": serialisation.encode_numpy(self.x),
            "z": serialisation.encode_numpy(self.z),
            "f": serialisation.encode_numpy(self.f),
            "activ": self.activ.name,
            "m_w": serialisation.encode_numpy(self.m_w),
            "v_w": serialisation.encode_numpy(self.v_w),
            "m_b": serialisation.encode_numpy(self.m_b),
            "v_b": serialisation.encode_numpy(self.v_b),
            "s_w": serialisation.encode_numpy(self.s_w),
            "s_b": serialisation.encode_numpy(self.s_b),
            "vw_momentum": serialisation.encode_numpy(self.vw_momentum),  # CORRECTION
            "vb_momentum": serialisation.encode_numpy(self.vb_momentum)   # CORRECTION
        }
    

    def serialise(self,name):
        serialisation.serialise(name,self.to_json())
    

    @classmethod
    def dict_to_layer(cls,d):
        res=cls(d["n_input"],
                   d["n_neurone"],
                   Activation.ActivationF.creation_with_name(d["activ"]))
        res.biais=serialisation.decode_numpy(d["biais"])
        res.w=serialisation.decode_numpy(d["w"])
        res.x=serialisation.decode_numpy(d["x"])
        res.z=serialisation.decode_numpy(d["z"])
        res.f=serialisation.decode_numpy(d["f"])
        res.m_w=serialisation.decode_numpy(d["m_w"])
        res.v_w=serialisation.decode_numpy(d["v_w"])
        res.m_b=serialisation.decode_numpy(d["m_b"])
        res.v_b=serialisation.decode_numpy(d["v_b"])
        res.s_w=serialisation.decode_numpy(d["s_w"])
        res.s_b=serialisation.decode_numpy(d["s_b"])
        res.vw_momentum=serialisation.decode_numpy(d["vw_momentum"])
        res.vb_momentum=serialisation.decode_numpy(d["vb_momentum"])
        return res


    def forward(self, x):
        self.x = np.array(x)
        
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        self.z = self.w @ self.x + self.biais.reshape(-1, 1)
        self.f = self.activ.function(self.z)
        return self.f

    def cleanWB(self):
        self.biais=np.random.randn(self.n_neurone)
        self.w = np.random.randn(self.n_neurone, self.n_input)

    def SGD_update(self,lr,g_w,g_b):
        self.w -= lr*g_w
        self.biais -= lr*g_b


    def SGDMomentum_update(self,grad_w, grad_b, lr, momentum=0.9):
        self.vw_momentum = momentum * self.vw_momentum + lr * grad_w
        self.vb_momentum = momentum * self.vb_momentum + lr * grad_b
        self.w -= self.vw_momentum
        self.biais -= self.vb_momentum


    def RMS_update(self,grad_w, grad_b, lr, beta=0.9, epsilon=1e-8):
        self.s_w = beta * self.s_w + (1 - beta) * (grad_w ** 2)
        self.s_b = beta * self.s_b + (1 - beta) * (grad_b ** 2)
        self.w -= lr * grad_w / (np.sqrt(self.s_w) + epsilon)
        self.biais -= lr * grad_b / (np.sqrt(self.s_b) + epsilon)

    def Adam_update(self,grad_w, grad_b, lr, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)

        m_w_hat = self.m_w / (1 - beta1 ** t)
        v_w_hat = self.v_w / (1 - beta2 ** t)
        m_b_hat = self.m_b / (1 - beta1 ** t)
        v_b_hat = self.v_b / (1 - beta2 ** t)

        self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.biais -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        
        
class Neural_Network:
    '''
    l:tableau de layer
    a:tableau des activation avec a[0] qui est l'entrée
    nbl:nombre de layer
    activ: tableau des fonctions d'activations correspondant à chaque layer
    ''' 


    def __init__(self,n_input_init,nb_n_l,activ,loss="mse"):
        if isinstance(activ, Activation.ActivationF):
            activ = [activ] * len(nb_n_l)
        else:
            assert len(activ) == len(nb_n_l)
        self.l=[]
        self.a=[]
        self.loss=loss
        self.n_input_init=n_input_init
        self.nb_n_l=nb_n_l
        self.activ=activ
        self.nbl=len(nb_n_l)
        n_input=n_input_init
        self.train_losses = []
        self.val_losses=[]
        for i,nb_n in enumerate(nb_n_l):
            self.l.append(Layer(n_input,nb_n,activ[i]))
            n_input=nb_n

        self.t = 0

    def to_json(self):
        return {
            "activ": self.activ[0].name,
            "nb_n_l": self.nb_n_l,
            "n_input_init": self.n_input_init,
            "l": [elt.to_json() for elt in self.l],
            "a": [serialisation.encode_numpy(elt) for elt in self.a],
            "nbl": self.nbl,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "t": self.t,
            "loss": self.loss
        }

    def serialise(self,name,mode='x'):
        serialisation.serialise(name,self.to_json(),mode)

    @classmethod
    def deserialise(cls,name):
        data=serialisation.deserialise(name)
        res=cls(data["n_input_init"],data["nb_n_l"],Activation.ActivationF.creation_with_name(data["activ"]))
        res.l=[Layer.dict_to_layer(d) for d in data["l"]]
        res.a=[serialisation.decode_numpy(elt) for elt in data["a"]]
        res.train_losses=data["train_losses"]
        res.nbl=data["nbl"]
        res.val_losses=data["val_losses"]
        res.t=data["t"]
        res.loss=data["loss"]  # CORRECTION : charger aussi loss
        return res
    


    def forward(self,x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        else:
            x = x.T
        
        self.a = [x]
        for neu in self.l:
            x=neu.forward(x)
            self.a.append(x)
        return x

    def cleanNetwork(self):
        for lay in self.l:
            lay.cleanWB()
        self.train_losses = []
        self.val_losses=[]

    def cross_entropy(self, y_pred, y_test):
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        # CORRECTION : Normaliser par le nombre d'échantillons
        n_samples = y_pred.shape[1] if y_pred.ndim > 1 else 1
        return -np.sum(y_test * np.log(y_pred_clipped)) / n_samples

    def MSE(self, y_pred, y):
        # CORRECTION : Normaliser par le nombre d'échantillons
        n_samples = y_pred.shape[1] if y_pred.ndim > 1 else 1
        return 0.5 * np.sum((y_pred - y) ** 2) / n_samples

    def compute_last_delta(self, y_pred, y, last_layer):
        if self.loss == "mse":
            return (y_pred - y) * last_layer.activ.derivative(last_layer.z)
        elif self.loss == "cross_entropy":
            # CORRECTION : avec softmax + cross-entropy, le gradient est simplement y_pred - y
            return (y_pred - y)
        else:
            raise ValueError("Unknown loss type: " + self.loss)


    def SGD(self,y,lr):
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        else:
            y = y.T
        
        y_pred=self.a[-1]
        L = self.nbl - 1
        delta=[None]*(L+1)
        
        neu=self.l[-1]
        delta[-1] = self.compute_last_delta(y_pred, y, neu)
        
        for i in range(L-1,-1,-1):
            neu=self.l[i]
            next_neu=self.l[i+1]
            delt=(next_neu.w.T @ delta[i+1])*neu.activ.derivative(neu.z)
            delta[i]=delt
            
        for (i,neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1)
            neu.SGD_update(lr,grad_w,grad_b)
        return y_pred
    
    def SGDMomentum(self, y, lr):
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        else:
            y = y.T
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        neu = self.l[-1]
        delta[-1] = self.compute_last_delta(y_pred, y, neu)

        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1)
            neu.SGDMomentum_update(grad_w, grad_b, lr)

        return y_pred


    def RMS(self, y, lr):
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        else:
            y = y.T
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        neu = self.l[-1]
        delta[-1] = self.compute_last_delta(y_pred, y, neu)

        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1)
            neu.RMS_update(grad_w, grad_b, lr)

        return y_pred

    

    def ADAM(self,y,lr):
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        else:
            y = y.T
        
        y_pred=self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        neu = self.l[-1]
        delta[-1] = self.compute_last_delta(y_pred, y, neu)
        
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)
        
        self.t += 1 
        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_b = np.sum(delta[i], axis=1)
            neu.Adam_update(grad_w, grad_b, lr, self.t)
        
        return y_pred


    def update_epoch_loss(self,epoch_loss, y_batch):
        if self.loss == "mse":
            return epoch_loss + self.MSE(self.a[-1], y_batch.T)
        elif self.loss == "cross_entropy":
            return epoch_loss + self.cross_entropy(self.a[-1], y_batch.T)


    def train_loss(self,epoch_loss, num_batches):
        train_loss = epoch_loss / num_batches
        self.train_losses.append(train_loss)
    
    def evaluate(self, x_test, y_test):
        y_pred = self.forward(x_test)
        y_test = np.array(y_test)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        else:
            y_test = y_test.T
            
        if self.loss == "mse":
            return self.MSE(y_pred, y_test)
        elif self.loss == "cross_entropy":
            return self.cross_entropy(y_pred, y_test)
        else:
            raise ValueError("Unknown loss type: " + self.loss)
        

    def train_SGD(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=True):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 10 == 0:  # CORRECTION : affichage tous les 10 epochs
                print(f"Epoch {k}/{epochs}")
            
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, Nb_v_entr, batch_size):
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                self.forward(x_batch)
                self.SGD(y_batch, lr)
                
                epoch_loss = self.update_epoch_loss(epoch_loss, y_batch)
                num_batches += 1
            
            self.train_loss(epoch_loss, num_batches)
            
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 10 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")


    def train_SGDMomentum(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=True):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 10 == 0:
                print(f"Epoch {k}/{epochs}")
            
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, Nb_v_entr, batch_size):
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                self.forward(x_batch)
                self.SGDMomentum(y_batch, lr)
                
                epoch_loss = self.update_epoch_loss(epoch_loss, y_batch)
                num_batches += 1
            
            self.train_loss(epoch_loss, num_batches)
            
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 10 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")


    def train_RMS(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=True):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 10 == 0:
                print(f"Epoch {k}/{epochs}")
            
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, Nb_v_entr, batch_size):
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                self.forward(x_batch)
                self.RMS(y_batch, lr)
                
                epoch_loss = self.update_epoch_loss(epoch_loss, y_batch)
                num_batches += 1
            
            self.train_loss(epoch_loss, num_batches)
            
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 10 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")


    def train_ADAM(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=True):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 10 == 0:  # CORRECTION : affichage tous les 10 epochs
                print(f"Epoch {k}/{epochs}")
            
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, Nb_v_entr, batch_size):
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                self.forward(x_batch)
                self.ADAM(y_batch, lr)
                
                epoch_loss = self.update_epoch_loss(epoch_loss, y_batch)
                num_batches += 1
            
            self.train_loss(epoch_loss, num_batches)
            
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 10 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")