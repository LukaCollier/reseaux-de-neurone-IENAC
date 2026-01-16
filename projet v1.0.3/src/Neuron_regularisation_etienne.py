import numpy as np
import src.Activation as Activation
import src.serialisation_pkl as serialisation_pkl
import src.serialisation as serialisation
import src.regularisation as regularisation

class Layer:
    '''
    n_input : input size
    n_neurone : number of neurons in the layer
    biais: vector of biases with |biais|=n_neurone
    w: weight matrix
    x : input
    z : result of Wx+b
    f : activation function on vectors
    '''

    
    def __init__(self, n_input, n_neurone, activ,f_regularisation,change_init):
        self.n_input = n_input
        self.n_neurone = n_neurone
        self.biais = np.zeros(n_neurone)
        
        # Initialization adapted according to the activation function

        if activ.name == "softmax":
            # Xavier for softmax
            self.w = np.random.randn(n_neurone, n_input) * np.sqrt(1.0 / n_input)
        elif not (change_init):
            # He for ReLU and others
            self.w = np.random.randn(n_neurone, n_input) * np.sqrt(2.0 / n_input)
        else:
            self.w = np.random.randn(n_neurone, n_input)
            
        self.x = np.zeros(n_input)
        self.z = np.zeros(n_neurone)
        self.f = np.zeros(n_neurone)
        self.activ = activ
        self.f_regularisation=f_regularisation
        print(type(f_regularisation))
        
        # For ADAM
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.biais)
        self.v_b = np.zeros_like(self.biais)
        
        # For RMSProp
        self.s_w = np.zeros_like(self.w)
        self.s_b = np.zeros_like(self.biais)
        
        # For Momentum
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
    

    def serialise(self, name):
        serialisation.serialise(name, self.to_json())
    

    @classmethod
    def dict_to_layer(cls, d,f_regularisation,lambd):
        res = cls(d["n_input"],
                  d["n_neurone"],
                  Activation.ActivationF.creation_with_name(d["activ"]),f_regularisation,lambd)
        res.biais = serialisation.decode_numpy(d["biais"])
        res.w = serialisation.decode_numpy(d["w"])
        res.x = serialisation.decode_numpy(d["x"])
        res.z = serialisation.decode_numpy(d["z"])
        res.f = serialisation.decode_numpy(d["f"])
        res.m_w = serialisation.decode_numpy(d["m_w"])
        res.v_w = serialisation.decode_numpy(d["v_w"])
        res.m_b = serialisation.decode_numpy(d["m_b"])
        res.v_b = serialisation.decode_numpy(d["v_b"])
        res.s_w = serialisation.decode_numpy(d["s_w"])
        res.s_b = serialisation.decode_numpy(d["s_b"])
        res.vw_momentum = serialisation.decode_numpy(d["vw_momentum"])
        res.vb_momentum = serialisation.decode_numpy(d["vb_momentum"])
        return res
  


    def forward(self, x):
        # x must have shape: (n_input, batch_size) or (n_input, 1)
        self.x = np.array(x)
        
        # If x is 1D (n_input,), convert it to (n_input, 1)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        # z = W @ x + b (broadcast the bias)
        self.z = self.w @ self.x + self.biais.reshape(-1, 1)
        self.f = self.activ.function(self.z)
        return self.f

    def cleanWB(self):
        self.biais = np.zeros(self.n_neurone)
        if self.activ.name == "softmax":
            self.w = np.random.randn(self.n_neurone, self.n_input) * np.sqrt(1.0 / self.n_input)
        else:
            self.w = np.random.randn(self.n_neurone, self.n_input) * np.sqrt(2.0 / self.n_input)

    
    def SGD_update(self, lr, g_w, g_b):
        self.w -= lr * g_w
        self.biais -= lr * g_b


    def SGDMomentum_update(self, grad_w, grad_b, lr, momentum=0.9):
        # CORRECTION: Updates velocities (without lr)
        self.vw_momentum = momentum * self.vw_momentum + grad_w
        self.vb_momentum = momentum * self.vb_momentum + grad_b

        # Update weights and biases (with lr)
        self.w -= lr * self.vw_momentum
        self.biais -= lr * self.vb_momentum


    def RMS_update(self, grad_w, grad_b, lr, beta=0.9, epsilon=1e-8):
        # Updates moving averages of the squared gradients
        self.s_w = beta * self.s_w + (1 - beta) * (grad_w ** 2)
        self.s_b = beta * self.s_b + (1 - beta) * (grad_b ** 2)

        # Update weights and biases
        self.w -= lr * grad_w / (np.sqrt(self.s_w) + epsilon)
        self.biais -= lr * grad_b / (np.sqrt(self.s_b) + epsilon)

    def Adam_update(self, grad_w, grad_b, lr, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Updates the moments
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)

        # Correction of the bias
        m_w_hat = self.m_w / (1 - beta1 ** t)
        v_w_hat = self.v_w / (1 - beta2 ** t)
        m_b_hat = self.m_b / (1 - beta1 ** t)
        v_b_hat = self.v_b / (1 - beta2 ** t)

        # Update weights and biases
        self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.biais -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        
        
class Neural_Network:
    '''
    l: layers array
    a: Array of activations with a[0] being the input
    nbl:number of layer
    activ: Array of activation functions corresponding to each layer
    ''' 


    def __init__(self, n_input_init, nb_n_l, activ, loss="MSE",name_f_regularisation='L0',lambda_regularisation=0,change_init=False):
        '''
        Note: isinstance checks that `activ` is indeed of the same type as the object
        '''
        if isinstance(activ, Activation.ActivationF):
            # Allows avoiding repeating the activation each time for a network with a single activation
            activ = [activ] * len(nb_n_l)
        else:
            assert len(activ) == len(nb_n_l)  # Checks that there are enough activations for the layers
        
        self.l = []
        self.a = []
        self.nbl = len(nb_n_l)
        n_input = n_input_init
        self.train_losses = []
        self.val_losses = []
        self.loss = loss
        self.loss_name = self.loss.replace("_", " ").upper()
        self.n_input_init = n_input_init
        self.nb_n_l = nb_n_l
        self.activ = activ
        self.f_regularisation=regularisation.RegularisationF.creation_with_name(name_f_regularisation,lambda_regularisation)
        
        for i, nb_n in enumerate(nb_n_l):
            self.l.append(Layer(n_input, nb_n, activ[i],f_regularisation=self.f_regularisation,change_init=change_init))
            n_input = nb_n

        # For ADAM
        self.t = 0


    def forward(self, x):
        x = np.array(x)
        # Convert x to format (n_input, batch_size)
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # (n_input, 1)
        else:
            x = x.T  # (batch_size, n_input) -> (n_input, batch_size)
        
        self.a = [x]
        for neu in self.l:
            x = neu.forward(x)
            self.a.append(x)
        return x

    def cleanNetwork(self):
        for lay in self.l:
            lay.cleanWB()
        self.train_losses = []
        self.val_losses = []
        self.t = 0


    def MSE(self, y_pred, y):
        return 0.5 * np.sum((y_pred - y) ** 2)

    def cross_entropy(self, y_pred, y):
        # y_pred: (n_output, batch_size), y: (n_output, batch_size) one-hot
        eps = 1e-12
        # CORRECTION: Clip only downwards to avoid log(0)
        y_pred = np.clip(y_pred, eps, None)
        ce = -np.sum(y * np.log(y_pred)) / y.shape[1]
        return ce

    def get_loss_value(self, y_pred, y):
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy":
            return self.cross_entropy(y_pred, y)
        else:
            return self.MSE(y_pred, y)
        
    def serialise_pkl(self, name, mode='xb'):
        serialisation_pkl.serialise_pkl(self, name, mode)
        
    @classmethod       
    def deserialise_pkl(cls, name): 
        return serialisation_pkl.deserialise_pkl(name)
    
    def SGD(self, y, lr):
        '''
        y : expected result possibly |y|> 1 if multiple neurons on the highest layer
        lr : step size for weight and bias updates
        Principle: first compute delta for the last layer since it cannot be computed in a recursive loop, then iterate back to the first layer
        '''
        y = np.array(y, dtype=float)
        # Convert y to format (n_output, batch_size)
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # (n_output, 1)
        else:
            y = y.T  # (batch_size, n_output) -> (n_output, batch_size)
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)
        
        # Top layer
        neu = self.l[-1]
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy" and neu.activ.name == "softmax":
            delta[-1] = (y_pred - y)
        else:
            delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)
        
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delt = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)
            delta[i] = delt
            
        for (i, neu) in enumerate(self.l):
            # Gradient: delta[i] is of shape (n_neurons, batch_size) self.a[i] is of shape (n_inputs, batch_size)
            # Matrix product: (n_neurons, batch_size) @ (batch_size, n_inputs) = (n_neurons, n_inputs)
            grad_w = delta[i] @ self.a[i].T
            grad_w = neu.f_regularisation.function(neu.w,grad_w) 
            grad_b = np.sum(delta[i], axis=1)
            neu.SGD_update(lr, grad_w, grad_b)
        return y_pred
    
    def SGDMomentum(self, y, lr):
        y = np.array(y, dtype=float)
        # Convert y to format (n_output, batch_size)
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # (n_output, 1)
        else:
            y = y.T  # (batch_size, n_output) -> (n_output, batch_size)
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        # Top layer - CORRECTION: shortcut for softmax+CE
        neu = self.l[-1]
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy" and neu.activ.name == "softmax":
            delta[-1] = (y_pred - y)
        else:
            delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)

        # For the other layers
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        # Update Momentum
        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_w = neu.f_regularisation.function(neu.w,grad_w) 
            grad_b = np.sum(delta[i], axis=1)
            neu.SGDMomentum_update(grad_w, grad_b, lr)

        return y_pred


    def RMS(self, y, lr):
        y = np.array(y, dtype=float)
        # Convert y to format (n_output, batch_size)
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # (n_output, 1)
        else:
            y = y.T  # (batch_size, n_output) -> (n_output, batch_size)
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        # Top layer - CORRECTION: shortcut for softmax+CE
        neu = self.l[-1]
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy" and neu.activ.name == "softmax":
            delta[-1] = (y_pred - y)
        else:
            delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)

        # Backward propagation
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        # Update RMSProp
        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_w = neu.f_regularisation.function(neu.w,grad_w) 
            grad_b = np.sum(delta[i], axis=1)
            neu.RMS_update(grad_w, grad_b, lr)

        return y_pred

    

    def ADAM(self, y, lr):
        y = np.array(y, dtype=float)
        # Convert y to format (n_output, batch_size)
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # (n_output, 1)
        else:
            y = y.T  # (batch_size, n_output) -> (n_output, batch_size)
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        # Top layer
        neu = self.l[-1]
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy" and neu.activ.name == "softmax":
            delta[-1] = (y_pred - y)
        else:
            delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)
        
        # For the other layers
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        # Additions by ADAM compared to the backward method
        self.t += 1 
        for (i, neu) in enumerate(self.l):
            grad_w = delta[i] @ self.a[i].T
            grad_w = neu.f_regularisation.function(neu.w,grad_w) 
            grad_b = np.sum(delta[i], axis=1)
            neu.Adam_update(grad_w, grad_b, lr, self.t)
        
        return y_pred


    def train_loss(self, epoch_loss, num_batches):
        train_loss = epoch_loss / num_batches
        self.train_losses.append(train_loss)
    
    def evaluate(self, x_test, y_test):
        y_pred = self.forward(x_test)  # y_pred has shape (n_output, n_samples)
        # Convert y_test to format (n_output, n_samples)
        y_test = np.array(y_test)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        else:
            y_test = y_test.T  # (n_samples, n_output) -> (n_output, n_samples)
        loss = self.get_loss_value(y_pred, y_test)
        return loss


    def train_SGD(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 100 == 0:
                print(f"Epoch {k}/{epochs}")
            
            # Mix the data at each epoch
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            # Itinerate over mini-batches
            for i in range(0, Nb_v_entr, batch_size):
                # Extract the batch
                # Computation of the end index of the batch
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # Forward and backward on the batch
                self.forward(x_batch)
                self.SGD(y_batch, lr)
                # Calculate the loss for this batch
                epoch_loss += self.get_loss_value(self.a[-1], y_batch.T)
                num_batches += 1
            
            # Mean training loss for this epoch
            self.train_loss(epoch_loss, num_batches)
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 100 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")


    def train_SGDMomentum(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 100 == 0:
                print(f"Epoch {k}/{epochs}")
            
            # Mix the data at each epoch
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            # Iterate over mini-batches
            for i in range(0, Nb_v_entr, batch_size):
                # Extract the batch
                # Computation of the end index of the batch
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # Forward and backward on the batch
                self.forward(x_batch)
                self.SGDMomentum(y_batch, lr)
                # Calculate the loss for this batch
                epoch_loss += self.get_loss_value(self.a[-1], y_batch.T)
                num_batches += 1

            # Mean training loss for this epoch
            self.train_loss(epoch_loss, num_batches)
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 100 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")


    def train_RMS(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 100 == 0:
                print(f"Epoch {k}/{epochs}")
            
            # Mix the data at each epoch
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            # Iterate over mini-batches
            for i in range(0, Nb_v_entr, batch_size):
                # Extraire le batch
                # Computation of the end index of the batch
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # Forward and backward on the batch
                self.forward(x_batch)
                self.RMS(y_batch, lr)
                # Calculate the loss for this batch
                epoch_loss += self.get_loss_value(self.a[-1], y_batch.T)
                num_batches += 1

            # Mean training loss for this epoch
            self.train_loss(epoch_loss, num_batches)
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 100 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")


    def train_ADAM(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 100 == 0:
                print(f"Epoch {k}/{epochs}")
            
            # Mix the data at each epoch
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            # Iterate over mini-batches
            for i in range(0, Nb_v_entr, batch_size):
                # Extract the batch
                # Computation of the end index of the batch
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # Forward and backward on the batch
                self.forward(x_batch)
                self.ADAM(y_batch, lr)
                # Calculate the loss for this batch
                epoch_loss += self.get_loss_value(self.a[-1], y_batch.T)
                num_batches += 1

            # Mean training loss for this epoch
            self.train_loss(epoch_loss, num_batches)
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 100 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")