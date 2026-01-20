import numpy as np
from . import Activation
from . import serialisation_pkl
from . import serialisation
from . import regularisation

class Layer:
    '''
    n_input: input size
    n_neurone: number of neurons
    bias: bias vector with |bias| = n_neurone
    w: weight matrix
    x: cached input
    z: Wx + b result
    f: activation output
    activ: vector activation function
    '''

    
    def __init__(self, n_input, n_neurone, activ, f_regularisation, flag=True):
        self.n_input = n_input
        self.n_neurone = n_neurone
        self.biais = np.zeros(n_neurone)
        self.flag=flag
        # Initialization adapted to the activation function
        if flag:
            if activ.name == "softmax":
                # Xavier for softmax
                self.w = np.random.randn(n_neurone, n_input) * np.sqrt(1.0 / n_input)
            else:
                # He for ReLU and others
                self.w = np.random.randn(n_neurone, n_input) * np.sqrt(2.0 / n_input)
        else:
            self.w = np.random.randn(n_neurone, n_input)
        self.x = np.zeros(n_input)
        self.z = np.zeros(n_neurone)
        self.f = np.zeros(n_neurone)
        self.activ = activ
        self.f_regularisation = f_regularisation
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
        """Serialize layer parameters and optimizer states to a JSON-serializable dict."""
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
    

    def serialize(self, name):
        """Persist layer parameters to disk."""
        serialisation.serialize(name, self.to_json())
    

    @classmethod
    def dict_to_layer(cls, d):
        """Rebuild a Layer instance from a serialized dictionary."""
        res = cls(d["n_input"],
                  d["n_neurone"],
                  Activation.ActivationF.creation_with_name(d["activ"]))
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
        """Run forward pass; supports single sample or batch as columns."""
        # x must have shape: (n_input, batch_size) or (n_input, 1)
        self.x = np.array(x)
        
        # If x is 1D (n_input,), reshape to (n_input, 1)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        # z = W @ x + b (bias is broadcast)
        self.z = self.w @ self.x + self.biais.reshape(-1, 1)
        self.f = self.activ.function(self.z)
        return self.f

    def cleanWB(self):
        """Reset weights, biases, and all optimizer accumulators."""
        self.biais = np.zeros(self.n_neurone)
        
        # Weight reset depending on activation
        if self.activ.name == "softmax":
            self.w = np.random.randn(self.n_neurone, self.n_input) * np.sqrt(1.0 / self.n_input)
        else:
            self.w = np.random.randn(self.n_neurone, self.n_input) * np.sqrt(2.0 / self.n_input)
        
        # Reset ALL accumulators
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
    def SGD_update(self, lr, g_w, g_b):
        """Vanilla SGD update."""
        self.w -= lr * g_w
        self.biais -= lr * g_b


    def SGDMomentum_update(self, grad_w, grad_b, lr, momentum=0.9):
        """SGD with momentum update."""
        # Update velocities (without lr)
        self.vw_momentum = momentum * self.vw_momentum + grad_w
        self.vb_momentum = momentum * self.vb_momentum + grad_b

        # Update weights and biases (with lr)
        self.w -= lr * self.vw_momentum
        self.biais -= lr * self.vb_momentum


    def RMS_update(self, grad_w, grad_b, lr, beta=0.9, epsilon=1e-8):
        """RMSProp update."""
        # Update moving averages of squared gradients
        self.s_w = beta * self.s_w + (1 - beta) * (grad_w ** 2)
        self.s_b = beta * self.s_b + (1 - beta) * (grad_b ** 2)

        # Update weights and biases
        self.w -= lr * grad_w / (np.sqrt(self.s_w) + epsilon)
        self.biais -= lr * grad_b / (np.sqrt(self.s_b) + epsilon)

    def Adam_update(self, grad_w, grad_b, lr, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam update with bias correction."""
        # Update moments
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)

        # Bias correction
        m_w_hat = self.m_w / (1 - beta1 ** t)
        v_w_hat = self.v_w / (1 - beta2 ** t)
        m_b_hat = self.m_b / (1 - beta1 ** t)
        v_b_hat = self.v_b / (1 - beta2 ** t)

        # Update weights and biases
        self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.biais -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        
        
class Neural_Network:
    '''
    l: list of layers
    a: list of activations with a[0] = input
    nbl: number of layers
    activ: activation functions per layer
    ''' 


    def __init__(self, n_input_init, nb_n_l, activ, loss="MSE", name_regularisation="L0", lambda_regularisation=0, flag=True):
        '''
        If a single activation is provided, it is duplicated for all layers.
        '''
        if isinstance(activ, Activation.ActivationF):
            # Avoid repeating the activation for networks with a single activation choice
            activ = [activ] * len(nb_n_l)
        else:
            assert len(activ) == len(nb_n_l)  # Ensure one activation per layer
        
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
        self.f_regularisation=regularisation.RegularisationF.creation_with_name(name_regularisation, lambda_regularisation)
        self.flag=flag
        
        for i, nb_n in enumerate(nb_n_l):
            self.l.append(Layer(n_input, nb_n, activ[i], self.f_regularisation, flag))
            n_input = nb_n

        # For ADAM time step
        self.t = 0
    def copy_with_regularisation_changes(self, n_f_regularisation, l_regularisation):
        """Clone the network while changing regularisation type and strength."""
        res=Neural_Network(self.n_input_init,
                              self.nb_n_l,
                              self.activ[0],loss=self.loss,
                              name_regularisation=n_f_regularisation,
                              lambda_regularisation=l_regularisation,
                              flag =self.flag)
        res.l=self.l.copy()
        return res


    def forward(self, x):
        """Forward pass through all layers; returns output."""
        x = np.array(x)
        # Convert x to shape (n_input, batch_size)
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
        """Reset all layers and tracked losses/optimizer states."""
        for lay in self.l:
            lay.cleanWB()
        self.train_losses = []
        self.val_losses = []
        self.t = 0


    def MSE(self, y_pred, y):
        """Mean squared error."""
        return 0.5 * np.sum((y_pred - y) ** 2)

    def cross_entropy(self, y_pred, y):
        """Cross-entropy loss for one-hot targets."""
        # y_pred: (n_output, batch_size), y: (n_output, batch_size) one-hot
        eps = 1e-12
        # Only clip the lower bound to avoid log(0)
        y_pred = np.clip(y_pred, eps, None)
        ce = -np.sum(y * np.log(y_pred)) / y.shape[1]
        return ce

    def get_loss_value(self, y_pred, y):
        """Select proper loss function (cross-entropy or MSE)."""
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy":
            return self.cross_entropy(y_pred, y)
        else:
            return self.MSE(y_pred, y)

    def serialize_pkl(self, name, mode='xb'):
        """Serialize the full network with pickle."""
        serialisation_pkl.serialize_pkl(self, name, mode)
        
    @classmethod       
    def deserialize_pkl(cls, name):
        """Load a network serialized with pickle."""
        return serialisation_pkl.deserialize_pkl(name)


    def optimizer(self, y, lr, method):
        """Backpropagation and parameter update using chosen optimizer."""
        y = np.array(y, dtype=float)
        # Convert y to shape (n_output, batch_size)
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # (n_output, 1)
        else:
            y = y.T  # (batch_size, n_output) -> (n_output, batch_size)
        
        y_pred = self.a[-1]
        L = self.nbl - 1
        delta = [None] * (L + 1)

        # Output layer
        neu = self.l[-1]
        if isinstance(self.loss, str) and self.loss.lower() == "cross_entropy" and neu.activ.name == "softmax":
            delta[-1] = (y_pred - y)
        else:
            delta[-1] = (y_pred - y) * neu.activ.derivative(neu.z)
        
        # Hidden layers
        for i in range(L - 1, -1, -1):
            neu = self.l[i]
            next_neu = self.l[i + 1]
            delta[i] = (next_neu.w.T @ delta[i + 1]) * neu.activ.derivative(neu.z)

        # Optimizer choice
        if method == "ADAM":
            self.t += 1 
            for (i, neu) in enumerate(self.l):
                grad_w = delta[i] @ self.a[i].T
                grad_w = neu.f_regularisation.function(neu.w,grad_w)
                grad_b = np.sum(delta[i], axis=1)
                neu.Adam_update(grad_w, grad_b, lr, self.t)
        elif method == "SGD":
            for (i, neu) in enumerate(self.l):
                grad_w = delta[i] @ self.a[i].T
                grad_w = neu.f_regularisation.function(neu.w,grad_w)
                grad_b = np.sum(delta[i], axis=1)
                neu.SGD_update(lr, grad_w, grad_b)
        elif method == "RMS":
            for (i, neu) in enumerate(self.l):
                grad_w = delta[i] @ self.a[i].T
                grad_w = neu.f_regularisation.function(neu.w,grad_w)
                grad_b = np.sum(delta[i], axis=1)
                neu.RMS_update(grad_w, grad_b, lr)
        elif method == "SGDMomentum":
            for (i, neu) in enumerate(self.l):
                grad_w = delta[i] @ self.a[i].T
                grad_w = neu.f_regularisation.function(neu.w,grad_w)
                grad_b = np.sum(delta[i], axis=1)
                neu.SGDMomentum_update(grad_w, grad_b, lr)
        return y_pred
            

    def train_loss(self, epoch_loss, num_batches):
        """Store mean training loss for an epoch."""
        train_loss = epoch_loss / num_batches
        self.train_losses.append(train_loss)
    
    def evaluate(self, x_test, y_test):
        """Compute loss on a validation/test set."""
        y_pred = self.forward(x_test)  # y_pred aura shape (n_output, n_samples)
        # Convertir y_test en format (n_output, n_samples)
        y_test = np.array(y_test)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        else:
            y_test = y_test.T  # (n_samples, n_output) -> (n_output, n_samples)
        loss = self.get_loss_value(y_pred, y_test)
        return loss
    
    
    def train(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, method="SGD", verbose=False):
        """Train the network with mini-batches and optional validation tracking."""
        Nb_v_entr = x_train.shape[0]
        for k in range(epochs):
            if verbose and k % 100 == 0:
                print(f"Epoch {k}/{epochs}")
            
            # Shuffle data each epoch
            indices = np.random.permutation(Nb_v_entr)
            epoch_loss = 0
            num_batches = 0
            # Mini-batch loop
            for i in range(0, Nb_v_entr, batch_size):
                # Extract the batch
                # Calculate end index of the batch
                end_idx = min(i + batch_size, Nb_v_entr)
                batch_indices = indices[i:end_idx]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # Forward and backward on the batch
                self.forward(x_batch)
                self.optimizer(y_batch, lr, method=method)
                # Accumulate loss for this batch
                epoch_loss += self.get_loss_value(self.a[-1], y_batch.T)
                num_batches += 1
            # Mean training loss for this epoch
            self.train_loss(epoch_loss, num_batches)
            if x_val is not None and y_val is not None:
                val_loss = self.evaluate(x_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and k % 100 == 0:
                    print(f"  Train Loss: {self.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")

                    
# backward compatibility for training function names with older versions
    def train_RMS(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        self.train(x_train, y_train, epochs, lr, batch_size, x_val, y_val, method="RMS", verbose=verbose)
    def train_ADAM(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        self.train(x_train, y_train, epochs, lr, batch_size, x_val, y_val, method="ADAM", verbose=verbose)
    def train_SGDMomentum(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        self.train(x_train, y_train, epochs, lr, batch_size, x_val, y_val, method="SGDMomentum", verbose=verbose)
    def train_SGD(self, x_train, y_train, epochs, lr, batch_size, x_val=None, y_val=None, verbose=False):
        self.train(x_train, y_train, epochs, lr, batch_size, x_val, y_val, method="SGD", verbose=verbose)
