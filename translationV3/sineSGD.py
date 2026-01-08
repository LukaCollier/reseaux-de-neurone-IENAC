import src.Neuron as Neuron
import src.Activation as Activation
import numpy as np
import matplotlib.pyplot as plt
'''
Regression problem using the parameters from the PDF example

'''
activ_tanh = Activation.ActivationF.tanh()
activ_leaky_relu = Activation.ActivationF.leaky_relu()
#network=neuroneopti.Neural_Network(1,[16,16,1],[activ_tanh,activ_tanh,activ_tanh])
network = Neuron.Neural_Network(1, [8, 8, 1], activ_tanh)
'''
NeuralNetwork Training
'''
#Nb_v_entr=500
Nb_v_entr = 2000  # instead of 500
Epoch=500 #instead of 2000
lr=1e-2
v_entr=np.random.uniform(0,2*np.pi,Nb_v_entr).reshape(-1, 1)  # Shape: (2000, 1)
res_th=np.sin(v_entr.flatten()).reshape(-1, 1)  # Shape: (2000, 1)

# Generation of validation data
Nb_v_val = 500
v_val = np.random.uniform(0, 2*np.pi, Nb_v_val).reshape(-1, 1)  # Shape: (500, 1)
res_val = np.sin(v_val.flatten()).reshape(-1, 1)  # Shape: (500, 1)

network.train_SGD(v_entr,res_th,Epoch,lr, batch_size=16, x_val=v_val, y_val=res_val) # Training with batch size 16 — note: do not reshape the input

train_losses = network.train_losses

# Display the loss curve
plt.figure(figsize=(15, 5))


# Plot 1: Loss curves
plt.subplot(1, 3, 1)
plt.plot(network.train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(network.val_losses, label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss curves (SGD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Comparison on validation data
plt.subplot(1, 3, 2)
val_pred = network.forward(v_val).flatten()  # forward now expects (n_samples, n_features)
plt.plot(v_val.flatten(), res_val.flatten(), 'o', label="np.sin", color="blue", markersize=4, alpha=0.5)
plt.plot(v_val.flatten(), val_pred, 'o', label="Neural Network", color="red", markersize=4, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Validation: sin(x) vs Neural Network')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Error distribution
plt.subplot(1, 3, 3)
errors = np.abs(res_val.flatten() - val_pred).flatten()  # Ensure it is a 1D array
plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Absolute error')
plt.ylabel('Frequency')
plt.title('Validation error distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display the final loss
print(f"\nLoss finale: {train_losses[-1]:.6f}")
