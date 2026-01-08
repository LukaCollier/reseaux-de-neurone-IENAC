import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import src.Neuron as Neuron
import src.Activation as Activation
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
data = pd.read_csv("framingham_clean.csv")


X = data.drop(columns=['age']).values.T
Y = data['age'].values.reshape(1, -1)

# Optimized split: 70% train, 15% validation, 15% test
# First, split off 30% for validation + test

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X.T, Y.T, test_size=0.3, random_state=42, shuffle=True
)

# Second, split the 30% into 50% validation and 50% test (15% each)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42, shuffle=True
)

print(f"Dataset split:")
print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X.T)*100:.1f}%)")
print(f"  Validation:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X.T)*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X.T)*100:.1f}%)")

# Normalization of features

mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)
X_train_norm = (X_train - mean_X) / (std_X + 1e-8)
X_val_norm = (X_val - mean_X) / (std_X + 1e-8)
X_test_norm = (X_test - mean_X) / (std_X + 1e-8)

# Normalization of Y
mean_Y = Y_train.mean()
std_Y = Y_train.std()
Y_train_norm = (Y_train - mean_Y) / std_Y
Y_val_norm = (Y_val - mean_Y) / std_Y
Y_test_norm = (Y_test - mean_Y) / std_Y

# Parameters
Epoch = 500
lr = 0.001
batch_size = 32

# DO NOT transpose X and Y - keep them in (n_samples, n_features) format
# pythonThe train_ADAM code expects this format for indexing by batch_indexes

# Create the network with the correct input size
activ_Relu = Activation.ActivationF.relu()
activ_idd = Activation.ActivationF.identity()

network = Neuron.Neural_Network(X_train_norm.shape[1], [16, 8, 1], [activ_Relu, activ_Relu, activ_idd])

# Training with NON-transposed data
network.train_ADAM(X_train_norm, Y_train_norm, Epoch, lr, batch_size, X_val_norm, Y_val_norm)

# Predictions on the validation set
Y_val_pred_norm = network.forward(X_val_norm)

# Denormalize the predictions
Y_val_pred = Y_val_pred_norm * std_Y + mean_Y
Y_val_denorm = Y_val_norm * std_Y + mean_Y

# Predictions on the test set
Y_test_pred_norm = network.forward(X_test_norm)
Y_test_pred = Y_test_pred_norm * std_Y + mean_Y
Y_test_denorm = Y_test_norm * std_Y + mean_Y

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1 : Loss curves
plt.subplot(1, 3, 1)
plt.plot(network.train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(network.val_losses, label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss curves (ADAM)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2 : Predictions vs Reality
plt.subplot(1, 3, 2)
plt.scatter(Y_val_denorm.flatten(), Y_val_pred.flatten(), alpha=0.6, color='blue', edgecolors='k', s=50)
plt.plot([Y_val_denorm.min(), Y_val_denorm.max()], 
         [Y_val_denorm.min(), Y_val_denorm.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Real age (years)')
plt.ylabel('Predicted age (years)')
plt.title('Validation: Predictions vs Reality')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3 : Error distribution
plt.subplot(1, 3, 3)
errors = np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten())
plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Absolute error (years)')
plt.ylabel('Frequency')
plt.title('Distribution of validation Errors')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final metrics
train_loss_final = network.train_losses[-1]
val_loss_final = network.val_losses[-1]
mae_val = np.mean(np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten()))
mae_test = np.mean(np.abs(Y_test_denorm.flatten() - Y_test_pred.flatten()))

print(f"\n{'='*50}")
print(f"FINAL RESULTS")
print(f"{'='*50}")
print(f"Training loss (normalized): {train_loss_final:.6f}")
print(f"Validation loss (normalized): {val_loss_final:.6f}")
print(f"MAE Validation: {mae_val:.2f} years")
print(f"MAE Test: {mae_test:.2f} years")
print(f"{'='*50}")