import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import src.Neuron as Neuron
import src.Activation as Activation
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
data = pd.read_csv("boston.csv")

X = data.drop(columns=['MEDV']).values.T
Y = data['MEDV'].values.reshape(1, -1)

# Split: 60% train, 20% validation, 20% test
X_temp, X_test, Y_temp, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

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
Epoch = 1000
lr = 0.001
lrSGD = 0.0001
batch_size = 32

# DO NOT transpose X and Y - keep them in (n_samples, n_features) format
# pythonThe train_ADAM code expects this format for indexing by batch_indexes

# Create the network with the correct input size
activ_Relu = Activation.ActivationF.relu()
activ_idd = Activation.ActivationF.identity()

network = Neuron.Neural_Network(X_train_norm.shape[1], [16, 8, 1], [activ_Relu, activ_Relu, activ_idd])


optimizers = {
    "ADAM": network.train_ADAM,
    "RMS": network.train_RMS,
    "SGD": network.train_SGD
}

# Loop through the 3 optimizers

for name, train_func in optimizers.items():
    if name == "SGD":
        train_func(X_train_norm, Y_train_norm, Epoch, lrSGD, batch_size, X_val_norm, Y_val_norm)
    else:
        train_func(X_train_norm, Y_train_norm, Epoch, lr, batch_size, X_val_norm, Y_val_norm)

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

    # Plot 1: Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(network.train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(network.val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Loss({name}) curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Predictions vs Actual Values (Validation)
    plt.subplot(1, 3, 2)
    plt.scatter(Y_val_denorm.flatten(), Y_val_pred.flatten(), alpha=0.6, color='blue', edgecolors='k', s=50)
    plt.plot([Y_val_denorm.min(), Y_val_denorm.max()], 
         [Y_val_denorm.min(), Y_val_denorm.max()], 
         'r--', linewidth=2, label='Perfect predictions')
    plt.xlabel('Real prices (k$)')
    plt.ylabel('Predicted prices (k$)')
    plt.title('Validation: Predictions vs Actual Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Error distribution
    plt.subplot(1, 3, 3)
    errors = np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten())
    plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute error (k$)')
    plt.ylabel('Frequency')
    plt.title('Validation error distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Final metrics
    train_loss_final = network.train_losses[-1]
    val_loss_final = network.val_losses[-1]
    mae_val = np.mean(np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten()))
    mae_test = np.mean(np.abs(Y_test_denorm.flatten() - Y_test_pred.flatten()))

    print(f"\n{'='*50}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*50}")
    print(f"Normalized train loss : {train_loss_final:.6f}")
    print(f"Normalized validation loss: {val_loss_final:.6f}")
    print(f"MAE validation: {mae_val:.2f}k$")
    print(f"MAE test: {mae_test:.2f}k$")
    print(f"{'='*50}")
    network.cleanNetwork()  # Reset the network for the next optimizer