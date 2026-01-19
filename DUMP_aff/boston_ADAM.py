import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import src.Neurone as Neurone
import src.Activation as Activation
from sklearn.model_selection import train_test_split

# ===================== Chargement des données =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
data = pd.read_csv("boston.csv")

X = data.drop(columns=['MEDV']).values
Y = data['MEDV'].values.reshape(-1, 1)

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

# ===================== Normalisation =====================
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)
X_train_norm = (X_train - mean_X) / (std_X + 1e-8)
X_val_norm = (X_val - mean_X) / (std_X + 1e-8)
X_test_norm = (X_test - mean_X) / (std_X + 1e-8)

mean_Y = Y_train.mean()
std_Y = Y_train.std()
Y_train_norm = (Y_train - mean_Y) / std_Y
Y_val_norm = (Y_val - mean_Y) / std_Y
Y_test_norm = (Y_test - mean_Y) / std_Y

# ===================== Paramètres =====================
Epoch = 100
lr = 0.001
batch_size = 32
N_runs = 20

# ===================== Stockage des runs =====================
all_train_curves = []
all_val_curves = []
activ_Relu = Activation.ActivationF.relu()
activ_idd = Activation.ActivationF.identity()
network = Neurone.Neural_Network(X_train_norm.shape[1], [16, 8, 1], [activ_Relu, activ_Relu, activ_idd])
# ===================== Boucle ADAM =====================
for run in range(N_runs):
    print(f"Run ADAM {run+1}/{N_runs}")
    network.cleanNetwork()

    network.train_ADAM(X_train_norm, Y_train_norm, Epoch, lr, batch_size, X_val_norm, Y_val_norm)

    all_train_curves.append(network.train_losses.copy())
    all_val_curves.append(network.val_losses.copy())

# ===================== Moyenne =====================
mean_train = np.mean(all_train_curves, axis=0)
mean_val = np.mean(all_val_curves, axis=0)

# ===================== Graphique convergence =====================
plt.figure(figsize=(12, 7))

# Courbes individuelles (transparentes)
for i in range(N_runs):
    plt.plot(all_val_curves[i], color='gray', alpha=0.3)

# Courbe moyenne
plt.plot(mean_val, color='red', linewidth=3, label="Validation moyenne (20 runs)")
plt.plot(mean_train, color='blue', linewidth=2, linestyle='--', label="Train moyen")

# Époque optimale
best_epoch = np.argmin(mean_val)
plt.axvline(best_epoch, color='black', linestyle=':', linewidth=2, label=f"Epoch optimal ≈ {best_epoch}")

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Stabilité et convergence d'ADAM sur Boston Housing (20 initialisations)")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\n==============================")
print(f"Époque optimale moyenne : {best_epoch}")
print(f"Loss validation minimale moyenne : {mean_val[best_epoch]:.6f}")
print("==============================")
