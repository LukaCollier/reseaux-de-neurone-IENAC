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

mean_Y = Y_train.mean()
std_Y = Y_train.std()
Y_train_norm = (Y_train - mean_Y) / std_Y
Y_val_norm = (Y_val - mean_Y) / std_Y

# ===================== Paramètres =====================
Epoch = 100
lr = 0.001
lrSGD = 0.0001
batch_size = 32
N_runs = 20

optimizers = {
    "ADAM": ("train_ADAM", lr, "red"),
    "RMSProp": ("train_RMS", lr, "green"),
    "SGD": ("train_SGD", lrSGD, "blue")
}

all_val_curves_global = []

# ===================== Boucle par optimiseur =====================
for opt_name, (opt_func, opt_lr, color) in optimizers.items():
    print(f"\nAnalyse de convergence : {opt_name}")

    all_val_curves = []
    all_train_curves = []

    for run in range(N_runs):
        print(f"Run {run+1}/{N_runs}")

        activ_Relu = Activation.ActivationF.relu()
        activ_idd = Activation.ActivationF.identity()
        network = Neurone.Neural_Network(X_train_norm.shape[1], [16, 8, 1],
                                         [activ_Relu, activ_Relu, activ_idd])

        getattr(network, opt_func)(X_train_norm, Y_train_norm,
                                   Epoch, opt_lr, batch_size,
                                   X_val_norm, Y_val_norm)

        all_train_curves.append(network.train_losses.copy())
        all_val_curves.append(network.val_losses.copy())
        all_val_curves_global.append((network.val_losses.copy(), color, opt_name))

    # ===================== Courbe moyenne =====================
    mean_val = np.mean(all_val_curves, axis=0)
    mean_train = np.mean(all_train_curves, axis=0)
    best_epoch = np.argmin(mean_val)

    # ===================== Graphique individuel =====================
    plt.figure(figsize=(12, 7))

    for i in range(N_runs):
        plt.plot(all_val_curves[i], color='gray', alpha=0.3)

    plt.plot(mean_val, color=color, linewidth=3, label=f"{opt_name} - Validation moyenne")
    plt.plot(mean_train, color=color, linestyle='--', linewidth=2, label=f"{opt_name} - Train moyen")
    plt.axvline(best_epoch, color='black', linestyle=':', linewidth=2,
                label=f"Epoch optimal ≈ {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Stabilité et convergence de {opt_name} (20 initialisations)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Epoch optimal moyen {opt_name} ≈ {best_epoch}")
    print(f"Loss val minimale moyenne = {mean_val[best_epoch]:.6f}")

# ===================== GRAPHIQUE GLOBAL : 60 COURBES =====================
plt.figure(figsize=(13, 8))

for curve, color, name in all_val_curves_global:
    plt.plot(curve, color=color, alpha=0.25)

plt.plot([], [], color='red', label='ADAM (20 runs)')
plt.plot([], [], color='green', label='RMSProp (20 runs)')
plt.plot([], [], color='blue', label='SGD (20 runs)')

plt.xlabel("Epoch")
plt.ylabel("Loss validation (MSE)")
plt.title("Comparaison globale des 60 convergences (ADAM vs RMSProp vs SGD)")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
