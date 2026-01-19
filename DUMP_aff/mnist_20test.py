import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import src.Neurone as Neurone
import src.Activation as Activation
import os

# --- preparation of datas ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

X_train = train_data.iloc[:, 1:].values / 255.0
y_train_labels = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test_labels = test_data.iloc[:, 0].values

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_train = one_hot_encode(y_train_labels)
y_test = one_hot_encode(y_test_labels)

print(f"Données d'entraînement : {X_train.shape[0]} exemples")
print(f"Données de test : {X_test.shape[0]} exemples")

# --- Paramètres ---
epochs = 20
batch_size = 128
lr = 0.0005
lrSGD = 0.0005
N_runs = 20
n_classes = 10

optimizers = {
    "ADAM": ("train_ADAM", lr, "blue"),
    "RMSProp": ("train_RMS", lr, "green"),
    "SGD": ("train_SGD", lrSGD, "orange")
}

# Stockage global de toutes les courbes de validation
all_val_curves_global = []

# =============================
# Boucle par optimiseur
# =============================
for opt_name, (opt_func, opt_lr, color) in optimizers.items():
    print(f"\n==============================")
    print(f"Analyse de convergence : {opt_name}")
    print(f"==============================")

    all_val_curves = []

    for run in range(N_runs):
        print(f"Run {run+1}/{N_runs}")

        # Initialisation du réseau
        relu = Activation.ActivationF.relu()
        softmax = Activation.ActivationF.softmax()
        nn = Neurone.Neural_Network(784, [128, 10], [relu, softmax], loss="cross_entropy")

        # Entraînement epoch par epoch pour collecter les losses
        for epoch in range(epochs):
            current_lr = lrSGD if opt_name == "SGD" else lr
            getattr(nn, opt_func)(X_train, y_train, epochs=1, lr=current_lr,
                                  batch_size=batch_size, x_val=X_test, y_val=y_test)

        all_val_curves.append(nn.val_losses.copy())
        all_val_curves_global.extend([(nn.val_losses.copy(), color, opt_name)])

    # ===================== Graphique moyen par optimiseur =====================
    mean_val = np.mean(all_val_curves, axis=0)
    best_epoch = np.argmin(mean_val)

    plt.figure(figsize=(12, 6))
    for curve in all_val_curves:
        plt.plot(curve, color='gray', alpha=0.3)
    plt.plot(mean_val, color=color, linewidth=3, label=f"{opt_name} - Validation moyenne")
    plt.axvline(best_epoch, color='black', linestyle=':', linewidth=2,
                label=f"Epoch optimal ≈ {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title(f"Stabilité et convergence de {opt_name} (20 runs)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Epoch optimal moyen {opt_name} ≈ {best_epoch}")
    print(f"Loss validation minimale moyenne = {mean_val[best_epoch]:.6f}")

# ===================== Graphique global des 60 courbes =====================
plt.figure(figsize=(13, 8))
for curve, color, name in all_val_curves_global:
    plt.plot(curve, color=color, alpha=0.25)

# Pour légende
plt.plot([], [], color='blue', label='ADAM (20 runs)')
plt.plot([], [], color='green', label='RMSProp (20 runs)')
plt.plot([], [], color='orange', label='SGD (20 runs)')

plt.xlabel("Epoch")
plt.ylabel("Loss validation (Cross-Entropy)")
plt.title("Comparaison globale des 60 courbes de validation (MNIST)")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
