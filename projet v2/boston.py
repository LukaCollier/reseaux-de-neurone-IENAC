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

# Split: 60% train, 20% validation, 20% test
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
Epoch = 400
lr = 0.001
lrSGD = 0.0001
batch_size = 32

# ===================== Réseau =====================
activ_Relu = Activation.ActivationF.relu()
activ_idd = Activation.ActivationF.identity()
network = Neurone.Neural_Network(X_train_norm.shape[1], [16, 8, 1], [activ_Relu, activ_Relu, activ_idd])

optimizers = {
    "ADAM": network.train_ADAM,
    "RMS": network.train_RMS,
    "SGD": network.train_SGD
}

# Pour le graphique comparatif final
all_train_losses = {}
all_val_losses = {}

# ===================== Boucle sur les optimiseurs =====================
for name, train_func in optimizers.items():
    network.cleanNetwork()
    print(f"\nEntraînement avec {name}...")

    if name == "SGD":
        train_func(X_train_norm, Y_train_norm, Epoch, lrSGD, batch_size, X_val_norm, Y_val_norm)
    else:
        train_func(X_train_norm, Y_train_norm, Epoch, lr, batch_size, X_val_norm, Y_val_norm)

    # Sauvegarde des losses pour le graphique final
    all_train_losses[name] = network.train_losses.copy()
    all_val_losses[name] = network.val_losses.copy()

    # ===================== Prédictions =====================
    Y_val_pred_norm = network.forward(X_val_norm)
    Y_val_pred = Y_val_pred_norm * std_Y + mean_Y
    Y_val_denorm = Y_val_norm * std_Y + mean_Y

    Y_test_pred_norm = network.forward(X_test_norm)
    Y_test_pred = Y_test_pred_norm * std_Y + mean_Y
    Y_test_denorm = Y_test_norm * std_Y + mean_Y

    # ===================== Visualisation par optimiseur =====================
    plt.figure(figsize=(15, 5))

    # 1) Courbes de loss
    plt.subplot(1, 3, 1)
    plt.plot(network.train_losses, label='Train Loss')
    plt.plot(network.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Courbes de perte ({name})')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2) Prédictions vs Réalité
    plt.subplot(1, 3, 2)
    plt.scatter(Y_val_denorm.flatten(), Y_val_pred.flatten(), alpha=0.6, edgecolors='k')
    plt.plot([Y_val_denorm.min(), Y_val_denorm.max()],
             [Y_val_denorm.min(), Y_val_denorm.max()],
             'r--', label='Parfait')
    plt.xlabel('Prix réels (k$)')
    plt.ylabel('Prix prédits (k$)')
    plt.title('Validation: Prédictions vs Réalité')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3) Histogramme des erreurs
    plt.subplot(1, 3, 3)
    errors = np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten())
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Erreur absolue (k$)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des erreurs')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ===================== Métriques =====================
    print(f"\n{'='*50}")
    print(f"RÉSULTATS FINAUX ({name})")
    print(f"{'='*50}")
    print(f"Loss train (norm): {network.train_losses[-1]:.6f}")
    print(f"Loss val   (norm): {network.val_losses[-1]:.6f}")
    print(f"MAE Val : {np.mean(errors):.2f} k$")
    print(f"MAE Test: {np.mean(np.abs(Y_test_denorm.flatten() - Y_test_pred.flatten())):.2f} k$")
    print(f"{'='*50}")

    network.cleanNetwork()

# ===================== GRAPHIQUE COMPARATIF FINAL =====================
plt.figure(figsize=(12, 7))

for name in optimizers.keys():
    plt.plot(all_train_losses[name], label=f"{name} - Train")
    plt.plot(all_val_losses[name], linestyle='--', label=f"{name} - Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Comparaison des optimiseurs : ADAM vs RMSProp vs SGD")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
