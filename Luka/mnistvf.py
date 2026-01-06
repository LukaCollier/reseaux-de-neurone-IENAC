import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import src.Neuronev2 as Neurone
import src.Activation as Activation
import os

# --- Préparation des données ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

data = pd.read_csv("mnist.csv")
X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_encoded = one_hot_encode(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# --- Initialisation du réseau ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy"
)

# --- Paramètres d'entraînement ---
epochs = 50
batch_size = 32
lr = 0.001
lrSGD = 0.0001

optimizers = {
    "ADAM": nn.train_ADAM,
    "RMS": nn.train_RMS,
    "SGD": nn.train_SGD
}

# Couleurs pour chaque optimisateur
colors = {
    "ADAM": "blue",
    "RMS": "green",
    "SGD": "orange"
}

# --- Figures pour les graphiques dynamiques ---
plt.ion()
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.tight_layout(pad=3.0)
for ax_row in axes:
    for ax in ax_row:
        ax.grid(True, alpha=0.3)

# --- NOUVELLE figure pour la comparaison ---
fig_comp, axes_comp = plt.subplots(1, 3, figsize=(18, 5))
fig_comp.suptitle('Comparaison des 3 optimisateurs', fontsize=16, fontweight='bold')
fig_comp.tight_layout(pad=3.0)
for ax in axes_comp:
    ax.grid(True, alpha=0.3)

# Dictionnaire pour stocker les résultats
all_results = {
    "ADAM": {"train_losses": [], "val_losses": [], "true_labels": None, "pred_labels": None, "errors": None},
    "RMS": {"train_losses": [], "val_losses": [], "true_labels": None, "pred_labels": None, "errors": None},
    "SGD": {"train_losses": [], "val_losses": [], "true_labels": None, "pred_labels": None, "errors": None}
}

# --- Boucle sur les optimisateurs ---
for row_idx, (name, train_func) in enumerate(optimizers.items()):
    nn.cleanNetwork()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        lr_current = lrSGD if name == "SGD" else lr
        train_func(X_train, y_train, epochs=1, lr=lr_current, batch_size=batch_size,
                   x_val=X_val, y_val=y_val)

        train_losses.append(nn.train_losses[-1])
        val_losses.append(nn.val_losses[-1])
        
        # Stocker pour les graphiques de comparaison
        all_results[name]["train_losses"] = train_losses.copy()
        all_results[name]["val_losses"] = val_losses.copy()

        # ---- Graphe 1: Loss dynamique ----
        ax1 = axes[row_idx, 0]
        ax1.clear()
        min_len = min(len(train_losses), len(val_losses))
        ax1.plot(range(min_len), train_losses[:min_len], label="Train Loss", color="blue")
        ax1.plot(range(min_len), val_losses[:min_len], label="Validation Loss", color="red")
        ax1.set_title(f"{name} - Loss dynamique")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ---- Graphe 2: Prédiction vs réalité ----
        ax2 = axes[row_idx, 1]
        ax2.clear()
        Y_val_pred = nn.forward(X_val)
        true_labels = np.argmax(y_val, axis=1)
        pred_labels = np.argmax(Y_val_pred, axis=1)
        
        n_samples = min(len(true_labels), len(pred_labels))
        ax2.scatter(true_labels[:n_samples], pred_labels[:n_samples], alpha=0.6, color="blue", edgecolors="k", s=50)
        ax2.plot([0, 9], [0, 9], "r--", linewidth=2, label="Prédiction parfaite")
        ax2.set_title(f"{name} - Prédiction vs Réalité")
        ax2.set_xlabel("Étiquette vraie")
        ax2.set_ylabel("Étiquette prédite")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ---- Graphe 3: Distribution des erreurs ----
        ax3 = axes[row_idx, 2]
        ax3.clear()
        errors = np.abs(true_labels[:n_samples] - pred_labels[:n_samples])
        ax3.hist(errors, bins=range(11), color="purple", alpha=0.7, edgecolor="black")
        ax3.set_title(f"{name} - Distribution des erreurs")
        ax3.set_xlabel(nn.loss_name)
        ax3.set_ylabel("Fréquence")
        ax3.grid(True, alpha=0.3)
        
        # Stocker les dernières prédictions
        all_results[name]["true_labels"] = true_labels[:n_samples]
        all_results[name]["pred_labels"] = pred_labels[:n_samples]
        all_results[name]["errors"] = errors

        plt.tight_layout(pad=3.0)
        plt.pause(0.01)
    
    # Mise à jour des graphiques de comparaison après chaque optimisateur
    # Graphe comparatif 1: Loss dynamique (Train + Val)
    axes_comp[0].clear()
    for opt_name in optimizers.keys():
        if len(all_results[opt_name]["train_losses"]) > 0:
            axes_comp[0].plot(range(len(all_results[opt_name]["train_losses"])), 
                             all_results[opt_name]["train_losses"], 
                             label=f"{opt_name} - Train", color=colors[opt_name], linewidth=2, linestyle='-')
            axes_comp[0].plot(range(len(all_results[opt_name]["val_losses"])), 
                             all_results[opt_name]["val_losses"], 
                             label=f"{opt_name} - Val", color=colors[opt_name], linewidth=2, linestyle='--', alpha=0.7)
    axes_comp[0].set_title("Loss dynamique - Comparaison", fontsize=14, fontweight='bold')
    axes_comp[0].set_xlabel("Epoch")
    axes_comp[0].set_ylabel("Loss")
    axes_comp[0].set_yscale("log")
    axes_comp[0].legend()
    axes_comp[0].grid(True, alpha=0.3)
    
    # Graphe comparatif 2: Prédiction vs réalité
    axes_comp[1].clear()
    for opt_name in optimizers.keys():
        if all_results[opt_name]["true_labels"] is not None:
            axes_comp[1].scatter(all_results[opt_name]["true_labels"], 
                               all_results[opt_name]["pred_labels"], 
                               alpha=0.4, color=colors[opt_name], s=20, label=opt_name)
    axes_comp[1].plot([0, 9], [0, 9], "r--", linewidth=2, label="Prédiction parfaite")
    axes_comp[1].set_title("Prédiction vs Réalité - Comparaison", fontsize=14, fontweight='bold')
    axes_comp[1].set_xlabel("Étiquette vraie")
    axes_comp[1].set_ylabel("Étiquette prédite")
    axes_comp[1].legend()
    axes_comp[1].grid(True, alpha=0.3)
    
    # Graphe comparatif 3: Distribution des erreurs
    axes_comp[2].clear()
    for opt_name in optimizers.keys():
        if all_results[opt_name]["errors"] is not None:
            axes_comp[2].hist(all_results[opt_name]["errors"], bins=range(11), 
                            color=colors[opt_name], alpha=0.5, edgecolor="black", label=opt_name)
    axes_comp[2].set_title("Distribution des erreurs - Comparaison", fontsize=14, fontweight='bold')
    axes_comp[2].set_xlabel("Erreur")
    axes_comp[2].set_ylabel("Fréquence")
    axes_comp[2].legend()
    axes_comp[2].grid(True, alpha=0.3)
    
    fig_comp.tight_layout()
    plt.pause(0.01)

# --- Affichage dynamique des images ---
fig_img, ax_img = plt.subplots(figsize=(3, 3))
correct_count = 0
nbf = X_val.shape[0]

for idx in range(nbf):
    image = X_val[idx].reshape(28, 28)
    true_label = np.argmax(y_val[idx])
    pred = nn.forward(X_val[idx].reshape(1, -1))
    pred_label = np.argmax(pred)
    if pred_label == true_label:
        correct_count += 1

    ax_img.clear()
    ax_img.imshow(image, cmap="gray")
    ax_img.set_title(f"Image {idx}/{nbf-1}\nVraie: {true_label}, Prédiction: {pred_label}")
    ax_img.axis("off")
    plt.pause(0.01)

plt.ioff()
plt.show()

print(f"Précision sur le jeu de validation : {correct_count/nbf*100:.2f}%")