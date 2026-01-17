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

# Chargement des données d'entraînement et de test depuis des fichiers séparés
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# Extraction des features et labels pour l'entraînement
X_train = train_data.iloc[:, 1:].values / 255.0
y_train_labels = train_data.iloc[:, 0].values

# Extraction des features et labels pour le test
X_test = test_data.iloc[:, 1:].values / 255.0
y_test_labels = test_data.iloc[:, 0].values

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_train = one_hot_encode(y_train_labels)
y_test = one_hot_encode(y_test_labels)

print(f"Données d'entraînement : {X_train.shape[0]} exemples")
print(f"Données de test : {X_test.shape[0]} exemples")

# --- initialisation ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy",
    name_regularisation='L0',
    lambda_regularisation=0,
    flag=True
)


epochs =  20 #15
batch_size = 128 #64
lr = 0.0005 #0.001
lrSGD = 0.0005 #0.001
lambda_regularisation_0=1e-3
lambda_regularisation_1=1e-2
lambda_regularisation_2=1e-1

colors = {
    f"ADAM_L2_{lambda_regularisation_0}": "blue",
    f"ADAM_L2_{lambda_regularisation_1}": "green",
    f"ADAM_L2_{lambda_regularisation_2}": "red"
}

lambdas = {
    f"ADAM_L2_{lambda_regularisation_0}": lambda_regularisation_0,
    f"ADAM_L2_{lambda_regularisation_1}": lambda_regularisation_1,
    f"ADAM_L2_{lambda_regularisation_2}": lambda_regularisation_2
}

regularisations={
    f"ADAM_L2_{lambda_regularisation_0}": "L2",
    f"ADAM_L2_{lambda_regularisation_1}": "L2",
    f"ADAM_L2_{lambda_regularisation_2}": "L2"
}

names=[f"ADAM_L2_{lambda_regularisation_0}",f"ADAM_L2_{lambda_regularisation_1}",f"ADAM_L2_{lambda_regularisation_2}"]
nn_list=[nn.copy_with_regularisation_changes(regularisations[name],lambdas[name]) for name in names]

optimizers = {names[i]: nn_list[i].train_ADAM for i in range(len(nn_list))}
# Dictionnaire pour stocker TOUT : courbes + prédictions
results = {}
n_classes = 10

# =============================
# ENTRAÎNEMENT ET ÉVALUATION POUR CHAQUE OPTIMISEUR
# =============================
for i,name in enumerate(names):
    train_func=optimizers[name]
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT ET ÉVALUATION - {name}")
    print(f"{'='*60}")
    
    nn_bis=nn_list[i]
    train_losses = []
    test_losses = []

    # Entraînement
    print(f"Entraînement en cours...")
    for epoch in range(epochs):
        lr_current = lrSGD if name == "SGD" else lr

        train_func(
            X_train, y_train,
            epochs=1,
            lr=lr_current,
            batch_size=batch_size,
            x_val=X_test,
            y_val=y_test
        )
        
        train_losses.append(nn_bis.train_losses[-1])
        test_losses.append(nn_bis.val_losses[-1])
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}")

    # Évaluation immédiate après l'entraînement
    print("\nÉvaluation sur le jeu de test...")
    predictions = []
    true_labels = []

    for idx in range(X_test.shape[0]):
        pred = nn.forward(X_test[idx].reshape(1, -1))
        pred_label = np.argmax(pred)
        true_label = np.argmax(y_test[idx])
        predictions.append(pred_label)
        true_labels.append(true_label)

    # Calcul de la précision globale
    accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100
    print(f"Précision finale sur le test set : {accuracy:.2f}%")

    # Création de la matrice de confusion
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(true_labels, predictions):
        cm[true_label, pred_label] += 1

    # Stocker TOUT dans results
    results[name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'predictions': predictions,
        'true_labels': true_labels,
        'cm': cm,
        'accuracy': accuracy
    }
nn
# =============================
# AFFICHAGE DES GRAPHIQUES
# =============================

# 1. GRAPHIQUE DES LOSSES
print("\n" + "="*60)
print("AFFICHAGE DES GRAPHIQUES")
print("="*60)

plt.figure(figsize=(10, 6))
plt.title("Comparaison des losses – Adam / RMSprop / SGD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

for name in optimizers.keys():
    plt.plot(results[name]['train_losses'], label=f"{name} - Train", color=colors[name])
    plt.plot(results[name]['test_losses'], label=f"{name} - Test", color=colors[name], linestyle="--")

plt.legend()
plt.tight_layout()
plt.show()

# 2. MATRICES DE CONFUSION POUR CHAQUE OPTIMISEUR
for name in optimizers.keys():
    print(f"\n{'='*60}")
    print(f"RÉSULTATS DÉTAILLÉS - {name}")
    print(f"{'='*60}")
    
    cm = results[name]['cm']
    predictions = results[name]['predictions']
    true_labels = results[name]['true_labels']
    accuracy = results[name]['accuracy']
    
    print(f"\nPrécision finale : {accuracy:.2f}%")
    print(f"Erreurs : {len(predictions) - int(accuracy * len(predictions) / 100)}/{len(predictions)}")

    # Calcul des métriques par classe
    print("\nMétriques par classe :")
    print(f"{'Classe':<8} {'Précision':<12} {'Rappel':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        print(f"{i:<8} {precision*100:>10.2f}% {recall*100:>10.2f}% {f1:>10.4f} {support:>10}")

    # Visualisation de la matrice de confusion
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Matrices de confusion - {name} (Précision: {accuracy:.2f}%)', fontsize=16, fontweight='bold')

    # Matrice de confusion avec valeurs absolues
    im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[0].set_xticks(range(n_classes))
    axes[0].set_yticks(range(n_classes))
    axes[0].set_xlabel('Prédiction')
    axes[0].set_ylabel('Vraie classe')
    axes[0].set_title('Matrice de confusion (valeurs absolues)')

    # Ajouter les valeurs dans les cellules
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)

    plt.colorbar(im1, ax=axes[0], label='Nombre de prédictions')

    # Matrice de confusion normalisée (par ligne)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(n_classes):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            cm_normalized[i, :] = cm[i, :] / row_sum

    im2 = axes[1].imshow(cm_normalized, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_xticks(range(n_classes))
    axes[1].set_yticks(range(n_classes))
    axes[1].set_xlabel('Prédiction')
    axes[1].set_ylabel('Vraie classe')
    axes[1].set_title('Matrice de confusion (normalisée)')

    # Ajouter les valeurs dans les cellules
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color=text_color)

    plt.colorbar(im2, ax=axes[1], label='Proportion')
    plt.tight_layout()
    plt.show()

    # Analyse des confusions les plus fréquentes
    print("\n" + "="*50)
    print("Top 10 des confusions les plus fréquentes :")
    print("="*50)

    # Utiliser defaultdict pour compter les confusions
    confusion_dict = defaultdict(int)
    for true_label, pred_label in zip(true_labels, predictions):
        if true_label != pred_label:
            confusion_dict[(true_label, pred_label)] += 1

    # Trier par fréquence
    sorted_confusions = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)

    for idx, ((true_class, pred_class), count) in enumerate(sorted_confusions[:10], 1):
        percentage = count / cm[true_class, :].sum() * 100
        print(f"{idx:2d}. Vrai: {true_class} → Prédit: {pred_class} | {count:4d} fois ({percentage:5.2f}%)")

print("\n" + "="*60)
print("ANALYSE TERMINÉE")
print("="*60)