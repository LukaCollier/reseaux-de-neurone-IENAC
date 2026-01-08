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

print("Chargement des données CIFAR-10...")
# Chargement des données d'entraînement et de test depuis des fichiers séparés
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# DEBUG: Afficher les premières lignes pour vérifier la structure
print("\n=== DEBUG: Structure des données ===")
print(f"Shape train_data: {train_data.shape}")
print(f"Shape test_data: {test_data.shape}")
print(f"\nPremières colonnes train_data:\n{train_data.iloc[:5, :5]}")
print(f"\nDernières colonnes train_data:\n{train_data.iloc[:5, -5:]}")

# Vérifier si le label est en première ou dernière colonne
first_col_unique = train_data.iloc[:, 0].unique()
last_col_unique = train_data.iloc[:, -1].unique()

print(f"\nValeurs uniques première colonne: {sorted(first_col_unique)[:15]}")
print(f"Valeurs uniques dernière colonne: {sorted(last_col_unique)[:15]}")

# Déterminer automatiquement où est le label
if len(first_col_unique) <= 10 and all(0 <= x <= 9 for x in first_col_unique):
    print("\n✓ Label détecté en PREMIÈRE colonne (format MNIST)")
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train_labels = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test_labels = test_data.iloc[:, 0].values
elif len(last_col_unique) <= 10 and all(0 <= x <= 9 for x in last_col_unique):
    print("\n✓ Label détecté en DERNIÈRE colonne (format CIFAR-10)")
    X_train = train_data.iloc[:, :-1].values / 255.0
    y_train_labels = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values / 255.0
    y_test_labels = test_data.iloc[:, -1].values
else:
    raise ValueError("Impossible de détecter la colonne des labels!")

print(f"\nShape après extraction:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train_labels: {y_train_labels.shape}")

# Vérification et correction de la dimension (3071 -> 3072)
def fix_shape(X):
    if X.shape[1] == 3071:
        print(f"  ⚠ Correction dimension: {X.shape[1]} -> 3072 (padding)")
        return np.pad(X, ((0, 0), (0, 1)), mode='constant')
    return X

X_train = fix_shape(X_train)
X_test = fix_shape(X_test)

def one_hot_encode(y, n_classes=10):
    y = np.clip(y.astype(int), 0, n_classes - 1)
    return np.eye(n_classes)[y]

y_train = one_hot_encode(y_train_labels)
y_test = one_hot_encode(y_test_labels)

cifar_classes = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 
                 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Mapping officiel CIFAR-10 (pour référence)
# 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer
# 5: dog, 6: frog, 7: horse, 8: ship, 9: truck

print(f"Données d'entraînement : {X_train.shape[0]} exemples")
print(f"Données de test : {X_test.shape[0]} exemples")
print(f"Dimension des images : {X_train.shape[1]} pixels (32x32x3)")

# Vérification de la distribution des classes
print("\n=== Distribution des classes (Train) ===")
for i in range(10):
    count = np.sum(y_train_labels == i)
    percentage = count / len(y_train_labels) * 100
    print(f"{i} - {cifar_classes[i]:12s}: {count:5d} ({percentage:5.2f}%)")

print("\n=== Distribution des classes (Test) ===")
for i in range(10):
    count = np.sum(y_test_labels == i)
    percentage = count / len(y_test_labels) * 100
    print(f"{i} - {cifar_classes[i]:12s}: {count:5d} ({percentage:5.2f}%)")

# Visualiser quelques exemples AVANT l'entraînement pour vérifier
print("\n=== Vérification visuelle de quelques exemples ===")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Vérification des labels AVANT entraînement', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    img_data = X_train[idx][:3072]
    
    # CIFAR-10 peut être stocké de 2 façons:
    # Format 1: R(1024) G(1024) B(1024) puis reshape (3, 32, 32)
    # Format 2: Pixels entrelacés (32, 32, 3)
    
    # Essayons le format standard CIFAR-10
    img_rgb = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
    
    ax.imshow(np.clip(img_rgb, 0, 1))
    label = int(y_train_labels[idx])
    ax.set_title(f"Label: {label} - {cifar_classes[label]}", fontsize=10, weight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Vérification alternative si les couleurs semblent bizarres
print("\n⚠️  Si les couleurs des images ci-dessus semblent incorrectes,")
print("   cela peut indiquer un problème de format de stockage RGB.")

input("\n✓ Vérifiez que les labels correspondent bien aux images. Appuyez sur ENTRÉE pour continuer...")

# --- initialisation ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=3072,  # 32x32x3 = 3072
    nb_n_l=[512, 256, 10],   # Architecture plus profonde pour CIFAR-10
    activ=[relu, relu, softmax],  # ReLU pour les couches cachées
    loss="cross_entropy"
)

epochs = 20  # Plus d'epochs pour CIFAR-10
batch_size = 256  # Batch size plus petit pour meilleure convergence
lr = 0.001   # Learning rate augmenté

optimizers = {
    "ADAM": nn.train_ADAM
}

colors = {
    "ADAM": "blue"
}

# Dictionnaire pour stocker TOUT : courbes + prédictions
results = {}
n_classes = 10

# =============================
# ENTRAÎNEMENT ET ÉVALUATION POUR CHAQUE OPTIMISEUR
# =============================
for name, train_func in optimizers.items():
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT ET ÉVALUATION - {name}")
    print(f"{'='*60}")
    
    nn.cleanNetwork()
    train_losses = []
    test_losses = []

    # Entraînement
    print(f"Entraînement en cours...")
    for epoch in range(epochs):
        train_func(
            X_train, y_train,
            epochs=1,
            lr=lr,
            batch_size=batch_size,
            x_val=X_test,
            y_val=y_test
        )

        train_losses.append(nn.train_losses[-1])
        test_losses.append(nn.val_losses[-1])
        
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

# =============================
# AFFICHAGE DES GRAPHIQUES
# =============================

# 1. GRAPHIQUE DES LOSSES
print("\n" + "="*60)
print("AFFICHAGE DES GRAPHIQUES")
print("="*60)

plt.figure(figsize=(10, 6))
plt.title("Évolution des losses – ADAM (CIFAR-10)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

plt.plot(results['ADAM']['train_losses'], label="Train", color=colors['ADAM'], linewidth=2)
plt.plot(results['ADAM']['test_losses'], label="Test", color=colors['ADAM'], linestyle="--", linewidth=2)

plt.legend()
plt.tight_layout()
plt.show()

# 2. MATRICE DE CONFUSION
name = 'ADAM'
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
print(f"{'Classe':<12} {'Précision':<12} {'Rappel':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 65)

for i in range(n_classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    support = cm[i, :].sum()
    
    print(f"{cifar_classes[i]:<12} {precision*100:>10.2f}% {recall*100:>10.2f}% {f1:>10.4f} {support:>10}")

# Visualisation de la matrice de confusion
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Matrices de confusion - {name} (Précision: {accuracy:.2f}%)', fontsize=16, fontweight='bold')

# Matrice de confusion avec valeurs absolues
im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
axes[0].set_xticks(range(n_classes))
axes[0].set_yticks(range(n_classes))
axes[0].set_xticklabels(cifar_classes, rotation=45, ha='right')
axes[0].set_yticklabels(cifar_classes)
axes[0].set_xlabel('Prédiction')
axes[0].set_ylabel('Vraie classe')
axes[0].set_title('Matrice de confusion (valeurs absolues)')

# Ajouter les valeurs dans les cellules
for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color, fontsize=8)

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
axes[1].set_xticklabels(cifar_classes, rotation=45, ha='right')
axes[1].set_yticklabels(cifar_classes)
axes[1].set_xlabel('Prédiction')
axes[1].set_ylabel('Vraie classe')
axes[1].set_title('Matrice de confusion (normalisée)')

# Ajouter les valeurs dans les cellules
for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=8)

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
    true_name = cifar_classes[true_class]
    pred_name = cifar_classes[pred_class]
    print(f"{idx:2d}. Vrai: {true_name:<10} → Prédit: {pred_name:<10} | {count:4d} fois ({percentage:5.2f}%)")

# =============================
# VISUALISATION D'EXEMPLES
# =============================
print("\n" + "="*60)
print("VISUALISATION D'EXEMPLES DE PRÉDICTIONS")
print("="*60)

print(f"Précision ADAM: {results['ADAM']['accuracy']:.2f}%")

# Visualiser quelques exemples
n_examples = 15
indices = np.random.choice(len(X_test), n_examples, replace=False)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle(f'Exemples de prédictions - ADAM', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx >= n_examples:
        ax.axis('off')
        continue
    
    test_idx = indices[idx]
    
    # Reconstruction de l'image RGB 32x32x3
    img_data = X_test[test_idx][:3072]
    img_rgb = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
    
    pred_label = results['ADAM']['predictions'][test_idx]
    true_label = results['ADAM']['true_labels'][test_idx]
    
    ax.imshow(np.clip(img_rgb, 0, 1))
    
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"V: {cifar_classes[true_label]}\nP: {cifar_classes[pred_label]}", 
                 color=color, fontsize=9, weight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ANALYSE TERMINÉE")
print("="*60)