import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import src.Neurone as Neurone
import src.Activation as Activation
import os
from sklearn.model_selection import train_test_split

# --- preparation of datas ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Chargement des données CIFAR-10...")
train_data = pd.read_csv("train.csv")

# Train: dernière colonne = label
X_train_full = train_data.iloc[:, :-1].values / 255.0
y_train_full_labels = train_data.iloc[:, -1].values

print(f"\nShape après extraction:")
print(f"  X_train_full: {X_train_full.shape}")
print(f"  y_train_full: {y_train_full_labels.shape}")

# Créer un split train/validation à partir des données d'entraînement
X_train, X_val, y_train_labels, y_val_labels = train_test_split(
    X_train_full, y_train_full_labels, 
    test_size=0.2,  # 20% pour validation
    random_state=42,
    stratify=y_train_full_labels  # Garder la distribution des classes
)

print(f"\n=== Split Train/Validation ===")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")

def one_hot_encode(y, n_classes=10):
    y = np.clip(y.astype(int), 0, n_classes - 1)
    return np.eye(n_classes)[y]

y_train = one_hot_encode(y_train_labels)
y_val = one_hot_encode(y_val_labels)

cifar_classes = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 
                 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Vérification de la distribution des classes
print("\n=== Distribution des classes (Train) ===")
for i in range(10):
    count = np.sum(y_train_labels == i)
    percentage = count / len(y_train_labels) * 100
    print(f"{i} - {cifar_classes[i]:12s}: {count:5d} ({percentage:5.2f}%)")

print("\n=== Distribution des classes (Validation) ===")
for i in range(10):
    count = np.sum(y_val_labels == i)
    percentage = count / len(y_val_labels) * 100
    print(f"{i} - {cifar_classes[i]:12s}: {count:5d} ({percentage:5.2f}%)")

# --- initialisation ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=3072,
    nb_n_l=[512, 256, 10],
    activ=[relu, relu, softmax],
    loss="cross_entropy"
)

epochs = 30
batch_size = 128
lr = 0.001

print(f"\n{'='*60}")
print(f"ENTRAÎNEMENT - ADAM")
print(f"{'='*60}")
print(f"Architecture: 3072 -> 512 -> 256 -> 10")
print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
print(f"{'='*60}\n")

nn.cleanNetwork()
train_losses = []
val_losses = []

# Entraînement
for epoch in range(epochs):
    nn.train_ADAM(
        X_train, y_train,
        epochs=1,
        lr=lr,
        batch_size=batch_size,
        x_val=X_val,
        y_val=y_val
    )

    train_losses.append(nn.train_losses[-1])
    val_losses.append(nn.val_losses[-1])
    
    # Calcul accuracy toutes les 5 epochs
    if epoch % 5 == 0 or epoch == epochs - 1:
        # Prédiction sur un échantillon pour éviter les problèmes de mémoire
        sample_size = min(1000, len(X_val))
        sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
        
        pred_list = []
        true_list = []
        for idx in sample_indices:
            pred = nn.forward(X_val[idx].reshape(1, -1))
            pred_list.append(np.argmax(pred))
            true_list.append(np.argmax(y_val[idx]))
        
        accuracy_epoch = np.mean(np.array(pred_list) == np.array(true_list)) * 100
        
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_losses[-1]:.6f} | "
              f"Val Loss: {val_losses[-1]:.6f} | Val Acc: {accuracy_epoch:.2f}%")

# Évaluation finale sur validation
print(f"\n{'='*60}")
print("ÉVALUATION FINALE SUR VALIDATION")
print(f"{'='*60}")

predictions = []
true_labels_list = []

for idx in range(X_val.shape[0]):
    pred = nn.forward(X_val[idx].reshape(1, -1))
    pred_label = np.argmax(pred)
    true_label = np.argmax(y_val[idx])
    predictions.append(pred_label)
    true_labels_list.append(true_label)

accuracy = np.mean(np.array(predictions) == np.array(true_labels_list)) * 100
print(f"\nPrécision finale sur validation : {accuracy:.2f}%")

# Matrice de confusion
n_classes = 10
cm = np.zeros((n_classes, n_classes), dtype=int)
for true_label, pred_label in zip(true_labels_list, predictions):
    cm[true_label, pred_label] += 1

# =============================
# AFFICHAGE DES GRAPHIQUES
# =============================

# 1. GRAPHIQUE DES LOSSES
plt.figure(figsize=(10, 6))
plt.title("Évolution des losses – ADAM (CIFAR-10)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

plt.plot(train_losses, label="Train", color="blue", linewidth=2)
plt.plot(val_losses, label="Validation", color="blue", linestyle="--", linewidth=2)

plt.legend()
plt.tight_layout()
plt.show()

# 2. MÉTRIQUES PAR CLASSE
print("\n=== Métriques par classe ===")
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

# 3. MATRICE DE CONFUSION
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Matrices de confusion (Précision: {accuracy:.2f}%)', fontsize=16, fontweight='bold')

# Valeurs absolues
im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
axes[0].set_xticks(range(n_classes))
axes[0].set_yticks(range(n_classes))
axes[0].set_xticklabels(cifar_classes, rotation=45, ha='right')
axes[0].set_yticklabels(cifar_classes)
axes[0].set_xlabel('Prédiction')
axes[0].set_ylabel('Vraie classe')
axes[0].set_title('Matrice de confusion (valeurs absolues)')

for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color, fontsize=8)

plt.colorbar(im1, ax=axes[0], label='Nombre de prédictions')

# Normalisée
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

for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=8)

plt.colorbar(im2, ax=axes[1], label='Proportion')
plt.tight_layout()
plt.show()

# 4. TOP CONFUSIONS
print("\n" + "="*50)
print("Top 10 des confusions les plus fréquentes :")
print("="*50)

confusion_dict = defaultdict(int)
for true_label, pred_label in zip(true_labels_list, predictions):
    if true_label != pred_label:
        confusion_dict[(true_label, pred_label)] += 1

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

n_examples = 15
indices = np.random.choice(len(X_val), n_examples, replace=False)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle(f'Exemples de prédictions - Validation (Accuracy: {accuracy:.2f}%)', 
             fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx >= n_examples:
        ax.axis('off')
        continue
    
    val_idx = indices[idx]
    
    img_data = X_val[val_idx][:3072]
    img_rgb = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
    
    pred_label = predictions[val_idx]
    true_label = true_labels_list[val_idx]
    
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