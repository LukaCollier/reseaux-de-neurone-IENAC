import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import src.Neuronev2 as Neurone
import src.Activation as Activation
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Charger CSV
data = pd.read_csv("mnist.csv")
X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_encoded = one_hot_encode(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 64, 10],
    activ=[relu, relu, softmax],
    loss="cross_entropy"
)

epochs = 30
batch_size = 64
lr = 0.0005

print("Début de l'entraînement...")
nn.train_SGDMomentum(
    x_train=X_train,
    y_train=y_train,
    epochs=epochs,
    lr=lr,
    batch_size=batch_size,
    x_val=X_val,
    y_val=y_val,
    verbose=True
)

# Affichage des courbes de loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(nn.train_losses, label="Train Loss")
plt.plot(nn.val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Évolution de la loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Évaluation finale avec matrice de confusion ----------
print("\nÉvaluation finale sur le jeu de validation...")
predictions = []
true_labels = []

for idx in range(X_val.shape[0]):
    pred = nn.forward(X_val[idx].reshape(1, -1))
    pred_label = np.argmax(pred)
    true_label = np.argmax(y_val[idx])
    predictions.append(pred_label)
    true_labels.append(true_label)

# Calcul de la précision globale
accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100
print(f"\n{'='*50}")
print(f"Précision finale : {accuracy:.2f}%")
print(f"Erreurs : {len(predictions) - int(accuracy * len(predictions) / 100)}/{len(predictions)}")
print(f"{'='*50}\n")

# Création de la matrice de confusion manuellement
n_classes = 10
cm = np.zeros((n_classes, n_classes), dtype=int)

for true_label, pred_label in zip(true_labels, predictions):
    cm[true_label, pred_label] += 1

# Calcul des métriques par classe
print("Métriques par classe :")
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

# Analyse des confusions les plus fréquentes avec collections
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