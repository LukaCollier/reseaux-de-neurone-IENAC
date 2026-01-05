import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# AMÉLIORATION 1: Architecture plus profonde avec plus de neurones
nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128,64, 10],  # Plus de couches et plus de neurones
    activ=[relu, relu, softmax],
    loss="cross_entropy"
)

# AMÉLIORATION 2: Plus d'epochs et learning rate adapté
epochs = 30  # Plus d'epochs
batch_size = 64  # Batch size plus grand pour stabilité
lr = 0.0005  # Learning rate adapté

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

# AMÉLIORATION 3: Sauvegarder le meilleur modèle
#print("\nSauvegarde du modèle...")
#nn.serialise_pkl("mnist_best_model", mode='w')

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

# ---------- Évaluation finale sur le jeu de validation ----------
print("\nÉvaluation finale sur le jeu de validation...")
res = 0
predictions = []
true_labels = []

for idx in range(X_val.shape[0]):
    pred = nn.forward(X_val[idx].reshape(1, -1))
    pred_label = np.argmax(pred)
    true_label = np.argmax(y_val[idx])
    predictions.append(pred_label)
    true_labels.append(true_label)
    if pred_label == true_label:
        res += 1

accuracy = res/X_val.shape[0]*100
print(f"\n{'='*50}")
print(f"Précision finale : {accuracy:.2f}%")
print(f"Erreurs : {X_val.shape[0] - res}/{X_val.shape[0]}")
print(f"{'='*50}\n")

# AMÉLIORATION 5: Matrice de confusion pour voir les erreurs
from collections import Counter
errors_by_class = {i: 0 for i in range(10)}
total_by_class = {i: 0 for i in range(10)}

for true_label, pred_label in zip(true_labels, predictions):
    total_by_class[true_label] += 1
    if true_label != pred_label:
        errors_by_class[true_label] += 1

print("Précision par classe :")
for i in range(10):
    class_acc = (total_by_class[i] - errors_by_class[i]) / total_by_class[i] * 100
    print(f"  Chiffre {i}: {class_acc:.2f}% ({total_by_class[i] - errors_by_class[i]}/{total_by_class[i]})")