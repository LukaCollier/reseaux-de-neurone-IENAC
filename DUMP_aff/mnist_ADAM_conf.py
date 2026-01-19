import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.Neurone as Neurone
import src.Activation as Activation
import os

# --- préparation des données ---
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

n_classes = 10
epochs = 20
batch_size = 128
lr = 0.0005
N_runs = 20

# =====================
# Matrice de confusion moyenne ADAM
# =====================
all_confusions = []

for run in range(N_runs):
    print(f"Run {run+1}/{N_runs} - ADAM")

    relu = Activation.ActivationF.relu()
    softmax = Activation.ActivationF.softmax()
    nn = Neurone.Neural_Network(784, [128, 10], [relu, softmax], loss="cross_entropy")

    # Entraînement complet
    for epoch in range(epochs):
        nn.train_ADAM(X_train, y_train, epochs=1, lr=lr, batch_size=batch_size, x_val=X_test, y_val=y_test)

    # Prédictions
    predictions = np.argmax(nn.forward(X_test) if hasattr(nn, 'forward_batch') else np.array([nn.forward(x.reshape(1,-1)) for x in X_test]), axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Matrice de confusion normalisée
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(true_labels, predictions):
        cm[t, p] += 1
    # Normalisation par ligne (classe)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    all_confusions.append(cm_norm)

# Matrice moyenne
cm_mean = np.mean(all_confusions, axis=0)

# =====================
# Affichage
# =====================
plt.figure(figsize=(10, 8))
plt.imshow(cm_mean, cmap='YlOrRd', vmin=0, vmax=1)
plt.colorbar(label="Proportion")
plt.title("Matrice de confusion moyenne - ADAM (MNIST, 20 runs)")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.xticks(np.arange(n_classes))
plt.yticks(np.arange(n_classes))

# Ajouter les valeurs dans les cellules
for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm_mean[i, j] > 0.5 else 'black'
        plt.text(j, i, f'{cm_mean[i, j]:.2f}', ha='center', va='center', color=text_color)

plt.tight_layout()
plt.show()
print("\n" + "="*60)