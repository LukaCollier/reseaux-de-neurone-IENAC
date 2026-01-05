import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

import src.Neurone as Neurone
import src.Activation as Activation

from tensorflow.keras.datasets import fashion_mnist


# Chargement du dataset Fashion-MNIST

(X_full, y_full), (_, _) = fashion_mnist.load_data()

X_full = X_full / 255.0
X_full = X_full.reshape(-1, 784)

def one_hot(y, n=10):
    return np.eye(n)[y]

y_full = one_hot(y_full)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)


# Normalisation centrée

mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std


# Réseau de neurones

relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[256, 128, 10],
    activ=[relu, relu, softmax],
    loss="cross_entropy"
)


#  Entraînement avec Early Stopping

epochs = 200
batch_size = 64
lr = 0.0015

patience = 10
best_val_loss = np.inf
wait = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):

    nn.train_ADAM(
        x_train=X_train,
        y_train=y_train,
        epochs=1,
        lr=lr,
        batch_size=batch_size,
        x_val=X_val,
        y_val=y_val
    )

    train_loss = nn.train_losses[-1]
    val_loss = nn.val_losses[-1]

    #  Accuracy train 
    train_correct = 0
    for i in range(X_train.shape[0]):
        pred = nn.forward(X_train[i].reshape(1, -1))
        if np.argmax(pred) == np.argmax(y_train[i]):
            train_correct += 1
    train_acc = train_correct / X_train.shape[0]

    # Accuracy validation 
    val_correct = 0
    for i in range(X_val.shape[0]):
        pred = nn.forward(X_val[i].reshape(1, -1))
        if np.argmax(pred) == np.argmax(y_val[i]):
            val_correct += 1
    val_acc = val_correct / X_val.shape[0]

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:03d} | "
          f"Train loss: {train_loss:.4f} | "
          f"Val loss: {val_loss:.4f} | "
          f"Train acc: {train_acc*100:.2f}% | "
          f"Val acc: {val_acc*100:.2f}%")

    # Early stopping 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"\n Early stopping à l'epoch {epoch+1}")
            break


# Courbes Loss & Accuracy

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train loss")
plt.plot(epochs_range, val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss - Fashion MNIST")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train accuracy")
plt.plot(epochs_range, val_accs, label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy - Fashion MNIST")
plt.legend()

plt.tight_layout()
plt.show()


# Matrice de confusion

y_true, y_pred = [], []

for i in range(X_val.shape[0]):
    pred = nn.forward(X_val[i].reshape(1, -1))
    y_pred.append(np.argmax(pred))
    y_true.append(np.argmax(y_val[i]))

cm = confusion_matrix(y_true, y_pred)

labels = [
    "T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
    "Sandale", "Chemise", "Basket", "Sac", "Bottine"
]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Prédiction")
plt.ylabel("Vraie classe")
plt.title("Matrice de confusion - Fashion MNIST")
plt.tight_layout()
plt.show()
