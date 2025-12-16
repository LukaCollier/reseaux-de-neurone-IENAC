import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import src.Neurone as Neurone
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
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy"
)

# ---------- Matplotlib dynamique ----------
'''
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Évolution dynamique de la loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
'''
train_losses = []
val_losses = []
'''
line_train, = ax.plot([], [], label="train loss")
line_val, = ax.plot([], [], label="val loss")
ax.legend()
'''
# ---------- Entraînement manuel avec mise à jour Matplotlib ----------
epochs = 100
batch_size = 256
lr = 0.001

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

    train_losses.append(nn.train_losses[-1])
    val_losses.append(nn.val_losses[-1])
'''
    # Mise à jour du graphe
    line_train.set_data(range(len(train_losses)), train_losses)
    line_val.set_data(range(len(val_losses)), val_losses)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

plt.ioff()
plt.show()
'''
# Choisir un index au hasard
plt.ion()  # mode interactif

fig, ax = plt.subplots(figsize=(3, 3))

for idx in range(X_val.shape[0]):
    image = X_val[idx].reshape(28, 28)
    true_label = np.argmax(y_val[idx])

    # Prédiction du réseau
    pred = nn.forward(X_val[idx].reshape(1, -1))
    pred_label = np.argmax(pred)

    # Affichage
    ax.clear()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Image {idx}/{X_val.shape[0]-1}\n"
                 f"Étiquette vraie : {true_label}\n"
                 f"Prédiction RN : {pred_label}")
    ax.axis("off")

    plt.pause(2)   # temps entre chaque image

plt.ioff()
plt.show()