
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import src.Neurone as Neurone
import src.Activation as Activation
import os

# --- preparation of datas ---
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

# --- initialisation ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy"
)

# --- training parameters ---
epochs = 50
batch_size = 32
lr = 0.001
lrSGD = 0.0001

optimizers = {
    "ADAM": nn.train_ADAM,
    "RMS": nn.train_RMS,
    "SGD": nn.train_SGD
}

colors = {
    "ADAM": "blue",
    "RMS": "green",
    "SGD": "orange"
}

# =============================
# FIGURE LOSS DYNAMIQUE
# =============================
plt.ion()
plt.figure(figsize=(10, 6))
plt.title("Comparaison des losses – Adam / RMSprop / SGD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

for name, train_func in optimizers.items():
    nn.cleanNetwork()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        lr_current = lrSGD if name == "SGD" else lr

        train_func(
            X_train, y_train,
            epochs=1,
            lr=lr_current,
            batch_size=batch_size,
            x_val=X_val,
            y_val=y_val
        )

        train_losses.append(nn.train_losses[-1])
        val_losses.append(nn.val_losses[-1])

    plt.plot(train_losses, label=f"{name} - Train", color=colors[name])
    plt.plot(val_losses, label=f"{name} - Val", color=colors[name], linestyle="--")

plt.legend()
plt.ioff()
plt.tight_layout()
plt.show()
 
