import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.Neuron_regularisation_etienne as Neuron
import src.Activation as Activation

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
train_data = pd.read_csv("fashion-mnist_train.csv")
test_data = pd.read_csv("fashion-mnist_test.csv")

X_train_full = train_data.iloc[:, 1:].values / 255.0
y_train_full = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_train_full_encoded = one_hot_encode(y_train_full)
y_test_encoded = one_hot_encode(y_test)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full_encoded, test_size=0.2, random_state=42
)

relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

# Network with smart initialization (He / Xavier)
nn_smart = Neuron.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy",
    change_init=False   # <- smart initialization
)

# Network with Gaussian initialization
nn_plain = Neuron.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy",
    change_init=True    # <- plain initialization = gaussian initialization
)
epochs = 15
batch_size = 64
lr = 0.001

nn_smart.train_ADAM(X_train, y_train, epochs, lr, batch_size, x_val=X_val, y_val=y_val)
nn_plain.train_ADAM(X_train, y_train, epochs, lr, batch_size, x_val=X_val, y_val=y_val)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(nn_smart.train_losses, label="Adam He/Xavier - Train", color="blue")
plt.plot(nn_smart.val_losses, label="Adam He/Xavier - Val", color="blue", linestyle="--")
plt.plot(nn_plain.train_losses, label="Adam simple - Train", color="red")
plt.plot(nn_plain.val_losses, label="Adam simple - Val", color="red", linestyle="--")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparison: smart initialization vs gaussian initialization")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# For the smart initialized network
y_pred_smart = nn_smart.forward(X_val)
y_pred_labels_smart = np.argmax(y_pred_smart, axis=0)
y_true_labels = np.argmax(y_val, axis=1)
accuracy_smart = (y_pred_labels_smart == y_true_labels).mean() * 100
print(f"Precision Adam He/Xavier: {accuracy_smart:.2f}%")

# For the plainly initialized network
y_pred_plain = nn_plain.forward(X_val)
y_pred_labels_plain = np.argmax(y_pred_plain, axis=0)
accuracy_plain = (y_pred_labels_plain == y_true_labels).mean() * 100
print(f"Precision Adam gaussian: {accuracy_plain:.2f}%")
