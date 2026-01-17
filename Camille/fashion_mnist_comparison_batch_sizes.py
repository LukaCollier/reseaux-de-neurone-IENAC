import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import src.Neurone as Neuron
import src.Activation as Activation

# --- Préparation des données ---
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

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full_encoded, test_size=0.2, random_state=42
)

# --- Initialisation du réseau ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neuron.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy"
)

epochs = 15
lr = 0.001
batch_sizes = [64, 1] 

results = {}
n_classes = 10

for batch_size in batch_sizes:
    print(f"\n{'='*60}")
    print(f"Training Adam - batch_size={batch_size}")
    print(f"{'='*60}")

    nn.cleanNetwork()
    train_losses = []
    val_losses = []

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

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

    # Évaluation sur validation
    predictions = []
    true_labels = []

    for idx in range(X_val.shape[0]):
        pred = nn.forward(X_val[idx].reshape(1, -1))
        pred_label = np.argmax(pred)
        true_label = np.argmax(y_val[idx])
        predictions.append(pred_label)
        true_labels.append(true_label)

    accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")

    results[batch_size] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracy': accuracy
    }

# --- Results display ---
plt.figure(figsize=(10, 6))
plt.title("Comparison of losses - Adam with different batch sizes")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

colors = {64: "blue", 1: "orange"}

for batch_size in batch_sizes:
    plt.plot(results[batch_size]['train_losses'], label=f"Train - batch={batch_size}", color=colors[batch_size])
    plt.plot(results[batch_size]['val_losses'], label=f"Val - batch={batch_size}", color=colors[batch_size], linestyle="--")

plt.legend()
plt.tight_layout()
plt.show()
