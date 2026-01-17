import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import src.Neuron_regularisation_etienne as Neuron
import src.Activation as Activation
from src import regularisation as Regularisation

import os

# --- preparation of datas ---
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

# --- initialization ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

epochs = 15
batch_size = 64
lr = 0.001

# Différents lambda pour L1
lambda_values = [1e-5, 1e-4, 1e-3]  

regularizations = {
    f"L1_{lam}": {"name": "L1", "lambda": lam} for lam in lambda_values
}

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

results = {}
n_classes = 10

# =============================
# TRAINING AND EVALUATION FOR EACH REGULARIZATION
# =============================
for reg_label, reg_params in regularizations.items():

    print(f"\n{'='*60}")
    print(f"Training with {reg_label}")
    print(f"{'='*60}")

    regularisation_obj = Regularisation.RegularisationF.creation_with_name(
    name=reg_params["name"],
    lambda_regularisation=reg_params["lambda"]
)

    nn = Neuron.Neural_Network(
    n_input_init=784,
        nb_n_l=[128, 10],
        activ=[relu, softmax],
        loss="cross_entropy",
        regularisation=regularisation_obj
    )
    train_losses = []
    val_losses = []

    # Training loop
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

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Train: {train_losses[-1]:.6f}, Val: {val_losses[-1]:.6f}")

    # Evaluation
    predictions = []
    true_labels = []

    for idx in range(X_val.shape[0]):
        pred = nn.forward(X_val[idx].reshape(1, -1))
        pred_label = np.argmax(pred)
        true_label = np.argmax(y_val[idx])
        predictions.append(pred_label)
        true_labels.append(true_label)

    accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100
    print(f"Final accuracy: {accuracy:.2f}%")

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, predictions):
        cm[t, p] += 1

    results[reg_label] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'true_labels': true_labels,
        'cm': cm,
        'accuracy': accuracy
    }

# =============================
# DISPLAY GRAPHS
# =============================

# Loss plot
plt.figure(figsize=(10, 6))
plt.title("Loss comparison for L1 with different lambda")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

colors = {f"L1_{lam}": col for lam, col in zip(lambda_values, ["blue", "green", "red"])}

for reg_label in regularizations.keys():
    plt.plot(results[reg_label]['train_losses'], label=f"{reg_label} - Train", color=colors[reg_label])
    plt.plot(results[reg_label]['val_losses'], linestyle="--", label=f"{reg_label} - Val", color=colors[reg_label])

plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrices
for reg_label in regularizations.keys():
    cm = results[reg_label]['cm']
    predictions = results[reg_label]['predictions']
    true_labels = results[reg_label]['true_labels']
    accuracy = results[reg_label]['accuracy']

    print(f"\n{'='*60}")
    print(f"Detailed results - {reg_label}")
    print(f"{'='*60}")
    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"Errors: {len(predictions) - int(accuracy * len(predictions) / 100)}/{len(predictions)}")

    # Per-class metrics
    print("\nPer-class metrics:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
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

    # Confusion matrix plot (absolute + normalized)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Confusion matrices - {reg_label} (Accuracy: {accuracy:.2f}%)', fontsize=16, fontweight='bold')

    im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[0].set_xticks(range(n_classes))
    axes[0].set_yticks(range(n_classes))
    axes[0].set_xlabel('Prediction')
    axes[0].set_ylabel('True class')
    axes[0].set_title('Absolute values')
    for i in range(n_classes):
        for j in range(n_classes):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im1, ax=axes[0], label='Predictions number')

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im2 = axes[1].imshow(cm_norm, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_xticks(range(n_classes))
    axes[1].set_yticks(range(n_classes))
    axes[1].set_xlabel('Prediction')
    axes[1].set_ylabel('True class')
    axes[1].set_title('Normalized')
    for i in range(n_classes):
        for j in range(n_classes):
            axes[1].text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.colorbar(im2, ax=axes[1], label='Proportion')
    plt.tight_layout()
    plt.show()
