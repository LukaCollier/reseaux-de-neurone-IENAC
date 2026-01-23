import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import src.Neuron as Neurone
import src.Activation as Activation
import os

# --- data preparation ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load training and test data from separate files
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# Extract features and labels for training
X_train = train_data.iloc[:, 1:].values / 255.0
y_train_labels = train_data.iloc[:, 0].values

# Extract features and labels for testing
X_test = test_data.iloc[:, 1:].values / 255.0
y_test_labels = test_data.iloc[:, 0].values

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_train = one_hot_encode(y_train_labels)
y_test = one_hot_encode(y_test_labels)

print(f"Training data: {X_train.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")

# --- initialization ---
relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy",
    name_regularisation='L0',
    lambda_regularisation=0,
    flag=False
)

epochs = 20  # 15
batch_size = 64  # 64
lr = 0.0005  # 0.001
lrSGD = 0.0005  # 0.001
lambda_regularisation = 1e-3

colors = {
    "ADAM_L0": "blue",
    "ADAM_L1": "green",
    "ADAM_L2": "red"
}

lambdas = {
    "ADAM_L0": 0,
    "ADAM_L1": lambda_regularisation,
    "ADAM_L2": lambda_regularisation
}

regularisations = {
    "ADAM_L0": "L0",
    "ADAM_L1": "L1",
    "ADAM_L2": "L2"
}

names = ["ADAM_L0", "ADAM_L1", "ADAM_L2"]
nn_list = [nn.copy_with_regularisation_changes(regularisations[name], lambdas[name]) for name in names]

optimizers = {f"ADAM_L{i}": nn_list[i].train_ADAM for i in range(len(nn_list))}

# Dictionary to store EVERYTHING: curves + predictions
results = {}
n_classes = 10

# =============================
# TRAINING AND EVALUATION FOR EACH OPTIMIZER
# =============================
for i, name in enumerate(names):
    train_func = optimizers[name]
    print(f"\n{'='*60}")
    print(f"TRAINING AND EVALUATION - {name}")
    print(f"{'='*60}")
    
    nn_bis = nn_list[i]
    train_losses = []
    test_losses = []

    # Training
    print("Training in progress...")
    for epoch in range(epochs):
        lr_current = lrSGD if name == "SGD" else lr

        train_func(
            X_train, y_train,
            epochs=1,
            lr=lr_current,
            batch_size=batch_size,
            x_val=X_test,
            y_val=y_test
        )
        
        train_losses.append(nn_bis.train_losses[-1])
        test_losses.append(nn_bis.val_losses[-1])
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}")

    # Immediate evaluation after training
    print("\nEvaluation on the test set...")
    predictions = []
    true_labels = []

    for idx in range(X_test.shape[0]):
        pred = nn.forward(X_test[idx].reshape(1, -1))
        pred_label = np.argmax(pred)
        true_label = np.argmax(y_test[idx])
        predictions.append(pred_label)
        true_labels.append(true_label)

    # Compute global accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100
    print(f"Final accuracy on test set: {accuracy:.2f}%")

    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(true_labels, predictions):
        cm[true_label, pred_label] += 1

    # Store EVERYTHING in results
    results[name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'predictions': predictions,
        'true_labels': true_labels,
        'cm': cm,
        'accuracy': accuracy
    }

# =============================
# PLOTS
# =============================

# 1. LOSS CURVES
print("\n" + "="*60)
print("DISPLAYING PLOTS")
print("="*60)

plt.figure(figsize=(10, 6))
plt.title("Loss comparison – Adam / RMSprop / SGD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

for name in optimizers.keys():
    plt.plot(results[name]['train_losses'], label=f"{name} - Train", color=colors[name])
    plt.plot(results[name]['test_losses'], label=f"{name} - Test", color=colors[name], linestyle="--")

plt.legend()
plt.tight_layout()
plt.show()

# 2. CONFUSION MATRICES FOR EACH OPTIMIZER
for name in optimizers.keys():
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS - {name}")
    print(f"{'='*60}")
    
    cm = results[name]['cm']
    predictions = results[name]['predictions']
    true_labels = results[name]['true_labels']
    accuracy = results[name]['accuracy']
    
    print(f"\nFinal accuracy: {accuracy:.2f}%")
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

    # Visualization of confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Confusion matrices - {name} (Accuracy: {accuracy:.2f}%)', fontsize=16, fontweight='bold')

    # Confusion matrix with absolute values
    im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[0].set_xticks(range(n_classes))
    axes[0].set_yticks(range(n_classes))
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True class')
    axes[0].set_title('Confusion matrix (absolute values)')

    # Add values inside cells
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)

    plt.colorbar(im1, ax=axes[0], label='Number of predictions')

    # Normalized confusion matrix (row-wise)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(n_classes):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            cm_normalized[i, :] = cm[i, :] / row_sum

    im2 = axes[1].imshow(cm_normalized, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_xticks(range(n_classes))
    axes[1].set_yticks(range(n_classes))
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True class')
    axes[1].set_title('Confusion matrix (normalized)')

    # Add values inside cells
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color=text_color)

    plt.colorbar(im2, ax=axes[1], label='Proportion')
    plt.tight_layout()
    plt.show()

    # Analysis of most frequent confusions
    print("\n" + "="*50)
    print("Top 10 most frequent confusions:")
    print("="*50)

    # Use defaultdict to count confusions
    confusion_dict = defaultdict(int)
    for true_label, pred_label in zip(true_labels, predictions):
        if true_label != pred_label:
            confusion_dict[(true_label, pred_label)] += 1

    # Sort by frequency
    sorted_confusions = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)

    for idx, ((true_class, pred_class), count) in enumerate(sorted_confusions[:10], 1):
        percentage = count / cm[true_class, :].sum() * 100
        print(f"{idx:2d}. True: {true_class} → Predicted: {pred_class} | {count:4d} times ({percentage:5.2f}%)")

print("\n" + "="*60)
print("ANALYSIS COMPLETED")
print("="*60)
