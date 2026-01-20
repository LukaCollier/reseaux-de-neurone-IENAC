import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import src.Neuron as Neuron
import src.Activation as Activation
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

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full_encoded, test_size=0.2, random_state=42)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

n_classes = 10

# =============================
# LOADING PRE-TRAINED MODEL
# =============================
print(f"{'='*60}")
print(f"Loading pre-trained model - ADAM")
print(f"{'='*60}")

nn = Neuron.Neural_Network.deserialize_pkl("fashion_mnist_adam_model")
print("✓ Model loaded successfully!")

# =============================
# EVALUATION ON VALIDATION SET
# =============================
print("\nEvaluating on validation set...")
predictions = []
true_labels = []

for idx in range(X_val.shape[0]):
    pred = nn.forward(X_val[idx].reshape(1, -1))
    pred_label = np.argmax(pred)
    true_label = np.argmax(y_val[idx])
    predictions.append(pred_label)
    true_labels.append(true_label)

# Calculate overall accuracy
accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100
print(f"Précision finale : {accuracy:.2f}%")

# Create confusion matrix
cm = np.zeros((n_classes, n_classes), dtype=int)
for true_label, pred_label in zip(true_labels, predictions):
    cm[true_label, pred_label] += 1

# =============================
# DETAILED RESULTS
# =============================
print(f"\n{'='*60}")
print(f"Detailed results - ADAM")
print(f"{'='*60}")

print(f"\nFinal accuracy : {accuracy:.2f}%")
print(f"Errors : {len(predictions) - int(accuracy * len(predictions) / 100)}/{len(predictions)}")

# Calculate metrics per class
print("\nPer-class metrics :")
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

# =============================
# CONFUSION MATRIX VISUALIZATION
# =============================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Matrices de confusion - ADAM (Précision: {accuracy:.2f}%)', fontsize=16, fontweight='bold')

# Confusion matrix with absolute values
im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
axes[0].set_xticks(range(n_classes))
axes[0].set_yticks(range(n_classes))
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('True class')
axes[0].set_title('Confusion matrix (absolute values)')

# Add values in cells
for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)

plt.colorbar(im1, ax=axes[0], label='Predictions number')

# Confusion matrix normalized (by row)
cm_normalized = np.zeros_like(cm, dtype=float)
for i in range(n_classes):
    row_sum = cm[i, :].sum()
    if row_sum > 0:
        cm_normalized[i, :] = cm[i, :] / row_sum

im2 = axes[1].imshow(cm_normalized, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
axes[1].set_xticks(range(n_classes))
axes[1].set_yticks(range(n_classes))
axes[1].set_xlabel('Prediction')
axes[1].set_ylabel('True class')
axes[1].set_title('Confusion matrix (normalized)')

# Add values in cells
for i in range(n_classes):
    for j in range(n_classes):
        text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color=text_color)

plt.colorbar(im2, ax=axes[1], label='Proportion')
plt.tight_layout()
plt.show()

# =============================
# CONFUSION ANALYSIS
# =============================
print("\n" + "="*50)
print("Top 10 most frequent confusions :")
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

# =============================
# DYNAMIC DISPLAY OF 500 RANDOM IMAGES
# =============================
print("\n" + "="*60)
print("DYNAMIC DISPLAY - 500 RANDOM IMAGES")
print("="*60)

# Select 500 random indices from validation set
n_display = 500
random_indices = np.random.choice(X_val.shape[0], size=n_display, replace=False)

# Create figure for automatic display
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

def display_image(idx_in_list):
    """Display image at given index in the random selection"""
    ax.clear()
    
    # Get the actual index in validation set
    actual_idx = random_indices[idx_in_list]
    
    # Get image and reshape to 28x28
    img = X_val[actual_idx].reshape(28, 28)
    
    # Get true label and prediction
    true_label = np.argmax(y_val[actual_idx])
    pred = nn.forward(X_val[actual_idx].reshape(1, -1))
    pred_label = np.argmax(pred)
    pred_confidence = np.max(pred) * 100
    
    # Display image
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    
    # Set title with prediction info
    if true_label == pred_label:
        title_color = 'green'
        status = '✓ CORRECT'
    else:
        title_color = 'red'
        status = '✗ WRONG'
    
    title = f"Image {idx_in_list + 1}/{n_display}\n"
    title += f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}\n"
    title += f"Confidence: {pred_confidence:.1f}% - {status}"
    
    ax.set_title(title, fontsize=12, fontweight='bold', color=title_color, pad=20)
    
    plt.pause(2)  # Pause for 2 seconds

print("\nAutomatic display started (2 seconds per image)...")
print("Close the window to stop and continue the program")

plt.ion()  # Enable interactive mode

# Display all 500 images
for idx in range(n_display):
    if not plt.fignum_exists(fig.number):
        print("\nDisplay stopped by user")
        break
    display_image(idx)

plt.ioff()  # Disable interactive mode
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)