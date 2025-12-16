import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import src.Neurone as Neurone
import src.Activation as Activation
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
data = pd.read_csv("framingham_clean.csv")


X = data.drop(columns=['age']).values.T
Y = data['age'].values.reshape(1, -1)

# Split optimisé: 70% train, 15% validation, 15% test
# Première étape: séparer 30% pour validation+test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X.T, Y.T, test_size=0.3, random_state=42, shuffle=True
)

# Deuxième étape: diviser les 30% en 50% validation et 50% test (15% chacun)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42, shuffle=True
)

print(f"Dataset split:")
print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X.T)*100:.1f}%)")
print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X.T)*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X.T)*100:.1f}%)")

# Normalisation des features (recommandé pour la régression)
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)
X_train_norm = (X_train - mean_X) / (std_X + 1e-8)
X_val_norm = (X_val - mean_X) / (std_X + 1e-8)
X_test_norm = (X_test - mean_X) / (std_X + 1e-8)

# Normalisation de Y (recommandé pour MSE)
mean_Y = Y_train.mean()
std_Y = Y_train.std()
Y_train_norm = (Y_train - mean_Y) / std_Y
Y_val_norm = (Y_val - mean_Y) / std_Y
Y_test_norm = (Y_test - mean_Y) / std_Y

# Paramètres
Epoch = 500
lr = 0.001
batch_size = 32

# NE PAS transposer X et Y - les garder en format (n_samples, n_features)
# Le code train_ADAM attend ce format pour l'indexation par batch_indices

# Créer le réseau avec la bonne taille d'entrée
activ_Relu = Activation.ActivationF.relu()
activ_idd = Activation.ActivationF.identity()

network = Neurone.Neural_Network(X_train_norm.shape[1], [16, 8, 1], [activ_Relu, activ_Relu, activ_idd])

# Entraîner avec les données NON transposées
network.train_ADAM(X_train_norm, Y_train_norm, Epoch, lr, batch_size, X_val_norm, Y_val_norm)

# Prédictions sur l'ensemble de validation
Y_val_pred_norm = network.forward(X_val_norm)
# Dénormaliser les prédictions
Y_val_pred = Y_val_pred_norm * std_Y + mean_Y
Y_val_denorm = Y_val_norm * std_Y + mean_Y

# Prédictions sur l'ensemble de test
Y_test_pred_norm = network.forward(X_test_norm)
Y_test_pred = Y_test_pred_norm * std_Y + mean_Y
Y_test_denorm = Y_test_norm * std_Y + mean_Y

# Visualisation
plt.figure(figsize=(15, 5))

# Graphique 1: Courbes de perte
plt.subplot(1, 3, 1)
plt.plot(network.train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(network.val_losses, label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Courbes de perte (ADAM)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Graphique 2: Prédictions vs Valeurs réelles (Validation)
plt.subplot(1, 3, 2)
plt.scatter(Y_val_denorm.flatten(), Y_val_pred.flatten(), alpha=0.6, color='blue', edgecolors='k', s=50)
plt.plot([Y_val_denorm.min(), Y_val_denorm.max()], 
         [Y_val_denorm.min(), Y_val_denorm.max()], 
         'r--', linewidth=2, label='Prédiction parfaite')
plt.xlabel('Âge réel (années)')
plt.ylabel('Âge prédit (années)')
plt.title('Validation: Prédictions vs Réalité')
plt.legend()
plt.grid(True, alpha=0.3)

# Graphique 3: Distribution des erreurs
plt.subplot(1, 3, 3)
errors = np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten())
plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Erreur absolue (années)')
plt.ylabel('Fréquence')
plt.title('Distribution des erreurs de validation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Métriques finales
train_loss_final = network.train_losses[-1]
val_loss_final = network.val_losses[-1]
mae_val = np.mean(np.abs(Y_val_denorm.flatten() - Y_val_pred.flatten()))
mae_test = np.mean(np.abs(Y_test_denorm.flatten() - Y_test_pred.flatten()))

print(f"\n{'='*50}")
print(f"RÉSULTATS FINAUX")
print(f"{'='*50}")
print(f"Loss d'entraînement (normalisée): {train_loss_final:.6f}")
print(f"Loss de validation (normalisée): {val_loss_final:.6f}")
print(f"MAE Validation: {mae_val:.2f} années")
print(f"MAE Test: {mae_test:.2f} années")
print(f"{'='*50}")