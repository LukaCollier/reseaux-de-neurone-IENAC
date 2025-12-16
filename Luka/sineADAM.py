import src.Neurone as Neurone
import src.Activation as Activation
import numpy as np
import matplotlib.pyplot as plt

'''
Problème de régression reprennant les paramètres de l'exemple du pdf
Modifié pour utiliser l'optimiseur ADAM
'''

activ_tanh = Activation.ActivationF.tanh()
activ_leaky_relu = Activation.ActivationF.leaky_relu()

# Création du réseau
network = Neurone.Neural_Network(1, [8, 8, 1], activ_tanh)

'''
Entrainement avec ADAM
'''

Nb_v_entr = 2000
Epoch = 500
lr = 1e-3  # Learning rate généralement plus petit pour ADAM (1e-3 au lieu de 1e-2)
batch_size = 16

# Génération des données d'entraînement
v_entr = np.random.uniform(0, 2*np.pi, Nb_v_entr).reshape(-1, 1)  # Shape: (2000, 1)
res_th = np.sin(v_entr.flatten()).reshape(-1, 1)  # Shape: (2000, 1)

# Génération des données de validation
Nb_v_val = 500
v_val = np.random.uniform(0, 2*np.pi, Nb_v_val).reshape(-1, 1)  # Shape: (500, 1)
res_val = np.sin(v_val.flatten()).reshape(-1, 1)  # Shape: (500, 1)
# CHANGEMENT PRINCIPAL: train_ADAM au lieu de train_SGD
network.train_ADAM(v_entr, res_th, Epoch, lr, batch_size, v_val, res_val)
train_losses = network.train_losses

network.serialise("sine_ADAM_model")

# Tracer la courbe de perte
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

# Graphique 2: Comparaison sur données de validation
plt.subplot(1, 3, 2)
val_pred = network.forward(v_val).flatten()  # forward attend maintenant (n_samples, n_features)
plt.plot(v_val.flatten(), res_val.flatten(), 'o', label="np.sin", color="blue", markersize=4, alpha=0.5)
plt.plot(v_val.flatten(), val_pred, 'o', label="Neural Network", color="red", markersize=4, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Validation: sin(x) vs Neural Network')
plt.legend()
plt.grid(True, alpha=0.3)
# Graphique 3: Distribution des erreurs
plt.subplot(1, 3, 3)
errors = np.abs(res_val.flatten() - val_pred).flatten()  # S'assurer que c'est un tableau 1D
plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Erreur absolue')
plt.ylabel('Fréquence')
plt.title('Distribution des erreurs de validation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# Afficher la loss finale
print(f"\nLoss finale: {train_losses[-1]:.6f}")
