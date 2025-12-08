import Neurone
import Activation
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
v_entr = np.random.uniform(0, 2*np.pi, Nb_v_entr)
res_th = np.sin(v_entr)

# CHANGEMENT PRINCIPAL: train_ADAM au lieu de train_SGD
network.train_ADAM(v_entr, res_th, Epoch, lr, batch_size)

train_losses = network.train_losses

# Tracer la courbe de perte
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss (ADAM)', color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Courbe de perte pendant l\'entraînement (ADAM)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

'''
Création du test de vérification
'''
x_test = np.random.uniform(0, 2*np.pi, 2000)
res_test = np.sin(x_test)
res_nn_test = np.array([network.forward([x])[0] for x in x_test])

# Graphique de comparaison
plt.subplot(1, 2, 2)
plt.plot(x_test, res_test, 'o', label="np.sin", color="blue", markersize=2, alpha=0.5)
plt.plot(x_test, res_nn_test, 'o', label="NeuralNetwork (ADAM)", color="red", markersize=2, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparaison sin(x) vs Neural Network')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Afficher la loss finale
print(f"\nLoss finale: {train_losses[-1]:.6f}")
print(f"Erreur moyenne absolue: {np.mean(np.abs(res_test - res_nn_test)):.6f}")