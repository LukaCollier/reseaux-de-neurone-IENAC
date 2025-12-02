import neuroneopti
import Activation
import numpy as np
import matplotlib.pyplot as plt
'''
Problème de régression reprennant les paramètres de l'exemple du pdf

'''
activ_tanh = Activation.ActivationF.tanh()
activ_leaky_relu = Activation.ActivationF.leaky_relu()
#network=neuroneopti.Neural_Network(1,[16,16,1],[activ_tanh,activ_tanh,activ_tanh])
network = neuroneopti.Neural_Network(1, [8, 8, 1], [activ_tanh, activ_tanh, activ_tanh])
'''
Entrainement NeuralNetwork
'''
#Nb_v_entr=500
Nb_v_entr = 2000  # au lieu de 500
Epoch=500 #au lieu de 2000
lr=1e-2
v_entr=np.random.uniform(0,2*np.pi,Nb_v_entr)
res_th=np.sin(v_entr)
train_losses = []
val_losses = []
for k in range(Epoch):
    if k % 100 == 0:
        print(f"Epoch {k}/{Epoch}")
    
    # Mélanger les données à chaque epoch
    indices = np.random.permutation(Nb_v_entr)
    epoch_loss = 0
    num_batches = 0
    # Parcourir par mini-batches de 10
    for i in range(0, Nb_v_entr, 8):
        # Extraire le batch (max 10 éléments)
        end_idx = min(i + 16, Nb_v_entr)
        batch_indices = indices[i:end_idx]
        x_batch = v_entr[batch_indices]
        y_batch = res_th[batch_indices]
        # Forward et backward sur le batch
        network.forward(x_batch)
        network.backward(y_batch, lr)
        # Calculer la perte pour ce batch
        epoch_loss += network.MSE(network.a[-1].reshape(1,-1), y_batch.reshape(1, -1))
        num_batches += 1
    
    # Perte moyenne d'entraînement pour cette epoch
    train_loss = epoch_loss / num_batches
    train_losses.append(train_loss)

# Tracer la courbe de perte
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Courbe de perte pendant l\'entraînement')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Échelle log pour mieux voir l'évolution
    
'''
création du test de vérification

'''
x_test =np.random.uniform(0,2*np.pi,2000)
res_test=np.sin(x_test)
res_nn_test=np.array([network.forward([x])[0] for x in x_test])


plt.figure(figsize=(10, 6))
plt.plot(x_test, res_test, 'o', label="np.sin", color="blue", markersize=2)
plt.plot(x_test, res_nn_test, 'o', label="NeuralNetwork", color="red", markersize=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
