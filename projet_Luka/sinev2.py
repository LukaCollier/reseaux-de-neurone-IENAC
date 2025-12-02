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
network = neuroneopti.Neural_Network(1, [8, 8, 1], activ_tanh)
'''
Entrainement NeuralNetwork
'''
#Nb_v_entr=500
Nb_v_entr = 2000  # au lieu de 500
Epoch=500 #au lieu de 2000
lr=1e-2
v_entr=np.random.uniform(0,2*np.pi,Nb_v_entr)
res_th=np.sin(v_entr)
network.train(v_entr,res_th,Epoch,lr, batch_size=16) #entrainement par batch de taille 16 attention ne pas reshape l'entrée
train_losses = network.train_losses

# Tracer la courbe de perte
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
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
