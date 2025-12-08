import neuroneopti
import Activation
import numpy as np
import matplotlib.pyplot as plt
'''
Problème de régression reprennant les paramètres de l'exemple du pdf

'''
activ_tanh = Activation.ActivationF.tanh()
activ_leaky_relu = Activation.ActivationF.leaky_relu()
network=neuroneopti.Neural_Network(1,[16,16,1],[activ_tanh,activ_leaky_relu,activ_tanh])


'''
Entrainement NeuralNetwork
'''
Nb_v_entr=500
Epoch=2000
lr=5e-2
v_entr=np.random.uniform(0,2*np.pi,Nb_v_entr)
res_th=np.sin(v_entr)

for j in range(Epoch):
    print(j)
    for i in range(Nb_v_entr):
        network.forward([v_entr[i]])
        network.backward([res_th[i]],lr)
    
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

