import json
import numpy as np


def serialise(name,data,mode='x'):
    """
    name=nom sans extension du fichier que l'on veut créer où on mettra les données stockées
    data=donnée à stocker 
    """
    try: #à voir s'il vaut mieux utiliser le mode 'w' en prenant le risque d'effacer des données ou si on conserve le mode 'x' 
        with open(f"{name}.json",mode) as f: #ouverture en 'x' par mesure de sécurité pour éviter que des données utiles ne soient écrasées
            json.dump(data,f,indent=4)
    except FileExistsError:
        print("Le fichier existe déja, il ne peut donc pas être modifié (mesure de sécurité pour ne pas écraser des données utiles)")

def encode_numpy(arr): 
    """
    arr=np.array type en entrée, renvoie un dictionnaire 
    """
    return {
        "__type__": "numpy.ndarray",
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "data": arr.tolist()
    }

def decode_numpy(d): 
    """
    d=dictionnaire en entrée  
    """
    if d.get("__type__") == "numpy.ndarray":
        arr = np.array(d["data"], dtype=d["dtype"])
        return arr.reshape(d["shape"])
    return d

def deserialise(name): 
    """
    name=nom du fichier sans extension dont on veut extraire les infos 
    """
    with open(f"{name}.json",'r') as f:
        data=json.load(f)
        return data
