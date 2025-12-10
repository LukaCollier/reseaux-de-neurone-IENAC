import json
import numpy as np


def serialise(name,data):
    """
    name=nom sans extension du fichier que l'on veut créer où on mettra les données stockées
    data=donnée à stocker 
    """
    with open(f"{name}.json",'w') as f:
        json.dump(data,f,indent=4)

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

def test():
    x=np.array([0,5,4,9])
    x=encode_numpy(x)
    x=decode_numpy(x)
    print(x)
    print(type(x))
#test()