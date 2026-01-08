import json
import numpy as np


def serialise(name,data,mode='x'):
    """
    name= Filename without extension of the file to create, where the stored data will be saved
    data=data to be saved 
    """
    try: # To consider whether it’s better to use mode 'w', risking overwriting data, or keep mode 'x'
        with open(f"{name}.json",mode) as f: #Open in 'x' mode as a safety measure to avoid overwriting useful data
            json.dump(data,f,indent=4)
    except FileExistsError:
        print("The file already exists, it cannot be modified (safety measure to avoid overwriting useful data)")

def encode_numpy(arr): 
    """
    arr=np.array type as input, returns a dictionary
    """
    return {
        "__type__": "numpy.ndarray",
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "data": arr.tolist()
    }

def decode_numpy(d): 
    """
    d=dictionnaire as input
    """
    if d.get("__type__") == "numpy.ndarray":
        arr = np.array(d["data"], dtype=d["dtype"])
        return arr.reshape(d["shape"])
    return d

def deserialise(name): 
    """
    name=Filename without extension from which we want to extract the information
    """
    with open(f"{name}.json",'r') as f:
        data=json.load(f)
        return data
