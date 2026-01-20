import json
import numpy as np


def serialise(name, data, mode='x'):
    """
    name: filename without extension where data will be stored.
    data: object to store.
    """
    try:  # keep 'x' for safety to avoid overwriting useful data
        with open(f"{name}.json", mode) as f:  # open in 'x' to avoid accidental overwrite
            json.dump(data, f, indent=4)
    except FileExistsError:
        print("File already exists; it cannot be modified (safety to avoid overwriting useful data)")

def encode_numpy(arr):
    """
    arr: np.array input, returns a dictionary for JSON serialization.
    """
    return {
        "__type__": "numpy.ndarray",
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "data": arr.tolist()
    }

def decode_numpy(d):
    """
    d: dictionary input; reconstruct numpy array if encoded.
    """
    if d.get("__type__") == "numpy.ndarray":
        arr = np.array(d["data"], dtype=d["dtype"])
        return arr.reshape(d["shape"])
    return d

def deserialise(name):
    """
    name: filename without extension to load JSON data from.
    """
    with open(f"{name}.json", 'r') as f:
        data = json.load(f)
        return data
