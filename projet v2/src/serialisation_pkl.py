import cloudpickle


def serialize_pkl(data, name, mode):
    """Serialize any Python object to a pickle file."""
    with open(f"{name}.pkl", mode) as f:
        cloudpickle.dump(data, f)
    
def deserialize_pkl(name):
    """Load a Python object from a pickle file."""
    with open(f"{name}.pkl",'rb') as f:
        return cloudpickle.load(f)