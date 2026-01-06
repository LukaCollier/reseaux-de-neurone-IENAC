import cloudpickle


def serialise_pkl(data,name,mode):
    with open(f"{name}.pkl", mode) as f:
        cloudpickle.dump(data, f)
    
def deserialise_pkl(name):
    with open(f"{name}.pkl",'rb') as f:
        return cloudpickle.load(f)