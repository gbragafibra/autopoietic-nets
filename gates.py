import numpy as np

def AND(inputs):
    return np.all(inputs == 1, axis=0).astype(int)

def OR(inputs):
    return np.any(inputs == 1, axis=0).astype(int)

def NAND(inputs):
    return 1 - AND(inputs)

def NOR(inputs):
    return 1 - OR(inputs)

def XOR(inputs):
    return (np.sum(inputs, axis=0) % 2).astype(int)
    
def XNOR(inputs):
    return 1 - XOR(inputs)

def Tautology(inputs):
    return np.ones((inputs.shape[1], inputs.shape[1]), dtype=int)  # Return 1 for all outputs

def Contradiction(inputs):
    return np.zeros((inputs.shape[1], inputs.shape[1]), dtype=int)  # Return 0 for all outputs