"""
Includes a neighborhood introduced by a gaussian,
instead of Moore neighborhood.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


"""
Defining gates, behaviour needs
to be changed
"""
def AND(inputs):
    return np.all(inputs == 1).astype(int)

def OR(inputs):
    return np.any(inputs == 1).astype(int)

def XOR(inputs):
    return (np.sum(inputs) % 2).astype(int)

#----------------------------


gates = [AND, OR, XOR] #more interesting behaviour without neg gates

#Compute entropy of the net
def H(S):
	# S: state matrix
    counts = np.unique(S, return_counts=True)[1]
    p = counts/(N**2)
    return -np.sum(p * np.log(p))

#----------------------------
"""
Params and conditions
"""
N = 200 #neuron count -> N² neurons generated
N_iter = 20 #number of iterations
S = np.random.choice((0,1), size = (N, N)) #init state
fix = True #to have ε fixed
ε_fixed = 1#if ε fixed 
k = 4 #If not fixed -> used denominator -> At max ε -> N_iter/k
Φ = np.zeros((N, N), dtype=int) #To keep track of synchronization at each neuron/ensemble


radius = 2.5 #radius for consideration

"""
compare condition between Φ and ε
If True -> Ensemble when Φ >= ε
If False -> Ensemble when Φ == ε
"""
geq_cond = False 
#----------------------------


# Dynamics for ε
def dynamics(*args, fixed = False):
    if fixed: #fixed assignment of ε
        return ε_fixed
    else:
        return int((N_iter * (1 - H(S)))/k)

ε = dynamics(fixed = fix) #init ε
        

"""
Init plot
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
mat1 = ax1.imshow(S, cmap="gray", vmin=0, vmax=1)
mat2 = ax2.imshow(Φ, cmap="hot", vmin=0, vmax=N_iter)
ax1.set_title(f"Threshold (ε): {ε}")
ax1.axis("off")
ax2.set_title("Φ")
ax2.axis("off")
cbar2 = fig.colorbar(mat2, ax=ax2)
cbar2.set_label("Synchronization Count (Φ)")


#if to have individual gate assignment for each neuron
#only at start doesn't update over iterations
gate = np.random.choice(gates, (N, N))

def update(frame, *args):
    global S, Φ, ε, gate

    print(f"Iteration {frame + 1}/{N_iter}")

    # reallocate gate choice again (local-wise)
    # if synchronization count too high
    """
    if np.mean(Φ) >= ε:
        gate = np.random.choice(gates, (N, N))
    """
    #------
    #choose a gate randomly for each iteration (globally)
    #gate = np.random.choice(gates) 
    #------
    x, y = np.indices(S.shape)
    new_state = np.zeros(S.shape)

    d_mask = np.sqrt((x - N//2)**2 + (y - N//2)**2) <= radius #distance mask 


    #state update
    for i in range(N):
        for j in range(N):
            # shift the distance mask
            mask = np.roll(np.roll(d_mask, i - N//2, axis = 0), j - N//2, axis = 1)
            new_state[i, j] = gate[i, j](S[mask]) #local
            #new_state[i, j] = gate(S[mask]) #global


    sync = (new_state == S)
    Φ[sync] += 1 
    Φ[~sync] = 0

    if geq_cond:
        mask_ensemble = (Φ >= ε)
    else:
        mask_ensemble = (Φ == ε)
    ε = dynamics(fixed = fix)
    S = new_state #update state
    # update given any ensemble formation
    if np.any(mask_ensemble):
        ensemble_idxs = np.argwhere(mask_ensemble)
        for i, j in ensemble_idxs:
        	#update neighbors given central neuron forming ensemble
            S[np.roll(np.roll(d_mask, i - N//2, axis = 0), j - N//2, axis = 1)] = S[i, j]

    mat1.set_array(S)
    mat2.set_array(Φ)

    ax1.set_title(f"Threshold (ε): {ε}; Fixed (k: {k if not fix else None}): {fix}; Φ $\geq$ ε: {geq_cond}, R: {radius}")
    return mat1, mat2

ani = FuncAnimation(fig, update, frames=N_iter, interval=1000)
ani.save("autopoietic_net.gif", writer="pillow", fps=10)
print("Finished!")
#plt.show()
