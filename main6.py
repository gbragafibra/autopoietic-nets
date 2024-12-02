"""
Continuous version
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def AND(inputs):
    return np.prod(inputs)

def OR(inputs):
    return 1 - np.prod(1 - inputs)

def XOR(inputs): #not sure if this is the correct description of XOR (continuous)
    return np.sin(np.pi * np.sum(inputs))**2

#----------------------------
gates = [AND, OR, XOR] #without neg gates


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
N_iter = 50 #number of iterations

S = np.zeros((N, N))
S[80:120, 80:120] = np.random.rand() 

#S = np.random.rand(N, N) # ∈ [0, 1] #rand init

fix = True #to have ε fixed
ε_fixed = 2#if ε fixed 
k = 5 #If not fixed -> used denominator -> At max ε -> N_iter/k
Φ = np.zeros((N, N), dtype=int) #To keep track of synchronization at each neuron/ensemble


radius = 2.5#radius for consideration
θ = 1e-10 #precision


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

    """ vectorized code (gets process killed) -> Creating array with N⁴ elements!
    # Can't precompute the masks' shifting
    #array that contains each mask (neighborhood; True or False) for each point
    d_mask_shift = np.array([np.roll(np.roll(d_mask, i - N//2, axis=0), j - N//2, axis=1)
        for i in range(N) for j in range(N)]).reshape(N, N, N, N)

    new_state = np.array([[gate[i, j](S[d_mask_shift[i, j]]) for j in range(N)] for i in range(N)])
    """

    #state update (non-vectorized)
    for i in range(N):
        for j in range(N):
            mask = np.roll(np.roll(d_mask, i - N//2, axis=0), j - N//2, axis=1)
            new_state[i, j] = gate[i, j](S[mask])

    sync = np.isclose(new_state, S, atol=θ) #precision really affects emergent behaviour
    
    Φ = np.where(sync, Φ + 1, 0)

    mask_ensemble = Φ >= ε if geq_cond else Φ == ε
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

    ax1.set_title(f"Threshold (ε): {ε}; Fixed (k: {k if not fix else None}): {fix}; Φ $\geq$ ε: {geq_cond}; R: {radius}; Precision: {θ}")
    return mat1, mat2

ani = FuncAnimation(fig, update, frames=N_iter, interval=1000)
ani.save("autopoietic_net.gif", writer="pillow", fps=10)
print("Finished!")
#plt.show()
