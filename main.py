import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gates import *

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
N = 500 #neuron count -> N² neurons generated
N_iter = 20 #number of iterations
S = np.random.choice((0,1), size = (N, N)) #init state
fix = False #to have ε fixed
ε_fixed = 15#if ε fixed
k = 7 #If not fixed -> used denominator -> At max ε -> N_iter/k
Φ = np.zeros((N, N), dtype=int) #To keep track of synchronization at each neuron/ensemble
extended = True #Considering also diagonal neighbors

"""
compare condition between Φ and ε
If True -> Ensemble when Φ >= ε
If False -> Ensemble when Φ == ε
"""
geq_cond = True 
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

def update(*args):
    global S, Φ, ε
    #------
    #choose a gate randomly for each iteration (globally)
    #gate = np.random.choice(gates) 
    #------
    if extended:
        #Getting neighbors by shifting matrix
        left = np.roll(S, -1, axis = 1)
        right = np.roll(S, 1, axis = 1)
        up = np.roll(S, 1, axis = 0)
        down = np.roll(S, -1, axis = 0)
        upper_left = np.roll(np.roll(S, 1, axis = 0), -1, axis = 1)
        upper_right = np.roll(np.roll(S, 1, axis = 0), 1, axis = 1)
        bottom_left = np.roll(np.roll(S, -1, axis = 0), -1, axis = 1)
        bottom_right = np.roll(np.roll(S, -1, axis = 0), 1, axis = 1)

        # stack neighbor states
        neighbors = np.stack([left, right, up, down, upper_left,
         upper_right, bottom_left, bottom_right], axis = 0)


    else:
        #Getting neighbors by shifting matrix
        left = np.roll(S, -1, axis = 1)
        right = np.roll(S, 1, axis = 1)
        up = np.roll(S, 1, axis = 0)
        down = np.roll(S, -1, axis = 0)

        # stack neighbor states
        neighbors = np.stack([left, right, up, down], axis = 0)

    #return new state
    #new_state = gate(neighbors) #if for global gate assignment

    #for individual assignment of gates
    new_state = np.array([[gate[i, j](neighbors[:, i, j]) for j in range(N)] for i in range(N)])


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
            if extended: #all neighbors(including diagonals)
                S[(i-1) % N, (j-1) % N] = S[i, j]  #upper left 
                S[(i-1) % N, (j+1) % N] = S[i, j]  #upper right 
                S[(i+1) % N, (j+1) % N] = S[i, j]  #bottom right 
                S[(i+1) % N, (j-1) % N] = S[i, j]  #bottom left
                S[i, (j-1) % N] = S[i, j]  #left 
                S[i, (j+1) % N] = S[i, j]  #right 
                S[(i-1) % N, j] = S[i, j]  #up 
                S[(i+1) % N, j] = S[i, j]  #down 
                
            else: 
                S[i, (j-1) % N] = S[i, j]  #left 
                S[i, (j+1) % N] = S[i, j]  #right 
                S[(i-1) % N, j] = S[i, j]  #up 
                S[(i+1) % N, j] = S[i, j]  #down 
    mat1.set_array(S)
    mat2.set_array(Φ)

    ax1.set_title(f"Threshold (ε): {ε}; Fixed (k: {k if not fix else None}): {fix}; Φ $\geq$ ε: {geq_cond}; Extended: {extended}")
    return mat1, mat2

ani = FuncAnimation(fig, update, frames=N_iter, interval=1000)
ani.save("autopoietic_net.gif", writer="pillow", fps=10)
#plt.show()