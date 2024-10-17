import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# continuous version of gates
def AND(inputs):
    return np.prod(inputs)

def OR(inputs):
    return 1 - np.prod(1 - inputs)

def XOR(inputs): #not sure if this is the correct description of XOR (continuous)
    return np.sin(np.pi * np.sum(inputs))**2

#-----------------------------
gates = [AND, OR, XOR]


# Parameters
N = 200 # generating N² neurons
N_iter = 20  # num of iterations
S = np.random.rand(N, N) # ∈ [0, 1]

ε = 1
Φ = np.zeros((N, N), dtype=int)

radius = 2.5
σ = 3 #std for gaussian

"""
if mean of Φ >= ε -> update gates; gate_update = True
"""
gate_update = False

"""
compare condition between Φ and ε
If True -> Ensemble when Φ >= ε
If False -> Ensemble when Φ == ε
"""
geq_cond = False
#-----------------------------

# gaussian weights
x, y = np.indices((N, N)) - N // 2
gaussian_weights = np.exp(-((x**2 + y**2) / (2 * σ**2)))
gaussian_weights /= np.sum(gaussian_weights) #normalize


# init plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
mat1 = ax1.imshow(S, cmap="gray", vmin=0, vmax=1)
mat2 = ax2.imshow(Φ, cmap="hot", vmin=0, vmax=N_iter)
ax1.set_title(f"Threshold (ε): {ε}")
ax1.axis("off")
ax2.set_title("Φ")
ax2.axis("off")
cbar2 = fig.colorbar(mat2, ax=ax2)
cbar2.set_label("Synchronization Count (Φ)")

# random local assignment of gates
gate = np.random.choice(gates, (N, N))

#-----------------------------
def update(frame, *args):
    global S, Φ, ε, gate

    print(f"Iteration {frame + 1}/{N_iter}")

    if gate_update:
        if np.mean(Φ) >= ε:
            gate = np.random.choice(gates, (N, N))

    new_state = np.zeros(S.shape)
    
    # state update with gaussian weights
    for i in range(N):
        for j in range(N):
            #shift gaussian
            shifted_gaussian = np.roll(np.roll(gaussian_weights, i - N // 2, axis=0), j - N // 2, axis=1)
            weighted_inputs = S * shifted_gaussian #weigh given S
            new_state[i, j] = gate[i, j](weighted_inputs)

    # Synchronization update
    sync = np.isclose(new_state, S, atol=1e-6)
    Φ = np.where(sync, Φ + 1, 0)

    # Ensemble formation condition
    mask_ensemble = Φ >= ε if geq_cond else Φ == ε
    S = new_state

    # Update neighbors based on ensemble formation
    if np.any(mask_ensemble):
        ensemble_idxs = np.argwhere(mask_ensemble)
        for i, j in ensemble_idxs:
            shifted_gaussian = np.roll(np.roll(gaussian_weights, i - N // 2, axis=0), j - N // 2, axis=1)
            weights = shifted_gaussian / np.sum(shifted_gaussian) #normalize
            S = np.where(shifted_gaussian >= 0, S * (1 - weights) + S[i, j] * weights, S)

    mat1.set_array(S)
    mat2.set_array(Φ)

    ax1.set_title(f"Threshold (ε): {ε}; R: {radius}; σ: {σ}, Update gates: {gate_update}")
    return mat1, mat2

ani = FuncAnimation(fig, update, frames=N_iter, interval=1000)
ani.save("autopoietic_net.gif", writer="pillow", fps=10)
print("Finished!")
#plt.show()