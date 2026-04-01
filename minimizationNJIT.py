import numpy as np
from matplotlib import pyplot as plt
from numba import njit,jit

# Load crosslink positions
positions = np.loadtxt("crosslinks-positions.txt")

# Load connection table (first column = ID, next columns = connected IDs)
connections = np.loadtxt("connected_xlinks.txt", skiprows=1, delimiter="\t", dtype=int)


@njit
def func(positions, connections, L=256, max_steps=1000, tolerance=1e-4):
    npos = len(positions)
    box = np.array([L, L, L], dtype=positions.dtype)

    step = 0
    max_disp = np.inf

    # preallocate index array to use for shuffling (Fisher-Yates fallback)
    indices = np.arange(npos, dtype=np.int64)

    while max_disp > tolerance and step < max_steps:
        max_disp = 0.0

        indices = np.random.permutation(npos)   
        # main loop
        for ii in range(npos):
            i = indices[ii]
            pos = positions[i] % box
            sum_delta = np.zeros(3, dtype=positions.dtype)
            count = 0

            # iterate over connection table
            # connections assumed shape (nbonds, bond_size) with zeros meaning no connection
            nbonds = connections.shape[0]
            bsize = connections.shape[1]
            for jb in range(nbonds):
                for k in range(bsize):
                    conn_val = connections[jb, k]
                    if conn_val == 0:
                        continue
                    # if this bond contains i (1-based IDs in file)
                    if conn_val - 1 == i:
                        # add contributions from other members of the bond
                        for other in range(bsize):
                            conn2 = connections[jb, other]
                            if conn2 == 0 or other == k:
                                continue
                            idx_other = conn2 - 1
                            conn_pos = positions[idx_other] % box
                            delta = conn_pos - pos
                            # minimum-image
                            delta = (delta + 0.5 * box) % box - 0.5 * box
                            # elementwise accumulate
                            sum_delta[0] += delta[0]
                            sum_delta[1] += delta[1]
                            sum_delta[2] += delta[2]
                            count += 1

            # compute COM safely as an array of same dtype as positions
            if count == 0:
                COM0 = 0.0
                COM1 = 0.0
                COM2 = 0.0
            else:
                invc = 1.0 / count
                COM0 = sum_delta[0] * invc
                COM1 = sum_delta[1] * invc
                COM2 = sum_delta[2] * invc

            # new_pos computed from pos and COM (modulo box)
            new_pos0 = (pos[0] + COM0) % box[0]
            new_pos1 = (pos[1] + COM1) % box[1]
            new_pos2 = (pos[2] + COM2) % box[2]

            # assign back elementwise so Numba sees scalar stores (safe)
            positions[i, 0] = new_pos0
            positions[i, 1] = new_pos1
            positions[i, 2] = new_pos2

            # compute displacement length (norm of COM)
            disp = np.sqrt(COM0 * COM0 + COM1 * COM1 + COM2 * COM2)
            if disp > max_disp:
                max_disp = disp

        step += 1
        print("Step", step," max displacement = ",max_disp)
        print("")

    return 1
        
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='blue')
plt.savefig("NIJTinitial_positions.png", dpi=300)
if False:
    for i, j in bonds:
        x = [positions[i-1,0], positions[j-1,0]]
        y = [positions[i-1,1], positions[j-1,1]]
        z = [positions[i-1,2], positions[j-1,2]]
        ax.plot(x, y, z, color='red') 
plt.title("Initial state")
plt.show()
func(positions,connections)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='blue')
plt.savefig("NIJTequilibrated_positions.png", dpi=300)
if False:
    for i, j in bonds:
        x = [positions[i-1,0], positions[j-1,0]]
        y = [positions[i-1,1], positions[j-1,1]]
        z = [positions[i-1,2], positions[j-1,2]]
        ax.plot(x, y, z, color='red') 
plt.title("Equilibrium state")
plt.show()
np.savetxt("NIJTequilibrated_positions.txt", positions, fmt="%.8f")