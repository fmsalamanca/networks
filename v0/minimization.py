import numpy as np
from matplotlib import pyplot as plt
# Load crosslink positions
positions = np.loadtxt("crosslinks-positions.txt")

# Load connection table (first column = ID, next columns = connected IDs)
connections = np.loadtxt("connected_xlinks.txt", skiprows=1, delimiter="\t", dtype=int)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='blue')
plt.savefig("initial_positions.png", dpi=300)
if False:
    for i, j in bonds:
        x = [positions[i-1,0], positions[j-1,0]]
        y = [positions[i-1,1], positions[j-1,1]]
        z = [positions[i-1,2], positions[j-1,2]]
        ax.plot(x, y, z, color='red') 
plt.title("Initial state")
plt.show()

max_steps = 1000
tolerance = 1e-4  # convergence criterion (adjust as needed)

step = 0
max_displacement = np.inf
# box length (scalar) used in many toy sections above; if you have per-axis boxes use an array like [Lx, Ly, Lz]
L = 256

# make box an array for component-wise ops (works if L is scalar or array)
box = np.asarray(L) if np.ndim(L) > 0 else np.array([L, L, L])

while max_displacement > tolerance and step < max_steps:
    max_displacement = 0.0

    # random update order helps Monte Carlo convergence (Fisher-Yates via np.random.permutation)
    indices = np.random.permutation(len(positions))

    for i in indices:  # run over xlinks
        pos = positions[i] % box

        # accumulate neighbor displacements using minimum-image convention
        sum_delta = np.zeros(pos.shape)
        count = 0

        for j in range(len(connections)):
                    # check if i appears in this bond
            for k in range(len(connections[j])):
                if connections[j][k] == 0: #ignore 0 indices -- no connnection
                    continue
                if int(connections[j][k]) - 1 == i:
                            # map the other positions in the bond
                    for other_idx in range(len(connections[j])):
                        if connections[j][other_idx] == 0: #ignore 0 indices -- no connnection
                            continue
                        if other_idx != k:
                            conn_idx = int(connections[j][other_idx]) - 1
                            conn_pos = positions[conn_idx] % box
                            delta = conn_pos - pos
                            delta = (delta + 0.5 * box) % box - 0.5 * box  # minimum-image
                            sum_delta += delta
                            count += 1

                # compute average displacement
        if count == 0:
            COM = np.zeros_like(pos)
        else:
            COM = sum_delta / count

        new_pos = (pos + COM) % box
        positions[i] = new_pos

        displacement = np.linalg.norm(COM)
        if displacement > max_displacement:
            max_displacement = displacement

    step += 1
    print(f"Step {step}: max displacement = {max_displacement:.6e}")
    if step % 10 == 0:
        np.savetxt(f"equilibrated_positions_step{step}.txt", positions, fmt="%.4f")

# optional quick plot of equilibrated positions
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='blue')

if False:
    for i, j in bonds:
        x = [positions[i-1,0], positions[j-1,0]]
        y = [positions[i-1,1], positions[j-1,1]]
        z = [positions[i-1,2], positions[j-1,2]]
        ax.plot(x, y, z, color='red') 
plt.title("Equilibrated state")
plt.savefig("equilibrated_positions.png", dpi=300)