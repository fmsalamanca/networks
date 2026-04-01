import numpy as np
from matplotlib import pyplot as plt
from numba import njit

# Load crosslink positions
positions = np.loadtxt("crosslinks-positions.txt")

# Load connection table (first column = ID, next columns = connected IDs)
connections = np.loadtxt("connected_xlinks.txt", skiprows=1, delimiter="\t", dtype=int)

# Load the data
# Assuming tab-separated values and integer type

# Sort by the first column (xlinkID)
sorted_indices = np.argsort(connections[:, 0])
connections = connections[sorted_indices]-1  # convert to 0-based indexing

L=256
box = np.array([L, L, L], dtype=positions.dtype)


nxlink = len(positions)

@njit
def func(positions, connections, max_steps=1000, tolerance=1e-4):
    step = 0
    max_energy = np.inf
    while max_energy > tolerance and step < max_steps:
        max_energy = 0.0
        indices = np.random.permutation(nxlink)  # from 0 to nxlink-1

        # main loop
        for ii in range(nxlink):
            #print(ii)
            i = indices[ii]
            pos = positions[i] % box
            #if i == 8:
            #    print("")
            #    print("Step ",step," xlink ",i," pos ",pos[0],pos[1],pos[2])
            sum_delta = np.zeros(3, dtype=positions.dtype)
            count = 0
            # iterate over connection table
            # connections assumed shape (nbonds, bond_size) with zeros meaning no connection
            for k in range(connections.shape[1]):
                conn_val = connections[i,k]
                if conn_val == -1:
                    continue
                conn_pos = positions[conn_val] % box
                #if i == 8:
                #    print(" Step ",step," connected to ",conn_val," at pos ",conn_pos[0],conn_pos[1],conn_pos[2])
                delta = conn_pos - pos
                        # minimum-image
                delta = (delta + 0.5 * box) % box - 0.5 * box
                        # elementwise accumulate
                sum_delta[0] += delta[0]
                sum_delta[1] += delta[1]
                sum_delta[2] += delta[2]
                count += 1
                #print(sum_delta[0],sum_delta[1],sum_delta[2],count)

            invc = 1.0 / count
            force0 = sum_delta[0] * invc
            force1 = sum_delta[1] * invc
            force2 = sum_delta[2] * invc

                    # new_pos computed from pos and COM (modulo box)
            new_pos0 = (pos[0] + force0) % box[0]
            new_pos1 = (pos[1] + force1) % box[1]
            new_pos2 = (pos[2] + force2) % box[2]
            #if i == 8:
            #    print("")
            #    print("Step ",step," xlink ",i," pos ",pos[0],pos[1],pos[2])
                    # assign back elementwise so Numba sees scalar stores (safe)
            positions[i, 0] = new_pos0
            positions[i, 1] = new_pos1
            positions[i, 2] = new_pos2
            #if i == 8:
            #    print(" Step ",step," new pos ",new_pos0,new_pos1,new_pos2)

                    # compute displacement length (norm of COM)
            energy = force0 * force0 + force1 * force1 + force2 * force2

            if energy > max_energy:
                max_energy = energy

        step += 1
        print("Step", step," max energy = ",max_energy)
    return 1

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='blue')
plt.savefig("NIJTinitial_positions.png", dpi=300)
plt.title("Initial state")
plt.show()

#equilibration
func(positions, connections)  


# Prepare a list to save: one row per connection
saving = []

for row in connections:
    xlink_id = row[0]
    xlink_pos = positions[xlink_id]  # 0-based indexing

    for neighbor_id in row[1:]:
        if neighbor_id == -1:
            continue
        neighbor_pos = positions[neighbor_id]

        # Minimum image difference
        diff = neighbor_pos - xlink_pos
        diff = (diff + 0.5 * box) % box - 0.5 * box
        distance = np.linalg.norm(diff)

        # Append: xlinkID, neighborID, xlink_pos(x,y,z), neighbor_pos(x,y,z), distance
        saving.append([
            xlink_id, neighbor_id,
            xlink_pos[0], xlink_pos[1], xlink_pos[2],
            neighbor_pos[0], neighbor_pos[1], neighbor_pos[2],
            distance
        ])
# Convert to NumPy array
saving = np.array(saving)

# Save to file with 5 decimals
np.savetxt(
    "equilibrated_positions-displacements.txt",
    saving,
    fmt=["%d", "%d", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f"],
    delimiter="\t",
    header="xlinkID\tneighborID\txlink_x\txlink_y\txlink_z\tneighbor_x\tneighbor_y\tneighbor_z\tdistance",
    comments=''
)
np.savetxt(
    "equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    header="xlink_x\txlink_y\txlink_z",
    comments=''
)
print("Saved positions and distances in equilibrated_positions and equilibrated_positions-displacements.txt")

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='blue')
plt.savefig("NIJTequilibrated_positions.png", dpi=300)
plt.title("Equilibrium state")
plt.show()

# Apply uniaxial strain in z direction

strain = 100 #use percentage

scale_z = 1 + strain/100
scale_x = np.sqrt(1/scale_z) 
scale_y = np.sqrt(1/scale_z)


box_new = np.array([box[0]*scale_x,box[1]*scale_y, box[2]*scale_z])

positions_new = positions.copy()
positions_new[:,0] *= box_new[0]/box[0]
positions_new[:,1] *= box_new[1]/box[1]
positions_new[:,2] *= box_new[2]/box[2]

positions_new = positions_new % box_new

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions_new[:,0], positions_new[:,1], positions_new[:,2], color='blue')
plt.title("Deformed state: strain "+str(strain)+"%")
plt.show()

import numpy as np

def compute_equilibrium_lengths(positions_eq=positions, connections=connections, box=box):
    """
    Compute equilibrium bond lengths under PBC.
    """
    lengths = []
    N = connections.shape[0]
    for i in range(N):
        for col in range(1, connections.shape[1]):
            j = connections[i, col]
            if j == -1:
                continue
            delta = positions_eq[j] - positions_eq[i]
            delta -= box * np.round(delta / box)
            dist = np.linalg.norm(delta)
            lengths.append(dist)
    return np.array(lengths)

# Example usage:
lengths_eq = compute_equilibrium_lengths(positions, connections, box)
# Define rupture cutoff distances as 1.5 times the equilibrium lengths
import numpy as np

def compute_rc_per_bond(positions_eq=positions, connections=connections, box=box, alpha=1.5):
    """
    Build rc_per_bond dict: (i,j) -> alpha * r_eq(i,j), with i<j.
    Uses minimum image convention consistent with your code.
    """
    rc_per_bond = {}
    N = connections.shape[0]
    for i in range(N):
        for col in range(1, connections.shape[1]):
            j = connections[i, col]
            if j == -1:
                continue
            # only store once
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in rc_per_bond:
                continue
            # equilibrium distance under PBC
            delta = positions_eq[b] - positions_eq[a]
            delta = (delta + 0.5 * box) % box - 0.5 * box
            r_eq = np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
            rc_per_bond[(a, b)] = alpha * r_eq
    return rc_per_bond

import numpy as np

rc_per_bond = compute_rc_per_bond(positions, connections, box, alpha=1.5)
def breakage_rc(positions, connections, box, rc_per_bond):
    """
    Break bonds if stretched beyond their per-bond critical length.
    positions: (N,3) array of particle positions
    connections: (N,5) array, first col = particle id (0-based), next cols = connections
    box: array([Lx, Ly, Lz])
    rc_per_bond: dict mapping (i,j) with i<j to critical length
    """
    N = connections.shape[0]
    connections_new = connections.copy()
    for i in range(N):
        for col in range(1, connections.shape[1]):
            j = connections[i, col]
            if j == -1:
                continue
            # compute distance with minimum image convention
            delta = positions[j] - positions[i]
            delta = (delta + 0.5 * box) % box - 0.5 * box
            distance = np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
            # look up per-bond threshold
            pair = (i, j) if i < j else (j, i)
            rc = rc_per_bond.get(pair, None)
            if rc is None:
                continue  # skip if no threshold defined
            if distance > rc:
                # break i->j
                connections_new[i, col] = -1
                # also break j->i
                for col2 in range(1, connections_new.shape[1]):
                    if connections_new[j, col2] == i:
                        connections_new[j, col2] = -1
                        break
    return connections_new

#break bonds beyond rc
connections_new = breakage_rc(positions_new, connections, box_new, rc_per_bond) 
output_file = "connected_xlinks_new.txt"

# Open and write with header + tab-separated values
header = "xlinkID\txlink_1\txlink_2\txlink_3\txlink_4"

np.savetxt(output_file, connections_new, fmt="%d", delimiter="\t", header=header, comments="")

#equilibration after deformation
func(positions_new, connections_new) 

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions_new[:,0], positions_new[:,1], positions_new[:,2], color='blue')

plt.title("Deformed equilibrated state: strain "+str(strain)+"%")
plt.show()
