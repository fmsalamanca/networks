import numpy as np
from matplotlib import pyplot as plt

def count_bonds(connections):
    nxlink = len(connections)
    count = 0
    for i in range(nxlink):
        for k in range(connections.shape[1]):
            conn_val = connections[i,k]
            # skip invalid, self-connection, and count each bond only once
            if not (conn_val == -1 or conn_val <= i):
                count += 1
    return count

def plot_connection_with_pbc(ax, pos1, pos2, box, color, alpha, linewidth):
    """Plot connection respecting minimum image convention"""
    delta = pos2 - pos1
    
    # Check if any dimension exceeds half box size
    needs_pbc = np.any(np.abs(delta) > box / 2)
    
    if needs_pbc:
        # Find which boundary to connect to
        boundary_point = pos1.copy()
        for dim in range(3):
            if delta[dim] > box[dim] / 2:
                boundary_point[dim] = box[dim]
            elif delta[dim] < -box[dim] / 2:
                boundary_point[dim] = 0
        
        # Plot line from pos1 to boundary
        ax.plot([pos1[0], boundary_point[0]],
                [pos1[1], boundary_point[1]],
                [pos1[2], boundary_point[2]], 
                color=color, alpha=alpha, linewidth=linewidth)
    else:
        # Normal connection within half box size
        ax.plot([pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                [pos1[2], pos2[2]], 
                color=color, alpha=alpha, linewidth=linewidth)

# Load data
positions = np.loadtxt('final_equilibrated_positions.txt')
remaining = np.loadtxt('remaining_connections.txt', dtype=int)
broken = np.loadtxt('broken_connections.txt', dtype=int)
box_curr = np.loadtxt('box_curr.txt', dtype=float)

connections0 = np.loadtxt("connected_xlinks.txt", skiprows=1, delimiter="\t", dtype=int)
sorted_indices = np.argsort(connections0[:, 0])
connections0 = connections0[sorted_indices]-1  # convert to 0-based indexing
connections0 = count_bonds(connections0)  # total number of connections before breakage
connectionsCurr = count_bonds(remaining)  # total number of connections after breakage

sorted_broken = np.sort(broken, axis=1)  # sort by first column (i)
unique_broken, counts_broken = np.unique(sorted_broken, axis=0, return_counts=True)
has_duplicates = np.any(counts_broken > 1)

if has_duplicates:
    print("Warning: There are duplicate broken connections in the data.")

connectionsBroken = len(unique_broken)  # total number of unique broken connections

print(f"Total initial bonds: {connections0}"
      f"\nRemaining bonds: {connectionsCurr} ({100*connectionsCurr/connections0:.2f} %)"
      f"\nBroken bonds: {connectionsBroken} ({100*connectionsBroken/connections0:.2f} %)")

fig = plt.figure(figsize=(14, 6))

# Box 1: Remaining connections
ax1 = fig.add_subplot(121, projection='3d')

for row in remaining:
    i = row[0]
    for k in range(1, len(row)):
        conn = row[k]
        if conn != -1:
            plot_connection_with_pbc(ax1, positions[i], positions[conn], 
                                    box_curr, 'g', 0.3, 0.5)

ax1.set_xlim(0, box_curr[0])
ax1.set_ylim(0, box_curr[1])
ax1.set_zlim(0, box_curr[2])
ax1.set_box_aspect((1, 1, 2))
ax1.set_title('Remaining Connections: {} %'.format(connectionsCurr*100//connections0))

# Box 2: Broken connections
ax2 = fig.add_subplot(122, projection='3d')

for row in broken:
    i, j = row[0], row[1]
    plot_connection_with_pbc(ax2, positions[i], positions[j], 
                            box_curr, 'r', 0.3, 0.5)

ax2.set_xlim(0, box_curr[0])
ax2.set_ylim(0, box_curr[1])
ax2.set_zlim(0, box_curr[2])
ax2.set_box_aspect((1, 1, 2))
ax2.set_title('Broken Connections: {} %'.format(connectionsBroken*100//connections0))

plt.tight_layout()
plt.savefig('Box-deformed.png', dpi=300, bbox_inches='tight')
plt.close()
print('Done.')