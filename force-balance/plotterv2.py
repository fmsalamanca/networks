from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import time 

def count_bonds(connections):
    bonds = set()
    for i in range(len(connections)):
        for k in range(connections.shape[1]):
            conn_val = connections[i, k]
            if conn_val != -1 and conn_val != i:  # skip invalid and self
                # Add as sorted tuple to avoid double-counting
                bond = tuple(sorted([i, conn_val]))
                bonds.add(bond)
    return len(bonds)

def plot_connection_with_pbc(ax, pos1, pos2, box, color, alpha, linewidth):
    """Plot connection respecting minimum image convention"""
    delta = pos2 - pos1
    delta = (delta + 0.5 * box) % box - 0.5 * box  # Wrap to [-box/2, box/2]
    endpoint = pos1 + delta
    crosses_boundary = np.any((endpoint < 0) | (endpoint > box))
    
    if crosses_boundary and np.any(delta != 0):
        # Find which boundary we hit first
        t_upper = np.where(delta > 0, (box - pos1) / delta, np.inf)
        t_lower = np.where(delta < 0, -pos1 / delta, np.inf)
        
        # Take minimum t across all dimensions (first boundary hit)
        t = np.min(np.concatenate([t_upper, t_lower]))
        
        # Calculate boundary intersection point
        boundary_point = pos1 + t * delta
        
        # Clip to ensure it's exactly on boundary
        boundary_point = np.clip(boundary_point, 0, box)
        
        # Plot line from pos1 to boundary projection
        ax.plot([pos1[0], boundary_point[0]],
                [pos1[1], boundary_point[1]],
                [pos1[2], boundary_point[2]], 
                color=color, alpha=alpha, linewidth=linewidth)
    else:
        # Normal connection stays within box
        ax.plot([pos1[0], endpoint[0]],
                [pos1[1], endpoint[1]],
                [pos1[2], endpoint[2]], 
                color=color, alpha=alpha, linewidth=linewidth)

current_dir = Path.cwd()
raw_data_dir = current_dir.parent / 'raw-data'

steps = [1,35,98,135]

for a in range(len(steps)):
    target_dir = raw_data_dir / 'f6' / 'N64' / 'run00001' / 'output' / 'step_00{}'.format(steps[a])

    if steps[a] < 10:
        target_dir = raw_data_dir / 'f6' / 'N64' / 'run00001' / 'output' / 'step_000{}'.format(steps[a])
    if steps[a] >= 100:
        target_dir = raw_data_dir / 'f6' / 'N64' / 'run00001' / 'output' / 'step_0{}'.format(steps[a])

    t0 = time.time()

    # Load data
    positions = np.loadtxt(target_dir / 'positions.txt')
    remaining = np.loadtxt(target_dir / 'connections.txt', dtype=int)
    box_curr = np.loadtxt(target_dir / 'box.txt')

    connectionsCurr = count_bonds(remaining)  # total number of connections after breakage

    # Load broken connections
    if (target_dir / 'broken_connections.txt').exists():
        broken = np.loadtxt(target_dir / 'broken_connections.txt', dtype=int)

    # Sort each row to ensure consistent ordering [smaller_id, larger_id]
        sorted_broken = np.sort(broken, axis=1)

        # Find unique bonds
        unique_broken, counts_broken = np.unique(sorted_broken, axis=0, return_counts=True)

        # Total unique broken connections
        connectionsBroken = len(unique_broken)



    fig = plt.figure(figsize=(14, 6))

    # Box 1: Remaining connections
    ax1 = fig.add_subplot(121, projection='3d')

    for row in remaining:
        i = row[0]
        for k in range(1, len(row)):
            conn = row[k]
            if conn != -1 and i != conn:  # valid connection and not self
                plot_connection_with_pbc(ax1, positions[i], positions[conn], 
                                        box_curr, 'g', 0.3, 0.5)

    ax1.set_xlim(0, box_curr[0])
    ax1.set_ylim(0, box_curr[1])
    ax1.set_zlim(0, box_curr[2])
    ax1.grid(False)
    if True:
        if steps[a] == 1:
            ax1.set_box_aspect((1, 1, 1))
        if steps[a] == 35:
            ax1.set_box_aspect((1, 1, 2))
        if steps[a] == 98:
            ax1.set_box_aspect((1, 1, 4))
        if steps[a] == 135:
            ax1.set_box_aspect((1, 1, 6))


    #ax1.set_box_aspect((1, 1, 2))

    plt.tight_layout()
    plt.savefig(target_dir / f'Box-deformed_{steps[a]}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plotting completed in {(time.time() - t0)/60:.2f} minutes.")
    print('Done.')