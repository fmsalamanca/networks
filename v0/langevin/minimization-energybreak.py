import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os


@njit
def func(positions, connections, box, max_steps=100, tolerance=1e-4):
    nxlink = len(positions)

    step = 0
    max_energy = np.inf
    while max_energy > tolerance and step < max_steps:
        max_energy = np.inf
        indices = np.random.permutation(nxlink)  # from 0 to nxlink-1

        # main loop
        for ii in range(nxlink):
            i = indices[ii]
            pos = positions[i] % box

            sum_delta = np.zeros(3, dtype=positions.dtype)
            count = 0
            # iterate over connection table
            for k in range(connections.shape[1]):
                conn_val = connections[i,k]
                if conn_val != -1:
                
                    conn_pos = positions[conn_val] % box

                    delta = conn_pos - pos
                        # minimum-image
                    delta = (delta + 0.5 * box) % box - 0.5 * box
                        # elementwise accumulate
                    sum_delta[0] += delta[0]
                    sum_delta[1] += delta[1]
                    sum_delta[2] += delta[2]
                    if i != conn_val:
                        count += 1
            if count == 0:
                count = 1  # avoid division by zero
            force0 = sum_delta[0] / count
            force1 = sum_delta[1] / count
            force2 = sum_delta[2] / count
                    # new_pos computed from pos and COM (modulo box)
            new_pos0 = (pos[0] + force0) % box[0]
            new_pos1 = (pos[1] + force1) % box[1]
            new_pos2 = (pos[2] + force2) % box[2]

            positions[i, 0] = new_pos0
            positions[i, 1] = new_pos1
            positions[i, 2] = new_pos2

            energy = force0 * force0 + force1 * force1 + force2 * force2

            #if energy > max_energy:
            #    max_energy = energy

        step += 1
        #print("Step", step," max energy = ",max_energy)
    print("Equilibration finished in",step,"steps with energy =",energy)
    return positions

def breakage_potential(positions, connections, box, U_crit=10.0,Nc=2,n=32,log_file=None):
    nxlink = len(positions)
    indices = np.random.permutation(nxlink)  # from 0 to nxlink-1
    test=0
    for ii in range(nxlink):
        i   = indices[ii]
        pos = positions[i] % box
        for k in range(1,connections.shape[1]): # 1 is to ignore self connection
            conn_val = connections[i,k]
            if conn_val != -1:
            
                conn_pos = positions[conn_val] % box

            # current distance under PBC
                delta  = conn_pos - pos
                delta  = (delta + 0.5 * box) % box - 0.5 * box
                energy = delta[0]**2 + delta[1]**2 + delta[2]**2
                if energy >= U_crit:
                    # Remove connection from i to conn_val
                    connections[i, k] = -1
                    # Find and remove reverse connection (from conn_val to i)
                    reverse_idx = np.where(connections[conn_val] == i)[0]
                    if len(reverse_idx) > 0:
                        connections[conn_val, reverse_idx[0]] = -1
                    
                    # Write to file in canonical form (always smaller index first)
                    # This ensures each bond is written exactly once
                    bond_pair = sorted([i, conn_val])
                    with open(log_file, "a") as f:
                        f.write(f"{bond_pair[0]} {bond_pair[1]}\n")
                if test == 0 and False:
                    print("broken connection: ",i,k," with energy = ",energy)
                    print("positions of i ", positions[i])
                    print("positions of k ", positions[k])
                    test+=1
        if False:
            lambda_chain = np.sqrt((lambda_x**2+lambda_y**2+lambda_z**2)/3)


            if lambda_chain/np.sqrt(n) < 0.84136:
                beta = 1.31446*np.tan(1.58986*lambda_chain/np.sqrt(n)) + 0.91209*lambda_chain/np.sqrt(n)
            elif lambda_chain/np.sqrt(n) >= 0.84136 and lambda_chain/np.sqrt(n) < 1:
                beta = 1/(np.sin(lambda_chain/np.sqrt(n))-lambda_chain/np.sqrt(n))
            else:
                ValueError("Langevin inverse not defined.")
                break
            U_curr = Nc*np.sqrt(n)*(beta*lambda_chain-np.sqrt(n)*np.log(np.sinh(beta)/beta)) # in units of kT

        if False: # harmonic potential
            delta_curr = (delta_curr + 0.5*box) % box - 0.5*box
            r = np.linalg.norm(delta_curr)

            delta = positions[conn_val[k]] - positions[i]
            delta = (delta + 0.5 * box) % box - 0.5 * box
            r0 = np.linalg.norm(delta)
                # harmonic bond energy
            U_curr = 0.5 * k * r**2 - 0.5 * k * r0**2
            U_curr = 0.5 * k * r**2
            # print(U_curr)

    return connections

def compute_total_z_force(positions, connections, box):
    nxlink = len(positions)

    indices = np.random.permutation(nxlink)  # from 0 to nxlink-1

    b = np.sqrt(10)
    forcevector = np.zeros(3, dtype=positions.dtype)
    for ii in range(nxlink):
        i = indices[ii]
        pos = positions[i] % box

        sum_delta = np.zeros(3, dtype=positions.dtype)
        for k in range(connections.shape[1]):
            conn_val = connections[i,k]
            if conn_val != -1:
                
                conn_pos = positions[conn_val] % box
                delta = conn_pos - pos
                delta = np.abs((delta + 0.5*box) % box - 0.5 * box)

            sum_delta += delta
        #print(sum_delta[2] > 0)
        forcevector += sum_delta
        
    force = forcevector[2] - (forcevector[0]+forcevector[1])/2
    print(forcevector[0],forcevector[1],forcevector[2])
    return force

###################################################

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

# Load crosslink positions
positions = np.loadtxt("crosslinks-positions.txt")

# Load connection table (first column = ID, next columns = connected IDs)
connections0 = np.loadtxt("connected_xlinks.txt", skiprows=1, delimiter="\t", dtype=int)
connections=connections0.copy()
# Sort by the first column (xlinkID)
sorted_indices = np.argsort(connections[:, 0])
connections = connections[sorted_indices]-1  # convert to 0-based indexing

#print("first equilibration before deformation...")

L=256
box = np.array([L, L, L], dtype=positions.dtype)
positions = func(positions, connections,box=box)  #
fZtotal = compute_total_z_force(positions, connections, box)
print("Total Z force = ", np.round(fZtotal,0))

np.savetxt(
    "initial_equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    comments=''
)

#delta0 = d0(positions,connections,box=box)
total_initial_bonds = count_bonds(connections)
if False:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(positions[:,0], positions[:,1], positions[:,2], s=5)
    for i in range(len(connections)):
        p0 = connections[i, 0]
        p1 = connections[i, 1:]
        x = np.vstack((
            np.full(len(p1), positions[p0, 0]),
            positions[p1, 0]
        ))
        y = np.vstack((
            np.full(len(p1), positions[p0, 1]),
            positions[p1, 1]
        ))
        z = np.vstack((
            np.full(len(p1), positions[p0, 2]),
            positions[p1, 2]
        ))
        ax.plot(x, y, z, color='black', linewidth=0.3)
    #ax.set_xlim(0, box[0])
    #ax.set_ylim(0, box[1])
    #ax.set_zlim(0, box[2])
    plt.title("Undeformed state")
    plt.savefig("Box-Undeformed.png")

#print("strain-controlled deformation with bond breakage...")
# ----------------------------
# Parameters
# ----------------------------
total_strain = 1000     # total strain (%) applied in z
n_steps = 10             # number of increments
dstrain = total_strain / n_steps   # strain increment per step (%)
box_curr = box
kappa = 1
kB = 1
T = 1
n = 32
Nc = 2
b = np.sqrt(10)
U_crit = ((n+1)*b)**2

if os.path.exists('broken_connections.txt'):
    os.remove('broken_connections.txt')

#print(r0)
# initial state

# Store initial state for linear strain application
initial_box = box_curr.copy()
initial_positions = positions.copy()

# ----------------------------
# Incremental deformation loop
# ----------------------------
#print("Starting deformation loop...")
forceZ = [fZtotal]
for step in range(1, n_steps + 1):
    fZtotal = 0
    
    if step>1:
        #print("Starting equilibration...")
    #if step == 1:
    #    print(positions[7,:])
        positions = func(positions, connections,box=box_curr)  
        #print("Equilibration done.")
    # ---- 1) linear strain application ----
    # Calculate cumulative strain at this step for LINEAR growth
    cumulative_strain = step * dstrain  # %

    scale_z = 1 + cumulative_strain / 100
    scale_x = np.sqrt(1/scale_z)
    scale_y = np.sqrt(1/scale_z)

    # Update box linearly from initial state (not compounded)
    box_curr = initial_box * np.array([scale_x, scale_y, scale_z])

    # ---- 2) rescale positions linearly from initial state ----
    positions = initial_positions * np.array([scale_x, scale_y, scale_z])

    positions = positions % box_curr
    #if step == 1:
    #    print(positions[7,:])

    # ---- 4) bond breakage ----
    connections = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,
                                                        log_file='broken_connections.txt')

    remaining_bonds = count_bonds(connections)
    print(f"After step {step}, remaining bonds: {remaining_bonds}/{total_initial_bonds} ({100*remaining_bonds/total_initial_bonds:.2f} %)")
    fZtotal = compute_total_z_force(positions, connections, box_curr)
    forceZ.append(fZtotal)
    print("Total Z force = ", np.round(fZtotal,0))
    if fZtotal < 1e-6:
        print("broken")
        continue

strain = np.linspace(1,total_strain/100+1,n_steps+1)
stress = forceZ
plt.plot(strain,forceZ)
plt.xlabel(r"$\lambda$")
plt.ylabel('Force (a.u.)')
plt.title('((32+1)sqrt(10))**2 = ' + str(np.round(U_crit,0)))
plt.xlim(left=1)
plt.ylim(bottom=0)
plt.xticks(ticks=strain)
plt.show()
plt.savefig("Stress-Strain.png")

np.savetxt(
    "remaining_connections.txt",
    connections,
    fmt='%d',
    delimiter="\t",
    comments=''
)

np.savetxt(
    "final_equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    comments=''
)

np.savetxt(
    "box_curr.txt",
    box_curr,
    fmt='%d',
    delimiter="\t",
    comments=''
)


