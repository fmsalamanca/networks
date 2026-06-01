from time import time
import numpy as np
from numba import njit
import os
import shutil

@njit
def func(positions, connections, box, max_steps=10000, b=np.sqrt(10), N=None, tolerance=1e-4):
    nxlink = len(positions)
    step = 0
    prev_energy = np.inf
    while step < max_steps:
        current_energy = 0
        indices = np.random.permutation(nxlink)  # from 0 to nxlink-1
        indices = np.arange(nxlink)  # from 0 to nxlink-1

        new_positions = positions.copy()
        # main loop
        for ii in range(nxlink):
            i = indices[ii]
            pos = positions[i] # NO NEED TO APPLY PBC HERE, AS IT WILL BE DONE IN THE DELTA CALCULATION
            sum_delta = np.zeros(3, dtype=positions.dtype)
            count = 0
            for k in range(1,connections.shape[1]):
                conn_val = connections[i,k]
                if conn_val != -1:
                    conn_pos = positions[conn_val]
                    delta = conn_pos - pos
                    delta = (delta + 0.5 * box) % box - 0.5 * box
                    sum_delta[0] += delta[0]
                    sum_delta[1] += delta[1]
                    sum_delta[2] += delta[2]
                    if i != conn_val:
                        count += 1
            if count == 0:
                continue
            if count == 1:
                new_pos0 = positions[conn_val,0] 
                new_pos1 = positions[conn_val,1] 
                new_pos2 = positions[conn_val,2] 
            else:
                COM0 = sum_delta[0] / count
                COM1 = sum_delta[1] / count
                COM2 = sum_delta[2] / count

                new_pos0 = (pos[0] + COM0) % box[0]
                new_pos1 = (pos[1] + COM1) % box[1]
                new_pos2 = (pos[2] + COM2) % box[2]

            energy = (3/2)*(sum_delta[0]**2 + sum_delta[1]**2 + sum_delta[2]**2)/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain
            
            new_positions[i, 0] = new_pos0
            new_positions[i, 1] = new_pos1
            new_positions[i, 2] = new_pos2
            current_energy += energy

        energy_diff = np.abs(prev_energy-current_energy)
        if energy_diff < tolerance:
            positions = new_positions
            break
        if True:
            positions = new_positions
            prev_energy = current_energy
            step += 1
    print("Equilibration finished in",step,"steps with energy difference=",energy_diff,"and final energy=",current_energy)
    return positions

def breakage_potential(positions, connections, box, U_crit,b=np.sqrt(10),N=None,log_file=None):
    broken = False
    nxlink = len(positions)
    indices = np.random.permutation(nxlink)  # from 0 to nxlink-1
    indices = np.arange(nxlink)  # from 0 to nxlink-1
    test=1
    for ii in range(nxlink):
        i   = indices[ii]
        pos = positions[i] % box
        for k in range(1,connections.shape[1]): # 1 is to ignore self connection
            conn_val = connections[i,k]
            if conn_val != -1 and i != conn_val:
            
                conn_pos = positions[conn_val] % box

            # current distance under PBC
                delta  = conn_pos - pos
                delta  = (delta + 0.5 * box) % box - 0.5 * box
                energy = (3/2)*(delta[0]**2 + delta[1]**2 + delta[2]**2)/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain
                if energy >= U_crit:
                    broken = True
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
    return connections, broken

def compute_total_stress(positions, connections, box,N=None,b=np.sqrt(10)):
    nxlink = len(positions)

    indices = np.random.permutation(nxlink)  # from 0 to nxlink-1
    #indices = np.arange(nxlink)  # from 0 to nxlink-1

    b = np.sqrt(10)
    total_delta = np.zeros(3, dtype=positions.dtype)
    for ii in range(nxlink):
        i = indices[ii]
        pos = positions[i]

        sum_delta = np.zeros(3, dtype=positions.dtype)
        for k in range(1,connections.shape[1]):
            print("Connection found between", i, "and", k, flush=True)

            conn_val = connections[i,k]
            if conn_val != -1 and i != conn_val:
                conn_pos = positions[conn_val]
                delta = conn_pos - pos
                delta = np.abs((delta + 0.5*box) % box - 0.5 * box)
                
            sum_delta += delta
        total_delta += sum_delta
    forcevector = 3/(N*b**2)*total_delta    #eq 2.96 from rubinstein, in units of kT, using ideal chain model with Kuhn length b and n segments per chain
    stress = forcevector[2]/(box[0]*box[1]) - (forcevector[0]/(box[1]*box[2]) + forcevector[1]/(box[0]*box[2]))/2
    stress = [forcevector[0]/(box[1]*box[2]), forcevector[1]/(box[0]*box[2]), forcevector[2]/(box[0]*box[1])]

    return stress

def compute_total_force(positions, connections, box,N=None,b=np.sqrt(10)):
    nxlink = len(positions)

    indices = np.random.permutation(nxlink)  # from 0 to nxlink-1
    #indices = np.arange(nxlink)  # from 0 to nxlink-1

    b = np.sqrt(10)
    total_delta = np.zeros(3, dtype=positions.dtype)
    for ii in range(nxlink):
        i = indices[ii]
        pos = positions[i]

        sum_delta = np.zeros(3, dtype=positions.dtype)
        for k in range(1, connections.shape[1]):
            conn_val = connections[i,k]
            if conn_val != -1 and i != conn_val:
                conn_pos = positions[conn_val]
                delta = conn_pos - pos
                delta = np.abs((delta + 0.5*box) % box - 0.5 * box)

            sum_delta += delta
        total_delta += sum_delta
    forcevector = 3/(N*b**2)*total_delta    #eq 2.96 from rubinstein, in units of kT, using ideal chain model with Kuhn length b and n segments per chain
    return forcevector


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

###################################################

# Load crosslink positions
positions = np.loadtxt("crosslinks-positionsTEST2.txt")
t0 = time()
# Load connection table (first column = ID, next columns = connected IDs)
connections0 = np.loadtxt("connected_xlinksTEST2.txt", skiprows=1, delimiter="\t", dtype=int)
connections=connections0.copy()

connections = connections -1  # convert to 0-based indexing, with -1 for no connection
params = {}
with open("system.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(" = ")
        params[key] = float(value)

number_of_monomers      = params["number_of_monomers"]
box_x                   = params["box_x"]
box_y                   = params["box_y"]
box_z                   = params["box_z"]
number_of_linear_chains = params["number_of_linear_chains"]
number_of_crosslinkers  = params["number_of_crosslinkers"]
chainLength             = params["chainLength"]
#mcs                     = params["mcs"]

box = np.array([box_x, box_y, box_z], dtype=positions.dtype)

L_prev = box_x
pos_prev = func(positions, connections, box,N=chainLength+1)
stress_ref = 0
tol_stress = 1e-6
tol_L = 1e-6
max_iter = 100
fd_step = 1e-5
damping = 1

def relax_box_uniaxial(positions, box,N=None):
    L_prev = (box[0]+box[1])/2  # using x (same as y) as the free dimension
    box[0] = box[1] = L_prev        # enforce symmetry immediately
    # Initial relaxation at current box
    positions = func(positions, connections, box=box,N=N)
    stress_prev = compute_total_stress(positions, connections=connections, box=box,N=N)
    f_prev = (stress_prev[0]+stress_prev[1])/2 - stress_prev[2]  # target: sigma_x == sigma_y == sigma_z
    
    if abs(f_prev) < tol_stress:
        return positions, stress_prev, box

    # Perturb L slightly for secant initialization
    L_curr = L_prev * (1 + fd_step)
    # Scale only x,y positions
    positions[:, 0] *= L_curr / L_prev
    positions[:, 1] *= L_curr / L_prev
    box[0] = box[1] = L_curr

    positions = func(positions, connections, box=box,N=N)
    stress_curr = compute_total_stress(positions, connections=connections, box=box,N=N)
    f_curr = (stress_curr[0]+stress_curr[1])/2 - stress_curr[2]  # target: sigma_x == sigma_y == sigma_z

    for _ in range(max_iter):
        #print(f"L={L_curr:.8f}, f={f_curr:.6e}, sx={stress_curr[0]:.6e}, sy={stress_curr[1]:.6e}, sz={stress_curr[2]:.6e}")
        if abs(f_curr) < tol_stress:
            print("Stress difference converged with f =", f_curr,flush=True)
            return positions,stress_curr,box

        if abs(L_curr - L_prev) / L_curr < tol_L:
            print("Box length converged with f =", f_curr,flush=True)
            return positions,stress_curr,box

        if (f_curr - f_prev) == 0:
            raise RuntimeError('Secant method failed: zero derivative')

        L_next = L_curr - f_curr * (L_curr - L_prev) / (f_curr - f_prev)
        L_next = L_curr + damping * (L_next - L_curr)  # damping helps stability

        if L_next <= 0:
            L_next = L_curr * 0.5

        # Scale ONLY x and y positions, not z
        scale = L_next / L_curr
        positions[:, 0] *= scale
        positions[:, 1] *= scale
        box[0] = box[1] = L_next

        positions = func(positions, connections, box=box,N=N)
        stress_next = compute_total_stress(positions, connections=connections, box=box,N=N)
        f_next = (stress_next[0]+stress_next[1])/2 - stress_next[2]  # target: sigma_x == sigma_y == sigma_z

        L_prev, f_prev = L_curr, f_curr
        L_curr, f_curr = L_next, f_next
        stress_curr = stress_next

    raise RuntimeError('Secant method did not converge')
  
positions,stress,box = relax_box_uniaxial(positions,box,N=chainLength+1)
stress = stress[2] - (stress[0]+stress[1])/2
np.savetxt(
    "initial_equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    comments=''
)

total_initial_bonds = count_bonds(connections)


#print("strain-controlled deformation with bond breakage...")
# ----------------------------
# Parameters
# ----------------------------
total_strain = 600     # total strain (%) applied in z
n_steps = 200          # number of increments
dstrain = total_strain / n_steps   # strain increment per step (%)
box_curr = box

N = chainLength+1
b = np.sqrt(10)
U_crit = (N*b)**2/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain

os.makedirs("./output", exist_ok=True)

# Store initial state for linear strain application
initial_box = box_curr.copy()
initial_positions = positions.copy()

# ----------------------------
# Incremental deformation loop
# ----------------------------
stress_total = [stress]
strain_total = [1]
for step in range(1, n_steps + 1):
    stress = 0
    
    if step>1:
        positions = func(positions, connections,box=box_curr,N=chainLength+1)  
    # ---- 1) linear strain application ----
    # Calculate cumulative strain at this step for LINEAR growth
    cumulative_strain = step * dstrain  # %

    scale_z = 1 + cumulative_strain / 100
    scale_x = np.sqrt(1/scale_z)
    scale_y = np.sqrt(1/scale_z)

    # Update box linearly from initial state (not compounded)
    box_new = initial_box * np.array([scale_x, scale_y, scale_z])
    box_ratio = box_new / box_curr
    box_curr = box_new
    # ---- 2) rescale positions linearly from initial state ----
    positions = positions * box_ratio


    # ---- 3) bond breakage ----
    connections,broken = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,N=chainLength+1,
                                                        log_file='broken_connections.txt')
    
    # ---- 4) relax system with new bonds ----
    while broken:
        positions = func(positions, connections,box=box_curr,N=chainLength+1)
        connections,broken = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,N=chainLength+1,
                                                        log_file='broken_connections.txt')
        stress = compute_total_stress(positions, connections, box_curr,N=chainLength+1)
        stress = stress[2] - (stress[0]+stress[1])/2
        if stress < 1e-4 or stress_total[0] > stress or 0.5*stress_total[-1] > stress:
            print("Stress has dropped to zero during relaxation, stopping deformation.", flush=True)
            checkpoint_dir = f"./output/step_final"
            os.makedirs(checkpoint_dir, exist_ok=True)

            np.savetxt(
                os.path.join(checkpoint_dir, "positions.txt"),
                positions,
                fmt=["%.4f", "%.4f", "%.4f"],
                delimiter="\t",
                comments=''
            )
            np.savetxt(
                os.path.join(checkpoint_dir, "connections.txt"),
                connections,
                fmt='%d',
                delimiter="\t",
                comments=''
            )
            np.savetxt(
                os.path.join(checkpoint_dir, "box.txt"),
                box_curr,
                fmt='%.6f',
                delimiter="\t",
                comments=''
            )

            if os.path.exists('broken_connections.txt'):
                    shutil.copy('broken_connections.txt',
                                os.path.join(checkpoint_dir, "broken_connections.txt"))

            with open(os.path.join(checkpoint_dir, "summary.txt"), "w") as f:
                f.write(f"Step: {step}\n")
                f.write(f"Scale_z (lambda): {scale_z:.6f}\n")
                f.write(f"Cumulative strain: {cumulative_strain:.4f} %\n")
                f.write(f"Stress: {stress:.6f}\n")
                f.write(f"Remaining bonds: {count_bonds(connections)}/{total_initial_bonds}\n")

            print(f"  [Checkpoint saved → {checkpoint_dir}]", flush=True)
            break
    with open('./output/breakage_log.txt', 'a') as f:
        if step == 1:
            f.write("step lambda cumulative_broken remaining stress\n")
        f.write(f"{step} {scale_z:.6f} "
                f"{total_initial_bonds - count_bonds(connections)} "
                f"{count_bonds(connections)} "
                f"{stress:.6f}\n")
    
    remaining_bonds = count_bonds(connections)
    print(f"After step {step}, remaining bonds: {remaining_bonds}/{total_initial_bonds} ({100*remaining_bonds/total_initial_bonds:.2f} %)", flush=True)
    stress = compute_total_stress(positions, connections, box_curr,N=chainLength+1)
    stress = stress[2] - (stress[0]+stress[1])/2

    stress_total.append(stress)
    strain_total.append(scale_z)
    print("Total stress = ", np.round(stress,4), 'with','lambda', '=', np.round(scale_z,2), flush=True)
    print('', flush=True)
    # ---- 5) checkpoint saving ----
    
    checkpoint_dir = f"./output/step_{step:04d}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    np.savetxt(
            os.path.join(checkpoint_dir, "positions.txt"),
            positions,
            fmt=["%.4f", "%.4f", "%.4f"],
            delimiter="\t",
            comments=''
        )
    np.savetxt(
            os.path.join(checkpoint_dir, "connections.txt"),
            connections,
            fmt='%d',
            delimiter="\t",
            comments=''
        )
    np.savetxt(
            os.path.join(checkpoint_dir, "box.txt"),
            box_curr,
            fmt='%.6f',
            delimiter="\t",
            comments=''
        )

    if os.path.exists('broken_connections.txt'):
            shutil.copy('broken_connections.txt',
                        os.path.join(checkpoint_dir, "broken_connections.txt"))

    with open(os.path.join(checkpoint_dir, "summary.txt"), "w") as f:
        f.write(f"Step: {step}\n")
        f.write(f"Scale_z (lambda): {scale_z:.6f}\n")
        f.write(f"Cumulative strain: {cumulative_strain:.4f} %\n")
        f.write(f"Stress: {stress:.6f}\n")
        f.write(f"Remaining bonds: {count_bonds(connections)}/{total_initial_bonds}\n")

    print(f"  [Checkpoint saved → {checkpoint_dir}]", flush=True)
    if stress < 1e-4 or stress_total[0] > stress or 0.5*stress_total[-1] > stress:
        print("Stress has dropped to zero, stopping deformation.", flush=True)
        positions = func(positions, connections,box=box_curr,N=chainLength+1)  # final relaxation
        checkpoint_dir = f"./output/step_final"
        os.makedirs(checkpoint_dir, exist_ok=True)

        np.savetxt(
            os.path.join(checkpoint_dir, "positions.txt"),
            positions,
            fmt=["%.4f", "%.4f", "%.4f"],
            delimiter="\t",
            comments=''
        )
        np.savetxt(
            os.path.join(checkpoint_dir, "connections.txt"),
            connections,
            fmt='%d',
            delimiter="\t",
            comments=''
        )
        np.savetxt(
            os.path.join(checkpoint_dir, "box.txt"),
            box_curr,
            fmt='%.6f',
            delimiter="\t",
            comments=''
        )

        if os.path.exists('broken_connections.txt'):
                shutil.copy('broken_connections.txt',
                            os.path.join(checkpoint_dir, "broken_connections.txt"))

        with open(os.path.join(checkpoint_dir, "summary.txt"), "w") as f:
            f.write(f"Step: {step}\n")
            f.write(f"Scale_z (lambda): {scale_z:.6f}\n")
            f.write(f"Cumulative strain: {cumulative_strain:.4f} %\n")
            f.write(f"Stress: {stress:.6f}\n")
            f.write(f"Remaining bonds: {count_bonds(connections)}/{total_initial_bonds}\n")

        print(f"  [Checkpoint saved → {checkpoint_dir}]", flush=True)
        break


strain_total = np.array(strain_total)

np.savetxt(
    "./output/stress_strain.txt",
    np.column_stack((strain_total, stress_total)),
    fmt=['%.6f', '%.6f'],
    delimiter="\t",
    header="Lambda\tStress",
    comments=''
)

strain_total = strain_total**2-1/strain_total
if True:
    from matplotlib import pyplot as plt

    plt.plot(strain_total,stress_total)
    plt.xlabel(r"$\lambda^2-1/\lambda$")  
    plt.ylabel(r'$\sigma_z-\frac{1}{2}(\sigma_x+\sigma_y)$')
    plt.title(r'$(Nb)^2/(Nb^2) = $' + str(np.round(U_crit,0)))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    #plt.show()
    plt.savefig("./output/Stress-Strain.png")



np.savetxt(
    "./output/remaining_connections.txt",
    connections,
    fmt='%d',
    delimiter="\t",
    comments=''
)

np.savetxt(
    "./output/final_equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    comments=''
)

np.savetxt(
    "./output/box_curr.txt",
    box_curr,
    fmt='%d',
    delimiter="\t",
    comments=''
)

print(f"Finished in {(time() - t0)/60:.2f} minutes.", flush=True)

