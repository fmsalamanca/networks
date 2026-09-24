from time import time
import numpy as np
import numba
import os
import shutil
import random

# Lennard-Jones excluded-volume parameters.
# These are intentionally weak and easy to tune from the script.
LJ_SIGMA = 1.0
LJ_EPSILON = 0.05
LJ_CUTOFF = 2.5 * LJ_SIGMA
LJ_FORCE_SCALE = 0.05


def minimum_image(delta, box):
    return (delta + 0.5 * box) % box - 0.5 * box


def lennard_jones_pair_energy(r2, sigma=LJ_SIGMA, epsilon=LJ_EPSILON, cutoff=LJ_CUTOFF):
    if r2 <= 1e-12:
        return np.inf
    if r2 >= cutoff * cutoff:
        return 0.0
    sr2 = (sigma * sigma) / r2
    sr6 = sr2 ** 3
    sr12 = sr6 ** 2
    sr_cut = sigma / cutoff
    sr_cut6 = sr_cut ** 6
    sr_cut12 = sr_cut6 ** 2
    shift = 4.0 * epsilon * (sr_cut12 - sr_cut6)
    return 4.0 * epsilon * (sr12 - sr6) - shift


def lennard_jones_pair_force(delta, r2, sigma=LJ_SIGMA, epsilon=LJ_EPSILON, cutoff=LJ_CUTOFF):
    if r2 <= 1e-12 or r2 >= cutoff * cutoff:
        return np.zeros(3, dtype=delta.dtype)
    sr2 = (sigma * sigma) / r2
    sr6 = sr2 ** 3
    sr12 = sr6 ** 2
    prefactor = 24.0 * epsilon * (2.0 * sr12 - sr6) / r2
    return prefactor * delta


def compute_lj_energy(positions, box, sigma=LJ_SIGMA, epsilon=LJ_EPSILON, cutoff=LJ_CUTOFF):
    total = 0.0
    n = len(positions)
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = minimum_image(positions[j] - positions[i], box)
            r2 = np.dot(delta, delta)
            total += lennard_jones_pair_energy(r2, sigma=sigma, epsilon=epsilon, cutoff=cutoff)
    return total


def compute_lj_force_field(positions, box, sigma=LJ_SIGMA, epsilon=LJ_EPSILON, cutoff=LJ_CUTOFF):
    forces = np.zeros_like(positions)
    n = len(positions)
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = minimum_image(positions[j] - positions[i], box)
            r2 = np.dot(delta, delta)
            if r2 <= 1e-12 or r2 >= cutoff * cutoff:
                continue
            fij = lennard_jones_pair_force(delta, r2, sigma=sigma, epsilon=epsilon, cutoff=cutoff)
            forces[i] += fij
            forces[j] -= fij
    return forces


@numba.njit
def minimum_image_numba(delta, box):
    return (delta + 0.5 * box) % box - 0.5 * box


@numba.njit
def lj_pair_energy_numba(r2, sigma, epsilon, cutoff):
    if r2 <= 1e-12:
        return np.inf
    if r2 >= cutoff * cutoff:
        return 0.0
    sr2 = (sigma * sigma) / r2
    sr6 = sr2 ** 3
    sr12 = sr6 ** 2
    sr_cut = sigma / cutoff
    sr_cut6 = sr_cut ** 6
    sr_cut12 = sr_cut6 ** 2
    shift = 4.0 * epsilon * (sr_cut12 - sr_cut6)
    return 4.0 * epsilon * (sr12 - sr6) - shift


@numba.njit
def lj_pair_force_numba(delta, r2, sigma, epsilon, cutoff):
    if r2 <= 1e-12 or r2 >= cutoff * cutoff:
        return np.zeros(3, dtype=delta.dtype)
    sr2 = (sigma * sigma) / r2
    sr6 = sr2 ** 3
    sr12 = sr6 ** 2
    prefactor = 24.0 * epsilon * (2.0 * sr12 - sr6) / r2
    return prefactor * delta


@numba.njit
def compute_lj_energy_numba(positions, box, sigma, epsilon, cutoff):
    total = 0.0
    n = len(positions)
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = minimum_image_numba(positions[j] - positions[i], box)
            r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            total += lj_pair_energy_numba(r2, sigma, epsilon, cutoff)
    return total


@numba.njit
def compute_lj_force_field_numba(positions, box, sigma, epsilon, cutoff):
    forces = np.zeros_like(positions)
    n = len(positions)
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = minimum_image_numba(positions[j] - positions[i], box)
            r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            if r2 <= 1e-12 or r2 >= cutoff * cutoff:
                continue
            fij = lj_pair_force_numba(delta, r2, sigma, epsilon, cutoff)
            forces[i, 0] += fij[0]
            forces[i, 1] += fij[1]
            forces[i, 2] += fij[2]
            forces[j, 0] -= fij[0]
            forces[j, 1] -= fij[1]
            forces[j, 2] -= fij[2]
    return forces


@numba.njit
def func(positions, connections, box, max_steps=10000, b=np.sqrt(10), N=None, tolerance=1e-4):
    nxlink = len(positions)
    step = 0
    prev_energy = np.inf
    while step < max_steps:
        current_energy = 0
        #indices = np.random.permutation(nxlink)  # from 0 to nxlink-1
        rows = np.arange(connections.shape[0])
        indices = rows.copy()
        np.random.shuffle(indices)  # shuffle in-place for random order each MCS
        #indices = np.arange(nxlink)  # from 0 to nxlink-1
        
        new_positions = positions.copy()
        # main loop
        for row in indices:
            i = connections[row, 0]
            pos = positions[i] # NO NEED TO APPLY PBC HERE, AS IT WILL BE DONE IN THE DELTA CALCULATION
            sum_delta = np.zeros(3, dtype=positions.dtype)
            sum_distance_squared = 0.0
            count = 0
            
            for k in range(1,connections.shape[1]):

                conn_val = connections[row,k]
                if conn_val >= 0 and conn_val < len(positions):
                    conn_pos = positions[conn_val]
                    delta = conn_pos - pos
                    delta = (delta + 0.5 * box) % box - 0.5 * box
                    sum_delta[0] += delta[0]
                    sum_delta[1] += delta[1]
                    sum_delta[2] += delta[2]
                    sum_distance_squared += delta[0]**2 + delta[1]**2 + delta[2]**2
                    if i != conn_val:
                        count += 1
            if count == 0:
                continue

            COM0 = sum_delta[0] / count
            COM1 = sum_delta[1] / count
            COM2 = sum_delta[2] / count

            new_pos0 = (pos[0] + COM0) % box[0]
            new_pos1 = (pos[1] + COM1) % box[1]
            new_pos2 = (pos[2] + COM2) % box[2]

            energy = (3/2)*sum_distance_squared/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain
            
            new_positions[i, 0] = new_pos0
            new_positions[i, 1] = new_pos1
            new_positions[i, 2] = new_pos2
            current_energy += energy

        lj_forces = compute_lj_force_field_numba(new_positions, box, LJ_SIGMA, LJ_EPSILON, LJ_CUTOFF)
        for i in range(len(new_positions)):
            new_positions[i, 0] = (new_positions[i, 0] + LJ_FORCE_SCALE * lj_forces[i, 0]) % box[0]
            new_positions[i, 1] = (new_positions[i, 1] + LJ_FORCE_SCALE * lj_forces[i, 1]) % box[1]
            new_positions[i, 2] = (new_positions[i, 2] + LJ_FORCE_SCALE * lj_forces[i, 2]) % box[2]
        current_energy += compute_lj_energy_numba(new_positions, box, LJ_SIGMA, LJ_EPSILON, LJ_CUTOFF)

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
    rows = np.arange(connections.shape[0])
    np.random.shuffle(rows)
    for row in rows:
        i = connections[row, 0]
        if i < 0 or i >= len(positions):
            continue
        pos = positions[i] % box
        for k in range(1,connections.shape[1]): # 1 is to ignore self connection
            conn_val = connections[row, k]
            if conn_val >= 0 and conn_val < len(positions) and conn_val != i:
                conn_pos = positions[conn_val] % box
                delta  = conn_pos - pos
                delta  = (delta + 0.5 * box) % box - 0.5 * box
                ideal_energy = (3/2)*(delta[0]**2 + delta[1]**2 + delta[2]**2)/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain
                lj_energy = lennard_jones_pair_energy(np.dot(delta, delta), sigma=LJ_SIGMA, epsilon=LJ_EPSILON, cutoff=LJ_CUTOFF)
                energy = ideal_energy + lj_energy
                if energy >= U_crit:
                    broken = True
                    connections[row, k] = -1
                    
                    reverse_rows = np.where(connections[:, 0] == conn_val)[0]
                    if len(reverse_rows) > 0:
                        reverse_row = reverse_rows[0]
                        reverse_idx = np.where(connections[reverse_row, 1:] == i)[0]
                        if len(reverse_idx) > 0:
                            connections[reverse_row, reverse_idx[0] + 1] = -1
                    if log_file is not None:
                        bond_pair = sorted([i, conn_val])
                        with open(log_file, "a") as f:
                            f.write(f"{bond_pair[0]} {bond_pair[1]}\n")
    return connections, broken

def compute_total_stress(positions, connections, box,N=None,b=np.sqrt(10)):
    total_delta = np.zeros(3, dtype=positions.dtype)
    for row in range(connections.shape[0]):
        i = connections[row, 0]
        if i < 0 or i >= len(positions):
            continue
        pos = positions[i]
        sum_delta = np.zeros(3, dtype=positions.dtype)
        for k in range(1,connections.shape[1]):
            conn_val = connections[row, k]
            if conn_val >= 0 and conn_val < len(positions) and conn_val != i:
                conn_pos = positions[conn_val]
                delta = conn_pos - pos
                delta = np.abs((delta + 0.5*box) % box - 0.5 * box)
                sum_delta += delta
        total_delta += sum_delta
    forcevector = 3/(N*b**2)*total_delta    #eq 2.96 from rubinstein, in units of kT, using ideal chain model with Kuhn length b and n segments per chain

    lj_forces = compute_lj_force_field(positions, box)
    forcevector += lj_forces.sum(axis=0)

    stress = forcevector[2]/(box[0]*box[1]) - (forcevector[0]/(box[1]*box[2]) + forcevector[1]/(box[0]*box[2]))/2
    stress = [forcevector[0]/(box[1]*box[2]), forcevector[1]/(box[0]*box[2]), forcevector[2]/(box[0]*box[1])]

    return stress

def compute_total_force(positions, connections, box,N=None,b=np.sqrt(10)):
    rows = np.arange(connections.shape[0])
    np.random.shuffle(rows)
    total_delta = np.zeros(3, dtype=positions.dtype)
    for row in rows:
        i = connections[row, 0]
        if i < 0 or i >= len(positions):
            continue
        pos = positions[i]

        sum_delta = np.zeros(3, dtype=positions.dtype)
        for k in range(1, connections.shape[1]):
            conn_val = connections[row, k]
            if conn_val >= 0 and conn_val < len(positions) and conn_val != i:
                conn_pos = positions[conn_val]
                delta = conn_pos - pos
                delta = np.abs((delta + 0.5*box) % box - 0.5 * box)

                sum_delta += delta
        total_delta += sum_delta
    forcevector = 3/(N*b**2)*total_delta    #eq 2.96 from rubinstein, in units of kT, using ideal chain model with Kuhn length b and n segments per chain
    forcevector += compute_lj_force_field(positions, box).sum(axis=0)
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
positions = np.loadtxt("crosslinks-positions.txt")
t0 = time()
# Load connection table (first column = ID, next columns = connected IDs)
connections = np.loadtxt("connected_xlinks.txt", skiprows=1, delimiter="\t", dtype=int)

connections = connections -1  # convert to 0-based indexing, with -1 for no connection
connections = connections.astype(int)
#for i in range(1, len(connections)):
#    if connections[i, 0] != connections[i-1, 0] + 1:
#        guilty = connections[i, 0]  # the first overestimated ID
#        print(f"Found overestimated ID: {guilty} at row {i}, correcting...", flush=True)
#        break
if False:
    gap =np.where(np.diff(connections[:,0])!=1)[0]
    guilty = connections[gap[0]+1, 0]  # the first overestimated ID
    print(f"Found overestimated ID: {guilty} at row {gap[0]+1}, correcting...", flush=True)

    @numba.njit
    def fix_connections(connections,guilty):
        for i in range(len(connections)):
            for j in range(0, connections.shape[1]):
                if connections[i,j] >= guilty:
                    connections[i,j] -= 1
        return connections
    connections = fix_connections(connections,guilty)
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
print("here,356")

L_prev = box_x
pos_prev = func(positions, connections, box,N=chainLength+1)
print(pos_prev)
stress_ref = 0
tol_stress = 1e-6
tol_L = 1e-6
max_iter = 100
fd_step = 1e-5
damping = 1
def relax_box_uniaxial(positions, connections, box, N=None):
    L_prev = (box[0]+box[1])/2  # using x (same as y) as the free dimension
    box[0] = box[1] = L_prev        # enforce symmetry immediately
    # Initial relaxation at current box
    positions = func(positions, connections, box=box,N=N)
    stress_prev = compute_total_stress(positions, connections, box=box,N=N)
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
        print(f"L={L_curr:.8f}, f={f_curr:.6e}, sx={stress_curr[0]:.6e}, sy={stress_curr[1]:.6e}, sz={stress_curr[2]:.6e}")
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
        print(f"L={L_curr:.6f}, f={f_curr:.6e}, scale={scale:.6f}, "
      f"sx={stress_curr[0]:.4e}, sy={stress_curr[1]:.4e}, sz={stress_curr[2]:.4e}")
    raise RuntimeError('Secant method did not converge')
print("")
print("here is the box before relaxation: ",box)
positions,stress,box = relax_box_uniaxial(positions,connections,box,N=chainLength+1)
stress = stress[2] - (stress[0]+stress[1])/2
print("here is the box after relaxation: ",box)
print("")
np.savetxt(
    "initial_equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    comments=''
)

#print("strain-controlled deformation with bond breakage...")
# ----------------------------
# Parameters
# ----------------------------
total_strain = 700     # total strain (%) applied in z
n_steps = 200          # number of increments
dstrain = total_strain / n_steps   # strain increment per step (%)
box_curr = box

N = chainLength+1
b = np.sqrt(10)
U_crit = (N*b)**2/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain

if False:
    connections,broken = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,N=chainLength+1,
                                                            log_file='broken_connections.txt')
total_initial_bonds = count_bonds(connections)
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
    stop_deformation = False
    while broken:
        positions = func(positions, connections,box=box_curr,N=chainLength+1)
        connections,broken = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,N=chainLength+1,
                                                        log_file='broken_connections.txt')
        stress = compute_total_stress(positions, connections, box_curr,N=chainLength+1)
        stress = stress[2] - (stress[0]+stress[1])/2
        if stress < 0:
            stress = 0
        if stress_total[-1] > stress:
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
            stop_deformation = True
            break
    if stop_deformation:
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
    previous_stress = stress_total[-1]
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
    if previous_stress > stress:
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

