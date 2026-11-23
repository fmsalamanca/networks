from time import time
import numpy as np
import numba
import os
import shutil
import random
from matplotlib import pyplot as plt

CHAIN_EDGE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ideal-chains", "crosslink_chain_edges.txt")
)

@numba.njit
def func(positions, connections, box, max_steps=10000, b=np.sqrt(10), N=None, tolerance=1e-4, energy_output=None):
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
            #energy = sum_distance_squared
            new_positions[i, 0] = new_pos0
            new_positions[i, 1] = new_pos1
            new_positions[i, 2] = new_pos2
            current_energy += energy

        energy_diff = np.abs(prev_energy-current_energy)
        if energy_output is not None:
            energy_output[step] = current_energy
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
                energy = (3/2)*(delta[0]**2 + delta[1]**2 + delta[2]**2)/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain
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
    return forcevector


def bond_entries(connections, n_positions=None):
    sources = connections[:, :1]
    neighbors = connections[:, 1:]
    source_ids = np.broadcast_to(sources, neighbors.shape)
    valid = (
        (source_ids >= 0)
        & (neighbors >= 0)
        & (neighbors != source_ids)
    )
    if n_positions is not None:
        valid &= (source_ids < n_positions) & (neighbors < n_positions)
    return np.column_stack((source_ids[valid], neighbors[valid]))


def count_bonds(connections, n_positions=None):
    return len(bond_entries(connections, n_positions))

def young_modulus_affine(positions, connections, box, N, b=np.sqrt(10), axis=2):
    """
    Módulo de Young (tracción uniaxial incompresible, F = diag(lam, lam^-1/2, lam^-1/2)
    a lo largo de `axis`) a partir de las distancias entre partículas del estado equilibrado.
    Unidades: kT / (unidad de longitud)^3.
    """
    k = 1.0 / (N * b**2)                        # E_enlace = k r^2 / 2  (en kT)

    # lista de enlaces únicos (no asumo simetría ni cuento dos veces)
    conn = connections[:, 1:]
    ii = np.repeat(connections[:, 0], conn.shape[1])
    jj = conn.ravel()
    m = (jj >= 0) & (jj < len(positions)) & (jj != ii)
    pairs = np.unique(np.sort(np.column_stack((ii[m], jj[m])), axis=1), axis=0)

    # vectores de enlace con imagen mínima (igual que en func)
    d = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    d = (d + 0.5 * box) % box - 0.5 * box
    r2 = np.sum(d**2, axis=1)
    dx2 = d[:, axis]**2
    s = 2 * dx2 - (r2 - dx2)                    # 2Δ_par² − Δ_perp²

    EY  = k * np.sum(r2) / len(d)                    # d²E/dλ² / V
    
    dev = 0.5 * k * np.sum(s) / len(d)                # tensión desviadora (debe ser ~0)

    # fuerza residual por partícula: k·Σ_j Δ_ij (debe ser ~0 si está equilibrado)
    F = np.zeros_like(positions)
    np.add.at(F, pairs[:, 0],  k * d)
    np.add.at(F, pairs[:, 1], -k * d)
    fmax = np.abs(F).max()

    return EY, dev, fmax, len(pairs)

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

L_prev = box_x
first_call_energy = np.full(10000, np.nan, dtype=np.float64)
pos_prev = func(positions, connections, box, N=chainLength+1, energy_output=first_call_energy)
energy_history = first_call_energy[~np.isnan(first_call_energy)]
energy_steps = np.flatnonzero(~np.isnan(first_call_energy)) + 1

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

positions,stress,box = relax_box_uniaxial(positions,connections,box,N=chainLength+1)
os.makedirs("./output", exist_ok=True)
if len(energy_history) > 0:
    plt.figure()
    plt.plot(np.log(energy_steps), energy_history)
    plt.xlabel("log(MCS)")
    plt.ylabel("Total energy (kT)")
    plt.savefig("./output/energy_vs_logmcs.png")
    plt.close()

chain_edge_map = np.genfromtxt(CHAIN_EDGE_FILE, delimiter="\t", names=True, dtype=int)
if (
    chain_edge_map.dtype.names is None
    or not {"chainID", "xlinkA", "xlinkB"}.issubset(chain_edge_map.dtype.names)
):
    raise ValueError(f"{CHAIN_EDGE_FILE} must contain chainID, xlinkA, and xlinkB columns.")
chain_edge_map = np.atleast_1d(chain_edge_map)
chain_ids = chain_edge_map["chainID"]
xlink_a = chain_edge_map["xlinkA"]
xlink_b = chain_edge_map["xlinkB"]
if np.any(chain_ids < 1) or np.any(xlink_a < 0) or np.any(xlink_b < 0):
    raise ValueError(f"{CHAIN_EDGE_FILE} contains invalid chain or crosslink IDs.")

has_both_crosslinks = (xlink_a > 0) & (xlink_b > 0)
chain_pairs = np.column_stack(
    (xlink_a[has_both_crosslinks], xlink_b[has_both_crosslinks])
)
connection_pairs = bond_entries(connections, len(positions))
connection_pairs = connection_pairs[connection_pairs[:, 0] < connection_pairs[:, 1]] + 1
chain_order = np.lexsort((chain_pairs[:, 1], chain_pairs[:, 0]))
connection_order = np.lexsort((connection_pairs[:, 1], connection_pairs[:, 0]))
if not np.array_equal(
    chain_pairs[chain_order], connection_pairs[connection_order]
):
    raise ValueError("Chain edge mapping does not match connected_xlinks.txt.")
if np.any(chain_pairs > len(positions)):
    raise ValueError("Chain edge mapping contains crosslink IDs outside the positions table.")

bond_lengths = np.zeros(len(chain_ids), dtype=np.float64)
bond_pairs = np.column_stack(
    (xlink_a[has_both_crosslinks] - 1, xlink_b[has_both_crosslinks] - 1)
)
bond_delta = positions[bond_pairs[:, 1]] - positions[bond_pairs[:, 0]]
bond_delta = (bond_delta + 0.5 * box) % box - 0.5 * box
bond_lengths[has_both_crosslinks] = np.linalg.norm(bond_delta, axis=1)
np.savetxt(
    "./output/chain_bond_lengths.txt",
    np.column_stack((chain_ids, xlink_a, xlink_b, bond_lengths)),
    fmt=["%d", "%d", "%d", "%.8f"],
    delimiter="\t",
    header="chainID\txlinkA\txlinkB\tbondLength",
    comments="",
)

stress = stress[2] - (stress[0]+stress[1])/2
np.savetxt(
    "initial_equilibrated_positions.txt",
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t",
    comments=''
)

EY, dev, fmax, nb = young_modulus_affine(positions, connections, box, N=chainLength+1, axis=2)
f = connections.shape[1]-1  # número de conexiones por partícula (first col is id)
EYid = (f - 2) / f # Adjusted Young's modulus for ideal chains
print(f"E_Y = {EY:.6e} kT/L^3  (G = {EY/3:.6e})")
print(f"E_Y/E_Y(id) = {EY/EYid:.6e} (ideal chain model with f = {f})")
print(f"tensión desviadora = {dev:.3e}, fuerza residual máx = {fmax:.3e}, enlaces = {nb}")
print(count_bonds(connections, len(positions)), "bonds in the system")
#print("strain-controlled deformation with bond breakage...")
# ----------------------------
# Parameters
# ----------------------------
total_strain = 120     # total strain (%) applied in z
n_steps = 200          # number of increments
dstrain = total_strain / n_steps   # strain increment per step (%)
scale_2_step = min(
    range(n_steps + 1),
    key=lambda candidate_step: abs(1 + candidate_step * dstrain / 100 - 2.0))
scale_3_step = min(
    range(n_steps + 1),
    key=lambda candidate_step: abs(1 + candidate_step * dstrain / 100 - 3.0))
box_curr = box

N = chainLength+1
b = np.sqrt(10)
U_crit = (N*b)**2/(N*b**2) # in units of kT, using ideal chain model with Kuhn length b and n segments per chain

if False:
    connections,broken = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,N=chainLength+1,
                                                            log_file='broken_connections.txt')
total_initial_bonds = count_bonds(connections, len(positions))

# Store initial state for linear strain application
initial_box = box_curr.copy()
initial_positions = positions.copy()

# ----------------------------
# Incremental deformation loop
# ----------------------------
stress_total = [stress]
scale_z = 1
strain_total = [scale_z]
for step in range(1, n_steps + 1):
    stress = 0
    
    if step>1:
        positions = func(positions, connections,box=box_curr,N=chainLength+1)

    current_scale_step = step - 1
    if current_scale_step == scale_2_step:
        EY, dev, fmax, nb = young_modulus_affine(positions, connections, box_curr, N=chainLength+1, axis=2)
        print(f"Step {current_scale_step}: E_Y near 100% deformation "
            f"(actual {scale_z:.6f})= {EY:.6e} kT/L^3  (G = {EY/3:.6e})")
    if current_scale_step == scale_3_step:
        EY, dev, fmax, nb = young_modulus_affine(positions, connections, box_curr, N=chainLength+1, axis=2)
        print(f"Step {current_scale_step}: E_Y near 200% deformation "
            f"(actual {scale_z:.6f})= {EY:.6e} kT/L^3  (G = {EY/3:.6e})")
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
                f.write(f"Remaining bonds: {count_bonds(connections, len(positions))}/{total_initial_bonds}\n")

            print(f"  [Checkpoint saved → {checkpoint_dir}]", flush=True)
            stop_deformation = True
            break
    if stop_deformation:
        break
    with open('./output/breakage_log.txt', 'a') as f:
        if step == 1:
            f.write("step lambda cumulative_broken remaining stress\n")
        f.write(f"{step} {scale_z:.6f} "
                f"{total_initial_bonds - count_bonds(connections, len(positions))} "
                f"{count_bonds(connections, len(positions))} "
                f"{stress:.6f}\n")
    
    remaining_bonds = count_bonds(connections, len(positions))
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
        f.write(f"Remaining bonds: {count_bonds(connections, len(positions))}/{total_initial_bonds}\n")

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
            f.write(f"Remaining bonds: {count_bonds(connections, len(positions))}/{total_initial_bonds}\n")

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

