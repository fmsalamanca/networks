from time import time
import math
import numpy as np
import numba
import os
import shutil
import random
from matplotlib import pyplot as plt

OUTPUT_DIR = "output"
DAMPING_FACTOR = 1.0
BFM_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "force-balance")
)
BFM_EXTRACTED_TABLES_DIR = os.path.join(BFM_DATA_DIR, "extracted_tables")
POSITIONS_FILE = "crosslinks-positions.txt"
CONNECTIONS_FILE = "connected_xlinks.txt"
SYSTEM_FILE = "system.txt"
BFM_DISTANCE_TABLE_FILE = os.path.join(
    BFM_EXTRACTED_TABLES_DIR, "crosslink_squared_distances.txt"
)
BFM_CHAIN_EDGE_FILE = os.path.join(
    BFM_EXTRACTED_TABLES_DIR, "crosslink_chain_edges.txt"
)
energy_history = []
energy_steps = []
energy_recorded = False


def _compute_total_energy(positions, connections, box, N, b):
    row_energies = np.zeros(connections.shape[0], dtype=np.float64)
    for row in range(connections.shape[0]):
        i = connections[row, 0]
        if i < 0 or i >= len(positions):
            continue

        sum_distance_squared = 0.0
        for k in range(1, connections.shape[1]):
            conn_val = connections[row, k]
            if 0 <= conn_val < len(positions):
                delta0 = positions[conn_val, 0] - positions[i, 0]
                delta1 = positions[conn_val, 1] - positions[i, 1]
                delta2 = positions[conn_val, 2] - positions[i, 2]
                delta0 -= box[0] * math.floor(delta0 / box[0] + 0.5)
                delta1 -= box[1] * math.floor(delta1 / box[1] + 0.5)
                delta2 -= box[2] * math.floor(delta2 / box[2] + 0.5)
                sum_distance_squared += delta0**2 + delta1**2 + delta2**2
        row_energies[row] = 1.5 * sum_distance_squared / (N * b**2)

    # Match the GPU's 256-thread accumulation and tree-reduction order.
    partial_energies = np.zeros(256, dtype=np.float64)
    for thread in range(256):
        row = thread
        while row < len(row_energies):
            partial_energies[thread] += row_energies[row]
            row += 256

    stride = 128
    while stride > 0:
        for thread in range(stride):
            partial_energies[thread] += partial_energies[thread + stride]
        stride //= 2
    return partial_energies[0]


def func(
    positions,
    connections,
    box,
    max_steps=100000,
    b=2.636,
    N=None,
    tolerance=1e-8,
    damping_factor=DAMPING_FACTOR,
):
    global energy_recorded
    record_energy = not energy_recorded
    energy_recorded = True
    if connections.shape[0] == 0:
        return positions.copy()
    if N is None:
        raise ValueError("N must be specified for CPU relaxation.")
    if N <= 0 or b <= 0:
        raise ValueError("N and b must be positive for CPU relaxation.")
    if not 0.0 <= damping_factor <= 1.0:
        raise ValueError("damping_factor must be in the interval [0, 1].")

    positions = np.ascontiguousarray(positions)
    connections = np.ascontiguousarray(connections)
    box = np.ascontiguousarray(box)
    step = 0
    current_energy = _compute_total_energy(positions, connections, box, N, b)
    energy_diff = np.inf
    while step < max_steps:
        step_scale = 1.0
        backtracks = 0
        while True:
            new_positions = positions.copy()
            row_order = np.random.permutation(connections.shape[0])
            for row in row_order:
                i = connections[row, 0]
                pos0 = positions[i, 0]
                pos1 = positions[i, 1]
                pos2 = positions[i, 2]
                sum_delta0 = 0.0
                sum_delta1 = 0.0
                sum_delta2 = 0.0
                count = 0

                for k in range(1, connections.shape[1]):
                    conn_val = connections[row, k]
                    if 0 <= conn_val < len(positions):
                        delta0 = positions[conn_val, 0] - pos0
                        delta1 = positions[conn_val, 1] - pos1
                        delta2 = positions[conn_val, 2] - pos2
                        delta0 -= box[0] * math.floor(delta0 / box[0] + 0.5)
                        delta1 -= box[1] * math.floor(delta1 / box[1] + 0.5)
                        delta2 -= box[2] * math.floor(delta2 / box[2] + 0.5)
                        sum_delta0 += delta0
                        sum_delta1 += delta1
                        sum_delta2 += delta2
                        if i != conn_val:
                            count += 1

                if count > 0:
                    effective_damping = damping_factor * step_scale
                    mean_delta0 = sum_delta0 / count
                    mean_delta1 = sum_delta1 / count
                    mean_delta2 = sum_delta2 / count
                    new_pos0 = pos0 + effective_damping * mean_delta0
                    new_pos1 = pos1 + effective_damping * mean_delta1
                    new_pos2 = pos2 + effective_damping * mean_delta2
                    new_pos0 -= box[0] * math.floor(new_pos0 / box[0])
                    new_pos1 -= box[1] * math.floor(new_pos1 / box[1])
                    new_pos2 -= box[2] * math.floor(new_pos2 / box[2])
                    new_positions[i, 0] = new_pos0
                    new_positions[i, 1] = new_pos1
                    new_positions[i, 2] = new_pos2

            candidate_energy = _compute_total_energy(
                new_positions, connections, box, N, b
            )
            energy_diff = abs(current_energy - candidate_energy)
            if np.isfinite(candidate_energy) and candidate_energy <= current_energy:
                break
            if backtracks == 32:
                break
            step_scale *= 0.5
            backtracks += 1

        if not np.isfinite(candidate_energy) or candidate_energy > current_energy:
            break

        positions = new_positions
        current_energy = candidate_energy
        step += 1
        if record_energy:
            energy_steps.append(step)
            energy_history.append(current_energy)
        if energy_diff < tolerance:
            break

    print(
        f"Equilibration finished in {step} steps with "
        f"energy difference={energy_diff:.30f} and final energy={current_energy:.0f}"
    )
    return positions

def breakage_potential(positions, connections, box, U_crit,b=2.636,N=None,log_file=None):
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

def compute_total_stress(positions, connections, box,N=None,b=2.636):
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

def compute_total_force(positions, connections, box,N=None,b=2.636):
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

def young_modulus_affine(positions, connections, box, N, b=2.636, axis=2):
    """Return a strand-averaged normalized extension and affine diagnostics.

    For each eligible strand, ``r`` is the minimum-image end-to-end vector.
    The first result is

        g_bar = mean(|r|^2 / (N * b^2))

    where ``N * b^2`` is the ideal-chain mean-square size. Reciprocal
    connection entries represent one strand, parallel strands retain their
    multiplicity, and self-links are excluded. Thus ``g_bar`` is a
    dimensionless per-strand average, not the bulk modulus itself.

    Under the model ``G = g * nu * kBT``, form ``G`` by applying the chosen
    numerical correction to ``g_bar`` and multiplying by ``nu`` (the number
    density of the same strand population) and ``kBT``. ``dev`` is the
    strand-averaged deviatoric contribution along ``axis``; ``fmax`` is the
    maximum residual force magnitude. The return value also includes the
    number of eligible strand instances.
    """
    k = 1.0 / (N * b**2)  # inverse ideal-chain mean-square size
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    # Reciprocal rows describe one link; repeated entries preserve multiplicity.
    conn = connections[:, 1:]
    ii = np.repeat(connections[:, 0], conn.shape[1])
    jj = conn.ravel()
    m = (jj >= 0) & (jj < len(positions)) & (jj != ii)
    directed_pairs = np.column_stack((ii[m], jj[m]))
    pairs, pair_indices = np.unique(
        np.sort(directed_pairs, axis=1), axis=0, return_inverse=True
    )
    source_is_larger = directed_pairs[:, 0] > directed_pairs[:, 1]
    directional_counts = np.zeros((len(pairs), 2), dtype=np.int64)
    np.add.at(
        directional_counts,
        (pair_indices, source_is_larger.astype(np.intp)),
        1,
    )
    pairs = np.repeat(pairs, directional_counts.max(axis=1), axis=0)

    pair_count = len(pairs)
    if pair_count == 0:
        return 0.0, 0.0, 0.0, 0

    d = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    d -= box * np.floor(d / box + 0.5)
    r2 = d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2
    dx2 = d[:, axis]**2
    s = 2 * dx2 - (r2 - dx2)

    partial_r2 = np.zeros(256, dtype=np.float64)
    partial_s = np.zeros(256, dtype=np.float64)
    for thread in range(256):
        pair = thread
        while pair < pair_count:
            partial_r2[thread] += r2[pair]
            partial_s[thread] += s[pair]
            pair += 256
    stride = 128
    while stride > 0:
        for thread in range(stride):
            partial_r2[thread] += partial_r2[thread + stride]
            partial_s[thread] += partial_s[thread + stride]
        stride //= 2
    sum_r2 = partial_r2[0]
    sum_s = partial_s[0]
    g_bar = k * sum_r2 / pair_count
    dev = 0.5 * k * sum_s / pair_count

    F = np.zeros_like(positions)
    np.add.at(F, pairs[:, 0],  k * d)
    np.add.at(F, pairs[:, 1], -k * d)
    fmax = np.abs(F).max()

    print(
        pair_count,
        "strands, k =",
        k,
        "sum(r2) =",
        sum_r2,
        "g_bar =",
        g_bar,
    )
    return g_bar, dev, fmax, pair_count

###################################################

t0 = time()
# Load connection table (first column = ID, next columns = connected IDs)
connections = np.loadtxt(
    CONNECTIONS_FILE, skiprows=1, delimiter="\t", dtype=int
) - 1
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
with open(SYSTEM_FILE, "r") as f:
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

positions = np.loadtxt(POSITIONS_FILE, delimiter="\t", ndmin=2)
expected_position_rows = int(number_of_crosslinkers)
if positions.shape != (expected_position_rows, 3):
    raise ValueError(
        f"{POSITIONS_FILE} contains {len(positions)} rows; "
        f"expected {expected_position_rows} crosslink positions."
    )
if np.any(connections[:, 0] < 0) or np.any(connections[:, 0] >= len(positions)):
    raise ValueError("Connection row IDs must refer to valid crosslink positions.")
if connections.shape[0] != len(positions) or not np.array_equal(
    np.sort(connections[:, 0]), np.arange(len(positions))
):
    raise ValueError(
        "CPU relaxation requires exactly one connection row for every crosslink position."
    )

box = np.array([box_x, box_y, box_z], dtype=positions.dtype)
periodic = np.array(
    [bool(params.get(f"periodic_{axis}", 1)) for axis in ("x", "y", "z")]
)

distance_table = np.genfromtxt(
    BFM_DISTANCE_TABLE_FILE,
    delimiter="\t",
    names=True,
    dtype=None,
    encoding="utf-8",
)
distance_table = np.atleast_1d(distance_table)
required_distance_columns = {
    "xlinkID_BFM",
    "connectedXlinkID_BFM",
    "squaredDistance",
    "chainID",
}
if (
    distance_table.dtype.names is None
    or not required_distance_columns.issubset(distance_table.dtype.names)
):
    raise ValueError(
        f"{BFM_DISTANCE_TABLE_FILE} must contain "
        "xlinkID_BFM, connectedXlinkID_BFM, squaredDistance, and chainID columns."
    )

xlink_a_bfm = distance_table["xlinkID_BFM"].astype(int)
xlink_b_bfm = distance_table["connectedXlinkID_BFM"].astype(int)
distance_chain_ids = distance_table["chainID"].astype(int)
reference_squared_distances = distance_table["squaredDistance"].astype(float)
xlink_bfm_offset = int(chainLength * number_of_linear_chains)
xlink_a = xlink_a_bfm - xlink_bfm_offset
xlink_b = xlink_b_bfm - xlink_bfm_offset
if (
    np.any(distance_chain_ids < 1)
    or np.any(distance_chain_ids > number_of_linear_chains)
    or np.any(xlink_a < 1)
    or np.any(xlink_a > len(positions))
    or np.any(xlink_b < 1)
    or np.any(xlink_b > len(positions))
    or np.any(~np.isfinite(reference_squared_distances))
    or np.any(reference_squared_distances < 0)
):
    raise ValueError(f"{BFM_DISTANCE_TABLE_FILE} contains invalid IDs or distances.")

initial_delta = positions[xlink_b - 1] - positions[xlink_a - 1]
initial_delta[:, periodic] -= box[periodic] * np.floor(
    initial_delta[:, periodic] / box[periodic] + 0.5
)
computed_squared_distances = np.einsum("ij,ij->i", initial_delta, initial_delta)
if not np.allclose(
    computed_squared_distances, reference_squared_distances, rtol=0, atol=1e-8
):
    raise ValueError(
        f"{BFM_DISTANCE_TABLE_FILE} squared distances do not match "
        f"{POSITIONS_FILE}."
    )

chain_edge_map = np.genfromtxt(
    BFM_CHAIN_EDGE_FILE,
    delimiter="\t",
    names=True,
    dtype=None,
    encoding="utf-8",
)
chain_edge_map = np.atleast_1d(chain_edge_map)
required_chain_edge_columns = {"chainID", "xlinkA", "xlinkB"}
if (
    chain_edge_map.dtype.names is None
    or not required_chain_edge_columns.issubset(chain_edge_map.dtype.names)
):
    raise ValueError(
        f"{BFM_CHAIN_EDGE_FILE} must contain chainID, xlinkA, and xlinkB columns."
    )

chain_ids = chain_edge_map["chainID"].astype(int)
chain_xlink_a = chain_edge_map["xlinkA"].astype(int)
chain_xlink_b = chain_edge_map["xlinkB"].astype(int)
if (
    len(chain_ids) != int(number_of_linear_chains)
    or not np.array_equal(
        np.sort(chain_ids), np.arange(1, int(number_of_linear_chains) + 1)
    )
    or np.any(chain_xlink_a < 0)
    or np.any(chain_xlink_a > len(positions))
    or np.any(chain_xlink_b < 0)
    or np.any(chain_xlink_b > len(positions))
    or np.any((chain_xlink_a == 0) & (chain_xlink_b > 0))
):
    raise ValueError(f"{BFM_CHAIN_EDGE_FILE} contains invalid or incomplete chain rows.")

has_chain_pair = (chain_xlink_a > 0) & (chain_xlink_b > 0)
distance_edge_rows = np.column_stack((distance_chain_ids, xlink_a, xlink_b))
chain_edge_rows = np.column_stack(
    (
        chain_ids[has_chain_pair],
        chain_xlink_a[has_chain_pair],
        chain_xlink_b[has_chain_pair],
    )
)
distance_edge_order = np.lexsort(
    (distance_edge_rows[:, 2], distance_edge_rows[:, 1], distance_edge_rows[:, 0])
)
chain_edge_order = np.lexsort(
    (chain_edge_rows[:, 2], chain_edge_rows[:, 1], chain_edge_rows[:, 0])
)
if not np.array_equal(
    distance_edge_rows[distance_edge_order],
    chain_edge_rows[chain_edge_order],
):
    raise ValueError(
        f"{BFM_CHAIN_EDGE_FILE} does not match {BFM_DISTANCE_TABLE_FILE}."
    )

L_prev = box_x
positions = func(
    positions,
    connections,
    box,
    max_steps=100000,
    N=chainLength + 1,
    tolerance=1e-8,
    damping_factor=DAMPING_FACTOR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
bond_lengths_squared = np.zeros(len(chain_ids), dtype=np.float64)
bond_delta = positions[chain_xlink_b[has_chain_pair] - 1] - positions[
    chain_xlink_a[has_chain_pair] - 1
]
bond_delta[:, periodic] -= box[periodic] * np.floor(
    bond_delta[:, periodic] / box[periodic] + 0.5
)
bond_lengths_squared[has_chain_pair] = np.einsum(
    "ij,ij->i", bond_delta, bond_delta
)
chain_xlink_a_bfm = np.where(
    chain_xlink_a > 0, xlink_bfm_offset + chain_xlink_a, 0
)
chain_xlink_b_bfm = np.where(
    chain_xlink_b > 0, xlink_bfm_offset + chain_xlink_b, 0
)
bond_length_order = np.argsort(chain_ids)
np.savetxt(
    os.path.join(OUTPUT_DIR, "chain_bond_lengths.txt"),
    np.column_stack(
        (chain_ids, chain_xlink_a_bfm, chain_xlink_b_bfm, bond_lengths_squared)
    )[bond_length_order],
    fmt=["%d", "%d", "%d", "%.8f"],
    delimiter="\t",
    header="chainID\txlinkA\txlinkB\tbondLength**2",
    comments="",
)

g_bar, dev, fmax, nb = young_modulus_affine(
    positions, connections, box, N=chainLength + 1, axis=2
)
# For N=8, f=3: b=2.636 (doi:10.1103/physreve.76.011803),
# Rf=9.437 (doi:10.1021/acs.macromol.2c00660).
correction = 2.636**2 / 9.437
g_corrected = g_bar * correction

f = connections.shape[1] - 1
g_ideal = (f - 2) / f
print(f"Pre-box-relaxation corrected g = {g_corrected:.6e}")
print(
    f"Pre-box-relaxation corrected g/g_ideal = {g_corrected/g_ideal:.6e} "
    f"(ideal chain model with f = {f})"
)


np.savez_compressed(
    os.path.join(OUTPUT_DIR, "equilibration_state.npz"),
    positions=positions,
    connections=connections,
    box=box,
    energy_history=energy_history,
    energy_steps=energy_steps,
)
print(
    "Equilibration complete. "
    "Positions, connections, box dimensions, and energy history saved to "
    f"{OUTPUT_DIR}"
)
raise SystemExit(0)
