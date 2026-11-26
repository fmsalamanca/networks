from time import time
import math
import numpy as np
import numba
from numba import cuda, float64
import os
import shutil
import random
from matplotlib import pyplot as plt

OUTPUT_DIR = "output-gpu"
BROKEN_CONNECTIONS_LOG = os.path.join(OUTPUT_DIR, "broken_connections.txt")
BFM_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "force-balance"))
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


@cuda.jit
def _relax_step_kernel(
    positions, connections, row_order, box, N, b, new_positions, row_energies
):
    row_slot = cuda.grid(1)
    if row_slot >= row_order.shape[0]:
        return

    # Map the randomized execution slot back to its original connection row.
    row = row_order[row_slot]
    row_energies[row] = 0.0
    i = connections[row, 0]
    pos0 = positions[i, 0]
    pos1 = positions[i, 1]
    pos2 = positions[i, 2]
    sum_delta0 = pos0 * 0.0
    sum_delta1 = pos1 * 0.0
    sum_delta2 = pos2 * 0.0
    sum_distance_squared = 0.0
    count = 0

    for k in range(1, connections.shape[1]):
        conn_val = connections[row, k]
        if conn_val >= 0 and conn_val < positions.shape[0]:
            delta0 = positions[conn_val, 0] - pos0
            delta1 = positions[conn_val, 1] - pos1
            delta2 = positions[conn_val, 2] - pos2
            delta0 -= box[0] * math.floor(delta0 / box[0] + 0.5)
            delta1 -= box[1] * math.floor(delta1 / box[1] + 0.5)
            delta2 -= box[2] * math.floor(delta2 / box[2] + 0.5)
            sum_delta0 += delta0
            sum_delta1 += delta1
            sum_delta2 += delta2
            sum_distance_squared += delta0**2 + delta1**2 + delta2**2
            if i != conn_val:
                count += 1

    if count > 0:
        new_pos0 = pos0 + sum_delta0 / count
        new_pos1 = pos1 + sum_delta1 / count
        new_pos2 = pos2 + sum_delta2 / count
        new_pos0 -= box[0] * math.floor(new_pos0 / box[0])
        new_pos1 -= box[1] * math.floor(new_pos1 / box[1])
        new_pos2 -= box[2] * math.floor(new_pos2 / box[2])
        new_positions[i, 0] = new_pos0
        new_positions[i, 1] = new_pos1
        new_positions[i, 2] = new_pos2
        row_energies[row] = 1.5 * sum_distance_squared / (N * b**2)


@cuda.jit
def _reduce_energy_kernel(row_energies, total_energy, count):
    thread = cuda.threadIdx.x
    partial = 0.0
    index = thread
    while index < count:
        partial += row_energies[index]
        index += cuda.blockDim.x

    shared = cuda.shared.array(shape=256, dtype=float64)
    shared[thread] = partial
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if thread < stride:
            shared[thread] += shared[thread + stride]
        cuda.syncthreads()
        stride //= 2

    if thread == 0:
        total_energy[0] = shared[0]


@cuda.jit
def _breakage_candidates_kernel(positions, connections, box, N, b, U_crit, candidates, slot_count):
    flat_index = cuda.grid(1)
    if flat_index >= slot_count:
        return

    neighbor_count = connections.shape[1] - 1
    row = flat_index // neighbor_count
    k = flat_index % neighbor_count + 1
    candidates[row, k] = 0

    i = connections[row, 0]
    conn_val = connections[row, k]
    if i < 0 or i >= positions.shape[0] or conn_val < 0 or conn_val >= positions.shape[0] or conn_val == i:
        return

    pos0 = positions[i, 0] - box[0] * math.floor(positions[i, 0] / box[0])
    pos1 = positions[i, 1] - box[1] * math.floor(positions[i, 1] / box[1])
    pos2 = positions[i, 2] - box[2] * math.floor(positions[i, 2] / box[2])
    conn0 = positions[conn_val, 0] - box[0] * math.floor(positions[conn_val, 0] / box[0])
    conn1 = positions[conn_val, 1] - box[1] * math.floor(positions[conn_val, 1] / box[1])
    conn2 = positions[conn_val, 2] - box[2] * math.floor(positions[conn_val, 2] / box[2])
    delta0 = conn0 - pos0
    delta1 = conn1 - pos1
    delta2 = conn2 - pos2
    delta0 -= box[0] * math.floor(delta0 / box[0] + 0.5)
    delta1 -= box[1] * math.floor(delta1 / box[1] + 0.5)
    delta2 -= box[2] * math.floor(delta2 / box[2] + 0.5)
    energy = 1.5 * (delta0**2 + delta1**2 + delta2**2) / (N * b**2)
    if energy >= U_crit:
        candidates[row, k] = 1


@cuda.jit
def _stress_per_row_kernel(positions, connections, box, row_deltas):
    row = cuda.grid(1)
    if row >= connections.shape[0]:
        return

    row_deltas[row, 0] = 0.0
    row_deltas[row, 1] = 0.0
    row_deltas[row, 2] = 0.0
    i = connections[row, 0]
    if i < 0 or i >= positions.shape[0]:
        return

    pos0 = positions[i, 0]
    pos1 = positions[i, 1]
    pos2 = positions[i, 2]
    sum_delta0 = 0.0
    sum_delta1 = 0.0
    sum_delta2 = 0.0
    for k in range(1, connections.shape[1]):
        conn_val = connections[row, k]
        if conn_val >= 0 and conn_val < positions.shape[0] and conn_val != i:
            delta0 = positions[conn_val, 0] - pos0
            delta1 = positions[conn_val, 1] - pos1
            delta2 = positions[conn_val, 2] - pos2
            delta0 -= box[0] * math.floor(delta0 / box[0] + 0.5)
            delta1 -= box[1] * math.floor(delta1 / box[1] + 0.5)
            delta2 -= box[2] * math.floor(delta2 / box[2] + 0.5)
            sum_delta0 += abs(delta0)
            sum_delta1 += abs(delta1)
            sum_delta2 += abs(delta2)

    row_deltas[row, 0] = sum_delta0
    row_deltas[row, 1] = sum_delta1
    row_deltas[row, 2] = sum_delta2


@cuda.jit
def _reduce_stress_kernel(row_deltas, total_delta, row_count):
    thread = cuda.threadIdx.x
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0
    row = thread
    while row < row_count:
        sum0 += row_deltas[row, 0]
        sum1 += row_deltas[row, 1]
        sum2 += row_deltas[row, 2]
        row += cuda.blockDim.x

    shared = cuda.shared.array(shape=(3, 256), dtype=float64)
    shared[0, thread] = sum0
    shared[1, thread] = sum1
    shared[2, thread] = sum2
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if thread < stride:
            shared[0, thread] += shared[0, thread + stride]
            shared[1, thread] += shared[1, thread + stride]
            shared[2, thread] += shared[2, thread + stride]
        cuda.syncthreads()
        stride //= 2

    if thread == 0:
        total_delta[0] = shared[0, 0]
        total_delta[1] = shared[1, 0]
        total_delta[2] = shared[2, 0]


@cuda.jit
def _affine_pair_kernel(positions, pairs, box, k, axis, pair_r2, pair_s, force_x, force_y, force_z):
    pair = cuda.grid(1)
    if pair >= pairs.shape[0]:
        return

    i = pairs[pair, 0]
    j = pairs[pair, 1]
    delta0 = positions[j, 0] - positions[i, 0]
    delta1 = positions[j, 1] - positions[i, 1]
    delta2 = positions[j, 2] - positions[i, 2]
    delta0 -= box[0] * math.floor(delta0 / box[0] + 0.5)
    delta1 -= box[1] * math.floor(delta1 / box[1] + 0.5)
    delta2 -= box[2] * math.floor(delta2 / box[2] + 0.5)

    r2 = delta0**2 + delta1**2 + delta2**2
    if axis == 0:
        parallel2 = delta0**2
    elif axis == 1:
        parallel2 = delta1**2
    else:
        parallel2 = delta2**2

    pair_r2[pair] = r2
    pair_s[pair] = 2.0 * parallel2 - (r2 - parallel2)
    cuda.atomic.add(force_x, i, k * delta0)
    cuda.atomic.add(force_y, i, k * delta1)
    cuda.atomic.add(force_z, i, k * delta2)
    cuda.atomic.add(force_x, j, -k * delta0)
    cuda.atomic.add(force_y, j, -k * delta1)
    cuda.atomic.add(force_z, j, -k * delta2)


@cuda.jit
def _reduce_affine_kernel(pair_r2, pair_s, force_x, force_y, force_z, totals, pair_count, particle_count):
    thread = cuda.threadIdx.x
    sum_r2 = 0.0
    sum_s = 0.0
    index = thread
    while index < pair_count:
        sum_r2 += pair_r2[index]
        sum_s += pair_s[index]
        index += cuda.blockDim.x

    max_force = 0.0
    index = thread
    while index < particle_count:
        component_max = abs(force_x[index])
        if abs(force_y[index]) > component_max:
            component_max = abs(force_y[index])
        if abs(force_z[index]) > component_max:
            component_max = abs(force_z[index])
        if component_max > max_force:
            max_force = component_max
        index += cuda.blockDim.x

    shared = cuda.shared.array(shape=(3, 256), dtype=float64)
    shared[0, thread] = sum_r2
    shared[1, thread] = sum_s
    shared[2, thread] = max_force
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if thread < stride:
            shared[0, thread] += shared[0, thread + stride]
            shared[1, thread] += shared[1, thread + stride]
            if shared[2, thread + stride] > shared[2, thread]:
                shared[2, thread] = shared[2, thread + stride]
        cuda.syncthreads()
        stride //= 2

    if thread == 0:
        totals[0] = shared[0, 0]
        totals[1] = shared[1, 0]
        totals[2] = shared[2, 0]


def _compute_total_delta_gpu(positions, connections, box):
    if connections.shape[0] == 0:
        return np.zeros(3, dtype=positions.dtype)

    positions = np.ascontiguousarray(positions)
    connections = np.ascontiguousarray(connections)
    box = np.ascontiguousarray(box)
    threads_per_block = 256
    blocks = (connections.shape[0] + threads_per_block - 1) // threads_per_block
    device_positions = cuda.to_device(positions)
    device_connections = cuda.to_device(connections)
    device_box = cuda.to_device(box)
    row_deltas = cuda.device_array((connections.shape[0], 3), dtype=np.float64)
    total_delta = cuda.device_array(3, dtype=np.float64)

    _stress_per_row_kernel[blocks, threads_per_block](
        device_positions, device_connections, device_box, row_deltas
    )
    _reduce_stress_kernel[1, threads_per_block](
        row_deltas, total_delta, connections.shape[0]
    )
    return total_delta.copy_to_host()


def func(positions, connections, box, max_steps=10000, b=np.sqrt(10), N=None, tolerance=1e-8):
    global energy_recorded
    record_energy = not energy_recorded
    energy_recorded = True
    if connections.shape[0] == 0:
        return positions.copy()
    if N is None:
        raise ValueError("N must be specified for GPU relaxation.")

    positions = np.ascontiguousarray(positions)
    connections = np.ascontiguousarray(connections)
    box = np.ascontiguousarray(box)
    threads_per_block = 256
    blocks = (connections.shape[0] + threads_per_block - 1) // threads_per_block
    device_positions = cuda.to_device(positions)
    device_connections = cuda.to_device(connections)
    device_box = cuda.to_device(box)
    next_positions = cuda.device_array_like(device_positions)
    device_row_order = cuda.device_array(connections.shape[0], dtype=np.int32)
    row_energies = cuda.device_array(connections.shape[0], dtype=np.float64)
    total_energy = cuda.device_array(1, dtype=np.float64)

    step = 0
    prev_energy = np.inf
    while step < max_steps:
        next_positions.copy_to_device(device_positions)
        # Visit every cross-link row once per Monte Carlo step in random order.
        row_order = np.random.permutation(connections.shape[0]).astype(np.int32)
        device_row_order.copy_to_device(row_order)
        _relax_step_kernel[blocks, threads_per_block](
            device_positions,
            device_connections,
            device_row_order,
            device_box,
            N,
            b,
            next_positions,
            row_energies,
        )
        _reduce_energy_kernel[1, threads_per_block](
            row_energies, total_energy, connections.shape[0]
        )
        current_energy = total_energy.copy_to_host()[0]
        if record_energy:
            energy_steps.append(step + 1)
            energy_history.append(current_energy)

        energy_diff = np.abs(prev_energy-current_energy)
        device_positions, next_positions = next_positions, device_positions
        if energy_diff < tolerance:
            break
        prev_energy = current_energy
        step += 1

    print(
        f"Equilibration finished in {step} steps with "
        f"energy difference={energy_diff:.30f} and final energy={current_energy:.0f}"
    )
    return device_positions.copy_to_host()

def breakage_potential(positions, connections, box, U_crit,b=np.sqrt(10),N=None,log_file=None):
    if N is None:
        raise ValueError("N must be specified for GPU bond-breakage checks.")

    broken = False
    rows = np.arange(connections.shape[0])
    np.random.shuffle(rows)
    neighbor_count = connections.shape[1] - 1
    if neighbor_count <= 0 or connections.shape[0] == 0:
        return connections, broken

    positions = np.ascontiguousarray(positions)
    connections = np.ascontiguousarray(connections)
    box = np.ascontiguousarray(box)
    slot_count = connections.shape[0] * neighbor_count
    threads_per_block = 256
    blocks = (slot_count + threads_per_block - 1) // threads_per_block
    device_positions = cuda.to_device(positions)
    device_connections = cuda.to_device(connections)
    device_box = cuda.to_device(box)
    candidates = cuda.device_array(connections.shape, dtype=np.uint8)
    _breakage_candidates_kernel[blocks, threads_per_block](
        device_positions,
        device_connections,
        device_box,
        N,
        b,
        U_crit,
        candidates,
        slot_count,
    )
    candidates = candidates.copy_to_host()
    row_by_id = np.full(len(positions), -1, dtype=int)
    row_by_id[connections[:, 0]] = np.arange(connections.shape[0])
    broken_pairs = []

    for row in rows:
        i = connections[row, 0]
        if i < 0 or i >= len(positions):
            continue
        for k in range(1,connections.shape[1]): # 1 is to ignore self connection
            conn_val = connections[row, k]
            if (
                candidates[row, k]
                and conn_val >= 0
                and conn_val < len(positions)
                and conn_val != i
            ):
                broken = True
                connections[row, k] = -1

                reverse_row = row_by_id[conn_val]
                if reverse_row >= 0:
                    reverse_idx = np.where(connections[reverse_row, 1:] == i)[0]
                    if len(reverse_idx) > 0:
                        connections[reverse_row, reverse_idx[0] + 1] = -1
                broken_pairs.append(tuple(sorted((i, conn_val))))

    if log_file is not None and broken_pairs:
        with open(log_file, "a") as f:
            for bond_pair in broken_pairs:
                f.write(f"{bond_pair[0]} {bond_pair[1]}\n")

    return connections, broken

def compute_total_stress(positions, connections, box,N=None,b=np.sqrt(10)):
    if N is None:
        raise ValueError("N must be specified for GPU stress calculation.")
    total_delta = _compute_total_delta_gpu(positions, connections, box)
    forcevector = 3/(N*b**2)*total_delta    #eq 2.96 from rubinstein, in units of kT, using ideal chain model with Kuhn length b and n segments per chain
    stress = [forcevector[0]/(box[1]*box[2]), forcevector[1]/(box[0]*box[2]), forcevector[2]/(box[0]*box[1])]

    return stress

def compute_total_force(positions, connections, box,N=None,b=np.sqrt(10)):
    if N is None:
        raise ValueError("N must be specified for GPU force calculation.")
    total_delta = _compute_total_delta_gpu(positions, connections, box)
    forcevector = 3/(N*b**2)*total_delta    #eq 2.96 from rubinstein, in units of kT, using ideal chain model with Kuhn length b and n segments per chain
    return forcevector


def bond_entries(connections, n_positions):
    sources = connections[:, :1]
    neighbors = connections[:, 1:]
    source_ids = np.broadcast_to(sources, neighbors.shape)
    valid = (
        (source_ids >= 0)
        & (source_ids < n_positions)
        & (neighbors >= 0)
        & (neighbors < n_positions)
        & (neighbors != source_ids)
    )
    return np.column_stack((source_ids[valid], neighbors[valid]))


def count_bonds(connections, n_positions):
    return len(bond_entries(connections, n_positions))


def connection_table_for_output(connections):
    ordered = connections[np.argsort(connections[:, 0])]
    return ordered + 1


def young_modulus_affine(positions, connections, box, N, b=np.sqrt(10), axis=2):
    """
    Módulo de Young (tracción uniaxial incompresible, F = diag(lam, lam^-1/2, lam^-1/2)
    a lo largo de `axis`) a partir de las distancias entre partículas del estado equilibrado.
    Unidades: kT / (unidad de longitud)^3.
    """
    k = 1.0 / (N * b**2)                        # E_enlace = k r^2 / 2  (en kT)
    V = np.prod(box)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    # lista de enlaces únicos (no asumo simetría ni cuento dos veces)
    conn = connections[:, 1:]
    ii = np.repeat(connections[:, 0], conn.shape[1])
    jj = conn.ravel()
    m = (jj >= 0) & (jj < len(positions)) & (jj != ii)
    pairs = np.unique(np.sort(np.column_stack((ii[m], jj[m])), axis=1), axis=0)

    positions = np.ascontiguousarray(positions)
    pairs = np.ascontiguousarray(pairs)
    box = np.ascontiguousarray(box)
    pair_count = len(pairs)
    if pair_count == 0:
        return 0.0, 0.0, 0.0, 0

    threads_per_block = 256
    pair_blocks = (pair_count + threads_per_block - 1) // threads_per_block
    device_positions = cuda.to_device(positions)
    device_pairs = cuda.to_device(pairs)
    device_box = cuda.to_device(box)
    pair_r2 = cuda.device_array(pair_count, dtype=np.float64)
    pair_s = cuda.device_array(pair_count, dtype=np.float64)
    force_x = cuda.to_device(np.zeros(len(positions), dtype=np.float64))
    force_y = cuda.to_device(np.zeros(len(positions), dtype=np.float64))
    force_z = cuda.to_device(np.zeros(len(positions), dtype=np.float64))
    totals = cuda.device_array(3, dtype=np.float64)

    _affine_pair_kernel[pair_blocks, threads_per_block](
        device_positions,
        device_pairs,
        device_box,
        k,
        axis,
        pair_r2,
        pair_s,
        force_x,
        force_y,
        force_z,
    )
    _reduce_affine_kernel[1, threads_per_block](
        pair_r2,
        pair_s,
        force_x,
        force_y,
        force_z,
        totals,
        pair_count,
        len(positions),
    )
    sum_r2, sum_s, fmax = totals.copy_to_host()

    EY = k * sum_r2 / pair_count  # d²E/dλ² / V

    dev = 0.5 * k * sum_s / pair_count  # tensión desviadora (debe ser ~0)
    return EY, dev, fmax, pair_count

###################################################

t0 = time()
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

connections = np.loadtxt(
    CONNECTIONS_FILE, skiprows=1, delimiter="\t", dtype=int
)
connections = connections - 1  # Convert IDs to 0-based; empty slots become -1.
if np.any(connections[:, 0] < 0) or np.any(connections[:, 0] >= len(positions)):
    raise ValueError("Connection row IDs must refer to valid crosslink positions.")
if connections.shape[0] != len(positions) or not np.array_equal(
    np.sort(connections[:, 0]), np.arange(len(positions))
):
    raise ValueError(
        "GPU relaxation requires exactly one connection row for every crosslink position."
    )
if not cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled Numba environment and verify the NVIDIA driver.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    (chain_ids[has_chain_pair], chain_xlink_a[has_chain_pair], chain_xlink_b[has_chain_pair])
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

state_path = os.path.join(OUTPUT_DIR, "equilibration_state.npz")
if not os.path.isfile(state_path):
    raise FileNotFoundError(
        f"Missing {state_path}; run equilibration-gpu.py before this script."
    )
with np.load(state_path, allow_pickle=False) as state:
    positions = state["positions"].copy()
    connections = state["connections"].copy()
    box = state["box"].copy()
    energy_history = state["energy_history"].copy()
    energy_steps = state["energy_steps"].copy()
energy_recorded = True
if positions.ndim != 2 or positions.shape[1] != 3 or box.shape != (3,):
    raise ValueError(f"{state_path} contains invalid positions or box data.")
if connections.ndim != 2 or connections.shape[0] != len(positions):
    raise ValueError(f"{state_path} contains invalid connection data.")

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

EY, dev, fmax, nb = young_modulus_affine(
    positions, connections, box, N=chainLength + 1, axis=2
)
f = connections.shape[1] - 1
EYid = (f - 2) / f
print(f"Pre-box-relaxation E_Y = {EY:.6e} kT/L^3  (G = {EY/3:.6e})")
print(f"Pre-box-relaxation E_Y/E_Y(id) = {EY/EYid:.6e} (ideal chain model with f = {f})")
print(
    f"Pre-box-relaxation deviatoric stress = {dev:.3e}, "
    f"maximum residual force = {fmax:.3e}, bonds = {nb}"
)

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
if energy_history.size > 0:
    plt.figure()
    plt.plot(np.log(energy_steps), energy_history)
    plt.xlabel("log(MCS)")
    plt.ylabel("Total energy (kT)")
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_vs_logmcs.png"))
    plt.close()

connection_pairs = bond_entries(connections, len(positions))
connection_pairs = connection_pairs[connection_pairs[:, 0] < connection_pairs[:, 1]] + 1
has_distinct_crosslinks = xlink_a != xlink_b
chain_pairs = np.column_stack(
    (xlink_a[has_distinct_crosslinks], xlink_b[has_distinct_crosslinks])
)
chain_order = np.lexsort((chain_pairs[:, 1], chain_pairs[:, 0]))
connection_order = np.lexsort((connection_pairs[:, 1], connection_pairs[:, 0]))
if not np.array_equal(
    chain_pairs[chain_order], connection_pairs[connection_order]
):
    raise ValueError("BFM distance table does not match the non-self connection table.")

connection_self_entries = np.count_nonzero(connections[:, 1:] == connections[:, :1])
distance_table_self_edges = np.count_nonzero(~has_distinct_crosslinks)
if (
    connection_self_entries % 2 != 0
    or connection_self_entries // 2 != distance_table_self_edges
):
    raise ValueError("BFM distance table does not match self-connections in the connection table.")

stress = stress[2] - (stress[0]+stress[1])/2
np.savetxt(
    os.path.join(OUTPUT_DIR, "initial_equilibrated_positions.txt"),
    positions,
    fmt=["%.4f", "%.4f", "%.4f"],
    delimiter="\t")

print(count_bonds(connections, len(positions)), "bonds in the system")
#print("strain-controlled deformation with bond breakage...")
# ----------------------------
# Parameters
# ----------------------------
total_strain = 240     # total strain (%) applied in z
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
                                                            log_file=BROKEN_CONNECTIONS_LOG)
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
if True:
    for step in range(1, n_steps + 1):
        stress = 0
        
        if step>1:
            positions = func(positions, connections,box=box_curr,N=chainLength+1)  
        # ---- 1) linear strain application ----
        # Calculate cumulative strain at this step for LINEAR growth
        current_scale_step = step - 1
        if current_scale_step == scale_2_step:
            EY, dev, fmax, nb = young_modulus_affine(positions, connections, box_curr, N=chainLength+1, axis=2)
            print(f"Step {current_scale_step}: E_Y near 100% deformation "
                f"(actual {scale_z:.6f})= {EY:.6e} kT/L^3  (G = {EY/3:.6e})")
        if current_scale_step == scale_3_step:
            EY, dev, fmax, nb = young_modulus_affine(positions, connections, box_curr, N=chainLength+1, axis=2)
            print(f"Step {current_scale_step}: E_Y near 200% deformation "
                f"(actual {scale_z:.6f})= {EY:.6e} kT/L^3  (G = {EY/3:.6e})")
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
                                                            log_file=BROKEN_CONNECTIONS_LOG)
        
        # ---- 4) relax system with new bonds ----
        stop_deformation = False
        while broken:
            positions = func(positions, connections,box=box_curr,N=chainLength+1)
            connections,broken = breakage_potential(positions=positions, connections=connections, box=box_curr, U_crit=U_crit,N=chainLength+1,
                                                            log_file=BROKEN_CONNECTIONS_LOG)
            stress = compute_total_stress(positions, connections, box_curr,N=chainLength+1)
            stress = stress[2] - (stress[0]+stress[1])/2
            if stress < 0:
                stress = 0
            if stress_total[-1] > stress:
                print("Stress has dropped to zero during relaxation, stopping deformation.", flush=True)
                checkpoint_dir = os.path.join(OUTPUT_DIR, "step_final")
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
                    connection_table_for_output(connections),
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

                if os.path.exists(BROKEN_CONNECTIONS_LOG):
                        shutil.copy(BROKEN_CONNECTIONS_LOG,
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
        with open(os.path.join(OUTPUT_DIR, "breakage_log.txt"), 'a') as f:
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
        
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"step_{step:04d}")
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
                connection_table_for_output(connections),
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

        if os.path.exists(BROKEN_CONNECTIONS_LOG):
                shutil.copy(BROKEN_CONNECTIONS_LOG,
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
            checkpoint_dir = os.path.join(OUTPUT_DIR, "step_final")
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
                connection_table_for_output(connections),
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

            if os.path.exists(BROKEN_CONNECTIONS_LOG):
                    shutil.copy(BROKEN_CONNECTIONS_LOG,
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
        os.path.join(OUTPUT_DIR, "stress_strain.txt"),
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
        plt.savefig(os.path.join(OUTPUT_DIR, "Stress-Strain.png"))



    np.savetxt(
        os.path.join(OUTPUT_DIR, "remaining_connections.txt"),
        connection_table_for_output(connections),
        fmt='%d',
        delimiter="\t",
        comments=''
    )

    np.savetxt(
        os.path.join(OUTPUT_DIR, "final_equilibrated_positions.txt"),
        positions,
        fmt=["%.4f", "%.4f", "%.4f"],
        delimiter="\t",
        comments=''
    )

    np.savetxt(
        os.path.join(OUTPUT_DIR, "box_curr.txt"),
        box_curr,
        fmt='%d',
        delimiter="\t",
        comments=''
    )

print(f"Finished in {(time() - t0)/60:.2f} minutes.", flush=True)

