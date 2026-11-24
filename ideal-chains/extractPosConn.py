import numpy as np
import time
from collections import Counter

t0 = time.time()

# ── 1. Parse header ──────────────────────────────────────────────────────────
info = []

with open("LastConfig2.bfm", "r") as file:
    data = file.readlines()
for i, line in enumerate(data):
    if (line.startswith("!") or line.startswith("#!")) and "=" in line:
        textstring = line.strip().lstrip("#!")   # strips both '!' and '#!' cleanly
        name, raw_value = textstring.split("=", 1)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        name = name.strip()
        info.append([name, value, i])

# ── 2. Named lookup (avoids fragile positional indexing) ────────────────────
info_dict  = {entry[0]: entry[1] for entry in info}
info_index = {entry[0]: entry[2] for entry in info}  # line index in file

number_of_monomers      = info_dict["number_of_monomers"]
box_x                   = info_dict["box_x"]
box_y                   = info_dict["box_y"]
box_z                   = info_dict["box_z"]
number_of_linear_chains = info_dict["number_of_linear_chains"]
number_of_crosslinkers  = info_dict["number_of_crosslinkers"]
chainLength             = info_dict["chainLength"]

params = {
    "number_of_monomers":      number_of_monomers,
    "box_x":                   box_x,
    "box_y":                   box_y,
    "box_z":                   box_z,
    "number_of_linear_chains": number_of_linear_chains,
    "number_of_crosslinkers":  number_of_crosslinkers,
    "chainLength":             chainLength,
}

with open("system.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key} = {value}\n")

# ── 3. Parse bonds ────────────────────────────────────────────────────────────
bonds_raw = "".join(data[data.index("!bonds\n") + 1 : info_index["box_x"]])
bonds = np.fromstring(bonds_raw, sep=" ", dtype=float).reshape(-1, 2)

# Column order not guaranteed in the file — sort each row so the chain monomer
# (always the smaller global ID) is column 0, crosslinker is column 1.
bonds = np.sort(bonds, axis=1)

# ── 4. Derive chain ID and local crosslinker ID ──────────────────────────────
chain_monomers = bonds[:, 0]
xlink_globals  = bonds[:, 1]

n_linear_chains = int(number_of_linear_chains)
n_crosslinkers = int(number_of_crosslinkers)
chain_length = int(chainLength)
chainsID = ((chain_monomers - 1) // chain_length + 1).astype(int)
xlink = (xlink_globals - chain_length * n_linear_chains).astype(int)
if np.any((chainsID < 1) | (chainsID > n_linear_chains)):
    raise ValueError("Bond data contains an invalid linear-chain ID.")
if np.any((xlink < 1) | (xlink > n_crosslinkers)):
    raise ValueError("Bond data contains an invalid crosslinker ID.")

# Include all chains, including those with zero or one crosslinker.
chain_to_xlinks = [set() for _ in range(n_linear_chains + 1)]
for chain_id, xlink_id in zip(chainsID, xlink):
    chain_to_xlinks[chain_id].add(xlink_id)
chains_xlinks = [
    [chain_id] + sorted(chain_to_xlinks[chain_id])
    for chain_id in range(1, n_linear_chains + 1)
]

max_chain_xlinks = max((len(row) - 1 for row in chains_xlinks), default=0)
chain_rows = [
    row + [0] * (max_chain_xlinks - (len(row) - 1))
    for row in chains_xlinks
]
chain_header = "\t".join(
    ["chainID"] + [f"xlink_{i+1}" for i in range(max_chain_xlinks)]
)
np.savetxt(
    "chain_xlinks.txt",
    np.asarray(chain_rows, dtype=int),
    fmt="%d",
    delimiter="\t",
    header=chain_header,
    comments="",
)

chain_link_counts = Counter(len(row) - 1 for row in chains_xlinks)
print(f"Linear chains accounted for: {len(chains_xlinks)}")
print(f"Chains with two or more crosslinkers: {sum(count for n, count in chain_link_counts.items() if n >= 2)}")
print(f"Chains with one crosslinker: {chain_link_counts[1]}")
print(f"Chains with no crosslinkers: {chain_link_counts[0]}")

# ── 6. Build crosslinker connection graph — PRESERVE MULTI-EDGES ────────────
# Using a list (not a set): if two different chains both bridge the same pair
# of crosslinkers, that pair appears twice, three times, etc. This is required
# for stress calculations, where each physical strand is a separate
# load-bearing connection even if it links the same two crosslinkers as
# another strand.
connections = {xlink_id: [] for xlink_id in range(1, n_crosslinkers + 1)}
pair_counts = Counter()
chain_edge_rows = []

for row in chains_xlinks:
    chain_id = row[0]
    xlinks = [int(x) for x in row[1:]]
    if len(xlinks) < 2:
        # Zero marks a missing endpoint for chains that do not bridge two crosslinkers.
        chain_edge_rows.append([chain_id, xlinks[0] if xlinks else 0, 0])
        continue
    for i in range(len(xlinks)):
        for j in range(i + 1, len(xlinks)):
            x1, x2 = xlinks[i], xlinks[j]
            connections[x1].append(x2)
            connections[x2].append(x1)
            pair = tuple(sorted((x1, x2)))
            pair_counts[pair] += 1
            chain_edge_rows.append([chain_id, pair[0], pair[1]])

# ── 7. Build padded output table (duplicates kept, sorted for readability) ──
rows = []
for xlink_id, conn_list in sorted(connections.items()):
    rows.append([xlink_id] + sorted(conn_list))

max_len = max(len(r) - 1 for r in rows)
rows_padded = np.array(
    [r + [0] * (max_len - (len(r) - 1)) for r in rows],
    dtype=int
)

header = "\t".join(["xlinkID"] + [f"xlink_{i+1}" for i in range(max_len)])
np.savetxt("connected_xlinks.txt", rows_padded, fmt="%.0f",
           delimiter="\t", header=header, comments="")

# ── 7b. Save each chain with crosslink endpoint IDs and its exact chain ID ───
chain_edge_rows = np.asarray(chain_edge_rows, dtype=int).reshape(-1, 3)
np.savetxt(
    "crosslink_chain_edges.txt",
    chain_edge_rows,
    fmt="%d",
    delimiter="\t",
    header="chainID\txlinkA\txlinkB",
    comments="",
)

# ── 7c. Save unique crosslink pairs with their multiplicity ──────────────────
edge_rows = np.array(
    [[a, b, w] for (a, b), w in sorted(pair_counts.items())],
    dtype=int
)
np.savetxt("crosslink_edges_weighted.txt", edge_rows, fmt="%d",
           delimiter="\t", header="xlinkA\txlinkB\tn_strands", comments="")

n_multi = sum(1 for w in pair_counts.values() if w > 1)
print(f"{n_multi} crosslinker pairs are bridged by more than one strand")

# ── 8. Parse and save crosslinker positions ──────────────────────────────────
# NOTE: this final slice still relies on the original positional indices
# (info[13], info[8]) because the header entry marking the start of the
# conformation block doesn't have a clean, reliably-named key in this file
# format (it appears to be the same line used for "mcs"). If you can confirm
# the exact !key name for that line, this can be converted to a named lookup
# like the rest of the script — tell me the key and I'll fix it.
crosslinks = "".join(data[info[13][2] + 1 + int(info[8][1]):])
crosslinks = np.fromstring(crosslinks, sep=" ", dtype=float).reshape(-1, 3)
np.savetxt("crosslinks-positions.txt", crosslinks, fmt="%.0f")

bond_lengths = np.zeros(len(chain_edge_rows), dtype=float)
has_both_crosslinks = (chain_edge_rows[:, 1] > 0) & (chain_edge_rows[:, 2] > 0)
bond_pairs = chain_edge_rows[has_both_crosslinks, 1:3] - 1
bond_delta = crosslinks[bond_pairs[:, 1]] - crosslinks[bond_pairs[:, 0]]
box = np.array([box_x, box_y, box_z])
bond_delta -= box * np.floor(bond_delta / box + 0.5)
bond_lengths[has_both_crosslinks] = np.linalg.norm(bond_delta, axis=1)
bond_length_order = np.lexsort(
    (chain_edge_rows[:, 2], chain_edge_rows[:, 1], chain_edge_rows[:, 0])
)
np.savetxt(
    "chain_bond_lengths.txt",
    np.column_stack((chain_edge_rows, bond_lengths))[bond_length_order],
    fmt=["%d", "%d", "%d", "%.8f"],
    delimiter="\t",
    header="chainID\txlinkA\txlinkB\tbondLength",
    comments="",
)

print(f"Execution time: {time.time() - t0:.2f} seconds")