import numpy as np
import time
from collections import defaultdict

t0 = time.time()

# ── 1. Parse header ──────────────────────────────────────────────────────────
info = []

with open("LastConfig2.bfm", "r") as file:
    data = file.readlines()
for i, line in enumerate(data):
    if (line.startswith("!") or line.startswith("#!")) and "=" in line:
        textstring = line.strip().lstrip("#!")   # strips both '!' and '#!' cleanly
        name  = textstring[:textstring.find("=")]
        value = float(textstring[textstring.find("=") + 1:])
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
chainLength              = info_dict["chainLength"]

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

chainsID = (chain_monomers - 1) // chainLength + 1
xlink    = xlink_globals - chainLength * number_of_linear_chains

bondss = np.column_stack([chainsID, xlink])
sorted_bonds = bondss[bondss[:, 0].argsort()]

# ── 5. Build chains → crosslinkers map ───────────────────────────────────────
chains = sorted_bonds[:, 0]
unique_chains, inverse = np.unique(chains, return_inverse=True)

chains_xlinks = []
for i, chain in enumerate(unique_chains):
    mask = inverse == i
    associated = sorted_bonds[mask, 1]

    #if associated.size < 2:       # skip chains bonded to only 1 crosslinker (dangling end)
    #    continue

    chains_xlinks.append([chain] + associated.tolist())

# ── 6. Build crosslinker connection graph — PRESERVE MULTI-EDGES ────────────
# Using a list (not a set): if two different chains both bridge the same pair
# of crosslinkers, that pair appears twice, three times, etc. This is required
# for stress calculations, where each physical strand is a separate
# load-bearing connection even if it links the same two crosslinkers as
# another strand.
connections = defaultdict(list)

for row in chains_xlinks:
    xlinks = [int(x) for x in row[1:]]
    for i in range(len(xlinks)):
        for j in range(i + 1, len(xlinks)):
            x1, x2 = xlinks[i], xlinks[j]
            connections[x1].append(x2)
            connections[x2].append(x1)

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
np.savetxt("connected_xlinksTEST3-2.txt", rows_padded, fmt="%.0f",
           delimiter="\t", header=header, comments="")

# ── 7b. Also save an explicit edge list with multiplicity (weight) ──────────
# One row per unique pair, with a count of how many strands bridge it.
# This is often more directly useful for stress: weight = number of strands.
from collections import Counter
pair_counts = Counter()
for row in chains_xlinks:
    xlinks = [int(x) for x in row[1:]]
    for i in range(len(xlinks)):
        for j in range(i + 1, len(xlinks)):
            pair = tuple(sorted((xlinks[i], xlinks[j])))
            pair_counts[pair] += 1

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
np.savetxt("crosslinks-positionsTEST3.txt", crosslinks, fmt="%.0f")

print(f"Execution time: {time.time() - t0:.2f} seconds")