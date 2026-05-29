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
        textstring = line.strip()[1:]
        name  = textstring[:textstring.find("=")]
        value = float(textstring[textstring.find("=") + 1:])
        info.append([name, value, i])

# ── 2. Build named lookup (fixes Bug 3) ─────────────────────────────────────
info_dict  = {entry[0]: entry[1] for entry in info}
info_index = {entry[0]: entry[2] for entry in info}  # line index in file

number_of_monomers      = info_dict["number_of_monomers"]
box_x                   = info_dict["box_x"]
box_y                   = info_dict["box_y"]
box_z                   = info_dict["box_z"]
try:
    number_of_linear_chains = info_dict["number_of_linear_chains"]
    number_of_crosslinkers  = info_dict["number_of_crosslinkers"]
    chainLength             = info_dict["chainLength"]
    #mcs                     = info_dict[-1][1]
    #print(mcs)  # last entry
except KeyError as e:
    number_of_linear_chains = info_dict["!number_of_linear_chains"]
    number_of_crosslinkers  = info_dict["!number_of_crosslinkers"]
    chainLength             = info_dict["!chainLength"]

params = {
    "number_of_monomers":      number_of_monomers,
    "box_x":                   box_x,
    "box_y":                   box_y,
    "box_z":                   box_z,
    "number_of_linear_chains": number_of_linear_chains,
    "number_of_crosslinkers":  number_of_crosslinkers,
    "chainLength":             chainLength,
    #"mcs":                     mcs,
}

with open("system.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key} = {value}\n")

# ── 3. Parse bonds ───────────────────────────────────────────────────────────
bonds_raw = "".join(data[data.index("!bonds\n") + 1 : info_index["box_x"]])
bonds = np.fromstring(bonds_raw, sep=" ", dtype=float).reshape(-1, 2)

# Fix Bug 1: column order not guaranteed — sort each row so chain monomer
# (smaller, 1-based global index ≤ chainLength*N_chains) is always column 0
bonds = np.sort(bonds, axis=1)

# ── 4. Derive chain ID and crosslinker local ID ──────────────────────────────
chain_monomers = bonds[:, 0]   # global index of the chain endpoint monomer
xlink_globals  = bonds[:, 1]   # global index of the crosslinker monomer

chainsID = (chain_monomers - 1) // chainLength + 1          # 1-based chain ID
xlink    = xlink_globals - chainLength * number_of_linear_chains  # 1-based local xlink ID

bondss = np.column_stack([chainsID, xlink])
sorted_bonds = bondss[bondss[:, 0].argsort()]

# ── 5. Build chains → crosslinkers map ──────────────────────────────────────
chains = sorted_bonds[:, 0]
unique_chains, inverse = np.unique(chains, return_inverse=True)

chains_xlinks = []
for i, chain in enumerate(unique_chains):
    mask = inverse == i
    associated = sorted_bonds[mask, 1]

    if associated.size < 2:       # skip chains bonded to only 1 crosslinker
        continue

    chains_xlinks.append([chain] + associated.tolist())

# ── 6. Build crosslinker connection graph (fixes Bug 2) ─────────────────────
connections = defaultdict(set)   # set → no duplicates

for row in chains_xlinks:
    xlinks = [int(x) for x in row[1:]]
    for i in range(len(xlinks)):
        for j in range(i + 1, len(xlinks)):
            x1, x2 = xlinks[i], xlinks[j]
            connections[x1].add(x2)
            connections[x2].add(x1)

# ── 7. Build padded output table ─────────────────────────────────────────────
rows = []
for xlink_id, conn_set in sorted(connections.items()):
    conn_list = sorted(conn_set)
    rows.append([xlink_id] + conn_list)

max_len = max(len(r) - 1 for r in rows)
rows_padded = np.array(
    [r + [0] * (max_len - (len(r) - 1)) for r in rows],
    dtype=int
)

header = "\t".join(["xlinkID"] + [f"xlink_{i+1}" for i in range(max_len)])
np.savetxt("connected_xlinksTEST2.txt", rows_padded, fmt="%.0f",
           delimiter="\t", header=header, comments="")

# ── 8. Parse and save crosslinker positions ──────────────────────────────────
bondvectors = "".join(data[data.index("!set_of_bondvectors\n") + 1 : data.index("!attributes\n") - 1])
attributes  = "".join(data[data.index("!attributes\n") + 1      : data.index("!reactivity\n") - 1])
#reactivity  = "".join(data[data.index("!reactivity\n") + 1      : info_index["number_of_linear_chains"]])

#conformations = "".join(
#    data[info_index["mcs"] + 1 : info_index["mcs"] + 1 + int(number_of_linear_chains)]
#)
crosslinks = "".join(data[info[13][2]+1+int(info[8][1]):])
crosslinks = np.fromstring(crosslinks, sep=" ", dtype=float).reshape(-1, 3)
np.savetxt("crosslinks-positionsTEST2.txt", crosslinks, fmt="%.0f")

print(f"Execution time: {time.time() - t0:.2f} seconds")
