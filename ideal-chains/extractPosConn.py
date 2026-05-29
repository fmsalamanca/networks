import numpy as np
import time
from collections import defaultdict

t0 = time.time()

info=[]
bonds=[]
with open("LastConfig2.bfm", "r") as file:
    data = file.readlines()
    for i,line in enumerate(data):
        if (line.startswith("!") or line.startswith("#!")) and "=" in line:
            textstring = line.strip()[1:]
            name = str(textstring[0:textstring.find("=")])
            value = float(textstring[textstring.find("=")+1:])
            index = i
            info.append([name,value,index])

number_of_monomers = info[1][1]
box_x = info[2][1]
box_y = info[3][1]
box_z = info[4][1]
number_of_linear_chains = info[8][1]
number_of_crosslinkers = info[9][1]
chainLength = info[10][1]
mcs = info[-1][1]

params = {
    "number_of_monomers":      number_of_monomers,
    "box_x":                   box_x,
    "box_y":                   box_y,
    "box_z":                   box_z,
    "number_of_linear_chains": number_of_linear_chains,
    "number_of_crosslinkers":  number_of_crosslinkers,
    "chainLength":             chainLength,
    "mcs":                     mcs,
}

with open("system.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key} = {value}\n")

bonds = "".join(data[data.index("!bonds\n")+1:info[2][2]]) #the end is when box x starts
bonds = np.fromstring(bonds,sep=" ",dtype=float).reshape(-1, 2)

bondvectors = "".join(data[data.index("!set_of_bondvectors\n")+1:data.index("!attributes\n")-1])

attributes = "".join(data[data.index("!attributes\n")+1:data.index("!reactivity\n")-1])

reactivity = "".join(data[data.index("!reactivity\n")+1:info[8][2]])

conformations = "".join(data[info[13][2]+1:info[13][2]+1+int(info[8][1])]) #the end is adding the number of chains to the first point

crosslinks = "".join(data[info[13][2]+1+int(info[8][1]):])
crosslinks = np.fromstring(crosslinks,sep=" ",dtype=float).reshape(-1, 3)

aa = bonds[:,0]-1
chainsID = aa//(chainLength)+1

xlink = bonds[:,1]-chainLength*number_of_linear_chains
bondss = np.array([chainsID,xlink]).T

sorted_bonds = bondss[bondss[:,0].argsort()]

chains = sorted_bonds[:,0]

# Find unique chains and indices
unique_chains, inverse = np.unique(chains, return_inverse=True)

chains_xlinks = []

for i, chain in enumerate(unique_chains):
    mask = inverse == i  # all rows for this chain
    associated = sorted_bonds[mask, 1]

    if associated.size < 2:  
        continue

    chains_xlinks.append([chain] + associated.tolist())

# Pad to max length
max_len = max(len(r) for r in chains_xlinks)
table1_padded = np.array([r + [np.nan]*(max_len-len(r)) for r in chains_xlinks])


connections = defaultdict(list)  # xlinkID -> set of connected xlinks

for row in chains_xlinks:
    # Remove NaNs
    xlinks = [int(x) for x in row[1:] if not np.isnan(x)]

    # For all pair combinations in this chain:
    for i in range(len(xlinks)):
        for j in range(i+1, len(xlinks)):
            x1, x2 = xlinks[i], xlinks[j]

            # Add both directions, but KEEP duplicates
            connections[x1].append(x2)
            connections[x2].append(x1)

# Determine padding length
max_len = max(len(v) for v in connections.values())

# Build padded rows
rows = []
for xlink, conn_list in connections.items():
    row = [xlink] + conn_list
    row += [0] * (max_len - len(conn_list))  # pad with zeros
    rows.append(row)

rows = np.array(rows, dtype=int)
header = "\t".join(["xlinkID"] + [f"xlink_{i+1}" for i in range(max_len)])
np.savetxt("connected_xlinksTEST1.txt", rows, fmt="%.0f", delimiter="\t", header=header, comments="")
np.savetxt('crosslinks-positionsTEST1.txt', crosslinks, fmt="%.0f")

print(f"Execution time: {time.time() - t0:.2f} seconds")