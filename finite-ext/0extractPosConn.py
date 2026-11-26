import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent


def get_bfm_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    for name in ("LastConfig2.bfm", "LastConfig.bfm"):
        candidate = OUTPUT_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find LastConfig2.bfm or LastConfig.bfm next to this script."
    )

HEADER_RE = re.compile(r"^#?!\s*([^=]+?)\s*=\s*(.*?)\s*$")
BOND_RE = re.compile(r"^\s*(\d+)\s+(\d+)")
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
POSITION_RE = re.compile(rf"^\s*({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})")


def read_bfm_file(path):
    metadata = []
    bond_rows = []
    position_rows = []
    reading_bonds = False
    reading_positions = False
    found_bonds = False
    found_positions = False

    with path.open(encoding="latin-1") as source:
        for line_number, line in enumerate(source, start=1):
            if reading_positions:
                if not line.strip():
                    continue
                match = POSITION_RE.match(line)
                if match is None:
                    raise ValueError(
                        f"Could not read three coordinates on line {line_number}."
                    )
                position_rows.append(tuple(float(value) for value in match.groups()))
                continue

            header_match = HEADER_RE.match(line)
            if header_match:
                key, value = header_match.groups()
                key = key.strip()
                metadata.append((key, value.strip()))
                if key == "box_x" and reading_bonds:
                    reading_bonds = False
                if key == "mcs":
                    reading_positions = True
                    found_positions = True
                continue

            if line.strip() == "!bonds":
                reading_bonds = True
                found_bonds = True
                continue

            if reading_bonds and line.strip():
                match = BOND_RE.match(line)
                if match is None:
                    raise ValueError(
                        f"Could not read a bond pair on line {line_number}."
                    )
                bond_rows.append(tuple(int(value) for value in match.groups()))

    if not found_bonds:
        raise ValueError(f"No !bonds section found in {path}.")
    if not found_positions:
        raise ValueError(f"No !mcs position section found in {path}.")

    return (
        dict(metadata),
        np.asarray(bond_rows, dtype=int).reshape(-1, 2),
        np.asarray(position_rows, dtype=float).reshape(-1, 3),
    )


def write_extracted_tables(
    metadata_rows,
    metadata,
    bonds,
    positions,
    output_dir,
    output_suffix="",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def suffixed(name):
        stem, dot, extension = name.rpartition(".")
        if dot:
            return f"{stem}{output_suffix}.{extension}"
        return f"{stem}{output_suffix}"

    with (OUTPUT_DIR / suffixed("system.txt")).open("w") as output:
        for key, value in metadata_rows:
            output.write(f"{key} = {value}\n")

    np.savetxt(
        tables_dir / suffixed("bonds.txt"),
        bonds,
        fmt="%d",
        delimiter="\t",
        header="monomerA\tmonomerB",
        comments="",
    )
    number_of_monomers = int(float(metadata["number_of_monomers"]))
    number_of_linear_chains = int(float(metadata["number_of_linear_chains"]))
    number_of_crosslinkers = int(float(metadata["number_of_crosslinkers"]))
    chain_length = int(float(metadata["chainLength"]))
    expected_position_rows = number_of_linear_chains + number_of_crosslinkers
    if len(positions) != expected_position_rows:
        raise ValueError(
            f"Found {len(positions)} position rows; expected "
            f"{number_of_linear_chains} chain positions plus "
            f"{number_of_crosslinkers} crosslink positions."
        )

    box = np.array(
        [float(metadata[f"box_{axis}"]) for axis in ("x", "y", "z")]
    )
    periodic = np.array(
        [bool(int(float(metadata[f"periodic_{axis}"]))) for axis in ("x", "y", "z")]
    )
    if np.any(periodic & (box <= 0)):
        raise ValueError("Periodic box dimensions must be positive.")

    if np.any(bonds < 1) or np.any(bonds > number_of_monomers):
        raise ValueError("The bond section contains monomer IDs outside the declared range.")

    ordered_bonds = np.sort(bonds, axis=1)
    chain_monomers = ordered_bonds[:, 0]
    crosslinker_globals = ordered_bonds[:, 1]
    chain_ids = (chain_monomers - 1) // chain_length + 1
    crosslinker_ids = crosslinker_globals - chain_length * number_of_linear_chains
    if np.any((chain_ids < 1) | (chain_ids > number_of_linear_chains)):
        raise ValueError("Bond data contains an invalid linear-chain ID.")
    if np.any((crosslinker_ids < 1) | (crosslinker_ids > number_of_crosslinkers)):
        raise ValueError("Bond data contains an invalid crosslinker ID.")

    chain_attachments = [[] for _ in range(number_of_linear_chains + 1)]
    for monomer_id, chain_id, crosslinker_id in zip(
        chain_monomers, chain_ids, crosslinker_ids
    ):
        chain_attachments[chain_id].append(
            (int(monomer_id), int(crosslinker_id))
        )
    chains = [
        [chain_id]
        + sorted(crosslinker_id for _, crosslinker_id in chain_attachments[chain_id])
        for chain_id in range(1, number_of_linear_chains + 1)
    ]

    max_crosslinkers = max(len(row) - 1 for row in chains)
    padded_chains = [
        row + [0] * (max_crosslinkers - len(row) + 1)
        for row in chains
    ]
    chain_header = "\t".join(
        ["chainID"] + [f"xlink_{index + 1}" for index in range(max_crosslinkers)]
    )
    np.savetxt(
        tables_dir / suffixed("chain_xlinks.txt"),
        np.asarray(padded_chains, dtype=int),
        fmt="%d",
        delimiter="\t",
        header=chain_header,
        comments="",
    )

    connections = {crosslinker_id: [] for crosslinker_id in range(1, number_of_crosslinkers + 1)}
    pair_counts = Counter()
    chain_edge_rows = []
    for chain_id in range(1, number_of_linear_chains + 1):
        attachments = chain_attachments[chain_id]
        matched_pairs = []
        for left in range(len(attachments)):
            monomer_a, source_xlink = attachments[left]
            for right in range(left + 1, len(attachments)):
                monomer_b, target_xlink = attachments[right]
                if abs(monomer_b - monomer_a) != chain_length - 1:
                    continue
                matched_pairs.append(tuple(sorted((source_xlink, target_xlink))))

        if not matched_pairs:
            attached_xlinks = chains[chain_id - 1][1:]
            chain_edge_rows.append(
                [chain_id, attached_xlinks[0] if len(attachments) == 1 else 0, 0]
            )
            continue

        for xlink_a, xlink_b in matched_pairs:
            connections[xlink_a].append(xlink_b)
            connections[xlink_b].append(xlink_a)
            pair_counts[(xlink_a, xlink_b)] += 1
            chain_edge_rows.append([chain_id, xlink_a, xlink_b])

    connection_rows = [
        [xlink_id] + sorted(neighbors)
        for xlink_id, neighbors in sorted(connections.items())
    ]
    max_neighbors = max(len(row) - 1 for row in connection_rows)
    padded_connections = [
        row + [0] * (max_neighbors - len(row) + 1)
        for row in connection_rows
    ]
    connection_header = "\t".join(
        ["xlinkID"] + [f"xlink_{index + 1}" for index in range(max_neighbors)]
    )
    np.savetxt(
        output_dir / suffixed("connected_xlinks.txt"),
        np.asarray(padded_connections, dtype=int),
        fmt="%d",
        delimiter="\t",
        header=connection_header,
        comments="",
    )

    chain_edge_array = np.asarray(chain_edge_rows, dtype=int).reshape(-1, 3)
    np.savetxt(
        tables_dir / suffixed("crosslink_chain_edges.txt"),
        chain_edge_array,
        fmt="%d",
        delimiter="\t",
        header="chainID\txlinkA\txlinkB",
        comments="",
    )

    has_connected_pair = (chain_edge_array[:, 1] > 0) & (chain_edge_array[:, 2] > 0)
    connected_pairs = chain_edge_array[has_connected_pair]
    local_xlink_a = connected_pairs[:, 1]
    local_xlink_b = connected_pairs[:, 2]
    crosslink_position_start = len(positions) - number_of_crosslinkers
    crosslink_positions = positions[crosslink_position_start:]
    np.savetxt(
        output_dir / suffixed("crosslinks-positions.txt"),
        crosslink_positions,
        fmt="%.8g",
        delimiter="\t",
    )
    position_a = crosslink_positions[local_xlink_a - 1]
    position_b = crosslink_positions[local_xlink_b - 1]
    deltas = position_b - position_a
    deltas[:, periodic] -= box[periodic] * np.floor(
        deltas[:, periodic] / box[periodic] + 0.5
    )
    squared_distances = np.einsum("ij,ij->i", deltas, deltas)
    bfm_xlink_offset = chain_length * number_of_linear_chains
    distance_rows = np.column_stack(
        (
            bfm_xlink_offset + local_xlink_a,
            bfm_xlink_offset + local_xlink_b,
            squared_distances,
            connected_pairs[:, 0],
        )
    )
    np.savetxt(
        tables_dir / suffixed("crosslink_squared_distances.txt"),
        distance_rows,
        fmt=["%d", "%d", "%.8f", "%d"],
        delimiter="\t",
        header="xlinkID_BFM\tconnectedXlinkID_BFM\tsquaredDistance\tchainID",
        comments="",
    )

    weighted_edges = np.asarray(
        [[xlink_a, xlink_b, count] for (xlink_a, xlink_b), count in sorted(pair_counts.items())],
        dtype=int,
    ).reshape(-1, 3)
    np.savetxt(
        tables_dir / suffixed("crosslink_edges_weighted.txt"),
        weighted_edges,
        fmt="%d",
        delimiter="\t",
        header="xlinkA\txlinkB\tn_strands",
        comments="",
    )

    print(f"Saved solver inputs in {output_dir}")
    print(f"Saved auxiliary tables in {tables_dir}")
    print(f"Linear chains accounted for: {len(chains)}")
    print(
        "Chains with two or more crosslinkers:",
        sum(len(row) >= 3 for row in chains),
    )
    print(
        "Self-connected chain pairs:",
        sum(count for (xlink_a, xlink_b), count in pair_counts.items() if xlink_a == xlink_b),
    )


def main():
    start = time.time()
    bfm_file = get_bfm_file()
    output_suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    if output_suffix and not output_suffix.startswith("_"):
        output_suffix = f"_{output_suffix}"
    output_dir = (
        bfm_file.parent
        if len(sys.argv) > 1
        else OUTPUT_DIR
    )
    metadata, bonds, positions = read_bfm_file(bfm_file)
    write_extracted_tables(
        list(metadata.items()),
        metadata,
        bonds,
        positions,
        output_dir,
        output_suffix,
    )
    print(f"Execution time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
