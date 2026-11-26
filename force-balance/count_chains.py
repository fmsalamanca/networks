import csv
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def main():
    parameters = {}
    with (DATA_DIR / "system.txt").open() as system_file:
        for line in system_file:
            key, value = line.strip().split(" = ", 1)
            parameters[key] = value

    chain_count = int(float(parameters["number_of_linear_chains"]))
    crosslink_count = int(float(parameters["number_of_crosslinkers"]))
    adjacency = {node: set() for node in range(1, crosslink_count + 1)}
    directed_edges = Counter()
    seen_sources = set()

    with (DATA_DIR / "connected_xlinks.txt").open(newline="") as connections_file:
        reader = csv.reader(connections_file, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError("connected_xlinks.txt is empty.")

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            source = int(row[0])
            if source not in adjacency:
                raise ValueError(f"Invalid crosslink ID {source} on line {line_number}.")
            if source in seen_sources:
                raise ValueError(f"Duplicate crosslink row for ID {source}.")
            seen_sources.add(source)

            for value in row[1:]:
                neighbor = int(value)
                if neighbor == 0:
                    continue
                if neighbor not in adjacency:
                    raise ValueError(f"Invalid crosslink ID {neighbor} on line {line_number}.")
                if neighbor == source:
                    continue
                directed_edges[source, neighbor] += 1
                adjacency[source].add(neighbor)
                adjacency[neighbor].add(source)

    for (source, neighbor), multiplicity in directed_edges.items():
        if directed_edges[neighbor, source] != multiplicity:
            raise ValueError(
                f"Connection multiplicity differs for crosslinks {source} and {neighbor}."
            )
    edge_count = sum(directed_edges.values()) // 2

    component_count = 0
    unvisited = set(adjacency)
    while unvisited:
        component_count += 1
        stack = [unvisited.pop()]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)

    print(f"Undirected connection edges (including duplicates): {edge_count}")
    print(f"Connected components among crosslinkers: {component_count}")
    print(f"Linear chains reported by system.txt: {chain_count}")


if __name__ == "__main__":
    main()
