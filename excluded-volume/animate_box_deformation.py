"""Animate box deformation and progressive crosslink breakage.

Run from the Ideal-chains directory, for example:

    python animate_box_deformation.py --output deformation.gif --fps 8

The coordinates in each output/step_XXXX/positions.txt file are indexed by
the particle IDs used by broken_connections.txt and connections.txt.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image


def as_2d(values, columns):
    values = np.atleast_2d(values)
    if values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns, got {values.shape[1]}")
    return values


def unique_bonds(connections):
    """Convert an adjacency table into unique, zero-based bond pairs."""
    connections = as_2d(connections, connections.shape[-1])
    bonds = set()
    for row in connections:
        first = int(row[0])
        for second in row[1:]:
            second = int(second)
            if second >= 0 and second != first:
                bonds.add(tuple(sorted((first, second))))
    return np.asarray(sorted(bonds), dtype=int)


def load_broken(path, count):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    values = np.atleast_2d(np.loadtxt(path, dtype=int))[:count]
    return {tuple(sorted((int(row[0]), int(row[1])))) for row in values}


def periodic_segments(positions, bonds, box):
    """Return line segments split at periodic box boundaries."""
    if len(bonds) == 0:
        return np.empty((0, 2, 3))
    positions = np.mod(positions, box)
    segments = []
    for first_id, second_id in bonds:
        first = positions[first_id]
        delta = (positions[second_id] - first + box / 2.0) % box - box / 2.0
        crossing_times = []
        for dimension in range(3):
            if delta[dimension] > 0:
                time = (box[dimension] - first[dimension]) / delta[dimension]
            elif delta[dimension] < 0:
                time = -first[dimension] / delta[dimension]
            else:
                continue
            if 0 < time < 1:
                crossing_times.append((time, dimension))

        breakpoints = [0.0] + sorted({time for time, _ in crossing_times}) + [1.0]
        for start_time, end_time in zip(breakpoints[:-1], breakpoints[1:]):
            start = np.mod(first + start_time * delta, box)
            end = np.mod(first + end_time * delta, box)
            for time, dimension in crossing_times:
                if np.isclose(start_time, time):
                    start[dimension] = 0.0 if delta[dimension] > 0 else box[dimension]
                if np.isclose(end_time, time):
                    end[dimension] = box[dimension] if delta[dimension] > 0 else 0.0
            segments.append((start, end))
    return np.asarray(segments)


def fit_positions_to_box(positions, source_box, target_box):
    """Wrap unwrapped coordinates and map them to the box being displayed."""
    wrapped = np.mod(positions, source_box)
    return wrapped * (target_box / source_box)


def box_segments(box):
    corners = np.array([
        [0, 0, 0], [box[0], 0, 0], [box[0], box[1], 0], [0, box[1], 0],
        [0, 0, box[2]], [box[0], 0, box[2]], [box[0], box[1], box[2]], [0, box[1], box[2]],
    ])
    edges = ((0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    return np.asarray([[corners[first], corners[second]] for first, second in edges])


def read_steps(output_dir):
    stress_strain = output_dir / "stress_strain.txt"
    stress_lambdas = []
    for line in stress_strain.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            try:
                stress_lambdas.append(float(fields[0]))
            except ValueError:
                continue

    breakage_log = output_dir / "breakage_log.txt"
    records = []
    for line in breakage_log.read_text().splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].isdigit():
            continue
        try:
            records.append((int(fields[0]), float(fields[1]), int(fields[2]), int(fields[3])))
        except ValueError:
            continue

    blocks = []
    current = []
    for record in records:
        if current and record[0] <= current[-1][0]:
            blocks.append(current)
            current = []
        current.append(record)
    if current:
        blocks.append(current)
    valid_blocks = [block for block in blocks if block and block[0][0] == 1]
    if not valid_blocks:
        raise FileNotFoundError(f"No coherent breakage block found in {breakage_log}")

    target_lambdas = np.asarray(stress_lambdas[1:])
    block = min(
        valid_blocks,
        key=lambda candidate: (
            abs(len(candidate) - len(target_lambdas)),
            abs(candidate[0][1] - target_lambdas[0])
            + abs(candidate[-1][1] - target_lambdas[-1]),
        ),
    )
    if len(block) != len(target_lambdas):
        raise ValueError("The selected breakage block does not match stress_strain.txt")

    steps = []
    for step_number, lambda_value, cumulative_broken, remaining_count in block:
        step_dir = output_dir / f"step_{step_number:04d}"
        required = (step_dir / "positions.txt", step_dir / "box.txt")
        if all(path.exists() for path in required):
            steps.append((step_number, lambda_value, cumulative_broken, remaining_count, step_dir))
    if not steps:
        raise FileNotFoundError(f"No complete step directories found in {output_dir}")
    return steps


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="box_deformation.gif", help="GIF or MP4 output path")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth simulation step to reduce render time")
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--connections", type=Path, default=Path("connected_xlinks.txt"))
    args = parser.parse_args()

    all_steps = read_steps(args.input_dir)
    stride = max(1, args.stride)
    steps = all_steps[::stride]
    if steps[-1] != all_steps[-1]:
        steps.append(all_steps[-1])
    first_connections = unique_bonds(np.loadtxt(steps[0][4] / "connections.txt", dtype=int))
    initial_bond_set = set(map(tuple, first_connections))
    remaining_by_step = [
        set(map(tuple, unique_bonds(np.loadtxt(step_dir / "connections.txt", dtype=int))))
        for _, _, _, _, step_dir in steps
    ]
    broken_by_step = [set()]
    broken_by_step.extend(initial_bond_set - remaining for remaining in remaining_by_step)
    total_bonds = len(initial_bond_set)
    if not initial_bond_set:
        raise ValueError("No crosslink bonds were found")

    fig = plt.figure(figsize=(14, 6))
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]

    def draw_axis(ax, positions, box, bonds, color, title):
        for collection in list(ax.collections):
            collection.remove()
        ax.set_autoscale_on(False)
        if isinstance(bonds, set):
            bonds = list(bonds)
        bond_array = np.asarray(bonds, dtype=int).reshape(-1, 2)
        segments = periodic_segments(positions, bond_array, box)
        if len(segments):
            ax.add_collection3d(Line3DCollection(
                segments, colors=color, linewidths=0.35, alpha=0.38, axlim_clip=True
            ))
        ax.add_collection3d(Line3DCollection(
            box_segments(box), colors="black", linewidths=0.7, alpha=0.55, axlim_clip=True
        ))
        ax.set_box_aspect(box / np.max(box))
        ax.set_xlim3d(0.0, float(box[0]))
        ax.set_ylim3d(0.0, float(box[1]))
        ax.set_zlim3d(0.0, float(box[2]))
        ax.set_autoscale_on(False)
        ax.grid(False)
        ax.set_title(title, fontsize=14)

    def update(frame):
        if frame == 0:
            step_number, lambda_value, _, _, step_dir = steps[0]
            lambda_value = 1.0
            remaining = initial_bond_set
        else:
            step_number, lambda_value, _, _, step_dir = steps[frame - 1]
            remaining = remaining_by_step[frame - 1]
        positions = np.loadtxt(step_dir / "positions.txt", dtype=float)
        source_box = np.loadtxt(step_dir / "box.txt", dtype=float).reshape(3)
        if frame == 0:
            box = np.full(3, 256.0)
        else:
            box = source_box
        positions = fit_positions_to_box(positions, source_box, box)
        cumulative_broken = len(broken_by_step[frame])
        remaining_count = total_bonds - cumulative_broken
        broken = broken_by_step[frame]
        draw_axis(axes[0], positions, box, remaining, "tab:green",
              f"Remaining connections: {100 * remaining_count / total_bonds:.1f}%")
        draw_axis(axes[1], positions, box, broken, "tab:red",
              f"Broken connections: {100 * cumulative_broken / total_bonds:.1f}%")
        fig.suptitle(f"Box deformation, lambda = {lambda_value:.3f}", fontsize=16)
        return axes

    animation = FuncAnimation(fig, update, frames=len(steps) + 1, interval=1000 / args.fps, blit=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp4":
        animation.save(output_path, writer=FFMpegWriter(fps=args.fps), dpi=120)
    else:
        animation.save(output_path, writer=PillowWriter(fps=args.fps), dpi=100)
        gif = Image.open(output_path)
        frames = []
        try:
            while True:
                frames.append(gif.convert("P", palette=Image.Palette.ADAPTIVE))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, round(1000 / args.fps)),
            loop=0,
            optimize=False,
        )
        gif.close()
    plt.close(fig)
    print(f"Saved {len(steps)} frames to {output_path}")


if __name__ == "__main__":
    main()