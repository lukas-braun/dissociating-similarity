import networkx as nx

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

_original_draw = Axes.draw


def extract_data_limits(ax):
    all_x, all_y = [], []
    for line in ax.lines:
        xdata, ydata = line.get_xdata(orig=True), line.get_ydata(orig=True)
        all_x.extend(xdata)
        all_y.extend(ydata)
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets):
            all_x.extend(offsets[:, 0])
            all_y.extend(offsets[:, 1])
    data_xmin = min(all_x) if all_x else None
    data_xmax = max(all_x) if all_x else None
    data_ymin = min(all_y) if all_y else None
    data_ymax = max(all_y) if all_y else None
    return data_xmin, data_xmax, data_ymin, data_ymax


def update_spine_bounds(ax):
    x_margin = plt.rcParams["axes.xmargin"]
    y_margin = plt.rcParams["axes.ymargin"]

    def apply_margin(vmin, vmax, margin):
        vrange = vmax - vmin
        pad = vrange * margin
        return vmin - pad, vmax + pad

    data_xmin, data_xmax, data_ymin, data_ymax = extract_data_limits(ax)

    if not hasattr(ax, "_raw_xlim"):
        if ax.get_autoscalex_on():
            ax._raw_xlim = (data_xmin, data_xmax)
        else:
            ax._raw_xlim = ax.get_xlim()

    if not hasattr(ax, "_raw_ylim"):
        if ax.get_autoscaley_on():
            ax._raw_ylim = (data_ymin, data_ymax)
        else:
            ax._raw_ylim = ax.get_ylim()

    raw_xlim = ax._raw_xlim
    raw_ylim = ax._raw_ylim

    if raw_xlim[0] is not None and raw_xlim[1] is not None:
        new_xlim = apply_margin(raw_xlim[0], raw_xlim[1], x_margin)
        ax.set_xlim(new_xlim)
        ax.spines["bottom"].set_bounds(raw_xlim[0], raw_xlim[1])

    if raw_ylim[0] is not None and raw_ylim[1] is not None:
        new_ylim = apply_margin(raw_ylim[0], raw_ylim[1], y_margin)
        ax.set_ylim(new_ylim)
        ax.spines["left"].set_bounds(raw_ylim[0], raw_ylim[1])


def patched_draw(self, renderer, *args, **kwargs):
    update_spine_bounds(self)
    return _original_draw(self, renderer, *args, **kwargs)


Axes.draw = patched_draw


def create_hierarchical_graph(tree, radius_step=1.):
    def polar(radius, angle):
        return radius * np.cos(angle), radius * np.sin(angle)

    def traverse(tree, level=1, angle_start=0, angle_end=2*np.pi):
        node, children = next(iter(tree.items()))
        pos = {node: (0., 0.)}
        edges = []
        queue = [(node, children, level, angle_start, angle_end)]

        while queue:
            parent, subtree, level, theta_start, theta_end = queue.pop()
            items = list(subtree.items())
            step = (theta_end - theta_start) / max(1, len(items))
            for i, (child, child_tree) in enumerate(items):
                theta = theta_start + i * step + step / 2.
                pos[child] = polar(level * radius_step, theta)
                edges.append((parent, child))
                if child_tree:
                    t1 = theta - step / 2.
                    t2 = theta + step / 2.
                    queue.append((child, child_tree, level + 1, t1, t2))
        return pos, edges

    pos, edges = traverse(tree)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    root = next(iter(tree))
    dists = nx.single_source_shortest_path_length(G, root)
    return G, pos, dists
