# ruff: noqa: E741
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import matplotlib as mpl


REF_X = 95.047
REF_Y = 100.000
REF_Z = 108.883


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) - 0.055)


def rgb_to_xyz(rgb):
    rgb = srgb_to_linear(rgb)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = np.dot(M, rgb) * 100
    return xyz


def xyz_to_rgb(xyz):
    xyz = xyz / 100
    M_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb_linear = np.dot(M_inv, xyz)
    return linear_to_srgb(rgb_linear)


def f(t):
    delta = 6 / 29
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)


def f_inv(t):
    delta = 6 / 29
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))


def xyz_to_lab(xyz):
    x, y, z = xyz[0] / REF_X, xyz[1] / REF_Y, xyz[2] / REF_Z
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b])


def lab_to_xyz(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    x = REF_X * f_inv(fx)
    y = REF_Y * f_inv(fy)
    z = REF_Z * f_inv(fz)
    return np.array([x, y, z])


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return np.array([int(hex_str[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def rgb_to_hex(rgb):
    rgb = np.clip(rgb, 0, 1)
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in rgb))


def hex_to_lab(hex_str):
    return xyz_to_lab(rgb_to_xyz(hex_to_rgb(hex_str)))


def lab_to_hex(lab):
    return rgb_to_hex(xyz_to_rgb(lab_to_xyz(lab)))


def bezier_interp(lab_points, t):
    points = lab_points[:]
    while len(points) > 1:
        points = [(1 - t) * p0 + t * p1 for p0, p1 in zip(points[:-1], points[1:])]
    return points[0]


def correct_lightness(lab_colors):
    L_start = lab_colors[0][0]
    L_end = lab_colors[-1][0]
    corrected = []
    for i, lab in enumerate(lab_colors):
        t = i / (len(lab_colors) - 1) if len(lab_colors) > 1 else 0
        L = L_start + (L_end - L_start) * t
        corrected.append(np.array([L, lab[1], lab[2]]))
    return corrected


def bezier_palette(hex_colors, steps):
    lab_points = [hex_to_lab(h) for h in hex_colors]
    sampled = [bezier_interp(lab_points, i / (steps - 1)) for i in range(steps)]
    sampled = correct_lightness(sampled)
    return [lab_to_hex(lab) for lab in sampled]


def brighten(hex_color):
    l, a, b = hex_to_lab(hex_color)
    l = np.clip(l + 30, 0, 100)
    return lab_to_hex([l, a, b])


def darken(hex_color):
    l, a, b = hex_to_lab(hex_color)
    l = np.clip(l - 30, 0, 100)
    return lab_to_hex([l, a, b])


def sequential(hex_colors, steps):
    if isinstance(hex_colors, str):
        hex_colors = [darken(hex_colors), brighten(hex_colors)]
    return np.asarray([hex_to_rgb(c) for c in bezier_palette(hex_colors, steps)])


def auto_gradient(color_hex):
    lab = hex_to_lab(color_hex)
    l_range = 100.0 * (0.95 - 1.0 / 3)
    l_step = l_range / (3 - 1)
    l_start = (100.0 - l_range) / 2.0
    Ls = [l_start + i * l_step for i in range(3)]
    return [np.array([L, lab[1], lab[2]]) for L in Ls]


def auto_colors(color_hex, reverse=False):
    pts = auto_gradient(color_hex) + [hex_to_lab("#f5f5f5")]
    if reverse:
        pts.reverse()
    return pts


def diverging(color1, color2, N):
    even = N % 2 == 0
    n_left = int(np.ceil(N / 2.0) + (1 if even else 0))
    n_right = n_left

    gen_left = auto_colors(color1, False)
    gen_right = auto_colors(color2, True)

    gen_left = correct_lightness(gen_left)
    gen_right = correct_lightness(gen_right)

    left_samples = [bezier_interp(gen_left, i / (n_left - 1)) for i in range(n_left)]
    right_samples = [
        bezier_interp(gen_right, i / (n_right - 1)) for i in range(n_right)
    ]

    if even:
        left_samples = left_samples[:-1]
    right_samples = right_samples[1:]

    full = left_samples + right_samples
    return [lab_to_hex(lab) for lab in full]


hex_colors = diverging("#0071b2", "#009e73", 267)
mpl.colormaps.register(cmap=ListedColormap(hex_colors, name="pretty"))


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
