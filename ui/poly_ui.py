# https://complex-analysis.com/content/domain_coloring.html
# example usage
# import ui.poly_ui
# ui.poly_ui.show([0, 0, 1], [1+1j, 2+2j, 3+3j], [1-4j, 8+2j, -2+6j])

import numpy as np
from numpy.polynomial import Polynomial
from ui.dcolor import DColor

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
from collections import Counter

SAMPLING = 1_000  # 'quality' of figure
FIGURE_SIZE = 8
DEGREE_COLOR = {
    1: "blue",
    2: "red",
    3: "brown",
    4: "pink",
    5: "yellow",
}
DEFAULT_COLOR = "purple"


def _find_bounds(all_zeroes):
    all_parts = np.array(all_zeroes).view(float)
    _min = min(all_parts)
    _max = max(all_parts)
    margin = (_max - _min) * 0.1
    return (_min - margin, _max + margin)


def _make_domain(min, max):
    x = np.linspace(min, max, SAMPLING)
    y = np.linspace(min, max, SAMPLING)
    return np.meshgrid(x, y)


def _to_complex(a: tuple):
    return a[0] + 1j * a[1]


def _make_color_model(zz):
    H = _normalize(np.angle(zz) % (2.0 * np.pi))  # Hue determined by arg(z)
    r = np.log2(1.0 + np.abs(zz))
    S = (1.0 + np.abs(np.sin(2.0 * np.pi * r))) / 2.0
    V = (1.0 + np.abs(np.cos(2.0 * np.pi * r))) / 2.0

    return H, S, V


def _normalize(arr):
    """Used for normalizing data in array based on min/max values"""
    arrMin = np.min(arr)
    arrMax = np.max(arr)
    arr = arr - arrMin
    return arr / (arrMax - arrMin)


def show(coeffs, factual_zeroes, predicted_zeroes):
    """
    Parameters:
    - coeffs: list of coefficients
    - factual_zeroes: list of factual zeroes
    - predicted_zeroes: list of predicted zeroes
    """
    coord_min, coord_max = _find_bounds(factual_zeroes + predicted_zeroes)
    print(f"Bounds: coord_min={coord_min}, coord_max={coord_max}")

    p = Polynomial(coeffs)
    zz = p(_to_complex(_make_domain(coord_min, coord_max)))
    H, S, V = _make_color_model(zz)
    rgb = hsv_to_rgb(np.dstack((H, S, V)))

    fig = plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE), dpi=100)
    ax = fig.gca()
    ax.set_xlabel("real")
    ax.set_ylabel("imag")

    # domain coloring
    ax.imshow(rgb, extent=[coord_min, coord_max, coord_min, coord_max], origin="lower")

    # factual zeroes
    factual_zeroes_rounded = np.round(factual_zeroes, decimals=6)
    counts = Counter(factual_zeroes_rounded)
    unique_factual_zeroes = list(set(factual_zeroes_rounded))
    for zero in unique_factual_zeroes:
        degree = counts[zero]
        color = DEGREE_COLOR.get(degree, DEFAULT_COLOR)
        ax.scatter(
            zero.real,
            zero.imag,
            c=color,
            marker="o",
            s=80,
            label=f"Factual (x{degree})",
            alpha=0.5,
        )

    for zero in predicted_zeroes:
        ax.scatter(zero.real, zero.imag, c="red", marker="1", s=140, label=f"Predicted")

    #    ax.invert_yaxis()  # make CCW orientation positive
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Polynomial predicted vs factual zeroes")
    plt.show()
