# https://complex-analysis.com/content/domain_coloring.html
# example usage
# import poly_graphics
# poly_graphics.show([0, 0, 1], [1+1j, 2+2j, 3+3j], [1-4j, 8+2j, -2+6j])

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
from collections import Counter
import utils
import torch


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


def show(coeffs, factual_zeroes, predicted_zeroes, logarithmic = False):
    """
    Parameters:
    - coeffs: list of coefficients
    - factual_zeroes: list of factual zeroes
    - predicted_zeroes: list of predicted zeroes
    """
    print(coeffs)
    print(factual_zeroes)
    print(predicted_zeroes)
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
    if not logarithmic:
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
            marker="X",
            s=80,
            label=f"Factual (x{degree})",
        )

    # predicted zeroes
    for zero in predicted_zeroes:
        ax.scatter(
            zero.real, zero.imag, c="red", marker="1", s=140, label=f"Predicted"
        )

    matched_pred, matched_fact = utils.match_closest(
        torch.view_as_real(torch.tensor([predicted_zeroes], dtype=torch.complex64)),
        torch.view_as_real(
            torch.tensor([factual_zeroes_rounded], dtype=torch.complex64)
        ),
    )
    # Linie łączące (uwaga: zip zadziała poprawnie tylko jeśli pred i true mają ten sam porządek)
    for p, t in zip(matched_pred[0], matched_fact[0]):
        ax.plot([p[0], t[0]], [p[1], t[1]], "k--", c="black")

    # axes
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    if logarithmic:
        ax.set_xscale('symlog', linthresh=1.0)
        ax.set_yscale('symlog', linthresh=1.0)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    ax.set_title("Polynomial predicted vs factual zeroes")

    # legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    plt.show()
