import numpy as np
from numpy.polynomial import Polynomial
import os
import torch

MAX_FLOAT = torch.finfo(torch.float32).max
BOUNDS_MODIFIER = 1e-33
LIMIT = MAX_FLOAT * BOUNDS_MODIFIER


import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


rng = np.random.default_rng()


def generate_randomly_distributed_zeroes(n_rows, multiple=[]):

    def gen(n):
        real_parts = rng.uniform(
            -LIMIT, LIMIT, size=(n, CONFIG["polynomial_degree"])
        ).astype(np.float32)
        imag_parts = rng.uniform(
            -LIMIT, LIMIT, size=(n, CONFIG["polynomial_degree"])
        ).astype(np.float32)

        z = real_parts + 1j * imag_parts

        # --- multiple zeroes logic ---
        if multiple:
            current_idx = 0
            for r in multiple:
                if r > 1:
                    # copy zero of r degree, and copy it to next r-1 positions
                    source_idx = current_idx
                    target_end = current_idx + r

                    if target_end <= CONFIG["polynomial_degree"]:
                        # copy rest of original zeroes
                        z[:, source_idx + 1 : target_end] = z[
                            :, source_idx : source_idx + 1
                        ]
                        current_idx = target_end
                    else:
                        break
                else:
                    current_idx += 1
        return z

    zeroes = gen(n_rows)

    while True:
        # calculate abs of complex number and ignore those less than 1, to retrieve biggest possible combination
        filtered_elements = np.where(np.abs(zeroes) > 1, zeroes, 1.0)
        products_mag = np.abs(np.prod(filtered_elements, axis=1))

        mask = products_mag > MAX_FLOAT
        invalid_count = np.sum(mask)

        if invalid_count == 0:
            break

        print(f"Redrawing zeroes because of oveflow: {invalid_count}")
        # regenerate zeroes for those which were too big to fit float
        zeroes[mask] = gen(invalid_count)

    return zeroes
