import numpy as np
from numpy.polynomial import Polynomial
import os
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

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


def match_closest(a, b):
    """
    Solves assignments problem using Hungarian algorithm
    Parameters:
    - a, b: tensor.float32 in size [rows count, polynomial degree, 2 (real and imaginary part)]
        it can be manually tested like:
            > a = torch.tensor([[[1, 1], [-5, -5], [10, 10]]], dtype=torch.float32)
            > b = torch.tensor([[[-1, -1], [0, 0], [2, 2]]], dtype=torch.float32)
            > utils.assignments(a, b)
    """

    batch_size, num_points, _ = a.shape

    # 1. Calculate pairwise Euclidean distances between all points in each batch.
    # The Euclidean distance between (real, imag) pairs is mathematically equivalent
    # to the magnitude of the difference between complex numbers.
    # We use .detach() because the assignment step shouldn't track gradients.
    dist_matrix = torch.cdist(a.detach(), b.detach())

    # Convert to numpy for scipy's linear_sum_assignment
    dist_matrix_np = dist_matrix.cpu().numpy()

    # 2. Collect the optimal permutation indices for the factual tensor
    matched_b_indices = []
    for i in range(batch_size):
        cost_matrix = dist_matrix_np[i]

        # linear_sum_assignment finds the lowest cost bipartite matching.
        # col_ind tells us which factual element optimally maps to predicted[row_ind].
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_b_indices.append(col_ind)

    # Convert the matched indices back to a tensor, shape: (batch_size, num_points)
    matched_indices = torch.tensor(matched_b_indices, device=b.device, dtype=torch.long)

    # 3. Gather the factual values according to the matched indices
    # We need to expand matched_indices to gather along the last dimension (real/imag pairs)
    # matched_indices_expanded shape becomes: (x, polynomial_degree, 2)
    matched_indices_expanded = matched_indices.unsqueeze(-1).expand(-1, -1, 2)

    # Reorder 'factual' along dimension 1 (the polynomial_degree items) to align with 'predicted'
    matched_b = torch.gather(b, 1, matched_indices_expanded)
    return (a, matched_b)



