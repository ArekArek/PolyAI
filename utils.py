import numpy as np
from numpy.polynomial import Polynomial
import os
import torch
from scipy.optimize import linear_sum_assignment
import datetime

MAX_FLOAT = torch.finfo(torch.float32).max * 1e-2
MAX_LOG = 6

import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


rng = np.random.default_rng()


def dt(nice=False):
    if nice:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def __random_log_uniform_float32(size):
    sign = np.random.choice([-1, 1], size=size)
    exponent = np.random.uniform(-MAX_LOG, MAX_LOG, size=size)
    mantissa = np.random.uniform(1.0, 10.0, size=size)
    return (sign * mantissa * 10**exponent).astype(np.float32)


def generate_uniformly_distributed_zeroes(n_rows):
    def gen(n):
        weights = rng.random((n, CONFIG["polynomial_degree"]))
        weights = (weights / weights.sum(axis=1, keepdims=True)) * MAX_LOG

        phi = rng.uniform(0, 2 * np.pi, size=(n, CONFIG["polynomial_degree"])).astype(
            np.float32
        )
        magnitude = __random_log_uniform_float32((n, CONFIG["polynomial_degree"]))
        z = magnitude * np.exp(1j * phi)
        return z

    zeroes = gen(n_rows)

    while True:
        # calculate abs of complex number and ignore those less than 1, to retrieve biggest possible combination
        filtered_elements = np.where(np.abs(zeroes) > 1, zeroes, 1.0)
        products_mag = np.abs(np.prod(filtered_elements, axis=1))

        mask = (~np.isfinite(products_mag)) | (products_mag > MAX_FLOAT)
        invalid_count = np.sum(mask)

        if invalid_count == 0:
            break

        print(
            f"Redrawing uniformly distributed zeroes because of oveflow: {invalid_count}"
        )
        # regenerate zeroes for those which were too big to fit float
        zeroes[mask] = gen(invalid_count)

    return zeroes


def generate_randomly_distributed_zeroes(n_rows, multiple=[]):

    def gen(n):
        real_parts = __random_log_uniform_float32((n, CONFIG["polynomial_degree"]))
        imag_parts = __random_log_uniform_float32((n, CONFIG["polynomial_degree"]))

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
        
        mask = (~np.isfinite(products_mag)) | (products_mag > MAX_FLOAT)
        invalid_count = np.sum(mask)

        if invalid_count == 0:
            break

        print(
            f"Redrawing randomly distributed zeroes because of oveflow: {invalid_count}"
        )
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


def read_compiled_model(path):
    import torch

    state_dict = torch.load(path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")  # remove the prefix
        new_state_dict[name] = v
    return new_state_dict
