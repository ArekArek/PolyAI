import numpy as np
from numpy.polynomial import Polynomial
import os
import argparse

import utils
from utils import CONFIG

parser = argparse.ArgumentParser(description="Generate data")
parser.add_argument(
    "-o",
    "--out",
    type=str,
    help="Path to output directory",
    default=CONFIG["training"]["input_data_path"],
)
parser.add_argument(
    "--random",
    type=int,
    help="Size of output polynomials with totally random zeroes distribution",
    default=20,
)
args = parser.parse_args()

if not os.path.exists(args.out):
    os.makedirs(args.out)

print("Start data generation")
print(f"random zeroes: {args.random}")


randomly_distributed_zeroes = utils.generate_randomly_distributed_zeroes(args.random)

zeroes = randomly_distributed_zeroes

np.random.shuffle(zeroes)


# sort zeroes for each polynomial by distance from zero (abs)
order = np.argsort(np.abs(zeroes), axis=1)
zeroes_sorted = np.take_along_axis(zeroes, order, axis=1)

# calculate coefficients
coeffs = np.array(
    [Polynomial.fromroots(row).coef for row in zeroes], dtype=np.complex64
)


print(f"Saving coefficients to {args.out}coefficients.npy")
np.save(args.out + "coefficients.npy", coeffs)
print(f"Saving zeroes to {args.out}zeroes_sorted.npy")
np.save(args.out + "zeroes.npy", zeroes_sorted)

print(f"Generated data seved correctly.")
