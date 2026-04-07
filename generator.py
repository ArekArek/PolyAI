import numpy as np
from numpy.polynomial import Polynomial
import os
import argparse
import utils
from utils import CONFIG, dt
import logging
import itertools


def _zeroes_multiplicities_combinations():
    min_multiplicity = 2
    possible_multiplicity_values = range(
        min_multiplicity, CONFIG["polynomial_degree"] + 1
    )
    possible_permutation_lengths = range(
        1, (CONFIG["polynomial_degree"] // min_multiplicity) + 1
    )

    def combinations_of_length(length):
        combinations = itertools.combinations_with_replacement(
            possible_multiplicity_values, r=length
        )
        return filter(
            lambda combination: sum(combination) <= CONFIG["polynomial_degree"],
            combinations,
        )

    return list(
        itertools.chain.from_iterable(
            [combinations_of_length(length) for length in possible_permutation_lengths]
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Generate data")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Save output to default training dataset directory",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Save output to default test dataset directory",
    )
    parser.add_argument(
        "--random",
        type=int,
        help="Size of output polynomials with totally random zeroes distribution",
        default=CONFIG["training"]["randomly_distributed_zeroes"],
    )
    parser.add_argument(
        "--repeated",
        type=int,
        help="Size of output polynomials with random zeroes distribution with repeated root. Thats number of rows for each zeroes multiplicity combination",
        default=CONFIG["training"]["repeated_distributed_zeroes"],
    )
    parser.add_argument(
        "--uniform",
        type=int,
        help="Size of output polynomials with uniform zeroes distribution",
        default=CONFIG["training"]["uniformly_distributed_zeroes"],
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disables shuffle of different types of zeroes",
    )
    args = parser.parse_args()

    if not args.out:
        if args.test:
            args.out = CONFIG["evaluation"]["test_data_path"]
        elif args.training:
            args.out = CONFIG["training"]["input_data_path"]
        else:
            raise Exception(f"Incorrect arguments {args}")

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{CONFIG['logs_path']}{dt()}-generate.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 50)
    logging.info(" Generate data ".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info(f" {args} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f"defaults: {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    logging.info(f"random zeroes: {args.random}")
    randomly_distributed_zeroes = utils.generate_randomly_distributed_zeroes(
        args.random
    )

    logging.info(f"uniform zeroes: {args.uniform}")
    uniformly_distributed_zeroes = utils.generate_uniformly_distributed_zeroes(
        args.uniform
    )

    multiplicities_combinations = _zeroes_multiplicities_combinations()
    logging.info(
        f"random zeroes with multiplicity: {args.repeated} x {len(multiplicities_combinations)} = {args.repeated * len(multiplicities_combinations)}"
    )
    repeated_zeroes = []
    for multiplicities in multiplicities_combinations:
        new_repeated_zeroes = utils.generate_randomly_distributed_zeroes(
            args.repeated, multiplicities
        )
        repeated_zeroes.append(new_repeated_zeroes)
    repeated_zeroes = np.vstack(repeated_zeroes)

    logging.info("Merging all zeroes")
    zeroes = np.vstack(
        (randomly_distributed_zeroes, uniformly_distributed_zeroes, repeated_zeroes)
    )
    if not args.no_shuffle:
        np.random.shuffle(zeroes)

    # sort zeroes for each polynomial by distance from zero (abs)
    order = np.argsort(np.abs(zeroes), axis=1)
    zeroes_sorted = np.take_along_axis(zeroes, order, axis=1)

    # calculate coefficients
    logging.info("Calculating coefficients")
    coeffs = np.array(
        [Polynomial.fromroots(row).coef for row in zeroes], dtype=np.complex64
    )

    logging.info(f"Saving coefficients to {args.out}coefficients.npy")
    np.save(args.out + "coefficients.npy", coeffs)
    logging.info(f"Saving zeroes to {args.out}zeroes.npy")
    np.save(args.out + "zeroes.npy", zeroes_sorted)

    logging.info(f"{len(coeffs)} rows generated in total")
    logging.info(f"Generated data saved correctly.")


if __name__ == "__main__":
    main()
