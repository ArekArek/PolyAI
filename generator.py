import numpy as np
from numpy.polynomial import Polynomial
import os
import argparse
import utils
from utils import CONFIG, dt
import logging

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


logging.info(f"Saving coefficients to {args.out}coefficients.npy")
np.save(args.out + "coefficients.npy", coeffs)
logging.info(f"Saving zeroes to {args.out}zeroes.npy")
np.save(args.out + "zeroes.npy", zeroes_sorted)

logging.info(f"{coeffs.size} rows generated in total")
logging.info(f"Generated data saved correctly.")
