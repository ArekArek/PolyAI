from datetime import datetime
import time
import argparse
import utils
from utils import CONFIG, dt
from model_gru import ModelGRU
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import poly_graphics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=CONFIG["training"]["output_model_path"],
        help="Path to model",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=CONFIG["evaluation"]["test_data_path"],
        help="Path to test data",
    )
    parser.add_argument(
        "-i", "--index", type=int, default=0, help="Index of record to be shown"
    )
    parser.add_argument(
        "-log",
        "--logarithmic",
        action="store_true",
        help="Logarithmic scale",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info("=" * 50)
    logging.info(" Show model ".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info(f" {args} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f" {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    coeffs_np_complex = np.expand_dims(
        np.load(os.path.join(args.data, "coefficients.npy"))[args.index], 0
    )
    coeff_tensor_complex = torch.from_numpy(coeffs_np_complex)
    coeff_tensor = torch.view_as_real(coeff_tensor_complex)

    zeroes_np_complex = np.expand_dims(
        np.load(os.path.join(args.data, "zeroes.npy"))[args.index], 0
    )
    factual_zeroes_complex = torch.from_numpy(zeroes_np_complex)
    factual_zeroes = torch.view_as_real(factual_zeroes_complex)

    model = ModelGRU()
    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        return

    model.load_state_dict(utils.read_compiled_model(args.model))
    model.eval()

    with torch.no_grad():
        predicted_zeroes = model(coeff_tensor)
 
    matched_zeroes = utils.match_closest(torch.view_as_real(polar_to_complex(predicted_zeroes)), factual_zeroes)
    loss = F.l1_loss(*matched_zeroes)

    logging.info(f" {factual_zeroes.size} test rows used in total ".center(50, "="))
    logging.info("=" * 50)
    logging.info(f" Result loss is {loss} ".center(50, "="))
    logging.info("=" * 50)

    poly_graphics.show(
        coeffs_np_complex[0].tolist(),
        zeroes_np_complex[0].tolist(),
        polar_to_complex(predicted_zeroes[0]).tolist(),
        args.logarithmic
    )


if __name__ == "__main__":
    main()
