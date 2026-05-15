import numpy as np
from utils import CONFIG, dt
import logging
from joblib import Parallel, delayed
import os
from model_gru import ModelGRU
from model_mlp import ModelMLP
import utils
import torch
import argparse
import time

def _calculate(coeffs, zeroes, model, max_iter, tolerance):
    deg = CONFIG["polynomial_degree"]
    num_polys = len(coeffs)

    if model:
        coeffs_torch = torch.view_as_real(torch.from_numpy(coeffs.copy()))
        logging.info(f"Loading model: {model}")
        with torch.no_grad():
            model_output = model(coeffs_torch)
            starting_points = torch.view_as_complex(model_output).numpy()

    else:
        r = CONFIG["evaluation"]["dk_circle_radius"]
        logging.info(f"No model passed, generate starting point on unit circle with r = {r}")
        roots_indices = np.arange(deg)
        phi = (roots_indices * 2 * np.pi / deg) + 0.4 # 0.4 to break symmetry
        roots = r * np.exp(1j * phi)
        starting_points = np.tile(roots, (num_polys, 1))

    # reverse coefficients to be in decreasing order (for np.polyval)
    c_rev = coeffs[:, ::-1].astype(dtype=complex)
    r = starting_points.copy().astype(dtype=complex)
    
    f0 = np.array([np.polyval(c_rev[i], r[i]) for i in range(num_polys)])
    
    # mask to select only polynomials which havent finished
    mask = np.ones(num_polys, dtype=bool) # True/1 means active polynomial (not converged)
    iterations = np.zeros(num_polys, dtype=int)
    
    # calculate identity for the diffs
    eye = np.eye(deg)

    for iteration in range(max_iter):
        if iteration % 100 == 0:
            logging.info(f" iteration: {iteration}/{max_iter} ".center(50, "-"))
            logging.info(f" calculated polynomials: {num_polys - np.sum(mask)}/{num_polys} ".center(50, "-"))
            
        if not np.any(mask):
            break
            
        # update only polynomials that haven't converged
        active_r = r[mask]
        active_f0 = f0[mask]
        active_c = c_rev[mask]
        
         # (zi - zj), eye is added so for i=j we have (zi-zi)+1=1 what will not break product
        diffs = active_r[:, :, None] - active_r[:, None, :] + eye
        # product of (zi - zj)
        dens = np.prod(diffs, axis=2)
        
        new_r = active_r - (active_f0 / dens)
        
        r[mask] = new_r
        
        new_f0 = np.array([np.polyval(active_c[i], new_r[i]) for i in range(len(active_c))])
        delta = np.max(np.abs(new_r - active_r), axis=1)
        
        converged_in_step = (delta <= tolerance)
        
        # update global mask (for finished) and iteration counts (for active)
        active_indices = np.where(mask)[0]
        iterations[active_indices[~converged_in_step]] += 1
        mask[active_indices[converged_in_step]] = False
        
        f0[mask] = new_f0[~converged_in_step] # Update f0 for the next round

    if zeroes is not None and zeroes.any():
        order = np.argsort(np.abs(r), axis=1)
        r_sorted = np.take_along_axis(r, order, axis=1)
        for i in range(num_polys):
            logging.info(f"{i}) Coefficients: {coeffs[i]}")
            logging.info(f"{i}) Factual roots: {zeroes[i]}")
            logging.info(f"{i}) Roots found: {r_sorted[i]}")
    return iterations

def main():
    parser = argparse.ArgumentParser(description="Run Durand-Kerner algorithm for evaluation")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
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
        "-c",
        "--cores",
        type=int,
        default=1,
        help="How many cores have to be used for parallel computations.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        help="How many results should be printed.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{CONFIG['logs_path']}{dt()}-eval.log")
            ]
    )

    logging.info("=" * 50)
    logging.info(" Run Durand-Kerner algorithm for evaluation ".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info(f" {args} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f" {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    coeffs_path = os.path.join(args.data, "coefficients.npy")
    zeroes_path = os.path.join(args.data, "zeroes.npy")
    logging.info(f"Load coefficients from {coeffs_path}")
    if args.verbose:
        coeffs = np.load(coeffs_path)[0:args.verbose]
        zeroes = np.load(zeroes_path)[0:args.verbose]
        zeroes_splitted = np.array_split(zeroes, args.cores)
    else:    
        coeffs = np.load(coeffs_path)
        zeroes_splitted = [None]*args.cores
    coeffs_splitted = np.array_split(coeffs, args.cores)

    if args.model:
        logging.info(f"Load model from {args.model}")
        model = ModelGRU()
        if CONFIG["training"]["model_type"] == "mlp":
            model = ModelMLP()
        model.load_state_dict(utils.read_compiled_model(args.model))
        model.eval()
    else:
        logging.info(f"No model passed in args")
        model = None

    max_iter = CONFIG["evaluation"]["dk_max_iterations"]

        
    logging.info("START")
    start_time = time.perf_counter()
    iterations = Parallel(n_jobs=args.cores)(
        delayed(lambda z: _calculate(z[0], z[1], model, max_iter, CONFIG["evaluation"]["dk_tolerance"]))(z) for z in zip(coeffs_splitted, zeroes_splitted)
    )
    end_time = time.perf_counter()
    logging.info("FINISHED")
    
    logging.info(f"run took {end_time - start_time:.4f} seconds")
    iterations = np.concatenate(iterations)

    logging.info(f"polynomials_total: {len(iterations)}")
    logging.info(f"polynomials_out_of_iterations: {len(iterations[iterations >= max_iter])}")
    logging.info(f"iterations_completed_total: {np.sum(iterations[iterations < max_iter])}")
    logging.info(f"iterations_average: {np.average(iterations[iterations < max_iter])}")
    logging.info(f"iterations_median: {np.median(iterations[iterations < max_iter])}")
    
if __name__ == "__main__":
    main()
