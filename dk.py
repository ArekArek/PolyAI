import numpy as np
from utils import CONFIG, dt
import logging
from joblib import Parallel, delayed
import os
from model_gru import ModelGRU
import utils
import torch
import argparse
import time

def _calculate(coeffs, model, max_iter, tolerance):
    deg = CONFIG["polynomial_degree"]
    num_polys = len(coeffs)


    if model:
        coeffs_torch = torch.view_as_real(torch.from_numpy(coeffs))
        logging.info(f"Loading model: {model}")
        with torch.no_grad():
            roots = utils.polar_to_complex(model(coeffs_torch)).numpy()

    else:
        r = CONFIG["evaluation"]["dk_circle_radius"]
        logging.info(f"No model passed, generate starting point on unit circle with r = {r}")
        roots_indices = np.arange(deg)
        phi = (roots_indices * 2 * np.pi / deg) # 0.4 to break symmetry
        roots = r * np.exp(1j * phi)
        roots = np.tile(roots, (num_polys, 1))


    # reverse coefficients for np.polyval and ensure they are 2D
    c_rev = coeffs[:, ::-1].astype(dtype=complex) 
    r = roots.copy().astype(dtype=complex)
    
    f0 = np.array([np.polyval(c_rev[i], r[i]) for i in range(num_polys)])
    mask = np.ones(num_polys, dtype=bool) # True = still iterating
    iterations = np.zeros(num_polys, dtype=int)
    
    # calculate identity for the denominator trick
    eye = np.eye(deg)

    for iteration in range(max_iter):
        if iteration % 100 == 0:
            logging.info(f" iteration: {iteration}/{max_iter} ".center(50, "-"))
            logging.info(f" calculated polynomials: {num_polys - np.sum(mask)}/{num_polys} ".center(50, "-"))

        if not np.any(mask):
            break
            
        # Only update polynomials that haven't converged
        active_r = r[mask]
        active_f0 = f0[mask]
        active_c = c_rev[mask]
        
        # Vectorized Aberth step for the active batch
        # Shape of diffs: (active_batch, deg, deg)
        diffs = active_r[:, :, None] - active_r[:, None, :] + eye
        dens = np.prod(diffs, axis=2)

        # Find where the denominator is dangerously small
        #epsilon = 1e-12
        #small_dens_mask = np.abs(dens) < epsilon

        #if np.any(small_dens_mask):
        #    # Generate tiny complex noise
        #    noise = (np.random.randn(*dens.shape) + 1j * np.random.randn(*dens.shape)) * 1e-6

        #    # Apply noise ONLY to the roots that are colliding
        #    # active_r shape is (batch, deg), noise needs to match
        #    active_r[small_dens_mask] += noise[small_dens_mask]

        #    # Re-calculate diffs and dens after perturbation
        #    diffs = active_r[:, :, None] - active_r[:, None, :] + eye
        #    dens = np.prod(diffs, axis=2)
        
        # Update roots
        new_r = active_r - (active_f0 / dens)

        r[mask] = new_r
        
        # Re-evaluate
        new_f0 = np.array([np.polyval(active_c[i], new_r[i]) for i in range(len(active_c))])

        delta = np.max(np.abs(new_r - active_r), axis=1)
        
        # Update active status
        converged_in_step = (delta <= tolerance)
        
        # Logic to update global mask and iteration counts
        active_indices = np.where(mask)[0]
        iterations[active_indices[~converged_in_step]] += 1
        mask[active_indices[converged_in_step]] = False
        
        f0[mask] = new_f0[~converged_in_step] # Update f0 for the next round

    return iterations

def run(coeffs, model: Optional = None, max_iter = CONFIG["evaluation"]["dk_max_iterations"], tolerance = CONFIG["evaluation"]["dk_tolerance"], n_cores = 1):

    coeffs_splitted = np.array_split(coeffs, n_cores)
        
    logging.info("start")
    start_time = time.perf_counter()
    iterations = Parallel(n_jobs=n_cores)(
        delayed(lambda c: _calculate(c, model, max_iter, tolerance))(c) for c in coeffs_splitted
    )
    end_time = time.perf_counter()
    logging.info("finished")
    logging.info(f"run took {end_time - start_time:.4f} seconds")
    iterations = np.concatenate(iterations)

    logging.info(f"polynomials_total: {len(iterations)}")
    logging.info(f"polynomials_out_of_iterations: {len(iterations[iterations >= max_iter])}")
    logging.info(f"iterations_completed_total: {np.sum(iterations[iterations < max_iter])}")
    logging.info(f"iterations_average: {np.average(iterations[iterations < max_iter])}")
    logging.info(f"iterations_median: {np.median(iterations[iterations < max_iter])}")
    

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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info("=" * 50)
    logging.info(" Run Durand-Kerner algorithm for evaluation ".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info(f" {args} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f" {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    coeffs_path = os.path.join(args.data, "coefficients.npy")
    logging.info(f"Load coefficients from {coeffs_path}")
    coeffs_np_complex = np.load(coeffs_path)


    if args.model:
        logging.info(f"Load model from {args.model}")
        model = ModelGRU()
        model.load_state_dict(utils.read_compiled_model(args.model))
        model.eval()
    else:
        logging.info(f"No model passed in args")
        model = None

    run(coeffs_np_complex, model, n_cores = args.cores)

if __name__ == "__main__":
    main()
