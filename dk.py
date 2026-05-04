import numpy as np
from utils import CONFIG, dt
import logging
from joblib import Parallel, delayed
import os
from model_gru import ModelGRU
import utils
import torch
import argparse

def _calculate(coeffs, model, max_iter, tolerance):
    deg = CONFIG["polynomial_degree"]
    num_polys = len(coeffs)

    if model:
        coeffs_torch = torch.view_as_real(torch.from_numpy(coeffs))
        logging.info(f"Loading model: {model}")
        with torch.no_grad():
            r1 = torch.view_as_complex(model(coeffs_torch[0][None,:,:])).numpy()
            roots = np.tile(r1, (num_polys, 1))

    else:
        r = CONFIG["evaluation"]["dk_circle_radius"]
        logging.info(f"No model passed, generate starting point on unit circle with r = {r}")
        roots_indices = np.arange(deg)
        phi = (roots_indices * 2 * np.pi / deg) # 0.4 to break symmetry
        roots = r * np.exp(1j * phi)
        roots = np.tile(roots, (num_polys, 1))

    # reverse coefficients for np.polyval and ensure they are 2D
    c_rev = coeffs[:, ::-1] 
    r = roots.copy()
    
    f0 = np.array([np.polyval(c_rev[i], r[i]) for i in range(num_polys)])
    
    mask = np.ones(num_polys, dtype=bool) # True = still iterating
    iterations = np.zeros(num_polys, dtype=int)
    
    # calculate identity for the denominator trick
    eye = np.eye(deg)

    for _ in range(max_iter):
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
        
        # Update roots
        new_r = active_r - (active_f0 / dens)
        r[mask] = new_r
        
        # Re-evaluate
        new_f0 = np.array([np.polyval(active_c[i], new_r[i]) for i in range(len(active_c))])
        # Calculate convergence: 2-norm across the degree dimension
        # delta shape: (active_batch,)
        # delta = np.sqrt(np.sum(np.abs(new_f0 - active_f0)**2, axis=1) / deg)
        delta = np.mean(np.abs(new_f0 - active_f0), axis=1)
        # delta = np.linalg.norm((new_f0 - active_f0), ord=2, axis=1) / np.sqrt(deg)
        # delta = np.linalg.norm(np.max(new_f0 - active_f0, 0.00001), ord=2, axis=1) / np.sqrt(deg)
        
        # Update active status
        converged_in_step = (delta <= tolerance)
        
        # Logic to update global mask and iteration counts
        active_indices = np.where(mask)[0]
        iterations[active_indices[~converged_in_step]] += 1
        mask[active_indices[converged_in_step]] = False
        
        f0[mask] = new_f0[~converged_in_step] # Update f0 for the next round

    polynomials_out_of_iterations = np.sum(mask)
    iterations_completed_total = np.sum(iterations[~mask])
    return {
        "polynomials_total": num_polys,
        "polynomials_out_of_iterations": polynomials_out_of_iterations.item(),
        "polynomials_correct_roots": (num_polys - polynomials_out_of_iterations).item(),
        "iterations_completed_total": iterations_completed_total.item(),
        "iterations_avg": (iterations_completed_total / (num_polys - polynomials_out_of_iterations)).item()
    }

def run(coeffs, model: Optional = None, max_iter = CONFIG["evaluation"]["dk_max_iterations"], tolerance = CONFIG["evaluation"]["dk_tolerance"], n_cores = 1):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{CONFIG['logs_path']}{dt()}-eval.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info("=" * 50)
    logging.info(" Evaluate durand kerner ".center(50, "="))
    logging.info(f" model is not provided: {not model}".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f"defaults: {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    coeffs_splitted = np.array_split(coeffs, n_cores)
        
    logging.info("start")
    stats = Parallel(n_jobs=n_cores)(
        delayed(lambda c: _calculate(c, model, max_iter, tolerance))(c) for c in coeffs_splitted
    )
    
    summary = {
        "polynomials_total": sum(d["polynomials_total"] for d in stats),
        "polynomials_out_of_iterations": sum(d["polynomials_out_of_iterations"] for d in stats),
        "polynomials_correct_roots": sum(d["polynomials_correct_roots"] for d in stats),
        "iterations_completed_total": sum(d["iterations_completed_total"] for d in stats),
    }
    
    # Calculate the global average
    # We usea conditional to avoid DivisionByZero if no correct roots exist
    if summary["polynomials_correct_roots"] > 0:
        summary["iterations_avg"] = (
            summary["iterations_completed_total"] / summary["polynomials_correct_roots"]
        )
    else:
        summary["iterations_avg"] = -99999
    logging.info(f"Summary:\n{summary}")

    logging.info("finished")

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

    run(coeffs_np_complex[0:100], model)

if __name__ == "__main__":
    main()
