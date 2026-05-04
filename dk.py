import numpy as np
from utils import CONFIG, dt
import logging
from joblib import Parallel, delayed

def _calculate(coeffs, roots, max_iter, tolerance):
    deg = CONFIG["polynomial_degree"]
    num_polys = len(coeffs)
    
    if not roots:
        r = CONFIG["evaluation"]["dk_circle_radius"]
        roots_indices = np.arange(deg)
        roots = r * np.exp(1j * roots_indices * 2 * np.pi / deg)
        roots = np.tile(roots, (len(coeffs), 1))
        
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
        delta = np.sqrt(np.sum((new_f0 - active_f0)**2, axis=1) / deg)
        
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

def durand_kerner(coeffs, roots: Optional = None, max_iter = CONFIG["evaluation"]["dk_max_iterations"], tolerance = CONFIG["evaluation"]["dk_tolerance"], n_cores = 1):
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
    logging.info(f" roots missing: {not roots}".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f"defaults: {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    coeffs_splitted = np.array_split(coeffs, n_cores)
    if roots:
        roots_splitted = np.array_split(roots, n_cores)
    else:
        roots_splitted = [None]*n_cores
        
    logging.info("start")
    stats = Parallel(n_jobs=n_cores)(
        delayed(lambda row: _calculate(row[0], row[1], max_iter, tolerance))(row) for row in zip(coeffs_splitted, roots_splitted)
    )
    
    summary = {
        "polynomials_total": sum(d["polynomials_total"] for d in stats),
        "polynomials_out_of_iterations": sum(d["polynomials_out_of_iterations"] for d in stats),
        "polynomials_correct_roots": sum(d["polynomials_correct_roots"] for d in stats),
        "iterations_completed_total": sum(d["iterations_completed_total"] for d in stats),
    }
    
    # Calculate the global average
    # We use a conditional to avoid DivisionByZero if no correct roots exist
    if summary["polynomials_correct_roots"] > 0:
        summary["iterations_avg"] = (
            summary["iterations_completed_total"] / summary["polynomials_correct_roots"]
        )
    else:
        summary["iterations_avg"] = -99999
    logging.info(f"Summary:\n{summary}")

    logging.info("finished")

