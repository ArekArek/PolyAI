import numpy as np
from utils import CONFIG, dt
import logging

def durand_kerner(coeffs, factual_roots, roots: Optional = None, max_iter = CONFIG["evaluation"]["dk_max_iterations"], tolerance = CONFIG["evaluation"]["dk_tolerance"]):
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

    if not roots:
        r = CONFIG["evaluation"]["dk_circle_radius"]
        roots_indices = np.arange(CONFIG["polynomial_degree"])
        roots = r * np.exp(1j * roots_indices * 2 * np.pi / CONFIG["polynomial_degree"])
        roots = np.tile(roots, (len(coeffs), 1))
    
    iterations_total = 0
    polynomials_out_of_iterations = 0
    incorrect_roots = 0
    correct_roots=0

    for i in range(len(coeffs)):
        c = coeffs[i][::-1]
        r = roots[i]
        cnt = 0
        f0 = np.polyval(c, r)
        avg_delta = tolerance + 1  
        while avg_delta > tolerance and cnt < max_iter:
            cnt += 1
        
            # Simultaneous distance calculation using advanced indexing
            dens = np.prod(r[:, None] - r[None, :] + np.eye(CONFIG["polynomial_degree"]), axis=1)
            r = r - f0 / dens
        
            f1 = np.polyval(c, r)
            # 2-norm of the change in polyval
            avg_delta = np.sqrt(np.sum((f1 - f0)**2) / CONFIG["polynomial_degree"])
            f0 = f1
        
        if cnt >= max_iter:
            polynomials_out_of_iterations+=1
        else:
            iterations_total+=cnt
            correct_roots+=1
    print(f"correct: {correct_roots}\navg_iterations: {iterations_total/correct_roots}\nincorrect_roots: {incorrect_roots}\nout_of_max_iterations: {polynomials_out_of_iterations}")

