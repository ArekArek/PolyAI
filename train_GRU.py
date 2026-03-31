import numpy as np
import torch
import os
from scipy.optimize import linear_sum_assignment

from model_gru import ModelGRU
import utils
from utils import CONFIG, dt
import logging
import time
import torch.nn.functional as F


def main():
    RUN_DIR = f"{CONFIG['training']['output_model_path']}{dt()}"

    if not os.path.exists(RUN_DIR):
        os.makedirs(RUN_DIR)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{CONFIG['logs_path']}{dt()}-train.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 50)
    logging.info(" Train model ".center(50, "="))
    logging.info(f" {dt(1)} ".center(50, "="))
    logging.info("-" * 50)
    logging.info(f"defaults: {CONFIG} ".center(50, "="))
    logging.info("=" * 50)

    org_model = ModelGRU()
    # compression for performance improvements
    model = torch.compile(org_model, mode="reduce-overhead")
    model.train()

    # data load
    coeffs_np_complex = np.load(
        os.path.join(CONFIG["training"]["input_data_path"], "coefficients.npy")
    )
    coeff_tensor_complex = torch.from_numpy(coeffs_np_complex)
    coeff_tensor = torch.view_as_real(coeff_tensor_complex)

    zeroes_np_complex = np.load(
        os.path.join(CONFIG["training"]["input_data_path"], "zeroes.npy")
    )
    zeroes_tensor_complex = torch.from_numpy(zeroes_np_complex)
    zeroes_tensor = torch.view_as_real(zeroes_tensor_complex)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=10
    )

    best_loss = float("inf")
    early_stop_counter = 0

    dataset = torch.utils.data.TensorDataset(coeff_tensor, zeroes_tensor)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True
    )

    start_time = time.time()

    for epoch in range(CONFIG["training"]["epochs_count"]):
        epoch_loss = 0.0
        model.train()

        for coeff_batch, factual in data_loader:
            optimizer.zero_grad()
            preds = model(coeff_batch)

            matched_zeroes = utils.match_closest(preds, factual)
            loss = F.mse_loss(*matched_zeroes)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * coeff_batch.size(0)

        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)

        model_filename = f"model_{epoch + 1:03d}_{epoch_loss:.6f}.h5"
        torch.save(model.state_dict(), os.path.join(RUN_DIR, model_filename))

        # overwrite best model if loss is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.h5"))
            logging.info(f"Epoch {epoch + 1:03d}: New best_loss: {best_loss:.6f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= CONFIG["training"]["early_stop_epochs"]:
            logging.info("Early stopping!")
            break

        elapsed = time.time() - start_time
        logging.info(
            f"{dt(1)} |:| Epoch {epoch + 1:03d}/{CONFIG['training']['epochs_count']}, Loss: {epoch_loss:.6f}, Time: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
