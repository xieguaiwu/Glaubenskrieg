"""Quick training pipeline test with synthetic data (large scale)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.model.ctm_model import CTMStockModel
from src.model.losses import LossConfig
from src.train.advanced_trainer import (
    LossWrapper,
    validate_advanced,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg_path = "configs/scale_large.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    seq_len = cfg["model"]["seq_len"]
    input_dim = cfg["model"]["input_dim"]
    batch_size = 16
    train_size = 200
    val_size = 50

    print(f"Generating synthetic data: {train_size + val_size} samples")
    data = torch.randn(train_size + val_size, seq_len, input_dim)
    targets = torch.randn(train_size + val_size, seq_len, 1)

    train_data = data[:train_size]
    train_targ = targets[:train_size]
    val_data = data[train_size:]
    val_targ = targets[train_size:]

    model_cfg = cfg["model"]
    model_params = {
        "input_dim": input_dim,
        "model_dim": model_cfg.get("model_dim", 64),
        "state_dim": model_cfg.get("state_dim", 16),
        "conv_kernel": model_cfg.get("conv_kernel", 3),
        "n_layers": model_cfg.get("n_layers", 3),
        "output_dim": model_cfg.get("output_dim", 1),
        "dropout": model_cfg.get("dropout", 0.1),
        "use_decomp": model_cfg.get("use_decomp", False),
        "bidirectional": model_cfg.get("bidirectional", False),
        "parallel_scan": model_cfg.get("parallel_scan", False),
        "return_hidden": model_cfg.get("return_hidden", False),
    }

    loss_config = LossConfig(
        lambda_mse=1.0,
        lambda_sharpe=0.0,
        lambda_directional=0.0,
        lambda_pinball=0.0,
        lambda_reg=0.01,
    )

    def class_targets_fn(t):
        return torch.sign(t).long() + 1

    print(f"Model params: {model_params}")
    print(f"Model param count: {CTMStockModel(**model_params).param_count():,}")

    model = CTMStockModel(**model_params).to(device)

    train_ds = TensorDataset(train_data, train_targ)
    val_ds = TensorDataset(val_data, val_targ)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    loss_wrapper = LossWrapper(
        config=loss_config,
        model=model,
        class_targets_fn=class_targets_fn,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    n_epochs = 2

    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_wrapper(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        val_m = validate_advanced(model, val_loader, loss_wrapper, device)
        print(f"  Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.6f}  "
              f"val_loss={val_m['avg_loss']:.6f}  sharpe={val_m['sharpe_ratio']:.4f}")

    print("\nTraining test PASSED")
    print(f"Final val sharpe: {val_m['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
