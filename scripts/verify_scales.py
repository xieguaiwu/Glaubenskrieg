"""Verify forward + backward pass for all model scales."""
import time
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.model.ctm_model import CTMStockModel
from src.model.multiasset_ctm import MultiAssetCTM
from src.model.loop_ctm import RecurrentCTM


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_params(cfg):
    mc = cfg["model"]
    params = {
        "input_dim": mc.get("input_dim", 9),
        "model_dim": mc.get("model_dim", 64),
        "state_dim": mc.get("state_dim", 16),
        "conv_kernel": mc.get("conv_kernel", 3),
        "n_layers": mc.get("n_layers", 3),
        "output_dim": mc.get("output_dim", 1),
        "dropout": mc.get("dropout", 0.1),
        "use_decomp": mc.get("use_decomp", False),
        "bidirectional": mc.get("bidirectional", False),
        "parallel_scan": mc.get("parallel_scan", False),
        "return_hidden": mc.get("return_hidden", False),
    }
    return params


def verify_scale(name, cfg):
    print(f"\n{'='*60}")
    print(f"Scale: {name}")
    print(f"{'='*60}")

    scale = cfg.get("scale", name)
    mc = cfg["model"]
    sc = cfg.get("scaling", {})
    use_cross_attn = sc.get("use_cross_attention", True)
    embedding_dim = sc.get("embedding_dim", None)
    seq_len = mc.get("seq_len", 63)
    input_dim = mc.get("input_dim", 9)

    n_assets = sc.get("n_assets", 100)
    if scale == "portfolio":
        model = MultiAssetCTM(
            n_assets=n_assets,
            input_dim=input_dim,
            model_dim=mc.get("model_dim", 64),
            state_dim=mc.get("state_dim", 16),
            n_layers=mc.get("n_layers", 3),
            output_dim=mc.get("output_dim", 1),
            embedding_dim=embedding_dim if embedding_dim is not None else None,
            use_cross_attention=use_cross_attn,
            dropout=mc.get("dropout", 0.1),
        )
        B = 2
        x = torch.randn(B, n_assets, seq_len, input_dim)
    elif scale == "loop":
        model_params = build_model_params(cfg)
        model_params["n_loop_iters"] = mc.get("n_loop_iters", 3)
        model_params["loop_dropout"] = mc.get("loop_dropout", 0.1)
        model = RecurrentCTM(**model_params)
        B = 4
        x = torch.randn(B, seq_len, input_dim)
    else:
        model = CTMStockModel(**build_model_params(cfg))
        B = 4
        x = torch.randn(B, seq_len, input_dim)

    param_count = count_params(model)
    print(f"Parameters: {param_count:,}")
    if scale == "loop":
        print(f"  n_loop_iters={mc.get('n_loop_iters', 3)}")

    t0 = time.time()
    y = model(x)
    t_forward = time.time() - t0
    print(f"Forward pass: {y.shape} in {t_forward*1000:.1f}ms")
    assert torch.isfinite(y).all(), "Forward output has NaN/Inf"

    if scale == "portfolio":
        B, T, C = y.shape
        expected_c = n_assets * (mc.get("output_dim", 1) + 3)
        assert C == expected_c, f"Expected {expected_c} channels, got {C}"
    else:
        B, T, C = y.shape
        assert C == mc.get("output_dim", 1) + 3, f"Expected {mc.get('output_dim', 1) + 3} channels, got {C}"

    loss = y.pow(2).mean()
    t1 = time.time()
    loss.backward()
    t_backward = time.time() - t1
    print(f"Backward pass: {t_backward*1000:.1f}ms")

    nan_grads = []
    zero_grads = []
    for name_p, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                nan_grads.append(name_p)
            if p.grad.abs().max() < 1e-30:
                zero_grads.append(name_p)

    if nan_grads:
        print(f"WARNING: NaN gradients in: {nan_grads}")
    else:
        print("Gradients: all finite ✓")

    if zero_grads:
        print(f"WARNING: Zero gradients in: {zero_grads}")
    else:
        print("Gradients: all non-zero ✓")

    print(f"Total time: {(time.time()-t0)*1000:.0f}ms")
    print(f"Result: {'PASS' if not nan_grads else 'FAIL'}")

    return {
        "scale": name,
        "param_count": param_count,
        "output_shape": list(y.shape),
        "forward_ms": round(t_forward * 1000, 1),
        "backward_ms": round(t_backward * 1000, 1),
        "total_ms": round((time.time() - t0) * 1000, 0),
        "nan_grads": [n for n in nan_grads],
        "zero_grads": [n for n in zero_grads],
        "passed": not nan_grads,
    }


def main():
    warnings.filterwarnings("ignore")
    configs = {
        "small": "configs/default.yaml",
        "large": "configs/scale_large.yaml",
        "portfolio": "configs/scale_portfolio.yaml",
        "loop": "configs/scale_loop.yaml",
    }

    all_results = {}
    all_passed = True
    for name, path in configs.items():
        try:
            cfg = load_config(path)
            result = verify_scale(name, cfg)
            all_results[name] = result
            if not result["passed"]:
                all_passed = False
        except Exception as e:
            print(f"\nERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"scale": name, "error": str(e), "passed": False}
            all_passed = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        status = "PASS" if r.get("passed") else "FAIL"
        if "error" in r:
            print(f"  {name}: {status} — ERROR: {r['error']}")
        else:
            print(f"  {name}: {status} — {r['param_count']:,} params, "
                  f"fwd={r['forward_ms']}ms bwd={r['backward_ms']}ms "
                  f"out={r['output_shape']}")

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
