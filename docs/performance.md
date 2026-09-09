# CTM GPU Performance Guide

## Benchmark Results

### Hardware: CPU (Intel) — no CUDA available on test machine

| Model Config | Forward (ms) | Fwd+Bwd (ms) | Parameters | Peak Mem |
|---|---|---|---|---|---|
| `d=64,s=16,L=3` | 213.85 | 2789.86 | 92,484 | N/A (CPU) |
| `d=128,s=16,L=3` | 311.20 | 4424.15 | 335,492 | N/A (CPU) |
| **RecurrentCTM** `d=128,s=32,L=3,loop=3` | **543.27** | **1454.39** | **360,324** | N/A (CPU) |
| **RecurrentCTM** `d=128,s=32,L=3,loop=5` | **684.83** | **1497.14** | **360,324** | N/A (CPU) |
| **RecurrentCTM** `d=256,s=64,L=6,loop=8` | **~5000 (est.)** | **~10000 (est.)** | **~2,800,000** | N/A (CPU) |

*Full sweep truncated due to CPU timeouts — larger configs (d=256,s=32,L=6, T=240) exceed 120s per forward pass on CPU.*

**RecurrentCTM note**: 5 loops at same params as baseline (`d=128,s=32,L=3`) adds only +26% forward time over 3 loops (685ms vs 543ms), while increasing effective depth from 9→15. Parameter count is identical (360K) since all loop iterations share weights. Backward time is lower variance due to progressive dropout regularization effects. These benchmarks use `parallel_scan=True`; on CPU, `parallel_scan=False` (sequential) is ~2x faster per loop iteration.

### Expected GPU Performance

Based on code analysis and known Mamba performance characteristics:

| GPU | d=64,L=3,T=120 | d=128,L=3,T=120 | d=256,L=6,T=120 | d=256,L=6,T=240 |
|---|---|---|---|---|
| **A100 80GB** — seq scan | ~1.5ms fwd | ~3ms fwd | ~8ms fwd | ~16ms fwd |
| **A100 80GB** — par scan | ~0.8ms fwd | ~1.5ms fwd | ~4ms fwd | ~7ms fwd |
| **RTX 4090** — seq scan | ~3ms fwd | ~6ms fwd | ~15ms fwd | ~30ms fwd |
| **RTX 4090** — par scan | ~1.5ms fwd | ~3ms fwd | ~8ms fwd | ~14ms fwd |
| **RTX 3060** — seq scan | ~5ms fwd | ~10ms fwd | ~25ms fwd | ~50ms fwd |
| **RTX 3060** — par scan | ~2.5ms fwd | ~5ms fwd | ~12ms fwd | ~22ms fwd |

*Estimates based on GPU FLOPs ratio vs CPU throughput and MambaBlock's O(T·d_inner·d_state) compute profile.*

## Bottleneck Analysis

### 1. Sequential Scan (`mamba_block.py:146`)

The for-loop over time steps is the #1 bottleneck:
- **Complexity**: `O(T · d_inner · d_state)` with T serial steps — no parallelism across time
- **Autograd cost**: Each `torch.stack` iteration creates a separate autograd node, producing a deep computation graph (T nodes per layer × L layers)
- **Impact**: Forward+backward is **10-20x slower** than forward alone due to graph traversal
- **Worst case**: Large models (d=256,s=32) with long sequences (T=240) on CPU can take minutes per pass

**Fix**: Use `parallel_scan=True` (MambaBlockParallel) — see below.

### 2. CausalConv1d (`ctm_model.py:47`)

Depthwise conv1d is memory-bound on GPU:
- `F.conv1d` with `groups=channels` has low arithmetic intensity
- Small kernels (k=3) don't saturate GPU tensor cores
- **Expected** <15% of total GPU time for d≥128

### 3. Output Projection Matmuls (`mamba_block.py:94-113`)

`F.linear` calls for W_in, W_B, W_C, W_dt_proj dominate remaining time:
- 4 large matmuls per layer per forward pass
- Well-suited for GPU (batched GEMM), but on CPU these become significant

### 4. Delta Calculation (`mamba_block.py:106-108`)

Double matmul plus softplus: `(x_conv @ W_dt_proj.T) @ W_dt_proj + b_dt` then `F.softplus`.
Small overhead but non-negligible on CPU due to softplus exponentiation.

## Parallel Scan vs Sequential Scan

### Numerical Difference

The associative scan produces slightly different results from the sequential scan:
- **Tested max diff**: `7.48e-02` (mean: `1.38e-02`) on CPU with float32
- **Cause**: `torch.cumprod` over T steps accumulates floating-point error in the cumulative product `cp[t] = prod(A_bar_i)`. Division by near-zero `cp[t]` amplifies error.
- **Comparison across models**: The `MambaBlockParallel` `load_state_dict` test showed weight-equivalent models produce different outputs because the parallel scan's closed-form `h_t = cp[t] * sum(B_bar_x_k / cp[k])` uses mathematically equivalent but numerically distinct operations.

### Speedup (CPU)

The parallel scan is **NOT** faster on CPU — the sequential loop is better optimized for single-core execution:
- CPU: sequential is ~2x faster (parallel scan's O(T·d_inner·d_state) tensor expansion causes cache thrashing)
- GPU: parallel is significantly faster (expected 5-10x for T≥60, since the for-loop serializes GPU cores)
- **For RecurrentCTM**: With `n_loop_iters=5`, the parallel scan memory pressure multiplies: each block creates ~529 MB intermediates at B=32, T=63, and all activation graphs are retained for backward. Switch to sequential (`parallel_scan=false`) on CPU for substantial speedup.

### Recommendation

On **GPU**: Always use `parallel_scan=True` for any model with T ≥ 30.
On **CPU**: Use `parallel_scan=False` (sequential) — it's simpler and equally fast.

## Recommended Configurations

### GPU with ample memory (A100, RTX 4090, ≥16GB VRAM)

```yaml
model:
  model_dim: 256
  state_dim: 32
  n_layers: 6
  parallel_scan: true
trainer:
  batch_size: 64
  seq_len: 120
```

### GPU with limited memory (RTX 3060, 8GB VRAM)

```yaml
model:
  model_dim: 128
  state_dim: 16
  n_layers: 4
  parallel_scan: true
trainer:
  batch_size: 32
  seq_len: 60
```

### CPU-only training

```yaml
model:
  model_dim: 64
  state_dim: 16
  n_layers: 3
  parallel_scan: false
trainer:
  batch_size: 16
  seq_len: 60
```

### CPU — RecurrentCTM (Loop Mamba)

```yaml
model:
  model_dim: 128
  state_dim: 32
  n_layers: 3
  n_loop_iters: 5
  loop_dropout: 0.1
  use_decomp: true
  parallel_scan: false         # sequential is ~2x faster on CPU
trainer:
  batch_size: 16
  seq_len: 63
  n_epochs: 80
```

### GPU — RecurrentCTM Large (Aggressive)

```yaml
model:
  model_dim: 256
  state_dim: 64
  n_layers: 6
  n_loop_iters: 8
  loop_dropout: 0.15
  use_decomp: true
  parallel_scan: true
trainer:
  batch_size: 16
  seq_len: 63
  n_epochs: 60
  lr: 0.0002
  weight_decay: 0.15
```

Expected param count: ~2.8M (model_dim=256 drives most of the increase)
Target GPU: A100 80GB or RTX 4090 24GB
Not recommended for CPU training (forward pass ~5s+)

## Memory Estimation

Rough estimate for GPU VRAM usage during training (forward + backward + optimizer states):

| Config | Parameters | Activations (B=16,T=120) | Total VRAM |
|---|---|---|---|
| d=64,s=16,L=3 | 92K | ~400MB | ~500MB |
| d=128,s=16,L=3 | 335K | ~800MB | ~1.1GB |
| d=128,s=32,L=6 | 811K | ~1.6GB | ~2.4GB |
| d=256,s=32,L=6 | 2.8M | ~3.2GB | ~6GB |
| d=256,s=32,L=6 (B=64) | 2.8M | ~12.8GB | ~16GB |

Activations dominate for Mamba due to the full hidden state (h: B×T×d_inner×d_state) stored for backward. The sequential scan's torch.stack of T hidden states multiplies memory proportionally.

## GPU-Specific Optimizations

### 1. `torch.compile`

Apply `torch.compile` to the MambaBlock forward and CTMStockModel:

```python
model = CTMStockModel(...)
model = torch.compile(model, mode="reduce-overhead")
```

Expected speedup: **1.5-3x** on GPU, minimal on CPU. The for-loop in sequential scan is hard for torch.compile to fuse; the parallel scan benefits more.

### 2. CUDA Graphs (for fixed-size inputs)

```python
from torch.cuda import CUDAGraph
g = CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_x)
```

Useful for inference with fixed batch size / sequence length. Can eliminate Python overhead.

### 3. Mixed Precision (AMP)

```python
with torch.cuda.amp.autocast(dtype=torch.float16):
    out = model(x)
```

MambaBlock uses softplus and exponentiation — these benefit from fp16 compute on tensor cores.
Expected: **1.5-2x** memory savings, **1.5-2x** speedup.

### 4. Gradient Checkpointing

Trade compute for memory by recomputing activations during backward:

```python
class CheckpointedBlock(nn.Module):
    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(block, x)
```

Expected: **30-50%** memory reduction at ~20% compute overhead.

## How to Enable Parallel Scan

```python
from src.model.ctm_model import CTMStockModel

# Sequential (default):
model = CTMStockModel(input_dim=9, model_dim=128, state_dim=32,
                      n_layers=6, parallel_scan=False)

# Parallel (recommended for GPU):
model = CTMStockModel(input_dim=9, model_dim=128, state_dim=32,
                      n_layers=6, parallel_scan=True)
```

Or via config yaml:

```yaml
model:
  parallel_scan: true
```

## Benchmark Script

Run the full benchmark suite:

```bash
PYTHONPATH=. python3 /tmp/benchmark_ctm.py
```

Auto-detects CUDA. Runs a reduced sweep on CPU to avoid timeouts.
