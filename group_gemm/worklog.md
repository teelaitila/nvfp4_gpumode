# Group GEMM Kernel Optimization Worklog

NVFP4 block-scaled group GEMM on B200 (Blackwell SM100).

---

## Optimizations

### 1. Dynamic Pipeline Stages
Compute optimal A/B pipeline stages at runtime based on SMEM capacity (228KB). Maximizes TMA/MMA overlap.

### 2. TMA Store Epilogue
Replaced SIMT stores with TMA bulk transfers: `TMem → Registers → SMEM → GMEM (TMA)`. Uses tensormap updates for runtime tensor shapes.

### 3. Class-Based Architecture
Refactored to `GroupGemmKernel` class with compile-time N/K constants (only M varies in benchmarks). Cleaner TMA partitioning, better codegen.

---

## Benchmark Progress

**Baseline** (before optimizations):
| Case | Config | Time |
|------|--------|------|
| 1 | g=8, K=7168, N=4096 | 326 µs |
| 2 | g=8, K=2048, N=7168 | 388 µs |
| 3 | g=2, K=4096, N=3072 | 168 µs |
| 4 | g=2, K=1536, N=4096 | 154 µs |

**After dynamic stages + TMA store + class refactor:**
| Case | Config | Time | Δ |
|------|--------|------|---|
| 1 | g=8, K=7168, N=4096 | 256 µs | -21% |
| 2 | g=8, K=2048, N=7168 | 260 µs | -33% |
| 3 | g=2, K=4096, N=3072 | 130 µs | -23% |
| 4 | g=2, K=1536, N=4096 | 125 µs | -19% |
