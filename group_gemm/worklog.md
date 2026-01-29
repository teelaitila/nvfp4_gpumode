# Group GEMM Kernel Optimization Worklog

NVFP4 block-scaled group GEMM on B200 (Blackwell SM100).

---

## Optimizations

### 1. Dynamic Pipeline Stages
Compute optimal A/B pipeline stages at runtime based on SMEM capacity (228KB). Maximizes TMA/MMA overlap.

### 2. TMA Store Epilogue
Replaced SIMT stores with TMA bulk transfers: `TMem → Registers → SMEM → GMEM (TMA)`.

### 3. Class-Based Architecture + Config Map
Refactored to `GroupGemm` class with CONFIG_MAP exposing all hyperparameters per benchmark case. Includes m_values, tile sizes, cluster shape, cache policy, pipeline stages.

### 4. TMA Cache Eviction
Added TMA_CACHE_EVICT_FIRST/LAST/NORMAL policies to TMA loads.

### 5. Actual N,K in TMA Atoms
Changed from max shapes to actual N,K values (constant across groups). Reduces tensormap update work.

### 6. Per-GROUP Tensormap Storage
Changed from per-CTA to per-GROUP tensormap allocation. Reduces memory and allows potential sharing.

### 7. Constexpr Group Shapes
Moved CTA shape list construction into the JIT path using `cutlass.range_constexpr` and passed `problem_sizes`/`num_groups` as `Constexpr`.

### 8. ACC + C Pipeline Staging
Added accumulator staging (`num_acc_stage`) and epilogue C staging (`num_c_stage`) to better overlap compute and TMA stores. Initially set to 2 stages each.

---

## Gap Analysis: Speed of Light

| Case | SOL | Current | Gap |
|------|-----|---------|-----|
| 1 | 18.8 µs | 251 µs | **13x** |
| 2 | 10.7 µs | 255 µs | **24x** |
| 3 | 2.4 µs | 127 µs | **53x** |
| 4 | 1.5 µs | 123 µs | **82x** |

### Root Cause: Per-CTA Tensormap Setup
Each CTA does: 5x init_tensormap_from_atom + 1x update_tensormap + 1x fence. This is ~15 expensive operations before any compute starts.

For small problems (cases 3,4), setup time dominates actual compute. Dualgemmex/singlegemmex avoid this by having shapes known at JIT time.

### Potential Solutions (Not Yet Implemented)
1. **Create TMA atoms per group** - Requires passing multiple atoms to kernel
2. **Persistent kernel** - One CTA handles multiple tiles, amortizing setup
3. **Pre-initialize tensormaps on host** - Requires host-side tensormap API

---

## Benchmark Progress

benchmarks:
  1 {"m": [80, 176, 128, 72, 64, 248, 96, 160], "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "g": 8, "seed": 1111}
  2 {"m": [40, 76, 168, 72, 164, 148, 196, 160], "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048], "g": 8, "seed": 1111}
  3 {"m": [192, 320], "n": [3072, 3072], "k": [4096, 4096], "g": 2, "seed": 1111}
  4 {"m": [128, 384], "n": [4096, 4096], "k": [1536, 1536], "g": 2, "seed": 1111}

**Baseline:**
| Case | Time |
|------|------|
| 1 | 326 µs |
| 2 | 388 µs |
| 3 | 168 µs |
| 4 | 154 µs |

**Current (with all optimizations):**
| Case | Time | Δ |
|------|------|---|
| 1 | 126 µs | -61% |
| 2 | 129 µs | -67% |
| 3 | 90.5 µs | -46% |
| 4 | 91.0 µs | -41% |

**Fastest (B200 modal, 2026-01-28):**
| Case | Time | Δ vs Baseline | Speedup vs Baseline |
|------|------|---------------|---------------------|
| 1 | 126 µs | -61% | 2.59x |
| 2 | 129 µs | -67% | 3.01x |
| 3 | 90.5 µs | -46% | 1.86x |
| 4 | 91.0 µs | -41% | 1.69x |

**Current (B200, 2026-01-28, acc=1, c=1):**
| Case | Time | Δ vs Baseline | Speedup vs Baseline |
|------|------|---------------|---------------------|
| 1 | 137 µs | -58% | 2.38x |
| 2 | 138 µs | -64% | 2.81x |
| 3 | 87.9 µs | -48% | 1.91x |
| 4 | 84.2 µs | -45% | 1.83x |

TODO: Tinker with stages per benchmark case (AB/C/ACC) and re-measure.
