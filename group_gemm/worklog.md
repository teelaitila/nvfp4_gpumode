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




---

## 2026-01-31: (submissionv2) moved to nvidias group gemm implementation

### Changes
- Added `apache-tvm-ffi` install helper (mirrors `backup.py`).
- Enabled `--enable-tvm-ffi` in `cute.compile` options.

### Result (B200 benchmark)
`nvidia_group_ex.py`:
- Case 1: 112 ± 0.4 µs
- Case 2: 104 ± 0.3 µs
- Case 3: 71.0 ± 1.25 µs
- Case 4: 65.8 ± 0.24 µs

`submissionv2.py` (with TVM-FFI enabled):
- Case 1: 93.7 ± 0.09 µs
- Case 2: 84.9 ± 0.16 µs
- Case 3: 51.7 ± 0.18 µs
- Case 4: 49.0 ± 0.16 µs

### Notes
TVM-FFI reduces host-side launch/argument overhead. The kernel math is unchanged;
improvements come from a lower-overhead ABI and lighter launch path, which matters
more for many small grouped GEMMs.

---

## 2026-02-04: (dynamic.py) CLC Dynamic Persistent Scheduler

### Changes
- Replaced static persistent tile scheduler with CLC (Cluster Launch Control) dynamic persistent scheduler
- Added `PipelineClcFetchAsync` for tile scheduling coordination
- Added scheduler warp (`sched_warp_id = 6`) to handle CLC tile fetching

### What CLC Dynamic Scheduler Does in group gemm
- Uses hardware CLC instructions (`clusterlaunchcontrol.try_cancel`) to dynamically fetch next tile
- Provides better load balancing than static scheduler when tile counts vary across groups
- Adapts to available SMs rather than statically selected number

### Result (B200 benchmark)
`dynamic.py` (CLC dynamic persistent):
- Case 1: 70.8 ± 0.11 µs (⚡ 69.0 µs)
- Case 2: 65.5 ± 0.41 µs (⚡ 62.0 µs)
- Case 3: 44.1 ± 0.19 µs (⚡ 41.3 µs)
- Case 4: 41.5 ± 0.19 µs (⚡ 38.1 µs)

### Comparison vs Previous Best (submissionv2.py)
| Case | submissionv2 (static) | dynamic (CLC) | Improvement |
|------|----------------------|---------------|-------------|
| 1 | 93.7 µs | 70.8 µs | **24% faster** |
| 2 | 84.9 µs | 65.5 µs | **23% faster** |
| 3 | 51.7 µs | 44.1 µs | **15% faster** |
| 4 | 49.0 µs | 41.5 µs | **15% faster** |

### Notes
CLC dynamic scheduling improves performance especially for grouped GEMMs with varying tile counts.
The dynamic load balancing avoids idle SMs waiting for slower groups to finish.



cache evict first improves perf 


## Benchmarks:
```
g: 8; k: [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168]; m: [80, 176, 128, 72, 64, 248, 96, 160]; n: [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]; seed: 1111
 ⏱ 40.4 ± 0.39 µs
 ⚡ 39.7 µs 🐌 79.1 µs

g: 8; k: [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]; m: [40, 76, 168, 72, 164, 148, 196, 160]; n: [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168]; seed: 1111
 ⏱ 28.3 ± 0.56 µs
 ⚡ 27.4 µs 🐌 84.1 µs

g: 2; k: [4096, 4096]; m: [192, 320]; n: [3072, 3072]; seed: 1111
 ⏱ 14.2 ± 0.44 µs
 ⚡ 13.3 µs 🐌 58.1 µs

g: 2; k: [1536, 1536]; m: [128, 384]; n: [4096, 4096]; seed: 1111
 ⏱ 11.0 ± 0.45 µs
 ⚡ 10.6 µs 🐌 55.4 µs
```