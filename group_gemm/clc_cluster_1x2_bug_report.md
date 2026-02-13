# CLC Dynamic Scheduler Bug Report (`cluster_shape_mn=(1,2)`)

## Summary
When using the CLC dynamic scheduler in `submissionv3.py` with `CLUSTER_SHAPE_MN = (1,2)`, the kernel fails with `cudaErrorIllegalAddress` or produces large correctness mismatches.  
The same grouped GEMM workloads run correctly in `submissionv2.py` and `s.py`, which do not use this CLC path.

## Repro Pattern
- `cluster=(1,1)`: generally works.
- `cluster=(1,2)`: fails after enabling CLC scheduler path.
- Failures are strongly correlated with groups containing very small `M` (notably `m=40`), and with benchmark groups that require many cluster tiles.
- Typical runtime symptom:
  - `RuntimeError: CUDA Error: cudaErrorIllegalAddress`
  - or massive output mismatch vs reference.

## Observations
- The error can surface on later CUDA API calls unless `torch.cuda.synchronize()` is used after kernel launch.
- Swapping certain TMA partition cluster coordinate indices can remove crash but yields wrong results, indicating coordinate-space/mapping sensitivity.
- `PipelineClcFetchAsync` docs indicate:
  - producer runs in CTA 0,
  - consumers run across all CTAs in cluster,
  - consumers read response from local SMEM.

## Changes Attempted
1. **Tensormap workspace sizing fixes**
   - Expanded tensormap workspace from `SM_COUNT` to a larger fixed pool (`TENSORMAP_WORKSPACE_SLOTS`).
   - Added guard checks for required CTA slots.
   - Result: reduced one class of potential OOB risk, but did not resolve `(1,2)` illegal address.

2. **CLC response buffer sizing/alignment fixes**
   - Increased response storage to full 16-byte payload size.
   - Adjusted storage type to supported 16-byte equivalent (`Int64[2]`).
   - Result: necessary hygiene, but not sufficient to fix `(1,2)`.

3. **Persistent worker-count launch experiment (`128` workers)**
   - Implemented fixed worker pool launch to improve tail effects.
   - Result: introduced/triggered correctness issues on larger groups; reverted to standard CLC grid shape launch.

4. **Host/compile cleanup**
   - Improved cache key specificity.
   - Converted some values to `Constexpr` where appropriate.
   - Removed/adjusted stale signature args and runtime call mismatch.
   - Added serializable error wrapping for multiprocessing harness.
   - Result: improved debuggability and stability of benchmarking harness errors.

## Most Likely Root Cause (Current)
The `(1,2)` CLC integration appears to have a **cluster-wide response visibility/ownership bug**:
- CLC producer is in CTA 0,
- all CTAs consume,
- but response handling likely is not correctly shared/synchronized for non-CTA0 consumers in this kernel integration.

This aligns with:
- illegal addresses in dynamic scheduling flow,
- and shape-dependent sensitivity (small `M` surfaces bad tile coordinates earlier).

## Why `submissionv2.py` and `s.py` Work
They avoid this CLC dynamic path and therefore avoid the cluster-wide response-sharing hazard introduced in `submissionv3.py`.

## Current Practical Decision
Use `CLUSTER_SHAPE_MN = (1,1)` for CLC path (known-good operating mode), and avoid `(1,2)` with current implementation.

## Remaining Fix Options
1. **Proper cluster-shared CLC response path (recommended long-term)**
   - Ensure the 16-byte response produced by CTA 0 is made visible to peer CTA(s) each stage.
   - Likely requires explicit CTA0->cluster shared transfer or DSMEM path plus cluster synchronization.

2. **Restrict CLC usage by cluster shape**
   - Enable CLC only for `(1,1)`.
   - Use non-CLC/static scheduler when `cluster_size > 1`.

3. **Fallback scheduling for problematic groups**
   - Route groups with very small `M` (e.g., `<64`) through a safer scheduler path.

4. **Add internal debug assertions/checks**
   - Validate decoded tile coordinates before use.
   - Early-detect out-of-range tile/group indices to avoid silent corruption.

## Notes for Future Debug Session
To finish the proper `(1,2)` fix, gather a minimal CUTLASS DSL example showing:
- how to broadcast or map a CTA-local SMEM pointer/value to peer CTA(s) in cluster, or
- a working DSMEM/cluster-copy pattern for a 16-byte payload with barrier synchronization.

