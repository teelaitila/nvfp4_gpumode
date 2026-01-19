# TMA for Dummies (CuteDSL / CUTLASS Python DSL)

This is a condensed, practical guide to NVIDIA TMA adapted to CuteDSL
(the Python DSL for CuTe/CUTLASS). It focuses on what you actually write:
build a TMA atom + descriptor in Python, then issue one TMA op per CTA
inside the kernel, with the right sync.

---

## The 20-second mental model

- **TMA = bulk async copy** between GMEM and SMEM, started by **one thread**
  in the CTA, completed by hardware in the async proxy.
- **You build a descriptor once** (the "tensormap"). The kernel consumes it.
- **Load uses a barrier** (mbarrier). **Store uses a fence** (async proxy fence).
- **TMA handles OOB predication** if you use the TMA tensor coordinates.
- **Strides must be TMA-friendly**: one contiguous dimension; all other
  strides are multiples of 16 bytes.

---

## One-page cheat sheet (CuteDSL names)

TMA ops (tile mode):
- GMEM -> SMEM: `cpasync.CopyBulkTensorTileG2SOp`
- GMEM -> SMEM multicast: `cpasync.CopyBulkTensorTileG2SMulticastOp`
- SMEM -> GMEM: `cpasync.CopyBulkTensorTileS2GOp`
- SMEM -> GMEM reduce: `cpasync.CopyReduceBulkTensorTileS2GOp`

Core helpers:
- `cpasync.make_tiled_tma_atom(...)` -> `(atom, tma_tensor)`
- `cpasync.tma_partition(...)` -> `(smem_tile, gmem_tile)`
- `cpasync.create_tma_multicast_mask(...)` -> `Int16 mask`

Descriptor management:
- `cpasync.prefetch_descriptor(atom)`
- `cpasync.copy_tensormap(atom, ptr)`
- `cpasync.update_tma_descriptor(atom, gmem_tensor, desc_ptr)`
- `cpasync.fence_tma_desc_acquire(desc_ptr)`
- `cpasync.cp_fence_tma_desc_release(global_ptr, shared_ptr)`

---

## TMA load, CuteDSL-style (GMEM -> SMEM)

### Step 1: Build the TMA atom and tensor (Python/host side)

```python
from cutlass.cute.nvgpu import cpasync

# GMEM tensor and SMEM layout are normal Cute objects
op = cpasync.CopyBulkTensorTileG2SOp()
cta_tiler = cute.make_tile(CTA_M, CTA_N)   # or a Layout-derived tiler

tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    op,
    gmem_tensor,
    smem_layout,
    cta_tiler,
)
```

### Step 2: Issue the load in-kernel

```python
# Make the CTA-specific tiles
smem_tile, gmem_tile = cpasync.tma_partition(
    tma_atom,
    cta_coord,      # blockIdx-based coord
    cta_layout,     # CTA layout / tiler
    smem_tensor,
    tma_tensor,     # IMPORTANT: use TMA tensor, not raw gmem_tensor
)

if threadIdx.x == 0:
    # 1) init barrier + expect bytes
    # 2) issue async bulk tensor copy using the TMA atom
    # (barrier helpers are in cutlass.pipeline or arch wrappers)
    cute.copy(tma_atom, gmem_tile, smem_tile, mbarrier)

__syncthreads()
wait_mbarrier(mbarrier, phase=0)
```

Key points:
- Only **one thread** issues the TMA op.
- All threads **wait** on the mbarrier before using SMEM.
- Always partition using the **TMA tensor**, not the raw GMEM tensor.

---

## TMA store (SMEM -> GMEM)

TMA store looks the same structurally, but the sync is different:

```python
op = cpasync.CopyBulkTensorTileS2GOp()
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tiler)

smem_tile, gmem_tile = cpasync.tma_partition(
    tma_atom, cta_coord, cta_layout, smem_tensor, tma_tensor
)

__syncthreads()
fence_proxy_async_shared_cta()  # ensure SMEM writes are visible to async proxy

if threadIdx.x == 0:
    cute.copy(tma_atom, smem_tile, gmem_tile)
```

Notes:
- **Fence before store**. No mbarrier is required for correctness on store.
- If you need to **reuse SMEM after the store**, use a store-arrive/wait
  sequence (PipelineTmaStore in Cutlass pipeline APIs).

---

## TMA store reduce (SMEM -> GMEM with reduction)

If you want `dst = dst + src` (or max/min), use a reduce op:

```python
op = cpasync.CopyReduceBulkTensorTileS2GOp(
    reduction_kind=cute.ReductionOp.ADD  # or MAX / MIN
)
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tiler)
```

Everything else is identical to TMA store, but it emits
`cp.reduce.async.bulk` instead of `cp.async.bulk`.

---

## TMA load multicast (GMEM -> SMEM for multiple CTAs)

Multicast = one GMEM tile goes to multiple CTAs in the **same cluster**.

```python
op = cpasync.CopyBulkTensorTileG2SMulticastOp()
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    op, gmem_tensor, smem_layout, cta_tiler, num_multicast=cluster_size
)

mcast_mask = cpasync.create_tma_multicast_mask(
    cta_layout_vmnk, cta_coord_vmnk, mcast_mode
)

if threadIdx.x == 0:
    cute.copy(tma_atom, gmem_tile, smem_tile, mbarrier, mcast_mask)
```

Rules of thumb:
- **Cluster shape must divide grid dims**.
- **Each CTA uses its own slice** (ctaid influences the partition).
- Mask selects which CTAs in the cluster participate.

---

## Descriptor lifecycle (when shapes or base pointers change)

If the GMEM tensor base/shape/stride changes between launches:

```python
cpasync.update_tma_descriptor(tma_atom, gmem_tensor, tma_desc_ptr)
cpasync.fence_tma_desc_release()
```

If the descriptor lives in shared memory, use:

```python
cpasync.cp_fence_tma_desc_release(global_desc_ptr, shared_desc_ptr)
cpasync.fence_tma_desc_acquire(shared_desc_ptr)
```

---

## Gotchas checklist

- **16-byte stride rule**: non-contiguous strides must be multiples of 16B.
- **Use TMA tensor coords** (`tma_tensor`) so OOB is predicated safely.
- **Single issuing thread** per CTA; sync the rest.
- **Load = barrier after**, **Store = fence before**.
- **Multicast requires cluster** + mask + per-CTA slice.
- If you reuse SMEM right after store, use a **store wait** or pipeline API.

---

## Minimal decision guide

- Want GMEM -> SMEM: `CopyBulkTensorTileG2SOp`
- Want GMEM -> SMEM + cluster: `CopyBulkTensorTileG2SMulticastOp`
- Want SMEM -> GMEM: `CopyBulkTensorTileS2GOp`
- Want SMEM -> GMEM + reduce: `CopyReduceBulkTensorTileS2GOp`

---

If you want, I can also add a tiny runnable CuteDSL kernel in this repo
that mirrors the load/store examples from the original article.
