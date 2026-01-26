import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

import functools
from typing import Tuple, List, Union

import torch
from task import input_t, output_t

# Kernel configuration parameters
# Size of tma descriptor in bytes
bytes_per_tensormap = 128
# Number of tensormaps: a, b, sfa, sfb, c
num_tensormaps = 5
# Tile sizes for M, N, K dimensions
mma_tiler_mnk = (128, 128, 256)  
# Shape of the K dimension for the MMA instruction
mma_inst_shape_k = 64
# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN  
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN  
# FP16 output type
c_dtype = cutlass.Float16  
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16  
# Warp roles for warp-specialized mainloop/epilogue
epilog_warp_id = (0, 1, 2, 3)
mma_warp_id = 4
tma_warp_id = 5
# Number of threads per CUDA thread block
threads_per_cta = 32 * len((mma_warp_id, tma_warp_id, *epilog_warp_id))
# Stage numbers - num_ab_stage computed dynamically based on SMEM
num_acc_stage = 1
# Total number of columns in tmem
num_tmem_alloc_cols = 512
# SMEM capacity for B200 (SM100) - 228KB
smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
# Target occupancy (CTAs per SM)
occupancy = 1


def compute_num_ab_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler: Tuple[int, int, int],
    a_dtype,
    b_dtype,
    sf_dtype,
    c_dtype,
    sf_vec_size: int,
    epi_tile,
    c_layout,
    smem_capacity: int,
    occupancy: int,
) -> int:
    """
    Compute optimal number of A/B pipeline stages based on SMEM capacity.
    
    More stages = better pipelining:
    - Stage 0: MMA consuming
    - Stage 1+: TMA prefetching next K tiles
    
    Returns the maximum number of stages that fit in SMEM.
    """
    # Compute SMEM size for single stage of each buffer
    a_smem_layout_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, a_dtype, 1)
    b_smem_layout_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, b_dtype, 1)
    sfa_smem_layout_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec_size, 1)
    sfb_smem_layout_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler, sf_vec_size, 1)
    c_smem_layout_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
    
    # Bytes per stage for A, B, SFA, SFB
    ab_bytes_per_stage = (
        cute.size_in_bytes(a_dtype, a_smem_layout_one)
        + cute.size_in_bytes(b_dtype, b_smem_layout_one)
        + cute.size_in_bytes(sf_dtype, sfa_smem_layout_one)
        + cute.size_in_bytes(sf_dtype, sfb_smem_layout_one)
    )
    
    # Fixed overhead: barriers, tensormap buffer, tmem holding buffer, etc.
    mbar_helpers_bytes = 2048  # Conservative estimate for barriers and metadata
    
    # C buffer (epilogue) - typically 1-2 stages
    num_c_stage = 1
    c_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_one) * num_c_stage
    
    # Compute max AB stages that fit
    available_for_ab = smem_capacity // occupancy - mbar_helpers_bytes - c_bytes
    num_ab_stage = max(1, available_for_ab // ab_bytes_per_stage)
    
    # Cap at reasonable maximum (diminishing returns beyond ~8 stages)
    num_ab_stage = min(num_ab_stage, 8)
    
    return num_ab_stage


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


def mainloop_s2t_copy_and_partition(
    sSF: cute.Tensor,
    tSF: cute.Tensor,
    sf_dtype: cutlass.Numeric,
    cta_group: tcgen05.CtaGroup,
) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(cta_group),
        sf_dtype,
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t, tCsSF_compact_s2t_
    )
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


def epilog_tmem_copy_and_partition(
    tidx: cutlass.Int32,
    tAcc: cute.Tensor,
    gC_mnl: cute.Tensor,
    epi_tile: cute.Tile,
    c_layout: utils.LayoutEnum,
    c_dtype: cutlass.Numeric,
    acc_dtype: cutlass.Numeric,
    cta_tile_shape_mnk: Tuple[int, int, int],
    use_2cta_instrs: cutlass.Boolean,
) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        cta_tile_shape_mnk,
        c_layout,
        c_dtype,
        acc_dtype,
        epi_tile,
        use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(
        tAcc[((None, None), 0, 0)],
        epi_tile,
    )
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
    gC_mnl_epi = cute.flat_divide(
        gC_mnl[((None, None), 0, 0)], epi_tile
    )
    tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
    tTR_rAcc = cute.make_fragment(
        tTR_gC[(None, None, None, 0, 0)].shape, acc_dtype
    )
    return tiled_copy_t2r, tTR_tAcc, tTR_rAcc


def epilog_smem_copy_and_partition(
    tiled_copy_t2r: cute.TiledCopy,
    tTR_rC: cute.Tensor,
    tidx: cutlass.Int32,
    sC: cute.Tensor,
    c_layout: utils.LayoutEnum,
    c_dtype: cutlass.Numeric,
    acc_dtype: cutlass.Numeric,
) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    copy_atom_r2s = sm100_utils.get_smem_store_op(
        c_layout, c_dtype, acc_dtype, tiled_copy_t2r
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sC = thr_copy_r2s.partition_D(sC)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    return tiled_copy_r2s, tRS_rC, tRS_sC


def epilog_gmem_copy_and_partition(
    atom: Union[cute.CopyAtom, cute.TiledCopy],
    gC_mnl: cute.Tensor,
    epi_tile: cute.Tile,
    sC: cute.Tensor,
) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
    gC_epi = cute.flat_divide(
        gC_mnl[((None, None), 0, 0)], epi_tile
    )
    tma_atom_c = atom
    sC_for_tma_partition = cute.group_modes(sC, 0, 2)
    gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
    bSG_sC, bSG_gC = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        sC_for_tma_partition,
        gC_for_tma_partition,
    )
    return tma_atom_c, bSG_sC, bSG_gC


# The CuTe reference implementation for NVFP4 block-scaled GEMM
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    tensor_of_abc_ptrs: cute.Tensor,
    tensor_of_sfasfb_ptrs: cute.Tensor,
    tensormaps: cute.Tensor,
    tensor_of_problem_sizes: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
    epi_tile: cute.Tile,
    cta_mn_list: List[Tuple[int, int]],
    num_tma_load_bytes: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
):
    """
    GPU device kernel performing the Group GEMM computation.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Delinearize bidz to coord_x, coord_y and group_idx for each CTA
    #
    bidx, bidy, bidz = cute.arch.block_idx()
    group_idx = 0
    find = False
    coord_x = 0
    coord_y = 0
    cta_rest = bidz
    for _, (cta_m, cta_n) in enumerate(cta_mn_list):
        if cta_rest >= (cta_m * cta_n):
            group_idx += 1
            cta_rest -= cta_m * cta_n
        else:
            if not find:
                coord_y = cta_rest // cta_m
                coord_x = cta_rest % cta_m
                cta_rest -= cta_m * cta_n
                find = True

    #
    # Construct C Tensor for each CTA
    #
    mC_mnl_iter = cute.make_ptr(
        c_dtype, tensor_of_abc_ptrs[group_idx, 2], cute.AddressSpace.gmem
    ).align(32)
    m = tensor_of_problem_sizes[group_idx, 0]
    n = tensor_of_problem_sizes[group_idx, 1]
    k = tensor_of_problem_sizes[group_idx, 2]
    l = tensor_of_problem_sizes[group_idx, 3]

    mC_mnl_layout = cute.make_layout(
        (m, n, l),
        stride=(cute.assume(n, 32), 1, cute.assume(m * n, 32),))
    real_mC_mnl = cute.make_tensor(mC_mnl_iter, mC_mnl_layout)
    # Local partition for global C Tensor (TMA tensor)
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        real_mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (coord_x, coord_y, 0)
    )

    #
    # Define shared storage for kernel (tensormap + barriers + tmem handle)
    size_tensormap_in_i64 = (
        num_tensormaps * bytes_per_tensormap // 8
    )
    @cute.struct
    class SharedStorage:
        tensormap_buffer: cute.struct.MemRange[
            cutlass.Int64, size_tensormap_in_i64
        ]
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]  # need both full & empty bar
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        tmem_holding_buf: cutlass.Int32
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
    tensormap_a_smem_ptr = tensormap_smem_ptr
    tensormap_b_smem_ptr = (
        tensormap_a_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfa_smem_ptr = (
        tensormap_b_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfb_smem_ptr = (
        tensormap_sfa_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_c_smem_ptr = (
        tensormap_sfb_smem_ptr
        + bytes_per_tensormap // 8
    )

    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )
    sC = smem.allocate_tensor(
        element_type=c_dtype,
        layout=c_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=c_smem_layout_staged.inner,
    )

    # Update tma descriptor with the correct shapes and strides
    tensormap_manager = utils.TensorMapManager(
        utils.TensorMapUpdateMode.SMEM,
        128,
    )
    tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 0, None)].iterator
    )
    tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 1, None)].iterator
    )
    tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 2, None)].iterator
    )
    tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 3, None)].iterator
    )
    tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 4, None)].iterator
    )

    mA_mkl_iter = cute.make_ptr(
        ab_dtype, tensor_of_abc_ptrs[group_idx, 0], cute.AddressSpace.gmem
    ).align(32)
    mB_nkl_iter = cute.make_ptr(
        ab_dtype, tensor_of_abc_ptrs[group_idx, 1], cute.AddressSpace.gmem
    ).align(32)
    sfa_mkl_iter = cute.make_ptr(
        sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 0], cute.AddressSpace.gmem
    ).align(32)
    sfb_nkl_iter = cute.make_ptr(
        sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], cute.AddressSpace.gmem
    ).align(32)
    mA_mkl_layout = cute.make_layout(
        (m, k, l), stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32),))
    mB_nkl_layout = cute.make_layout(
        (n, k, l), stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32),))

    atom_shape = ((32, 4), (sf_vec_size, 4))
    atom_stride = ((16, 4), (0, 1))
    sfa_layout = cute.tile_to_shape(
        cute.make_layout(atom_shape, stride=atom_stride),
        mA_mkl_layout.shape,
        (2, 1, 3),
    )
    sfb_layout = cute.tile_to_shape(
        cute.make_layout(atom_shape, stride=atom_stride),
        mB_nkl_layout.shape,
        (2, 1, 3),
    )
    real_tensor_a = cute.make_tensor(mA_mkl_iter, mA_mkl_layout)
    real_tensor_b = cute.make_tensor(mB_nkl_iter, mB_nkl_layout)
    real_tensor_sfa = cute.make_tensor(sfa_mkl_iter, sfa_layout)
    real_tensor_sfb = cute.make_tensor(sfb_nkl_iter, sfb_layout)
    real_tensor_c = real_mC_mnl

    if warp_idx == 0:
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_a, tensormap_a_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_b, tensormap_b_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfa, tensormap_sfa_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfb, tensormap_sfb_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_c, tensormap_c_smem_ptr, 0
        )
        tensormap_manager.update_tensormap(
            (
                real_tensor_a,
                real_tensor_b,
                real_tensor_sfa,
                real_tensor_sfb,
                real_tensor_c,
            ),
            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb, tma_atom_c),
            (
                tensormap_a_gmem_ptr,
                tensormap_b_gmem_ptr,
                tensormap_sfa_gmem_ptr,
                tensormap_sfb_gmem_ptr,
                tensormap_c_gmem_ptr,
            ),
            0,
            (
                tensormap_a_smem_ptr,
                tensormap_b_smem_ptr,
                tensormap_sfa_smem_ptr,
                tensormap_sfb_smem_ptr,
                tensormap_c_smem_ptr,
            ),
        )
        tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)

    cute.arch.barrier()

    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)
        cpasync.prefetch_descriptor(tma_atom_c)

    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = mma_tile_coord_v == 0
    cta_rank_in_cluster = cute.arch.make_warp_uniform(
        cute.arch.block_idx_in_cluster()
    )
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
        cta_rank_in_cluster
    )
    block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
        cta_rank_in_cluster
    )

    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_tma_producer = 1
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, num_tma_producer
    )
    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )

    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, len(epilog_warp_id)
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=32 * len((mma_warp_id, *epilog_warp_id)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilog_warp_id[0],
    )

    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

    gA_mkl_real = cute.local_tile(
        real_tensor_a, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl_real = cute.local_tile(
        real_tensor_b, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl_real = cute.local_tile(
        real_tensor_sfa, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl_real = cute.local_tile(
        real_tensor_sfb,
        cute.slice_((mma_tiler_mnk[0], cute.round_up(mma_tiler_mnk[1], 128), mma_tiler_mnk[2]), (0, None, None)),
        (None, None, None),
    )

    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl,
        cute.slice_((mma_tiler_mnk[0], cute.round_up(mma_tiler_mnk[1], 128), mma_tiler_mnk[2]), (0, None, None)),
        (None, None, None),
    )

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    k_block_cnt = cute.size(gA_mkl_real, mode=[3])

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB = thr_mma.partition_B(gB_nkl)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
    tCgC = thr_mma.partition_C(gC_mnl)

    a_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
    )
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    sfa_cta_layout = a_cta_layout
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        block_in_cluster_coord_vmnk[2],
        sfa_cta_layout,
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    sfb_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
    )
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        block_in_cluster_coord_sfb_vmnk[1],
        sfb_cta_layout,
        cute.group_modes(sSFB, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    pipeline_init_wait(cluster_shape_mn=(1, 1))

    mma_tile_coord_mnl = (coord_x, coord_y, 0)
    tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    slice_n = mma_tile_coord_mnl[1]
    if cutlass.const_expr(mma_tiler_mnk[1] == 64):
        slice_n = mma_tile_coord_mnl[1] // 2
    tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

    a_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
    )
    b_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
    )
    sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
    )
    sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
    )

    if warp_idx == tma_warp_id:
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_ab_stage
        )
        tma_desc_a = tensormap_manager.get_tensormap_ptr(
            tensormap_a_gmem_ptr, cute.AddressSpace.generic
        )
        tma_desc_b = tensormap_manager.get_tensormap_ptr(
            tensormap_b_gmem_ptr, cute.AddressSpace.generic
        )
        tma_desc_sfa = tensormap_manager.get_tensormap_ptr(
            tensormap_sfa_gmem_ptr, cute.AddressSpace.generic
        )
        tma_desc_sfb = tensormap_manager.get_tensormap_ptr(
            tensormap_sfb_gmem_ptr, cute.AddressSpace.generic
        )
        for k_block_idx in cutlass.range(0, k_block_cnt, 1, unroll=1):
            ab_pipeline.producer_acquire(ab_producer_state)
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, ab_producer_state.count)],
                tAsA[(None, ab_producer_state.index)],
                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                tma_desc_ptr=tma_desc_a,
                mcast_mask=a_full_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, ab_producer_state.count)],
                tBsB[(None, ab_producer_state.index)],
                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                tma_desc_ptr=tma_desc_b,
                mcast_mask=b_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA_slice[(None, ab_producer_state.count)],
                tAsSFA[(None, ab_producer_state.index)],
                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                tma_desc_ptr=tma_desc_sfa,
                mcast_mask=sfa_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB_slice[(None, ab_producer_state.count)],
                tBsSFB[(None, ab_producer_state.index)],
                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                tma_desc_ptr=tma_desc_sfb,
                mcast_mask=sfb_full_mcast_mask,
            )
            ab_producer_state.advance()

    elif warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=sf_dtype,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
            mainloop_s2t_copy_and_partition(sSFA, tCtSFA, sf_dtype, tcgen05.CtaGroup.ONE)
        )
        tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
            mainloop_s2t_copy_and_partition(sSFB, tCtSFB, sf_dtype, tcgen05.CtaGroup.ONE)
        )

        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_ab_stage
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_acc_stage
        )

        tCtSFB_mma = tCtSFB
        if cutlass.const_expr(mma_tiler_mnk[1] == 64):
            offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
            shifted_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                + offset,
                dtype=sf_dtype,
            )
            tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for _ in range(k_block_cnt):
            ab_pipeline.consumer_wait(ab_consumer_state)
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                ab_consumer_state.index,
            )
            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
            tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t_staged,
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t_staged,
                tCtSFB_compact_s2t,
            )
            num_kphases = cute.size(tCrA, mode=[2])
            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                kphase_coord = (
                    None,
                    None,
                    kphase_idx,
                    ab_consumer_state.index,
                )
                sf_kphase_coord = (None, None, kphase_idx)
                tiled_mma.set(
                    tcgen05.Field.SFA,
                    tCtSFA[sf_kphase_coord].iterator,
                )
                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB_mma[sf_kphase_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[kphase_coord],
                    tCrB[kphase_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
        acc_pipeline.producer_commit(acc_producer_state)

    elif warp_idx in epilog_warp_id:
        tmem.allocate(num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_acc_stage
        )
        acc_pipeline.consumer_wait(acc_consumer_state)
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[None, 0, 0])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[None, 0, 0])
        tDgC = thr_copy_t2r.partition_D(tCgC[None, 0, 0])
        tDrAcc = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
        tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)
        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
        acc_vec = tDrAcc.load()
        tDrC.store(acc_vec.to(c_dtype))
        simt_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=16
        )
        # Only epilogue warps participate, so use their thread count
        epilog_threads = 32 * len(epilog_warp_id)  # 128 threads
        thread_layout = cute.make_layout(
            (1, epilog_threads), stride=(epilog_threads, 1))
        value_layout = cute.make_layout((1, 1))
        tiled_copy_r2g = cute.make_tiled_copy_tv(
            simt_atom, thread_layout, value_layout
        )
        thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)
        cC = cute.make_identity_tensor(gC_mnl.shape)
        tDcC = thr_copy_r2g.partition_D(cC)
        tDpC = cute.make_rmem_tensor(tDrC.shape, cutlass.Boolean)
        residue_m = real_mC_mnl.shape[0] - cutlass.Int32(coord_x) * mma_tiler_mnk[0]
        residue_n = real_mC_mnl.shape[1] - cutlass.Int32(coord_y) * mma_tiler_mnk[1]
        for i in range(cute.size(tDrC.shape)):
            tDpC[i] = cute.elem_less(tDcC[i], (residue_n, residue_m))
        cute.copy(
            simt_atom,
            cute.flatten(tDrC),
            cute.flatten(tDgC),
            pred=cute.flatten(tDpC),
        )
        tmem.relinquish_alloc_permit()
        tmem.free(acc_tmem_ptr)


# Host-side JIT function to prepare tensors and launch GPU kernel.
@cute.jit
def my_kernel(
    ptr_of_tensor_of_problem_sizes: cute.Pointer,
    ptr_of_tensor_of_abc_ptrs: cute.Pointer,
    ptr_of_tensor_of_sfasfb_ptrs: cute.Pointer,
    ptr_of_tensor_of_tensormap: cute.Pointer,
    total_num_clusters: cutlass.Int32,
    problem_sizes: List[
        Tuple[int, int, int, int]
    ],  # Problem sizes for each group
    num_groups: cutlass.Int32,
):

    tensor_of_abc_ptrs = cute.make_tensor(
        ptr_of_tensor_of_abc_ptrs, cute.make_layout((num_groups, 3), stride=(3, 1))
    )
    tensor_of_sfasfb_ptrs = cute.make_tensor(
        ptr_of_tensor_of_sfasfb_ptrs, cute.make_layout((num_groups, 2), stride=(2, 1))
    )
    tensor_of_problem_sizes = cute.make_tensor(
        ptr_of_tensor_of_problem_sizes, cute.make_layout((num_groups, 4), stride=(4, 1))
    )
    tensor_of_tensormap = cute.make_tensor(
        ptr_of_tensor_of_tensormap,
        cute.make_layout((total_num_clusters, num_tensormaps, 16), stride=(num_tensormaps * 16, 16, 1)),
    )

    # Use fixed max shapes for initial TMA descriptor and atom setup.
    # The real TMA desc and atom will be updated during kernel execution.
    max_m = cutlass.Int32(512)
    max_n = cutlass.Int32(7168)
    max_k = cutlass.Int32(7168)
    min_a_shape = (
        max_m,
        cutlass.Int32(1),
        max_k,
        cutlass.Int32(1),
    )
    min_b_shape = (
        cutlass.Int32(1),
        max_n,
        max_k,
        cutlass.Int32(1),
    )
    initial_a = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_a_shape[0], cute.assume(min_a_shape[2], 32), min_a_shape[3]),
            stride=(
                cute.assume(min_a_shape[2], 32),
                1,
                cute.assume(min_a_shape[0] * min_a_shape[2], 32),
            ),
        ),
    )
    initial_b = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_b_shape[1], cute.assume(min_b_shape[2], 32), min_b_shape[3]),
            stride=(
                cute.assume(min_b_shape[2], 32),
                1,
                cute.assume(min_b_shape[1] * min_b_shape[2], 32),
            ),
        ),
    )
    initial_c = cute.make_tensor(
        cute.make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (max_m, max_n, cutlass.Int32(1)),
            stride=(
                cute.assume(max_n, 32),
                1,
                cute.assume(max_m * max_n, 32),
            ),
        ),
    )

    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_a.shape, sf_vec_size
    )
    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_b.shape, sf_vec_size
    )
    # Create initial SFA and SFB tensors with fake shape and null pointer.
    initial_sfa = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfa_layout)
    initial_sfb = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfb_layout)

    # Select MMA operation
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma.thr_id.shape,),
    )

    mma_inst_shape_mn_sfb = (
        mma_tiler_mnk[0],
        cute.round_up(mma_tiler_mnk[1], 128),
    )
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        sf_dtype,
        sf_vec_size,
        tcgen05.CtaGroup.ONE,
        mma_inst_shape_mn_sfb,
    )
    cluster_layout_sfb_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma_sfb.thr_id.shape,),
    )

    # Compute CTA tile shape and epilogue tile (needed for stage computation)
    c_layout = utils.LayoutEnum.ROW_MAJOR
    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )
    epi_tile = sm100_utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk,
        False,
        c_layout,
        c_dtype,
    )

    # Dynamically compute optimal number of pipeline stages based on SMEM capacity
    num_ab_stage = compute_num_ab_stages(
        tiled_mma=tiled_mma,
        mma_tiler=mma_tiler_mnk,
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        epi_tile=epi_tile,
        c_layout=c_layout,
        smem_capacity=smem_capacity,
        occupancy=occupancy,
    )

    # Compute A/B/SFA/SFB/C shared memory layout with computed stages
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
        c_dtype,
        c_layout,
        epi_tile,
        1,
    )
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # Setup TMA for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_a,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for B
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_b,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfa,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfb,
        sfb_smem_layout,
        (mma_inst_shape_mn_sfb[0], mma_inst_shape_mn_sfb[1], mma_tiler_mnk[2]),
        tiled_mma_sfb,
        cluster_layout_sfb_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
    ) * atom_thr_size

    # Setup TMA for C
    epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
    tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        initial_c,
        epi_smem_layout,
        epi_tile,
    )

    # Store CTA shape information for each Group in a List
    cta_mn_list = []
    for group_idx, (m, n, k, l) in enumerate(problem_sizes):
        x, y = cute.ceil_div(problem_sizes[group_idx][:2], mma_tiler_mnk[0:2])
        cta_mn_list.append((x, y))

    # Compute grid size
    grid = (1, 1, total_num_clusters)

    # Launch the kernel
    kernel(
        tiled_mma,
        tiled_mma_sfb,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sfa,
        tma_tensor_sfa,
        tma_atom_sfb,
        tma_tensor_sfb,
        tma_atom_c,
        tma_tensor_c,
        tensor_of_abc_ptrs,
        tensor_of_sfasfb_ptrs,
        tensor_of_tensormap,
        tensor_of_problem_sizes,
        cluster_layout_vmnk,
        cluster_layout_sfb_vmnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        c_smem_layout_staged,
        epi_tile,
        cta_mn_list,
        num_tma_load_bytes,
        num_ab_stage,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )
    return


# Global cache for compiled kernels (keyed by group size)
_compiled_kernel_cache = {}
# This function is used to compile the kernel once and cache it and then allow users to 
# run the kernel multiple times to get more accurate timing results.
def compile_kernel(problem_sizes):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    This should be called before any timing measurements.

    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache
    
    # Convert problem_sizes list to a hashable tuple for use as dictionary key
    cache_key = f"{len(problem_sizes)}"

    # Check if we already have a compiled kernel for these problem sizes
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    # Fake cluster numbers for compile only.
    total_num_clusters = cutlass.Int32(1)
    num_groups = cutlass.Int32(len(problem_sizes))
    # Each cluster needs its own set of tensormaps (one for A, B, SFA, SFB)
    # Shape: (total_num_clusters, num_tensormaps=5, bytes_per_tensormap/8=16)
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    compiled_func = cute.compile(
        my_kernel,
        cute_ptr_of_tensor_of_problem_sizes,
        cute_ptr_of_tensor_of_abc_ptrs,
        cute_ptr_of_tensor_of_sfasfb_ptrs,
        cute_ptr_of_tensor_of_tensormap,
        total_num_clusters,
        problem_sizes,
        num_groups
    )
    # Store compiled kernel in cache with problem_sizes as key
    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled group GEMM kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (abc_tensors, sfasfb_tensors, problem_sizes) where:
            abc_tensors: list of tuples (a, b, c) where 
                a is torch.Tensor[float4e2m1fn_x2] of shape [m, k // 2, l]
                b is torch.Tensor[float4e2m1fn_x2] of shape [n, k // 2, l]
                c is torch.Tensor[float16] of shape [m, n, l]
            sfasfb_tensors: list of tuples (sfa, sfb) where 
                sfa is torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l]
                sfb is torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l]
            problem_sizes: list of tuples (m, n, k, l)
            each group has its own a, b, c, sfa, sfb with different m, n, k, l problem sizes
            l should always be 1 for each group.
            list size is the number of groups.
    
    Returns:
        list of c tensors where c is torch.Tensor[float16] of shape [m, n, l] for each group
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    compiled_func = compile_kernel(problem_sizes)

    # Extract raw data pointers from all input tensors for each group
    # These will be passed to the GPU kernel to access the actual tensor data
    abc_ptrs = []
    sfasfb_ptrs = []
    for i, ((a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l)) in enumerate(zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)):
        # Store pointers to A, B, and C matrices for this group
        abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
        # Store pointers to scale factor tensors for this group
        sfasfb_ptrs.append((sfa_reordered.data_ptr(), sfb_reordered.data_ptr()))

    # Create torch tensor to store problem sizes for all groups
    # Shape: (num_groups, 4) where each row contains (m, n, k, l) for that group
    # Layout: (num_groups, 4):(4, 1) means row-major storage
    tensor_of_problem_sizes = torch.tensor(
        problem_sizes, dtype=torch.int32, device="cuda"
    )

    # Create torch tensors to store data pointers for all groups
    # These allow the GPU kernel to dynamically access different tensors per group
    # tensor_of_abc_ptrs: Shape (num_groups, 3) containing (a_ptr, b_ptr, c_ptr) per group
    # tensor_of_sfasfb_ptrs: Shape (num_groups, 2) containing (sfa_ptr, sfb_ptr) per group
    tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
    tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")

    # Compute the tile shape for each CUDA Thread Block (CTA)
    # cta_tile_shape_mn: [M_tile, N_tile] = [128, 128] for this kernel
    cta_tile_shape_mn = [128, mma_tiler_mnk[1]]
    # cluster_tile_shape_mn: Total tile shape per cluster (same as CTA since cluster is 1x1)
    cluster_tile_shape_mn = tuple(
        x * y for x, y in zip(cta_tile_shape_mn, (1, 1))
    )
    
    # Compute total number of cluster tiles needed across all groups
    # Each group's (m, n) dimensions are divided into tiles of size cluster_tile_shape_mn
    # This determines the total grid size (bidz dimension) for kernel launch
    total_num_clusters = 0
    num_groups = len(problem_sizes)
    for m, n, _, _ in problem_sizes:
        # Calculate number of tiles needed in M and N dimensions for this group
        num_clusters_mn = tuple(
            (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
        )
        # Multiply M_tiles * N_tiles to get total tiles for this group
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    # Allocate device memory for tensormap descriptors
    # Each cluster needs its own set of tensormaps (one for A, B, SFA, SFB)
    # Shape: (total_num_clusters, num_tensormaps=5, bytes_per_tensormap/8=16)
    # Tensormaps are hardware descriptors used by TMA for efficient memory transfers
    tensormap_shape = (
        total_num_clusters,
        num_tensormaps,
        bytes_per_tensormap // 8,
    )
    tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")

    # Create CuTe pointers to the metadata tensors that will be passed to the kernel
    # These allow the GPU kernel to read problem sizes and tensor pointers
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64,
        tensor_of_abc_ptrs.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64,
        tensor_of_sfasfb_ptrs.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32,
        tensor_of_problem_sizes.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64,
        tensor_of_tensormap.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Launch the JIT-compiled GPU kernel with all prepared data
    # The kernel will perform block-scaled group GEMM: C = A * SFA * B * SFB for all groups
    compiled_func(
        cute_ptr_of_tensor_of_problem_sizes, # Pointer to problem sizes array
        cute_ptr_of_tensor_of_abc_ptrs,      # Pointer to ABC tensor pointers array
        cute_ptr_of_tensor_of_sfasfb_ptrs,   # Pointer to scale factor pointers array
        cute_ptr_of_tensor_of_tensormap,     # Pointer to tensormap buffer
        total_num_clusters,                  # Total number of CTAs to launch
        problem_sizes,                       # Problem sizes list (for host-side processing)
        num_groups,                          # Number of groups in this batch
    )

    res = []
    for i in range(num_groups):
        res.append(abc_tensors[i][2])
    return res