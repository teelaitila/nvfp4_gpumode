from torch._higher_order_ops.torchbind import call_torchbind_fake
import cuda.bindings.driver as cuda

import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr
from cutlass.utils import LayoutEnum

# Kernel configuration parameters
# Tile sizes for M, N, K dimensions
mma_tiler_mnk= (128, 128, 256)  
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
# Number of threads per CUDA thread block
threads_per_cta = 128  
# Stage numbers of shared memory and tmem
num_acc_stage = 1
num_ab_stage = 3  # More stages to better hide TMA latency for large K dimensions
num_c_stage = 2   # Overlap R2S writes with TMA S2G stores
# Total number of columns in tmem
num_tmem_alloc_cols = 512

# Cluster shape for TMA multicast (M, N)
cluster_shape_mn = (1, 4)

# TMA Prefetch distance (how many tiles ahead to prefetch)
# Match num_ab_stage for optimal pipeline utilization
prefetch_dist = num_ab_stage

# Warp specialization IDs
tma_warp_id = 0   # Warp 0: TMA loads
mma_warp_id = 1   # Warp 1: MMA compute


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


#  GPU device kernel
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b1: cute.CopyAtom,
    mB_nkl1: cute.Tensor,
    tma_atom_b2: cute.CopyAtom,
    mB_nkl2: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb1: cute.CopyAtom,
    mSFB_nkl1: cute.Tensor,
    tma_atom_sfb2: cute.CopyAtom,
    mSFB_nkl2: cute.Tensor,
    tma_atom_c: cute.CopyAtom,  # TMA S2G atom for C
    mC_mnl: cute.Tensor,  # TMA tensor for C store (for global coords)
    cluster_layout_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    c_smem_layout_staged: cute.ComposedLayout,  # SMEM layout for C epilogue
    epi_tile: cute.Tile,  # Epilogue tile for partitioning
    c_layout: cutlass.Constexpr[utils.LayoutEnum],  # Layout enum for C tensor (compile-time)
    num_tma_load_bytes: cutlass.Constexpr[int],
    epilogue_op: cutlass.Constexpr = lambda x: x
    * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
):
    """
    GPU device kernel performing the dual GEMM computation with silu activation.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx = cute.arch.thread_idx()

    #
    # Setup cta/thread coordinates
    #
    # Coords inside cluster
    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = mma_tile_coord_v == 0
    cta_rank_in_cluster = cute.arch.make_warp_uniform(
        cute.arch.block_idx_in_cluster()
    )
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
        cta_rank_in_cluster
    )

    # Coords outside cluster
    cta_coord = (bidx, bidy, bidz)
    mma_tile_coord_mnl = (
        cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
        cta_coord[1],
        cta_coord[2],
    )
    # Coord inside cta
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Define shared storage for kernel
    #
    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_holding_buf: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB1 = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB2 = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_M, MMA_K, STAGE)
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB1 = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB2 = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )
    # (EPI_M, EPI_N, STAGE) - Epilogue buffer for C
    sC = smem.allocate_tensor(
        element_type=c_dtype,
        layout=c_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=c_smem_layout_staged.inner,
    )

    #
    # Initialize mainloop ab_pipeline, acc_pipeline and their states
    #
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_mcast_ctas_a = cluster_shape_mn[1] if cluster_shape_mn[1] > 1 else 1
    num_mcast_ctas_b = 1
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, num_tma_producer
    )
    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        mcast_mode_mn=(1, cluster_shape_mn[1]),
        defer_sync=True,
    )
    ab_producer, ab_consumer = ab_pipeline.make_participants()
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
    acc_producer, acc_consumer = acc_pipeline.make_participants()

    # Cluster arrive after barrier init
    if cutlass.const_expr(cluster_shape_mn[1] > 1):
        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)
    
    # C store pipeline for TMA S2G
    c_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        cute.arch.WARP_SIZE,  # One warp for TMA store
    )
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=num_c_stage,
        producer_group=c_producer_group,
    )

    #
    # Local_tile partition global tensors
    #
    # (bM, bK, RestM, RestK, RestL)
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl1 = cute.local_tile(
        mB_nkl1, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl2 = cute.local_tile(
        mB_nkl2, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl1 = cute.local_tile(
        mSFB_nkl1, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gSFB_nkl2 = cute.local_tile(
        mSFB_nkl2, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )
    k_tile_cnt = cute.size(gA_mkl, mode=[3])

    #
    # Partition global tensor for TiledMMA_A/B/SFA/SFB/C
    #
    # (MMA, MMA_M, MMA_K, RestK)
    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB1 = thr_mma.partition_B(gB_nkl1)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB2 = thr_mma.partition_B(gB_nkl2)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB1 = thr_mma.partition_B(gSFB_nkl1)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB2 = thr_mma.partition_B(gSFB_nkl2)
    # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCgC = thr_mma.partition_C(gC_mnl)

    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # TMA Partition_S/D for A (cluster multicast across N)
    a_cta_layout = cute.make_layout(cute.size(cluster_layout_vmnk, mode=[2]))
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA Partition_S/D for B1
    b_cta_layout = cute.make_layout(cute.size(cluster_layout_vmnk, mode=[1]))
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB1, tBgB1 = cpasync.tma_partition(
        tma_atom_b1,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB1, 0, 3),
        cute.group_modes(tCgB1, 0, 3),
    )
    # TMA Partition_S/D for B2
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB2, tBgB2 = cpasync.tma_partition(
        tma_atom_b2,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB2, 0, 3),
        cute.group_modes(tCgB2, 0, 3),
    )
    #  TMA Partition_S/D for SFA
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    # TMA Partition_S/D for SFB1
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB1, tBgSFB1 = cpasync.tma_partition(
        tma_atom_sfb1,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sSFB1, 0, 3),
        cute.group_modes(tCgSFB1, 0, 3),
    )
    tBsSFB1 = cute.filter_zeros(tBsSFB1)
    tBgSFB1 = cute.filter_zeros(tBgSFB1)
    # TMA Partition_S/D for SFB2
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB2, tBgSFB2 = cpasync.tma_partition(
        tma_atom_sfb2,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sSFB2, 0, 3),
        cute.group_modes(tCgSFB2, 0, 3),
    )
    tBsSFB2 = cute.filter_zeros(tBsSFB2)
    tBgSFB2 = cute.filter_zeros(tBgSFB2)

    #
    # Multicast masks (A/SFA only) across N dimension
    #
    a_full_mcast_mask = None
    sfa_full_mcast_mask = None
    if cutlass.const_expr(cluster_shape_mn[1] > 1):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )

    #
    # Partition shared/tensor memory tensor for TiledMMA_A/B/C
    #
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB1 = tiled_mma.make_fragment_B(sB1)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB2 = tiled_mma.make_fragment_B(sB2)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    #
    # Alloc tensor memory buffer
    # Make ACC1 and ACC2 tmem tensor
    # ACC1 += A @ B1
    # ACC2 += A @ B2
    #
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
    acc_tmem_ptr1 = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc1),
        dtype=cutlass.Float32,
    )
    tCtAcc2 = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)

    #
    # Make SFA/SFB1/SFB2 tmem tensor
    #
    # SFA tmem layout: (MMA, MMA_M, MMA_K)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    # Get SFA tmem ptr
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc2),
        dtype=sf_dtype,
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

    # SFB1, SFB2 tmem layout: (MMA, MMA_N, MMA_K)
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    # Get SFB1 tmem ptr
    sfb_tmem_ptr1 = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc2)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    tCtSFB1 = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout)
    # Get SFB2 tmem ptr
    sfb_tmem_ptr2 = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc2)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFB1),
        dtype=sf_dtype,
    )
    tCtSFB2 = cute.make_tensor(sfb_tmem_ptr2, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB1/SFB2
    #
    # Make S2T CopyAtom
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    # (MMA, MMA_MN, MMA_K)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB1_compact = cute.filter_zeros(sSFB1)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB1_compact = cute.filter_zeros(tCtSFB1)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB1_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB1_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB1_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB1_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB1_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB1_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB1_compact)

    # SFB2 S2T copy and partition
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB2_compact = cute.filter_zeros(sSFB2)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB2_compact = cute.filter_zeros(tCtSFB2)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB2_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB2_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB2_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB2_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB2_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB2_compact)

    #
    # Slice to per mma tile index
    #
    # ((atom_v, rest_v), RestK)
    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgB1 = tBgB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgB2 = tBgB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgSFB1 = tBgSFB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgSFB2 = tBgSFB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    #
    # WARP SPECIALIZATION: Split TMA and MMA into separate warps
    # With TMA prefetch for improved memory latency hiding
    #
    
    # TMA Warp: handles all memory loads
    if warp_idx == tma_warp_id:
        # Prefetch TMA descriptors to hide descriptor load latency
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b1)
        cpasync.prefetch_descriptor(tma_atom_b2)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb1)
        cpasync.prefetch_descriptor(tma_atom_sfb2)
        
        # Initial prefetch: prime the pipeline with first prefetch_dist tiles
        for pf_k_tile in cutlass.range(0, min(prefetch_dist, k_tile_cnt), unroll=4):
            cute.prefetch(tma_atom_a, tAgA[(None, pf_k_tile)])
            cute.prefetch(tma_atom_b1, tBgB1[(None, pf_k_tile)])
            cute.prefetch(tma_atom_b2, tBgB2[(None, pf_k_tile)])
            cute.prefetch(tma_atom_sfa, tAgSFA[((0, None), pf_k_tile)])
            cute.prefetch(tma_atom_sfb1, tBgSFB1[((0, None), pf_k_tile)])
            cute.prefetch(tma_atom_sfb2, tBgSFB2[((0, None), pf_k_tile)])
        
        # Reset producer state and peek for first iteration
        ab_producer.reset()
        peek_ab_empty_status = ab_producer.try_acquire()
        
        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=4):
            # Conditionally acquire empty buffer slot (non-blocking if peek succeeded)
            ab_empty = ab_producer.acquire_and_advance(peek_ab_empty_status)

            # TMA load A/B1/B2/SFA/SFB1/SFB2 to shared memory
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=a_full_mcast_mask,
            )
            cute.copy(
                tma_atom_b1,
                tBgB1[(None, ab_empty.count)],
                tBsB1[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b2,
                tBgB2[(None, ab_empty.count)],
                tBsB2[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[((0, None), ab_empty.count)],
                tAsSFA[((0, None), ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=sfa_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfb1,
                tBgSFB1[((0, None), ab_empty.count)],
                tBsSFB1[((0, None), ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfb2,
                tBgSFB2[((0, None), ab_empty.count)],
                tBsSFB2[((0, None), ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            
            # Rolling prefetch: issue prefetch for future tiles
            if ab_empty.count < k_tile_cnt - prefetch_dist:
                future_k_tile = ab_empty.count + prefetch_dist
                cute.prefetch(tma_atom_a, tAgA[(None, future_k_tile)])
                cute.prefetch(tma_atom_b1, tBgB1[(None, future_k_tile)])
                cute.prefetch(tma_atom_b2, tBgB2[(None, future_k_tile)])
                cute.prefetch(tma_atom_sfa, tAgSFA[((0, None), future_k_tile)])
                cute.prefetch(tma_atom_sfb1, tBgSFB1[((0, None), future_k_tile)])
                cute.prefetch(tma_atom_sfb2, tBgSFB2[((0, None), future_k_tile)])
            
            # Peek for next iteration (non-blocking check if buffer is available)
            peek_ab_empty_status = cutlass.Boolean(1)
            if ab_empty.count + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_producer.try_acquire()
        
        # Wait for all TMA loads to complete (producer tail)
        ab_producer.tail()

    # MMA Warp: handles all computation
    if warp_idx == mma_warp_id and is_leader_cta:
        # Acquire accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()
        # Set ACCUMULATE field to False for the first k_tile iteration
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        
        # Reset consumer state and peek for first iteration (non-blocking)
        ab_consumer.reset()
        peek_ab_full_status = ab_consumer.try_wait()
        
        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=4):
            # Conditionally wait for TMA data (non-blocking if peek succeeded)
            ab_full = ab_consumer.wait_and_advance(peek_ab_full_status)

            # Copy SFA/SFB1/SFB2 to tmem
            s2t_stage_coord = (None, None, None, None, ab_full.index)
            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
            tCsSFB1_compact_s2t_staged = tCsSFB1_compact_s2t[s2t_stage_coord]
            tCsSFB2_compact_s2t_staged = tCsSFB2_compact_s2t[s2t_stage_coord]
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t_staged,
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB1_compact_s2t_staged,
                tCtSFB1_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB2_compact_s2t_staged,
                tCtSFB2_compact_s2t,
            )

            # tCtAcc1 += tCrA * tCrSFA * tCrB1 * tCrSFB1
            # tCtAcc2 += tCrA * tCrSFA * tCrB2 * tCrSFB2
            num_kblocks = cute.size(tCrA, mode=[2])
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (
                    None,
                    None,
                    kblock_idx,
                    ab_full.index,
                )

                # Set SFA/SFB tensor to tiled_mma
                sf_kblock_coord = (None, None, kblock_idx)
                tiled_mma.set(
                    tcgen05.Field.SFA,
                    tCtSFA[sf_kblock_coord].iterator,
                )
                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB1[sf_kblock_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc1,
                    tCrA[kblock_coord],
                    tCrB1[kblock_coord],
                    tCtAcc1,
                )

                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB2[sf_kblock_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc2,
                    tCrA[kblock_coord],
                    tCrB2[kblock_coord],
                    tCtAcc2,
                )

                # Enable accumulate on tCtAcc1/tCtAcc2 after first kblock
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal TMA warp that this buffer slot is free
            ab_full.release()
            
            # Peek for next iteration (non-blocking check if data is ready)
            peek_ab_full_status = cutlass.Boolean(1)
            if ab_full.count + 1 < k_tile_cnt:
                peek_ab_full_status = ab_consumer.try_wait()
        
        # Commit accumulator to epilogue
        acc_empty.commit()

    #
    # Epilogue with TMA store
    #
    
    acc_dtype = cutlass.Float32
    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )
    
    # Get the TMEM load copy atom using helper
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        cta_tile_shape_mnk,
        c_layout,
        c_dtype,
        acc_dtype,
        epi_tile,
        False,  # use_2cta_instrs
    )
    
    # Flat divide accumulators by epilogue tile
    # Our accumulator has shape (MMA, MMA_M, MMA_N)
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
    tCtAcc1_epi = cute.flat_divide(
        tCtAcc1[((None, None), 0, 0)],
        epi_tile,
    )
    tCtAcc2_epi = cute.flat_divide(
        tCtAcc2[((None, None), 0, 0)],
        epi_tile,
    )
    
    # Create T2R tiled copy
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tCtAcc1_epi[(None, None, 0, 0)]
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    
    # Partition accumulators (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_tAcc1 = thr_copy_t2r.partition_S(tCtAcc1_epi)
    tTR_tAcc2 = thr_copy_t2r.partition_S(tCtAcc2_epi)
    
    # Flat divide global C by epilogue tile
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    gC_epi = cute.flat_divide(
        tCgC[((None, None), 0, 0, None, None, None)], epi_tile
    )
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tTR_gC = thr_copy_t2r.partition_D(gC_epi)
    
    # Register tensors for accumulator
    tTR_rAcc1 = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, acc_dtype
    )
    tTR_rAcc2 = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, acc_dtype
    )
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, c_dtype
    )
    
    # Setup R2S copy (Register to SMEM)
    copy_atom_r2s = sm100_utils.get_smem_store_op(
        c_layout, c_dtype, acc_dtype, tiled_copy_t2r
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    
    # Partition SMEM for R2S (R2S, R2S_M, R2S_N, STAGE)
    tRS_sC = thr_copy_r2s.partition_D(sC)
    # Retile register tensor for R2S
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    
    # Setup TMA partition for SMEM to Global - need separate partition per stage
    gC_for_tma = cute.group_modes(gC_epi, 0, 2)
    
    # Create TMA partitions for each stage
    sC_stage0 = cute.group_modes(sC[(None, None, 0)], 0, 2)
    sC_stage1 = cute.group_modes(sC[(None, None, 1)], 0, 2)
    bSG_sC_0, bSG_gC = cpasync.tma_partition(
        tma_atom_c,
        0,  # CTA coordinate
        cute.make_layout(1),  # Single CTA
        sC_stage0,
        gC_for_tma,
    )
    bSG_sC_1, _ = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        sC_stage1,
        gC_for_tma,
    )
    
    # Slice to current tile
    bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]

    # Wait for accumulator buffer full
    acc_full = acc_consumer.wait_and_advance()

    # Epilogue loop over subtiles with 2-stage pipelining
    epi_m_tiles = cute.size(tTR_tAcc1.shape, mode=[3])
    epi_n_tiles = cute.size(tTR_tAcc1.shape, mode=[4])
    
    # Track in-flight TMA stores (0 = none, 1 = stage 0, 2 = stage 1)
    stage_idx = cutlass.Int32(0)
    
    for epi_m in cutlass.range(epi_m_tiles):
        for epi_n in cutlass.range(epi_n_tiles):
            # Copy accumulator to register (T2R)
            cute.copy(tiled_copy_t2r, tTR_tAcc1[(None, None, None, epi_m, epi_n)], tTR_rAcc1)
            cute.copy(tiled_copy_t2r, tTR_tAcc2[(None, None, None, epi_m, epi_n)], tTR_rAcc2)

            # Silu activation on acc1 and multiply with acc2
            acc_vec1 = epilogue_op(tTR_rAcc1.load())
            acc_vec2 = tTR_rAcc2.load()
            acc_vec = acc_vec1 * acc_vec2
            tTR_rC.store(acc_vec.to(c_dtype))
            
            # Store from registers to SMEM using R2S copy (alternating stages)
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, stage_idx)])
            
            # Fence to ensure R2S completes before TMA reads SMEM
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier()
            
            # TMA store from SMEM to Global (one warp issues TMA)
            if warp_idx == 0:
                # Select source SMEM buffer based on stage
                if stage_idx == 0:
                    cute.copy(tma_atom_c, bSG_sC_0, bSG_gC[(None, epi_m, epi_n)])
                else:
                    cute.copy(tma_atom_c, bSG_sC_1, bSG_gC[(None, epi_m, epi_n)])
                cute.arch.cp_async_bulk_commit_group()
                
                # Wait for at most 1 TMA in flight (pipelining!)
                # This allows the previous TMA to complete while we process the next tile
                cute.arch.cp_async_bulk_wait_group(1, read=True)
            
            cute.arch.barrier()
            
            # Alternate stage for next iteration
            stage_idx = 1 - stage_idx
    
    # Wait for all remaining TMA stores to complete
    if warp_idx == 0:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    cute.arch.barrier()

    acc_full.release()
    # Deallocate TMEM
    cute.arch.barrier()
    tmem.free(acc_tmem_ptr)
    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b1_ptr: cute.Pointer,
    b2_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb1_ptr: cute.Pointer,
    sfb2_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
    epilogue_op: cutlass.Constexpr = lambda x: x
    * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, n, k, l = problem_size

    # Setup attributes that depend on gemm inputs
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    b_tensor1 = cute.make_tensor(
        b1_ptr,
        cute.make_layout(
            (n, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    b_tensor2 = cute.make_tensor(
        b2_ptr,
        cute.make_layout(
            (n, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n))
    )
    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        a_tensor.shape, sf_vec_size
    )
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        b_tensor1.shape, sf_vec_size
    )
    sfb_tensor1 = cute.make_tensor(sfb1_ptr, sfb_layout)
    sfb_tensor2 = cute.make_tensor(sfb2_ptr, sfb_layout)

    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma.thr_id.shape,),
    )

    # Compute A/B/SFA/SFB/C shared memory layout
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    # B1 and B2 have the same size thus share the same smem layout
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
    # SFB1 and SFB2 have the same size thus share the same smem layout
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)
    
    # Compute epilogue tile shape and C SMEM layout
    c_layout = LayoutEnum.from_tensor(c_tensor)
    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // atom_thr_size,
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )
    use_2cta_instrs = False
    epi_tile = sm100_utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk,
        use_2cta_instrs,
        c_layout,
        c_dtype,
    )
    c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
        c_dtype,
        c_layout,
        epi_tile,
        num_c_stage,
    )

    # Setup TMA for A (cluster-aware atom for multicast)
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        cluster_shape_mn, tiled_mma.thr_id
    )
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        a_tensor,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
    )
    # Setup TMA for B1
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        cluster_shape_mn, tiled_mma.thr_id
    )
    tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b_tensor1,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
    )
    # Setup TMA for B2
    tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b_tensor2,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged , (None, None, None, 0)
    )
    sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
        cluster_shape_mn, tiled_mma.thr_id
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        sfa_op,
        sfa_tensor,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB1
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged , (None, None, None, 0)
    )
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
        cluster_shape_mn, tiled_mma.thr_id
    )
    tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op,
        sfb_tensor1,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB2
    tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op,
        sfb_tensor2,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
        internal_type=cutlass.Int16,
    )
    
    # Setup TMA for C store (S2G - Shared to Global)
    c_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
    c_cta_tiler = cute.composition(cute.make_identity_layout(c_tensor.shape), epi_tile)
    tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        c_tensor,
        c_smem_layout,
        c_cta_tiler,
    )

    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
    ) * atom_thr_size

    # Compute grid size
    grid = (
        cute.ceil_div(c_tensor.shape[0], mma_tiler_mnk[0]),
        cute.ceil_div(c_tensor.shape[1], mma_tiler_mnk[1]),
        c_tensor.shape[2],
    )
    grid = cute.round_up(grid, (*cluster_shape_mn, 1))

    # Launch the kernel.
    kernel(
        # MMA (Matrix Multiply-Accumulate) configuration
        tiled_mma,                  # Tiled MMA object defining NVFP4 GEMM compute pattern
        
        # TMA (Tensor Memory Accelerator) atoms and tensors for shared input matrix A
        tma_atom_a,                 # TMA copy atom defining how to load A from global memory
        tma_tensor_a,               # Tensor descriptor for A matrix (m, k, l) - shared by both GEMMs
        
        # TMA atoms and tensors for first B matrix (B1)
        tma_atom_b1,                # TMA copy atom defining how to load B1 from global memory
        tma_tensor_b1,              # Tensor descriptor for B1 matrix (n, k, l) - first GEMM
        
        # TMA atoms and tensors for second B matrix (B2)
        tma_atom_b2,                # TMA copy atom defining how to load B2 from global memory
        tma_tensor_b2,              # Tensor descriptor for B2 matrix (n, k, l) - second GEMM
        
        # TMA atoms and tensors for scale factor A (shared)
        tma_atom_sfa,               # TMA copy atom for loading scale factors for A
        tma_tensor_sfa,             # Tensor descriptor for SFA (block scale factors for A) - shared
        
        # TMA atoms and tensors for scale factor B1
        tma_atom_sfb1,              # TMA copy atom for loading scale factors for B1
        tma_tensor_sfb1,            # Tensor descriptor for SFB1 (block scale factors for B1)
        
        # TMA atoms and tensors for scale factor B2
        tma_atom_sfb2,              # TMA copy atom for loading scale factors for B2
        tma_tensor_sfb2,            # Tensor descriptor for SFB2 (block scale factors for B2)
        
        # TMA atom for C store (S2G)
        tma_atom_c,                 # TMA copy atom for storing C from shared to global memory
        tma_tensor_c,               # TMA tensor for C store
        cluster_layout_vmnk,        # Cluster layout for multicast
        
        # Shared memory layouts with staging for pipelined execution
        a_smem_layout_staged,       # Staged shared memory layout for A (includes stage dimension)
        b_smem_layout_staged,       # Staged shared memory layout for B1/B2 (includes stage dimension)
        sfa_smem_layout_staged,     # Staged shared memory layout for SFA (includes stage dimension)
        sfb_smem_layout_staged,     # Staged shared memory layout for SFB1/SFB2 (includes stage dimension)
        c_smem_layout_staged,       # Staged shared memory layout for C epilogue
        epi_tile,                   # Epilogue tile for partitioning
        c_layout,                   # Layout enum for C tensor
        
        # Pipeline synchronization parameter
        num_tma_load_bytes,         # Total bytes to load per TMA transaction (for barrier setup)
        
        # Epilogue operation
        epilogue_op,                # Epilogue operation to apply to output (e.g., element-wise ops)
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(*cluster_shape_mn, 1),
    )
    return


# Global cache for compiled kernel
_compiled_kernel_cache = None
# This function is used to compile the kernel once and cache it and then allow users to 
# run the kernel multiple times to get more accurate timing results.
def compile_kernel():
    """
    Compile the kernel once and cache it.
    This should be called before any timing measurements.
    
    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache
    
    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache
    

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    b1_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    b2_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )

    # Compile the kernel
    _compiled_kernel_cache = cute.compile(my_kernel, a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (0, 0, 0, 0))
    
    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled dual GEMM kernel with silu activation,
    C = silu(A @ B1) * (A @ B2).
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn 
            b1: [n, k, l] - Input matrix in float4e2m1fn 
            b2: [n, k, l] - Input matrix in float4e2m1fn 
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb1_cpu: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb2_cpu: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb1_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb2_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, n, l] - Output vector in float16
    
    Returns:
        Output tensor c with computed results
    """
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
    
    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    compiled_func = compile_kernel()

    # Get dimensions from MxKxL layout
    _, k, _ = a.shape
    m, n, l = c.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2 

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b1_ptr = make_ptr(
        ab_dtype, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b2_ptr = make_ptr(
        ab_dtype, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, sfb1_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, sfb2_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    compiled_func(a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (m, n, k, l))

    return c