import torch
from task import input_t, output_t
from utils import make_match_reference

from torch._higher_order_ops.torchbind import call_torchbind_fake
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

# Kernel configuration parameters
# Tile sizes for M, N, K dimensions
mma_tiler_mnk = (256, 256, 256)  
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
# Cluster shape for 2CTA MMA
cluster_shape_mnk = (2, 1, 1)
# Stage numbers - can increase due to reduced SMEM per CTA with 2CTA
num_acc_stage = 1
num_ab_stage = 4  # Increased from 1 for better latency hiding
# Total number of columns in tmem
num_tmem_alloc_cols = 512


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# The CuTe reference implementation for NVFP4 block-scaled GEMM with 2CTA MMA
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    cta_layout_vmnk: cute.Layout,
    num_tma_load_bytes: cutlass.Constexpr[int],
):
    """
    GPU device kernel performing the batched GEMM computation with 2CTA MMA.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx = cute.arch.thread_idx()

    #
    # Setup cta/thread coordinates for 2CTA
    #
    bidx, bidy, bidz = cute.arch.block_idx()
    
    # Get CTA rank within cluster
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    
    # Compute MMA tile coordinates
    # For 2CTA: mma_coord_vmnk[0] is the peer CTA index (0 or 1)
    #           mma_coord_vmnk[1:] are the MMA tile coordinates
    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),  # V: peer CTA index within pair
        bidx // cute.size(cta_layout_vmnk, mode=[0]), # M: MMA tile M coordinate  
        bidy,                                          # N: MMA tile N coordinate
        bidz,                                          # L: batch index
    )
    
    # Determine if this is the leader CTA (even CTA in the pair)
    is_leader_cta = mma_coord_vmnk[0] == 0
    
    # MMA tile coordinate for indexing (M, N, L)
    mma_tile_coord_mnl = (mma_coord_vmnk[1], mma_coord_vmnk[2], mma_coord_vmnk[3])
    
    # Thread index
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Define shared storage for kernel with 2CTA support
    #
    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_dealloc_mbar_ptr: cutlass.Int64  # For 2CTA TMEM deallocation sync
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
    sB = smem.allocate_tensor(
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
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    #
    # Create TMA multicast masks for A and B only
    # Scale factors use non-multicast TMA due to different layout structure
    #
    tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    #
    # Initialize mainloop ab_pipeline, acc_pipeline and their states
    #
    # For 2CTA: need to account for multicast participants
    num_mcast_ctas_a = cute.size(cta_layout_vmnk, mode=[2])  # N mode
    num_mcast_ctas_b = cute.size(cta_layout_vmnk, mode=[1])  # M mode  
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1
    
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer),
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()
    
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            cute.size(cta_layout_vmnk, mode=[0]) * threads_per_cta,
        ),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    #
    # Local_tile partition global tensors
    #
    mma_coord_mnk = (mma_coord_vmnk[1], mma_coord_vmnk[2], None)
    
    # (bM, bK, RestK)
    gA_mkl = cute.local_tile(
        mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1)
    )
    # (bN, bK, RestK)
    gB_nkl = cute.local_tile(
        mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1)
    )
    # (bM, bN)
    gC_mnl = cute.local_tile(
        mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None)
    )
    # Scale factors
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1)
    )
    
    k_tile_cnt = cute.size(gA_mkl, mode=[2])

    #
    # Partition global tensor for TiledMMA - using peer CTA coordinate
    #
    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])
    
    # (MMA, MMA_M, MMA_K, RestK)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestK)
    tCgB = thr_mma.partition_B(gB_nkl)
    # (MMA, MMA_M, MMA_K, RestK)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestK)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC_mnl)

    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # TMA Partition_S/D for A - using cluster coordinate
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],  # N mode coordinate in cluster
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA Partition_S/D for B
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],  # M mode coordinate in cluster
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    # TMA Partition_S/D for SFA - non-multicast, simpler partitioning
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    # TMA Partition_S/D for SFB - non-multicast, simpler partitioning
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    #
    # Partition shared/tensor memory tensor for TiledMMA_A/B/C
    #
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    #
    # Alloc tensor memory buffer - with 2CTA support
    #
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
    )
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

    #
    # Make SFA/SFB tmem tensor
    #
    # Get SFA tmem ptr
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
        dtype=sf_dtype,
    )
    # (MMA, MMA_M, MMA_K)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
    # Get SFB tmem ptr
    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    # (MMA, MMA_N, MMA_K)
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB
    # SFA uses CtaGroup.TWO, SFB uses CtaGroup.ONE (matching their TMA atoms)
    #
    copy_atom_s2t_sfa = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
        sf_dtype,
    )
    copy_atom_s2t_sfb = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )
    
    # SFA S2T copy
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    # (MMA, MMA_MN, MMA_K)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t_sfa, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
    )
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    # SFB S2T copy - uses CtaGroup.ONE
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact = cute.filter_zeros(sSFB)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t_sfb, tCtSFB_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
    tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
    )
    tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

    #
    # Execute Data copy and Math computation in the k_tile loop
    # For 2CTA: only leader CTA (warp 0) issues TMA and MMA
    #
    if warp_idx == 0:
        # Wait for accumulator buffer empty - only leader CTA
        if is_leader_cta:
            acc_producer.acquire_and_advance()
            
        # Set ACCUMULATE field to False for the first k_tile iteration
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        
        # Execute k_tile loop
        for k_tile in range(k_tile_cnt):
            # Wait for AB buffer empty
            ab_empty = ab_producer.acquire_and_advance()

            # TMA load A/B/SFA/SFB to shared memory with multicast
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_a,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_b,
            )
            # Scale factors use non-multicast TMA
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, k_tile)],
                tAsSFA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, k_tile)],
                tBsSFB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

            # Only leader CTA executes MMA
            if is_leader_cta:
                # Wait for AB buffer full
                ab_full = ab_consumer.wait_and_advance()

                # Copy SFA/SFB from shared memory to TMEM
                s2t_stage_coord = (None, None, None, None, ab_full.index)
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

                # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
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
                        tCtSFB[sf_kblock_coord].iterator,
                    )

                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[kblock_coord],
                        tCrB[kblock_coord],
                        tCtAcc,
                    )
                    # Enable accumulate on tCtAcc after first kblock
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Async arrive AB buffer empty
                ab_full.release()
                
        # Signal that the accumulator is fully computed - only leader CTA
        if is_leader_cta:
            acc_producer.commit()

    #
    # Release TMEM allocation lock
    #
    tmem.relinquish_alloc_permit()

    #
    # Epilogue
    # Partition for epilogue
    #
    op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
    copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    # (T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
    # (T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_gC = thr_copy_t2r.partition_D(tCgC)
    # (T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None].shape, cutlass.Float32
    )
    # (T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None].shape, c_dtype
    )
    # STG Atom
    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)

    # Wait for accumulator buffer full
    acc_full = acc_consumer.wait_and_advance()

    # Copy accumulator to register
    cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
    acc_vec = tTR_rAcc.load().to(c_dtype)
    tTR_rC.store(acc_vec)
    # Store C to global memory
    cute.copy(simt_atom, tTR_rC, tTR_gC)

    acc_full.release()

    # Ensure used buffers are properly synchronized before producer exit
    if warp_idx == 0:
        ab_producer.tail()
        if is_leader_cta:
            acc_producer.tail()

    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(acc_tmem_ptr)

    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel with 2CTA MMA.
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
    b_tensor = cute.make_tensor(
        b_ptr,
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
        b_tensor.shape, sf_vec_size
    )
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

    # Create MMA operation with CtaGroup.TWO for 2CTA
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.TWO,  # 2CTA MMA
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    
    # For SFB: use separate tiled_mma with CtaGroup.ONE and adjusted tile shape
    # This is required because SFB has different layout requirements
    mma_inst_shape_mn_sfb = (
        mma_tiler_mnk[0] // 2,  # Halve M for 2CTA
        cute.round_up(mma_tiler_mnk[1], 128),
    )
    mma_op_sfb = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_inst_shape_mn_sfb[0], mma_inst_shape_mn_sfb[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,  # SFB always uses CtaGroup.ONE
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma_sfb = cute.make_tiled_mma(mma_op_sfb)

    # Construct the VMNK layout for 2CTA
    # This creates a 4D layout where mode 0 is the peer CTA index
    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(
        cta_layout_mnk,
        (tiled_mma.thr_id.shape,),
    )
    # Separate cluster layout for SFB (uses thr_id.shape = 1)
    cta_layout_sfb_vmnk = cute.tiled_divide(
        cta_layout_mnk,
        (tiled_mma_sfb.thr_id.shape,),
    )

    # Compute separate tiler for SFB (adjusted for 2CTA)
    mma_tiler_sfb = (
        mma_inst_shape_mn_sfb[0],
        mma_inst_shape_mn_sfb[1],
        mma_tiler_mnk[2],
    )
    
    # Compute A/B/SFA/SFB/C shared memory layout
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
    # SFB uses separate tiled_mma_sfb and adjusted tiler
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma_sfb,
        mma_tiler_sfb,
        sf_vec_size,
        num_ab_stage,
    )

    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # Setup TMA for A with multicast support
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO),
        a_tensor,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    # Setup TMA for B with multicast support
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO),
        b_tensor,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    # Setup TMA for SFA - use non-multicast since scale factors are small
    # and have different layout structure that doesn't map well to 2CTA multicast
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
        sfa_tensor,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB - uses separate tiled_mma_sfb with CtaGroup.ONE
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        sfb_tensor,
        sfb_smem_layout,
        mma_tiler_sfb,
        tiled_mma_sfb,
        cta_layout_sfb_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # Compute TMA load bytes - account for both CTAs in pair
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
    ) * atom_thr_size

    # Compute grid size - for 2CTA, M is divided by 2 since each MMA tile
    # is computed by 2 CTAs jointly (the accumulator is split in M)
    grid = (
        cute.ceil_div(c_tensor.shape[0], mma_tiler_mnk[0] // 2),  # Divide M by 2 for pair-UMMA
        cute.ceil_div(c_tensor.shape[1], mma_tiler_mnk[1]),
        c_tensor.shape[2],
    )
    
    # Round up grid to cluster size
    grid = cute.round_up(grid, cluster_shape_mnk)

    # Launch the kernel with 2CTA cluster
    kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sfa,
        tma_tensor_sfa,
        tma_atom_sfb,
        tma_tensor_sfb,
        c_tensor,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        cta_layout_vmnk,
        num_tma_load_bytes,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=cluster_shape_mnk,
    )
    return


# Global cache for compiled kernel
_compiled_kernel_cache = None

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
    b_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )

    # Compile the kernel
    _compiled_kernel_cache = cute.compile(my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0))
    
    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMM kernel with 2CTA MMA.
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data
    
    # Ensure kernel is compiled
    compiled_func = compile_kernel()

    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    n, _, _ = b.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2 

    # Create CuTe pointers
    a_ptr = make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c


def generate_input(
    m: int,
    n: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMM.
    """
    torch.manual_seed(seed)
    
    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        -6, 6, (l, m, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    b_ref = torch.randint(
        -6, 6, (l, n, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )
    
    def create_scale_factor_tensors(l, mn, sf_k):
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        ref_f8_random_int = torch.randint(-3, 3, ref_shape, dtype=torch.int8, device='cuda')
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        mma_permute_order = (3, 4, 1, 5, 2, 0)
        rand_int_tensor = torch.randint(-3, 3, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')
        
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')
        
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k
        
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]
        
        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_ref_permuted = create_scale_factor_tensors(l, n, sf_k)

    return (a_ref, b_ref, sfa_ref_cpu.to("cuda"), sfb_ref_cpu.to("cuda"), sfa_ref_permuted, sfb_ref_permuted, c_ref)


check_implementation = make_match_reference(custom_kernel, rtol=1e-03, atol=1e-03)

