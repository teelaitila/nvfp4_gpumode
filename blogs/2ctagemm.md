2 CTA GEMM on B200

04 Jan, 2026

Blackwell GPUs give the User the ability to perform UMMA operation on 2 CTAs (i.e. Thread blocks). I'll focus on the difference between 1 CTA and 2 CTA case. For general structure of Blackwell Gemm there are various examples in the CuTeDSL repository with focus on Blackwell. For simplicity I choose to not go with Multicast in TMA to isolate the adjustments needed for 2 CTA Gemm launch. For guidance on how to implement multicasted approach you can see the existing examples in the Cutlass repo.

Comparison

Setup

We configure our Gemm class like this:

class Gemm:
    def __init__(self):
        self.ab_dtype = cutlass.BFloat16
        self.c_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        self.mma_tiler_mnk = (128, 256, 64)  # Tiler for the MMA
        self.mma_inst_shape_mnk = (
            128,
            256,
            16,
        )  # UMMA Shape (MMA_INST_M, MMA_INST_N, MMA_INST_K)
        self.cluster_shape_mn = (1, 1)  # Cluster Shape for CTA
        self.threads_per_cta = 128  # Need 128 Threads for Epilogue

        self.ab_stages = 4  # TMA-Umma Pipeline
        self.acc_stages = 1  # Umma-Store Pipeline
As you can read in Table 39 of the PTX docs the instruction shape we chose is the largest possible instruction shape for an UMMA op with 1 CTA.

For the 2 CTA case we would adjust as follows:

class Gemm:
    def __init__(self):
        self.ab_dtype = cutlass.BFloat16
        self.c_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        self.mma_tiler_mnk = (256, 256, 64)  # Tiler for the MMA
        self.mma_inst_shape_mnk = (
            256,  # Up to 256
            128,
            16,
        )  # UMMA Shape (MMA_INST_M, MMA_INST_N, MMA_INST_K)
        self.cluster_shape_mn = (
            2,
            1,
        )  # Cluster Shape for CTA -> Increase to 2 for 2 CTA
        self.threads_per_cta = 128  # Need 128 Threads for Epilogue

        self.ab_stages = 4  # TMA-Umma Pipeline
        self.acc_stages = 1  # Umma-Store Pipeline
Note that we increase instruction shape in M mode by a factor of 2 because 2 CTA UMMA can handle larger tiles. Also we use a cluster in M mode where each cluster will than contain the 2 CTAs which collaborate on one UMMA.

In CuTeDSL we configure UMMA as follows:

tiled_mma = sm100_utils.make_trivial_tiled_mma(
	self.ab_dtype,
	OperandMajorMode.K,
	OperandMajorMode.K,
	self.acc_dtype,
	CtaGroup.ONE,
	self.mma_tiler_mnk[:2],
	OperandSource.SMEM,
)
The object looks as follows:

Tiled MMA
  Thr Layout VMNK: (1,1,1,1):(0,0,0,0)
  Permutation MNK: (_,_,_)
MMA Atom
  ThrID:           1:0
  Shape MNK:       (128,256,16)
  TV Layout A:     (1,(128,16)):(128,(1,128))
  TV Layout B:     (1,(256,16)):(256,(1,256))
  TV Layout C:     (1,(128,256)):(128,(1,128))
For the 2 CTA case we need to make an obvious adjustment:

tiled_mma = sm100_utils.make_trivial_tiled_mma(
	self.ab_dtype,
	OperandMajorMode.K,
	OperandMajorMode.K,
	self.acc_dtype,
	CtaGroup.TWO,
	self.mma_tiler_mnk[:2],
	OperandSource.SMEM,
)
This will than give us

Tiled MMA
  Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
  Permutation MNK: (_,_,_)
MMA Atom
  ThrID:           2:1
  Shape MNK:       (256,256,16)
  TV Layout A:     (2,(128,16)):(128,(1,256))
  TV Layout B:     (2,(128,16)):(128,(1,256))
  TV Layout C:     (2,(128,256)):(128,(1,256))
We see that we have the same Value Layouts as above. However here we have a 2 in the first component. This can be interpreted as each of the two CTAs holding half of the mma_tile.

Consequently we need to adjust the CTA tile which will be used to compute the grid

self.cta_tile_shape_mnk = (
	self.mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
	self.mma_tiler_mnk[1],
	self.mma_tiler_mnk[2],
)  # Rescale by factor of 2 in M dimension.
The SMEM Layouts we obtain from sm100_utils.make_smem_layout_a stay therefore as they are.

We can calculate the cluster_layout_vmnk as

self.cluster_layout_vmnk = cute.tiled_divide(
	cute.make_layout((*self.cluster_shape_mn, 1)),  
	(tiled_mma.thr_id.shape,),  # 1
) 
Note that in case of 2 CTA this will give us ((2), 1, 1, 1):((1), 0, 0, 0) compared to ((1), 1, 1, 1):((0), 0, 0, 0) in 1 CTA case.

Next we can either hardcode the TMA operation or use cluster_shape_to_tma_atom_A from the blackwell utilities the CuTeDSL provides. I did that.

The code is as follows:

@dsl_user_op
def cluster_shape_to_tma_atom_A(
    cluster_shape_mnk: cute.Shape, atom_thr_id: cute.Layout, *, loc=None, ip=None
) -> Union[CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp]:
	atom_sm_cnt = cute.size(atom_thr_id, loc=loc, ip=ip)
	mcast = not (cute.size(cluster_shape_mnk, mode=[1], loc=loc, ip=ip) == 1)
	cluster_size = cute.size(cluster_shape_mnk, loc=loc, ip=ip)
	
	if not isinstance(cluster_size, int) or not isinstance(atom_sm_cnt, int):
		raise ValueError(
			f"Dynamic cluster shape or atom SM count is not supported: {cluster_shape_mnk} and {atom_thr_id}"
		)
	
	if cute.size(cluster_shape_mnk, mode=[0], loc=loc, ip=ip) % atom_sm_cnt != 0:
		raise ValueError(
			f"Cluster shape not divisible by MMA size: {cluster_shape_mnk} and {atom_thr_id}"
		)
	
	if atom_sm_cnt == 2 and mcast:
		return CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)
	elif atom_sm_cnt == 2 and not mcast:
		return CopyBulkTensorTileG2SOp(CtaGroup.TWO)
	elif atom_sm_cnt == 1 and mcast:
		return CopyBulkTensorTileG2SMulticastOp(CtaGroup.ONE)
	elif atom_sm_cnt == 1 and not mcast:
		return CopyBulkTensorTileG2SOp(CtaGroup.ONE)
We see that we will choose the non multicasted operation in both cases. In the 2 CTA case we provide it with the Cta Group. You can read about the qualifier in the corresponding section of PTX docs.

A further adjustment we need to make is to join the kernel with the cluster and pass cluster_layout_vmnk to the kernel.

self.kernel(
	tiled_mma,
	tma_atom_a,
	tma_tensor_a,
	tma_atom_b,
	tma_tensor_b,
	c,
	a_smem_layout,
	b_smem_layout,
	self.cluster_layout_vmnk,  # Pass Cluster
).launch(
	grid=grid,
	block=(self.threads_per_cta, 1, 1),
	cluster=(*self.cluster_shape_mn, 1),  # Add Cluster
)
Note that so far we only had to make a few adjustments:

Increase MMA Tiler M mode
Increase Cluster M mode
Rescale CTA Layout
Pass Cluster Layout as kernel argument and launch with kernel with cluster
Prologue in Kernel

Inside the kernel we need to adjust calculation of our coordinates as follows:

mma_coord_mnk = (bidx, bidy, None)
to

mma_coord_vmnk = (
	bidx % cute.size(cta_layout_vmnk, mode=[0]),  # Either 0 or 1
	bidx // cute.size(cta_layout_vmnk, mode=[0]),  # 0,..,bM/2
	bidy,
	None,
)
mma_coord_mnk = mma_coord_vmnk[1:]
Note that we reorder the coordinates such that we have the CTA dim (i.e. V) as the fastest changing mode. We will see below how the V mode is used in slicing and getting the work for each Thread.

The code snipped below shows to further adjustments we need to do:

Allocate an additional MBarrier for deallocation of TMEM in 2 CTA case
Increase the number of tma bytes. Per UMMA we have double the tile size within one CTA now.
@cute.struct
class SharedStorage:
	ab_mbar_ptr: cute.struct.MemRange[
		cutlass.Int64, self.ab_stages * 2
	]  # Empty/Full TMA <-> UMMA
	acc_mbar_ptr: cute.struct.MemRange[
		cutlass.Int64, self.acc_stages * 2
	]  # Empty/Full UMMA <-> Store
	tmem_holding_buf: cutlass.Int32  # Tmem addr in SMEM
	tmem_dealloc_mbar_ptr: cutlass.Int64  # Needed for 2 CTA

storage = smem.allocate(SharedStorage)

num_tma_copy_bytes = cute.size_in_bytes(
	self.ab_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
) + cute.size_in_bytes(
	self.ab_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
)  # Add bytes for copy of one stage
num_tma_copy_bytes *= cute.size(
	cta_layout_vmnk, mode=[0]
)  # Double because 2 CTA
Within the two Pipelines (for TMA <-> Umma and Umma <-> Store) there are two more adjustments (apart from increase of num_tma_copy_bytes) we make: We provide the cta_layout_vmnk to each of the pipelines. We'll furthermore need to increase the size of the acc consumer group. That is because now we have 256 threads, each of them storing one row of the output tile.

In the TmemAllocator we prodvide two new arguments for two cta cases. Note that we also must include cute.arch.cluster_arrive() for the 2 CTA case as an additional fence after we allocated all mbarriers.

tmem = cutlass.utils.TmemAllocator(
	alloc_result_dst_smem_ptr=storage.tmem_holding_buf,
	barrier_for_retrieve=tmem_alloc_barrier,
	is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
	two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
)

cute.arch.mbarrier_init_fence()  # Fence after Mbarrier
cute.arch.cluster_arrive()  # Arrive on Cluster
There are only a few more adjustments in the Prologue we need to make:

thr_mma = tiled_mma.get_slice(0) -> thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0]), this is clear because above we saw that Thread dimension in above TV Layouts for MMA corresponds to number of CTAs. And we explicitly constructed the V mode of the coordinate such that it is exactly in the range of the number of CTAs.
Replace cute.arch.sync_threads() by cute.arch.cluster_wait() before TMEM alloc.
And that's already it for the Prologue. To summarise we needed to adjust:

Coordinate needs additional V dimension.
Additional TMEM mbar for deallocation
Multiply number of TMA copy bytes by 2
Tmem allocator needs to be provided with pointer for deallocation of TMEM
After usual mbarrier fence a cluster_arrive, before TMEM alloc an cluster_wait where we had sync_threads. Note that these two changes could be saved by using the pipeline_init{arrive|wait} abstraction in CuTe which wraps these up.
Mainloop in Kernel

The adjustments in the mainloop are surprisingly simple.

Before starting MMA calculation we need to determine if we are currently on the Leader by using is_leader_cta = mma_coord_vmnk[0] == 0.

We'll than only execute all UMMA related logic if this is the case:

if is_leader_cta:
	acc_producer.acquire_and_advance()
if is_leader_cta:
	ab_full = (
		ab_consumer.wait_and_advance()
	)  # Atomically wait for data and advance to next pipeline stage.
	# MMA
	num_k_blocks = cute.size(tCrA, mode=[2])
	for k_block in cutlass.range_constexpr(num_k_blocks):
		k_block_coord = (
			None,  # MMA
			None,  # MMA_{M/N}
			k_block,  # MMA_K
			ab_full.index,  # STAGE
		)
		cute.gemm(
			tiled_mma,
			tCtAcc,
			tCrA[k_block_coord],
			tCrB[k_block_coord],
			tCtAcc,
		)
		tiled_mma.set(tcgen05.Field.ACCUMULATE, True)  # Enable Accum

	# Tile done, release lock
	ab_full.release()
# Commit processed tile to epilogue for Store for Leader
if is_leader_cta:
	acc_producer.commit()
And thats it with adjustments of the Mainloop.

Epilogue

In the Epilogue there are no adjustments to be made in the main logic. We already adjusted the number of consumer threads for the Acc Pipeline above. The only additional step that is needed is a bookkeeping one to clean up the Pipelines:

        if warp_idx == 0:
            ab_producer.tail()  # Cleanup
            if is_leader_cta:
                acc_producer.tail()  # Cleanup from leader CTA
Performance

I benchmarked three cases using the code below with m, n, k = 8192, 8192, 8192.

def benchmark(callable, a_tensor, b_tensor, c_tensor):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_tensor, b_tensor, c_tensor),
        warmup_iterations=100,
        iterations=1000,
    )

    # Calculate metrics

    # Calculate total float ops calculated:
    # - M * N * K * 2 (FMA)
    total_float_ops = m * n * k * 2

    # Calculate achieved TFlops
    achieved_tflops = total_float_ops / (avg_time_us * 1000000)  # TFlops

    # Print results
    # ------------
    print("Performance Metrics:")
    print("-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_tflops:.2f} tflops")


def run(m: int, n: int, k: int):
    gemm = Gemm()

    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    a = make_tensors(m, k, cutlass_torch.dtype(gemm.ab_dtype))
    b = make_tensors(n, k, cutlass_torch.dtype(gemm.ab_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(gemm.c_dtype))

    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )  # K-Major
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )  # K-Major
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )  # N-Major

    gemm_compiled = cute.compile(gemm, a_tensor, b_tensor, c_tensor)
    gemm_compiled(a_tensor, b_tensor, c_tensor)

    benchmark(gemm_compiled, a_tensor, b_tensor, c_tensor)
    ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()
    torch.testing.assert_close(
        c.cpu(), ref.to(cutlass_torch.dtype(gemm.c_dtype)), atol=1e-01, rtol=1e-05
    )
    print("PASS")
1 CTA: 1323.19 tflops
2 CTA without multicast: 1399.98 tflops
2 CTA with multicast: 1396.41 tflops
This shows that we get a good boost from employing the 2CTA feature. Multicast is not needed for our simple example. Note that to archive best performance we could further adjust to use TMA for the store (right now we employ direct store from RMEM -> GMEM after we loaded from TMEM -> RMEM), implement a persistent tile scheduling etc.. Here I focused on writing simple GEMM baseline to study the 2 CTA feature more in depth.

Conclusion

We saw that we can quiet simply turn a 1 CTA kernel into a 2 CTA kernel in CuTeDSL. The experiments were performed on Verda, please check out their site if you want to program on Blackwell GPUs. If you like to exchange ideas you can contact me on Linkedin.