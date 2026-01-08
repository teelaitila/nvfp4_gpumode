Blackwell Pipelining with CuTeDSL

23 Dec, 2025

Introduction

Blackwell Pipelining with CuTeDSL When writing code for modern GPU architectures one frequently can overlap certain workloads due to the asynchronous instructions we can leverage. For example it is commonly known from the Hopper architecture that we can overlap TMA (i.e. memory transfer) with MMA (i.e. computation). On Blackwell we can introduce another level of overlapping namely we can now have TMA -> MMA -> Epilogue overlap. In this blogpost I show how it is done in CuTeDSL.

Pipelines

In persistent blockscaled example to synchronize between different asynchronous workloads we use multiple pipelines namely:

AB Pipeline

ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
ab_pipeline_consumer_group = pipeline.CooperativeGroup(
	pipeline.Agent.Thread, num_tma_producer
)
ab_pipeline = pipeline.PipelineTmaUmma.create(
	barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
	num_stages=self.num_ab_stage,
	producer_group=ab_pipeline_producer_group,
	consumer_group=ab_pipeline_consumer_group,
	tx_count=self.num_tma_load_bytes,
	cta_layout_vmnk=cluster_layout_vmnk,
	defer_sync=True,
)
Acc Pipeline

acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
num_acc_consumer_threads = len(self.epilog_warp_id) * (
	2 if use_2cta_instrs else 1
)
acc_pipeline_consumer_group = pipeline.CooperativeGroup(
	pipeline.Agent.Thread, num_acc_consumer_threads
)
acc_pipeline = pipeline.PipelineUmmaAsync.create(
	barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
	num_stages=self.num_acc_stage,
	producer_group=acc_pipeline_producer_group,
	consumer_group=acc_pipeline_consumer_group,
	cta_layout_vmnk=cluster_layout_vmnk,
	defer_sync=True,
)
C Pipeline

# Threads/warps participating in tma store pipeline
c_producer_group = pipeline.CooperativeGroup(
	pipeline.Agent.Thread,
	32 * len(self.epilog_warp_id),
)
c_pipeline = pipeline.PipelineTmaStore.create(
	num_stages=self.num_c_stage,
	producer_group=c_producer_group,
)
Blackwell can overlap the following processes:

Loading Tiles from GMEM to SMEM via TMA and computation of the MMA into TMEM (this is job of AB Pipeline)
Computation of the MMA into TMEM and storing of the results to RMEM (this is job of the Acc Pipeline)
Potentially we can introduce another level of asynchronous operation which is the job of the C Pipeline (i.e. we can store the result in a staged way by issuing a copy and immediately continuing with the next tile while the copy is busy) This approach lets us overlap independent operations: loading the next tile while tensor cores compute the current one, or storing a completed result while tensor cores have already moved on to the next.
PipelineTmaUmma

For this Pipeline the producer is the TMA and the consumer is the Umma as we can read here

class PipelineTmaUmma(PipelineAsync):
    """
    PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).
    """
The docstring of PipelineAsync contains valuable table:

+-----------+-----------+-----------+-----------+-----------+-----------+
| Barrier   | State     | p.acquire | p.commit  | c.wait    | c.release |
+===========+===========+===========+===========+===========+===========+
| empty_bar | empty     | <Return>  | n/a       | n/a       | -         |
+-----------+-----------+-----------+-----------+-----------+-----------+
| empty_bar | wait      | <Block>   | n/a       | n/a       | -> empty  |
+-----------+-----------+-----------+-----------+-----------+-----------+
| full_bar  | wait      | n/a       | -> full   | <Block >  | n/a       |
+-----------+-----------+-----------+-----------+-----------+-----------+
| full_bar  | full      | n/a       | -         | <Return>  | n/a       |
+-----------+-----------+-----------+-----------+-----------+-----------+

Where:

- p: producer
- c: consumer
- <Block>: This action is blocked until transition to a state allow it to proceed by other side
- e.g. ``p.acquire()`` is blocked until ``empty_bar`` transition to ``empty`` state by ``c.release()``
which we will frequently use in our analysis below. Come back to it if you feel lost.

We initialize their corresponding CooperativeGroups as follows:

ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
ab_pipeline_consumer_group = pipeline.CooperativeGroup(
	pipeline.Agent.Thread, num_tma_producer
)
We can learn about parameters for CooperativeGroup in the corresponding code:

class CooperativeGroup:
	...
    def __init__(self, agent: Agent, size: int = 1, alignment=None):
        if alignment is not None:
		...
        # Size indicates how many threads are participating in this CooperativeGroup
        self.size = size
        # Agent indicates the type of thread group
        self.agent = agent
We see that producer group only has one thread participating in the CooperativeGroup. In the consumer we may have multiple threads in the CooperativeGroup depending if we use multicast for TMA operation.

ab_pipeline = pipeline.PipelineTmaUmma.create(
	barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
	num_stages=self.num_ab_stage,
	producer_group=ab_pipeline_producer_group,
	consumer_group=ab_pipeline_consumer_group,
	tx_count=self.num_tma_load_bytes,
	cta_layout_vmnk=cluster_layout_vmnk,
	defer_sync=True,
)
We initialize the corresponding pipeline with the mbar pointer, the number of stages, the above defined groups, the total number of bytes we load with the TMA and the layout for the CTA. Note that we only provide the full bar pointer and the empty bar pointer is inferred from that by shifting by the number of stages. Two barriers are needed for proper synchronization as you can infer from the diagram above.

sync_object_full = PipelineAsync._make_sync_object(
	barrier_storage.align(min_align=8), num_stages, producer
)
sync_object_empty = PipelineAsync._make_sync_object(
	barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
)
Note that based on the PipelineOp we will select appropriate operations. See the relevant code:

# pipeline/sm100.py
####
producer_type = PipelineOp.TmaLoad
consumer_type = PipelineOp.TCGen05Mma

producer = (producer_type, producer_group)
consumer = (consumer_type, consumer_group)
####

# pipeline/helpers.py
####
if self.op_type is PipelineOp.AsyncThread:
	self.arrive_mbarrier(index, dst, loc=loc, ip=ip)
elif self.op_type is PipelineOp.TCGen05Mma:
	assert cta_group is not None, (
		"Error: CTA group must be provided for TCGen05Mma."
	)
	self.arrive_tcgen05mma(index, dst, cta_group, loc=loc, ip=ip)
elif self.op_type in [PipelineOp.TmaLoad]:
	self.arrive_and_expect_tx(index, self.tx_count)
elif self.op_type is PipelineOp.AsyncLoad:
	self.arrive_cp_async_mbarrier(index, loc=loc, ip=ip)
else:
	assert False, (
		f"Error: MbarrierArray is not supported for PipelineOp: {_get_pipeline_op(self.op_type)}."
	)
####
We can see distinction between the two arrive ops

TMA uses arrive op arrive and expect.

def arrive_and_expect_tx(self, index: int, tx_count: int) -> None:
	with cute.arch.elect_one():
		cute.arch.mbarrier_arrive_and_expect_tx(self.get_barrier(index), tx_count)
The producer on the other hand will use

def arrive_tcgen05mma(
	self,
	index: int,
	mask: Optional[int],
	cta_group: cute.nvgpu.tcgen05.CtaGroup,
	*,
	loc=None,
	ip=None,
) -> None:
	if mask is None:
		with cute.arch.elect_one(loc=loc, ip=ip):
			cute.nvgpu.tcgen05.commit(
				self.get_barrier(index, loc=loc, ip=ip), loc=loc, ip=ip
			)
	else:
		with cute.arch.elect_one(loc=loc, ip=ip):
			cute.nvgpu.tcgen05.commit(
				self.get_barrier(index, loc=loc, ip=ip),
				mask,
				cta_group,
				loc=loc,
				ip=ip,
			)
To communicate arrive.

The number of TMA load bytes is defined above and simply counts all the bytes we transfer within one stage (see the slice_ operation on the staged layout below).

a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
...
b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
...
sfa_smem_layout = cute.slice_(
	self.sfa_smem_layout_staged, (None, None, None, 0)
)
...
sfb_smem_layout = cute.slice_(
	self.sfb_smem_layout_staged, (None, None, None, 0)
)
...
a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
self.num_tma_load_bytes = (
	a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
) * atom_thr_size
Let us now see the whole pipeline in action and its usage.

if warp_idx == self.tma_warp_id:
	#
	# Persistent tile scheduling loop
	#
	tile_sched = utils.StaticPersistentTileScheduler.create(
		tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
	)
	work_tile = tile_sched.initial_work_tile_info()

	ab_producer_state = pipeline.make_pipeline_state(
		pipeline.PipelineUserType.Producer, self.num_ab_stage
	)

	while work_tile.is_valid_tile:
		# Get tile coord from tile scheduler
		cur_tile_coord = work_tile.tile_idx
		mma_tile_coord_mnl = (
			cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
			cur_tile_coord[1],
			cur_tile_coord[2],
		)

		#
		# Slice to per mma tile index
		#
		# ((atom_v, rest_v), RestK)
		tAgA_slice = tAgA[
			(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
		]
		# ((atom_v, rest_v), RestK)
		tBgB_slice = tBgB[
			(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
		]

		# ((atom_v, rest_v), RestK)
		tAgSFA_slice = tAgSFA[
			(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
		]

		slice_n = mma_tile_coord_mnl[1]
		if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
			slice_n = mma_tile_coord_mnl[1] // 2
		# ((atom_v, rest_v), RestK)
		tBgSFB_slice = tBgSFB[
			(None, slice_n, None, mma_tile_coord_mnl[2])
		]

		# Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
		ab_producer_state.reset_count()
		peek_ab_empty_status = cutlass.Boolean(1)
		if ab_producer_state.count < k_tile_cnt:
			peek_ab_empty_status = ab_pipeline.producer_try_acquire(
				ab_producer_state
			)
		#
		# Tma load loop
		#
		for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
			# Conditionally wait for AB buffer empty
			ab_pipeline.producer_acquire(
				ab_producer_state, peek_ab_empty_status
			)

			# TMA load A/B/SFA/SFB
			cute.copy(
				tma_atom_a,
				tAgA_slice[(None, ab_producer_state.count)],
				tAsA[(None, ab_producer_state.index)],
				tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
				mcast_mask=a_full_mcast_mask,
			)
			cute.copy(
				tma_atom_b,
				tBgB_slice[(None, ab_producer_state.count)],
				tBsB[(None, ab_producer_state.index)],
				tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
				mcast_mask=b_full_mcast_mask,
			)
			cute.copy(
				tma_atom_sfa,
				tAgSFA_slice[(None, ab_producer_state.count)],
				tAsSFA[(None, ab_producer_state.index)],
				tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
				mcast_mask=sfa_full_mcast_mask,
			)
			cute.copy(
				tma_atom_sfb,
				tBgSFB_slice[(None, ab_producer_state.count)],
				tBsSFB[(None, ab_producer_state.index)],
				tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
				mcast_mask=sfb_full_mcast_mask,
			)

			# Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
			ab_producer_state.advance()
			peek_ab_empty_status = cutlass.Boolean(1)
			if ab_producer_state.count < k_tile_cnt:
				peek_ab_empty_status = ab_pipeline.producer_try_acquire(
					ab_producer_state
				)

		#
		# Advance to next tile
		#
		tile_sched.advance_to_next_work()
		work_tile = tile_sched.get_current_work()

	#
	# Wait A/B buffer empty
	#
	ab_pipeline.producer_tail(ab_producer_state)
Before start to copy the tiles we initialize the pipeline state.

ab_producer_state = pipeline.make_pipeline_state(
	pipeline.PipelineUserType.Producer, self.num_ab_stage
)
This will create a Pipeline state:

PipelineState(
	stages,
	Int32(0, loc=loc, ip=ip),
	Int32(0, loc=loc, ip=ip),
	Int32(1, loc=loc, ip=ip),
)
with the given number of stages, the count and index equal to zero and the phase bit flipped on. The index will keep track of the current index in the circular buffer. Once the end of the buffer is reached we start from the beginning again. Once the end of the buffer is reached we start from the beginning again in a circular fashion.

We'll than start to process all valid tiles for the current SM. Note that the tile scheduler will give us information which tile in M and N direction we are currently processing. The K dimension needs to be looped over. Before we start doing so we perform a producer_try_aquire.

while work_tile.is_valid_tile:
	# Get coordinate for current tile, slice the tensor in GMEM
	# to get the current tile in M and N direction 
	# for each of the tensors we want 
	# to transfer via TMA onto SMEM
	...

	# Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
	ab_producer_state.reset_count()
	peek_ab_empty_status = cutlass.Boolean(1)
	if ab_producer_state.count < k_tile_cnt:
		peek_ab_empty_status = ab_pipeline.producer_try_acquire(
			ab_producer_state
		)
producer_try_aquire will simply return boolean telling us if the current position in the buffer is available.

We are now processing the K tiles. We start with producer_aquire. After being done with the copies (which we provide with the barrier for the current state) we advance the states and get the status of the next position in the buffer.

		# Tma load loop
		#
		for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
			# Conditionally wait for AB buffer empty
			ab_pipeline.producer_acquire(
				ab_producer_state, peek_ab_empty_status
			)

			# TMA load A/B/SFA/SFB
			cute.copy(
				tma_atom_a,
				tAgA_slice[(None, ab_producer_state.count)],
				tAsA[(None, ab_producer_state.index)],
				tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
				mcast_mask=a_full_mcast_mask,
			)
			...

			# Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
			ab_producer_state.advance()
			peek_ab_empty_status = cutlass.Boolean(1)
			if ab_producer_state.count < k_tile_cnt:
				peek_ab_empty_status = ab_pipeline.producer_try_acquire(
					ab_producer_state
				)
producer_aquire will be blocking operation until the consumer releases the corresponding section of the circular buffer. Under the hood it will look like this:
def producer_acquire(
	self, state: PipelineState, try_acquire_token: Optional[Boolean] = None
):
	"""
	TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
	"""
	if_generate(
		try_acquire_token is None or try_acquire_token == 0,
		lambda: self.sync_object_empty.wait(state.index, state.phase),
	)
	if_generate(
		self.is_leader_cta,
		lambda: self.sync_object_full.arrive(state.index, self.producer_mask),
	)
copy will internally signal via tma_bar_ptr the producer when it's finished by arriving on its barrier.
advance will increase count, flip phase and increase index with warp around.
At the end we (potentially) prefetch like above. Note we may schedule multiple copies via this mechanism. As long as the producer is not busy on a position in the circular buffer we can keep running through that until we reach a block.
	#
	# Advance to next tile
	#
	tile_sched.advance_to_next_work()
	work_tile = tile_sched.get_current_work()

#
# Wait A/B buffer empty
#
ab_pipeline.producer_tail(ab_producer_state)
At the end this is some bookkeeping to avoid dangling pointers.

We know analysed the Producer part of the ab_pipeline. Let us check how it's used in the consumer.

Below I stripped down the MMA section to only contain the parts relevant for the current pipeline.

# Specialized MMA warp
#
if warp_idx == self.mma_warp_id:
	# Setup tensors in TMEM, tile scheduler etc
	...
	ab_consumer_state = pipeline.make_pipeline_state(
		pipeline.PipelineUserType.Consumer, self.num_ab_stage
	)

	while work_tile.is_valid_tile:
		# Get tile coord from tile scheduler
		cur_tile_coord = work_tile.tile_idx
		mma_tile_coord_mnl = (
			cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
			cur_tile_coord[1],
			cur_tile_coord[2],
		)

		# Set TMEM buffer for current tile

		# Peek (try_wait) AB buffer full for k_tile = 0
		ab_consumer_state.reset_count()
		peek_ab_full_status = cutlass.Boolean(1)
		if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
			peek_ab_full_status = ab_pipeline.consumer_try_wait(
				ab_consumer_state
			)
		#
		# Wait for accumulator buffer empty
		#
		...
		# Handling of special cases, clear accumulator for MMA
		...
		#
		# Mma mainloop
		#
		for k_tile in range(k_tile_cnt):
			if is_leader_cta:
				# Conditionally wait for AB buffer full
				ab_pipeline.consumer_wait(
					ab_consumer_state, peek_ab_full_status
				)

				#  Copy SFA/SFB from smem to tmem
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

				# tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
				num_kblocks = cute.size(tCrA, mode=[2])
				for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
					kblock_coord = (
						None,
						None,
						kblock_idx,
						ab_consumer_state.index,
					)

					# Set SFA/SFB tensor to tiled_mma
					sf_kblock_coord = (None, None, kblock_idx)
					tiled_mma.set(
						tcgen05.Field.SFA,
						tCtSFA[sf_kblock_coord].iterator,
					)
					tiled_mma.set(
						tcgen05.Field.SFB,
						tCtSFB_mma[sf_kblock_coord].iterator,
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
				ab_pipeline.consumer_release(ab_consumer_state)

			# Peek (try_wait) AB buffer full for k_tile = k_tile + 1
			ab_consumer_state.advance()
			peek_ab_full_status = cutlass.Boolean(1)
			if ab_consumer_state.count < k_tile_cnt:
				if is_leader_cta:
					peek_ab_full_status = ab_pipeline.consumer_try_wait(
						ab_consumer_state
					)

	...
We see that the pattern mirrors the producer logic closely.

We first perform initialization

ab_consumer_state = pipeline.make_pipeline_state(
	pipeline.PipelineUserType.Consumer, self.num_ab_stage
)
PipelineState(
	stages,
	Int32(0, loc=loc, ip=ip),
	Int32(0, loc=loc, ip=ip),
	Int32(0, loc=loc, ip=ip),
)
It is kind of obvious that the phase bit is flipped off initially because we first need to produce something before we can consume it.

We'll than perform same non blocking check if next slot is free

# Peek (try_wait) AB buffer full for k_tile = 0
		ab_consumer_state.reset_count()
		peek_ab_full_status = cutlass.Boolean(1)
		if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
			peek_ab_full_status = ab_pipeline.consumer_try_wait(
				ab_consumer_state
			)
and than within the loop and for the leader CTA we do:

ab_pipeline.consumer_wait(
	ab_consumer_state, peek_ab_full_status
)
which will wait be blocked until the TMA copy from above finishes and we have transferred the tile for current position in the circular buffer to SMEM.

We'll than perform the MMA op and after doing so signalling the producer via

ab_pipeline.consumer_release(ab_consumer_state)
that the current slot of the circular buffer can be used for loading a new tile. Note that we must wait for the MMA to finish to perform the release because:

# (MMA, MMA_M, MMA_K, STAGE)
tCrA = tiled_mma.make_fragment_A(sA)
# (MMA, MMA_N, MMA_K, STAGE)
tCrB = tiled_mma.make_fragment_B(sB)
After that we make a peek again to perform non blocking lookup for the next position in the circular buffer.

That's it with analysis of the first Pipeline. We have seen that it is necessary to properly synchronize between the TMA Loads for the MMA inputs and the MMA operation (which uses them).

PipelineUmmaAsync

From the init of pipeline we can identify producer and consumer of this pipeline:

# Initialize acc_pipeline (barrier) and states
acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
num_acc_consumer_threads = len(self.epilog_warp_id) * (
	2 if use_2cta_instrs else 1
)
acc_pipeline_consumer_group = pipeline.CooperativeGroup(
	pipeline.Agent.Thread, num_acc_consumer_threads
)
acc_pipeline = pipeline.PipelineUmmaAsync.create(
	barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
	num_stages=self.num_acc_stage,
	producer_group=acc_pipeline_producer_group,
	consumer_group=acc_pipeline_consumer_group,
	cta_layout_vmnk=cluster_layout_vmnk,
	defer_sync=True,
)
Producer will be UMMA and consumer will be the epilogue (note that each warp uses respectively 1 or 2 threads depending on the fact if we use 2 CTA instruction).

We can also read that off from the implementation inside the create

producer_type = PipelineOp.TCGen05Mma
consumer_type = PipelineOp.AsyncThread

producer = (producer_type, producer_group)
consumer = (consumer_type, consumer_group)
Let us see this pipeline in action:

#
# Specialized MMA warp
#
if warp_idx == self.mma_warp_id:
	...
	acc_producer_state = pipeline.make_pipeline_state(
		pipeline.PipelineUserType.Producer, self.num_acc_stage
	)

	while work_tile.is_valid_tile:
		# Get tile coord from tile scheduler
		cur_tile_coord = work_tile.tile_idx
		mma_tile_coord_mnl = (
			cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
			cur_tile_coord[1],
			cur_tile_coord[2],
		)

		# Get accumulator stage index
		if cutlass.const_expr(self.overlapping_accum):
			acc_stage_index = acc_producer_state.phase ^ 1
		else:
			acc_stage_index = acc_producer_state.index

		# Set tensor memory buffer for current tile
		# (MMA, MMA_M, MMA_N)
		tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

		...
		#
		# Wait for accumulator buffer empty
		#
		if is_leader_cta:
			acc_pipeline.producer_acquire(acc_producer_state)

		...

		#
		# Reset the ACCUMULATE field for each tile
		#
		tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

		#
		# Mma mainloop
		#
		...

		#
		# Async arrive accumulator buffer full
		#
		if is_leader_cta:
			acc_pipeline.producer_commit(acc_producer_state)
		acc_producer_state.advance()

		#
		# Advance to next tile
		#
		tile_sched.advance_to_next_work()
		work_tile = tile_sched.get_current_work()

	#
	# Wait for accumulator buffer empty
	#
	acc_pipeline.producer_tail(acc_producer_state)
We see that pipelining logic is similar to the above example. We have initially a producer_acquire on the leader CTA. This be blocking in case that the current stage is not marked as empty.

We'll than perform MMA operation and producer_commit to signal we are ready with this operation. This tells the consumer that the accumulated result is ready and corresponding result can be stored.

At the end we have again bookkeeping producer_tail to avoid dangling pointers and clean up after our workload is finished.

Let's look at corresponding consumer logic

# Specialized epilogue warps
#
if warp_idx < self.mma_warp_id:
	...

	acc_consumer_state = pipeline.make_pipeline_state(
		pipeline.PipelineUserType.Consumer, self.num_acc_stage
	)

	...

	while work_tile.is_valid_tile:
		# Get tile coord from tile scheduler
		cur_tile_coord = work_tile.tile_idx
		mma_tile_coord_mnl = (
			cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
			cur_tile_coord[1],
			cur_tile_coord[2],
		)

		#
		# Slice to per mma tile index
		#
		# ((ATOM_V, REST_V), EPI_M, EPI_N)
		bSG_gC = bSG_gC_partitioned[
			(
				None,
				None,
				None,
				*mma_tile_coord_mnl,
			)
		]

		# Get accumulator stage index
		if cutlass.const_expr(self.overlapping_accum):
			acc_stage_index = acc_consumer_state.phase
			reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
		else:
			acc_stage_index = acc_consumer_state.index

		# Set tensor memory buffer for current tile
		# (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
		tTR_tAcc = tTR_tAcc_base[
			(None, None, None, None, None, acc_stage_index)
		]

		#
		# Wait for accumulator buffer full
		#
		acc_pipeline.consumer_wait(acc_consumer_state)

		tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
		bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

		#
		# Store accumulator to global memory in subtiles
		#
		subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
		num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
		for subtile_idx in cutlass.range(subtile_cnt):
			real_subtile_idx = subtile_idx
			if cutlass.const_expr(self.overlapping_accum):
				if reverse_subtile:
					real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
			#
			# Load accumulator from tensor memory buffer to register
			#
			tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
			cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

			#
			# Async arrive accumulator buffer empty ealier when overlapping_accum is enabled
			#
			if cutlass.const_expr(self.overlapping_accum):
				if subtile_idx == self.iter_acc_early_release_in_epilogue:
					# Fence for TMEM load
					cute.arch.fence_view_async_tmem_load()
					with cute.arch.elect_one():
						acc_pipeline.consumer_release(acc_consumer_state)
					acc_consumer_state.advance()

			#
			# Convert to C type
			#
			acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
			acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
			tRS_rC.store(acc_vec)

			#
			# Store C to shared memory
			#
			c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
			cute.copy(
				tiled_copy_r2s,
				tRS_rC,
				tRS_sC[(None, None, None, c_buffer)],
			)
			# Fence and barrier to make sure shared memory store is visible to TMA store
			cute.arch.fence_proxy(
				cute.arch.ProxyKind.async_shared,
				space=cute.arch.SharedSpace.shared_cta,
			)
			self.epilog_sync_barrier.arrive_and_wait()

			#
			# TMA store C to global memory
			#
			if warp_idx == self.epilog_warp_id[0]:
				cute.copy(
					tma_atom_c,
					bSG_sC[(None, c_buffer)],
					bSG_gC[(None, real_subtile_idx)],
				)
				# Fence and barrier to make sure shared memory store is visible to TMA store
				c_pipeline.producer_commit()
				c_pipeline.producer_acquire()
			self.epilog_sync_barrier.arrive_and_wait()

		#
		# Async arrive accumulator buffer empty
		#
		if cutlass.const_expr(not self.overlapping_accum):
			with cute.arch.elect_one():
				acc_pipeline.consumer_release(acc_consumer_state)
			acc_consumer_state.advance()

		#
		# Advance to next tile
		#
		tile_sched.advance_to_next_work()
		work_tile = tile_sched.get_current_work()

	#
	# Dealloc the tensor memory buffer
	#
	tmem.relinquish_alloc_permit()
	self.epilog_sync_barrier.arrive_and_wait()
	tmem.free(acc_tmem_ptr)
	#
	# Wait for C store complete
	#
	c_pipeline.producer_tail()
Initially we consumer_wait on accumulator buffer to wait until its ready to be processed.. We'll than store the result from TMEM to RMEM. In case of overlapping_accumulator (which is turned on by default for 2 CTA case and otherwise not) we immediately perform a fence operation and perform a consumer_release (i.e. signal that the next producer can start to produce result with MMA) perform the RMEM -> SMEM store. Note that in case of overlapping accumulator we only have one stage and assign the position in the buffer via the inverse (i.e. ^ 1 ) of the phase bit. In case where we don't overlap we have two stages and wait for the consumer_release after we performed the transfer from SMEM -> GMEM via the TMA.

To summarize the second pipeline: It is used to synchronize between the the computation of the UMMA and the corresponding store for the result of this computation. Note that the pattern is similar to the above producer/consumer pattern, except that the we have changed definitions of what the producer and what the consumer does.

PipelineTmaStore

# Threads/warps participating in tma store pipeline
c_producer_group = pipeline.CooperativeGroup(
	pipeline.Agent.Thread,
	32 * len(self.epilog_warp_id),
)
c_pipeline = pipeline.PipelineTmaStore.create(
	num_stages=self.num_c_stage,
	producer_group=c_producer_group,
)
Here we only have a producer:

@dataclass(frozen=True)
class PipelineTmaStore(PipelineAsync):
    """
    PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.
    """
As we saw above this pipeline is only used to commit the TMA store and than wait for its completion until the buffer is ready to be reused. Note that we have at least 2 c stages so we can continue with the next subtile until the buffer is cleared at the previous stage. This gives us another level of asynchronous options. Compared to simple TMEM -> RMEM -> GMEM way of doing the store it gave a good performance boost in my experiments.

if warp_idx == self.epilog_warp_id[0]:
	cute.copy(
		tma_atom_c,
		bSG_sC[(None, c_buffer)],
		bSG_gC[(None, real_subtile_idx)],
	)
	# Fence and barrier to make sure shared memory store is visible to TMA store
	c_pipeline.producer_commit()
	c_pipeline.producer_acquire()
self.epilog_sync_barrier.arrive_and_wait()
Conclusion

As we saw pipelining is important to understand internals of performant Blackwell kernels. By carefully examining the pattern it can be however understood and symmetries between different types of pipelines are interesting to observe and discover.

If you want to perform your own experiments on Blackwell GPUs you may consider Verda. I performed all the experiments on their platform and the experience is very good.

To exchange ideas on GPU programming or ML in in general I'd be happy to connect on my Linkedin.