Warp Specialisation in CuTeDSL

07 Jan, 2026

Warp specialisation is an optimisation that splits the GEMM mainloop into two parts: We have one warp that does the TMA (i.e. copy tiles to SMEM) and one warp that does the MMA (i.e. multiply together these tiles). CuTeDSLs pipelining abstraction makes it particularly convenient for the user to implement this optimisation. In this short note I briefly show how to turn ordinary non persistent Blackwell mainloops into warp specialised ones in the CuTeDSL.

Ordinary Mainloop 1 CTA

Below we see the ordinary one mainloop we can write for a B200 GPU. We briefly summarise the structure:

acc_empty: Aquire current buffer for the Producer of the UMMA <-> Store Pipeline. We will release this lock at the end once the producer is done, i.e. we have computed the full tile bM x bN For each tile:
ab_empty: Aquire current buffer for the Producer of the TMA <-> UMMA Pipeline. This simply tells the TMA: "Wait until the current stage is free and if so, copy the tiles over to SMEM for A and B".
ab_full: Here the UMMA waits for the tiles of it's current stage to be ready for UMMA. UMMA expects A and B to reside in SMEM. Once that is the case for the current stage the UMMA loop is triggered and we compute the result.
ab_full.release(): The consumer tells the producer that it's done with the current stage. It signals that it can take this slot in the buffer and start copying. acc_empty.commit(): The UMMA is now in role of producer (for the epilogue which consumes it by transferring the computed result to GMEM). commit() releases the lock on the stage and the epilogue will know it can start copying the result.
num_k_tiles = cute.size(gA_mk, mode=[2])
if warp_idx == 0:
	acc_empty = (
		acc_producer.acquire_and_advance()
	)  # Acquire the current buffer and advance to the next pipeline stage.
	for k_tile in cutlass.range(
		num_k_tiles, prefetch_stages=self.ab_stages - 2
	):
		# TMA
		ab_empty = (
			ab_producer.acquire_and_advance()
		)  # Acquire the current buffer and advance to the next pipeline stage.
		cute.copy(
			tma_atom_a,
			tAgA[(None, ab_empty.count)],  # Global count -> RestK
			tAsA[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
		cute.copy(
			tma_atom_b,
			tBgB[(None, ab_empty.count)],  # Global count -> RestK
			tBsB[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
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

	# Commit processed tile to epilogue for Store
	acc_empty.commit()
Note here we have a prefetch_stages parameter. This is a hint for the compiler to first issue prefetch_stages copy operations with the TMA. It is useful to introduce this parameter (or do the prefetch "by hand", if you are interested in doing it by hand see the Blackwell examples in CuTeDSL repo) because we want to keep the Tensor Core busy at all times and transfer from GMEM -> SMEM is relatively expensive compared to the very fast Tensor Core operations.

Warp specialisation

Actually to write the warp specialised version of the above kernel is almost trivial:

# Warp 0: TMA operations
if warp_idx == self.tma_warp_id:
	for k_tile in cutlass.range(num_k_tiles):
		# TMA - acquire empty barrier
		ab_empty = (
			ab_producer.acquire_and_advance()
		)  # Acquire the current buffer and advance to the next pipeline stage.
		cute.copy(
			tma_atom_a,
			tAgA[(None, ab_empty.count)],  # Global count -> RestK
			tAsA[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
		cute.copy(
			tma_atom_b,
			tBgB[(None, ab_empty.count)],  # Global count -> RestK
			tBsB[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)

# Warp 1: MMA operations
if warp_idx == self.mma_warp_id:
	acc_empty = (
		acc_producer.acquire_and_advance()
	)  # Acquire the current buffer and advance to the next pipeline stage.
	for k_tile in cutlass.range(num_k_tiles):
		# Wait for TMA data to be ready
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

	# Commit processed tile to epilogue for Store
	acc_empty.commit()
Note that we just put all logic related to the UMMA in one warp and all logic related to the TMA in the other warp. Each warp can than do it's job. We nicely decoupled the two workloads in the sense that one warp is responsible for scheduling copies and the other one is responsible for scheduling computation. The pipeline synchronisation which I explained above and in previous blogposts ensures that we only compute a tile once the corresponding copy is ready. We don't need a prefetch here because the TMA will simply schedule copies asynchronously in it's own warp.

The compared performance for both cases where I left all (untuned) parameters like number of stages etc exactly as they were:

1318.88 tflops for the non specialised version.
1376.56 tflops for the specialised version.
This is a good boost in performance! The 1 CTA version is now close in performance to the 2 CTA version (which gets 1395.39 tflops).

2 CTA Mainloop

2 CTA mainloop is almost identical. Main difference here is that we compute the leader CTA and he is responsible for the UMMA related logic. See my recent blogpost.

is_leader_cta = mma_coord_vmnk[0] == 0  # Only issue MMA from Leader.
num_k_tiles = cute.size(gA_mk, mode=[2])
if warp_idx == 0:
	if is_leader_cta:
		acc_producer.acquire_and_advance()
	for k_tile in cutlass.range(
		num_k_tiles, prefetch_stages=self.ab_stages - 2
	):
		# TMA
		ab_empty = (
			ab_producer.acquire_and_advance()
		)  # Acquire the current buffer and advance to the next pipeline stage.
		cute.copy(
			tma_atom_a,
			tAgA[(None, ab_empty.count)],  # Global count -> RestK
			tAsA[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
		cute.copy(
			tma_atom_b,
			tBgB[(None, ab_empty.count)],  # Global count -> RestK
			tBsB[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
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

	# Commit processed tile to epilogue for Store
	if is_leader_cta:
		acc_producer.commit()
At the very end of the kernel we need to do some "cleanup" like so

if warp_idx == 0:
	ab_producer.tail()  # Cleanup
	if is_leader_cta:
		acc_producer.tail()  # Cleanup from leader CTA
2 CTA warp specialised

If you understood the one CTA warp specialised version it will be obvious how to extend to the 2 CTA case:

is_leader_cta = mma_coord_vmnk[0] == 0  # Only issue MMA from Leader.
num_k_tiles = cute.size(gA_mk, mode=[2])

# TMA Warp
if warp_idx == self.tma_warp_id:
	for k_tile in cutlass.range(num_k_tiles):
		# TMA warp handles ab_empty logic
		ab_empty = (
			ab_producer.acquire_and_advance()
		)  # Acquire the current buffer and advance to the next pipeline stage.
		cute.copy(
			tma_atom_a,
			tAgA[(None, ab_empty.count)],  # Global count -> RestK
			tAsA[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
		cute.copy(
			tma_atom_b,
			tBgB[(None, ab_empty.count)],  # Global count -> RestK
			tBsB[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)

# MMA Warp (Only execute on leader)
if warp_idx == self.mma_warp_id:
	if is_leader_cta:
		acc_producer.acquire_and_advance()
		for k_tile in cutlass.range(num_k_tiles):
			# MMA warp handles ab_full logic
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

		# Commit processed tile to epilogue for Store
		acc_producer.commit()
The cleanup step at the end looks like this:

if warp_idx == self.tma_warp_id:
	ab_producer.tail()  # Cleanup
if warp_idx == self.mma_warp_id:
	if is_leader_cta:
		acc_producer.tail()  # Cleanup from leader CTA
Note that we could again simply archive warp specialisation by putting all TMA logic into one warp and all UMMA logic into the other warp. However for the 2 CTA version I didn't observe any speedup on the 8192 x 8192 x 8192 problem config. It would be interesting if this will always be the case.

Peeking

We can use peek technique as follows:

num_k_tiles = cute.size(gA_mk, mode=[2])

# Warp 0: TMA operations
if warp_idx == self.tma_warp_id:
	peek_ab_empty_status = ab_producer.try_acquire()  # Check Buffer available
	for k_tile in cutlass.range(num_k_tiles):
		# TMA - acquire empty barrier
		ab_empty = ab_producer.acquire_and_advance(
			peek_ab_empty_status
		)  # Acquire the current buffer and advance to the next pipeline stage.
		# If peek, non blocking!
		cute.copy(
			tma_atom_a,
			tAgA[(None, ab_empty.count)],  # Global count -> RestK
			tAsA[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)
		cute.copy(
			tma_atom_b,
			tBgB[(None, ab_empty.count)],  # Global count -> RestK
			tBsB[(None, ab_empty.index)],  # Index with wrap around -> STAGE
			tma_bar_ptr=ab_empty.barrier,  # Barrier to signal Bit transfer
		)

		# Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
		peek_ab_empty_status = cutlass.Boolean(1)
		if ab_empty.count + 1 < num_k_tiles:
			peek_ab_empty_status = ab_producer.try_acquire()

# Warp 1: MMA operations
if warp_idx == self.mma_warp_id:
	peek_ab_full_status = ab_consumer.try_wait()  # Peek
	acc_empty = (
		acc_producer.acquire_and_advance()
	)  # Acquire the current buffer and advance to the next pipeline stage.
	for k_tile in cutlass.range(num_k_tiles):
		# Wait for TMA data to be ready
		ab_full = ab_consumer.wait_and_advance(
			peek_ab_full_status
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
		# Peek (try_wait) AB buffer full for k_tile = k_tile + 1
		peek_ab_full_status = cutlass.Boolean(1)
		if ab_full.count + 1 < num_k_tiles:
			peek_ab_full_status = ab_consumer.try_wait()

	# Commit processed tile to epilogue for Store
	acc_empty.commit()
This will - as the name says - "peek" and see if the current buffer slot is free. We than provide this as a token to the producer ops and if the peek returned true the corresponding operation is nonblocking. You can read more about it in the CuTeDSL codebase. This technique is commonly used in the CuTeDSL examples. However I didn't observe positive effect on performance for my problem config.

Conclusion

I hope this short note showed how we can speed up (at least sometimes) simple non persistent Blackwell kernels by creating a warp specialised version of it. The next optimisation that comes to mind is persistent kernel design which I will probably elaborate on in the future. If you like to experiment with B200 yourself I suggest to check out Verda which provides convenient way of using B200. You can contact me on Linkedin to exchange ideas.