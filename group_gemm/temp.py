

# Global cache for compiled kernels
_compiled_kernel_cache = {}
# Cache for per-shape metadata tensors
_metadata_cache = {}
# Cache for per-data pointer tensors (keyed by id(data))
_data_ptr_cache = {}
_DATA_PTR_CACHE_MAX = 64


def compile_kernel(problem_sizes: List[Tuple[int, int, int, int]]):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    """
    global _compiled_kernel_cache

    num_groups = len(problem_sizes)
    problem_sizes_tuple = tuple(tuple(ps) for ps in problem_sizes)
    cache_key = (num_groups, problem_sizes_tuple)

    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    grouped_blockscaled_gemm = Sm100GroupedBlockScaledGemmKernel(
        SF_VEC_SIZE,
        MMA_TILER_MN,
        CLUSTER_SHAPE_MN,
    )

    max_active_clusters = _MAX_ACTIVE_CLUSTERS

    cta_tile_shape_mn = [MMA_TILER_MN[0], MMA_TILER_MN[1]]
    cluster_tile_shape_mn = tuple(
        x * y for x, y in zip(cta_tile_shape_mn, CLUSTER_SHAPE_MN)
    )
    total_num_clusters = 0
    for m, n, _, _ in problem_sizes:
        num_clusters_mn = tuple(
            (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
        )
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    problem_sizes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_groups, 4), stride_order=(1, 0)
    )
    strides_abc_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_groups, 3, 2), stride_order=(2, 1, 0)
    )
    ptrs_abc_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (num_groups, 3), stride_order=(1, 0)
    )
    ptrs_sfasfb_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (num_groups, 2), stride_order=(1, 0)
    )
    tensormap_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (
            SM_COUNT,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        ),
        stride_order=(2, 1, 0),
    )
    total_num_clusters = cutlass.Int32(total_num_clusters)

    compiled_func = cute.compile(
        grouped_blockscaled_gemm,
        num_groups,
        problem_sizes_fake,
        strides_abc_fake,
        ptrs_abc_fake,
        ptrs_sfasfb_fake,
        tensormap_fake,
        total_num_clusters,
        max_active_clusters,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled group GEMM kernel.
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    num_groups = len(problem_sizes)

    compiled_func = compile_kernel(problem_sizes)

    problem_sizes_key = tuple(tuple(ps) for ps in problem_sizes)
    cached_meta = _metadata_cache.get(problem_sizes_key)
    if cached_meta is None:
        tensor_of_problem_sizes = torch.tensor(
            problem_sizes, dtype=torch.int32, device="cuda"
        )
        strides_abc = [[(k, 1), (k, 1), (n, 1)] for _, n, k, _ in problem_sizes]
        tensor_of_strides_abc = torch.tensor(
            strides_abc, dtype=torch.int32, device="cuda"
        )

        cta_tile_shape_mn = [128, MMA_TILER_MN[1]]
        cluster_tile_shape_mn = tuple(
            x * y for x, y in zip(cta_tile_shape_mn, CLUSTER_SHAPE_MN)
        )
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

        sm_count = SM_COUNT
        max_active_clusters = _MAX_ACTIVE_CLUSTERS
        tensormap_shape = (
            sm_count,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        )
        tensor_of_tensormap = torch.empty(
            tensormap_shape, dtype=torch.int64, device="cuda"
        )

        cached_meta = (
            tensor_of_problem_sizes,
            tensor_of_strides_abc,
            tensor_of_tensormap,
            total_num_clusters,
            max_active_clusters,
        )
        _metadata_cache[problem_sizes_key] = cached_meta
    else:
        (
            tensor_of_problem_sizes,
            tensor_of_strides_abc,
            tensor_of_tensormap,
            total_num_clusters,
            max_active_clusters,
        ) = cached_meta

    data_id = id(data)
    first_ptr = abc_tensors[0][0].data_ptr()
    cached_ptrs = _data_ptr_cache.get(data_id)
    if cached_ptrs is not None:
        cached_first_ptr, cached_ps_key, tensor_of_abc_ptrs, tensor_of_sfasfb_ptrs = (
            cached_ptrs
        )
        if cached_first_ptr != first_ptr or cached_ps_key != problem_sizes_key:
            cached_ptrs = None

    if cached_ptrs is None:
        abc_ptrs = []
        sfasfb_ptrs = []
        for (a, b, c), (sfa_reordered, sfb_reordered), _ in zip(
            abc_tensors, sfasfb_reordered_tensors, problem_sizes
        ):
            abc_ptrs.append([a.data_ptr(), b.data_ptr(), c.data_ptr()])
            sfasfb_ptrs.append(
                [sfa_reordered.data_ptr(), sfb_reordered.data_ptr()]
            )

        tensor_of_abc_ptrs = torch.tensor(
            abc_ptrs, dtype=torch.int64, device="cuda"
        )
        tensor_of_sfasfb_ptrs = torch.tensor(
            sfasfb_ptrs, dtype=torch.int64, device="cuda"
        )

        if len(_data_ptr_cache) >= _DATA_PTR_CACHE_MAX:
            _data_ptr_cache.pop(next(iter(_data_ptr_cache)))
        _data_ptr_cache[data_id] = (
            first_ptr,
            problem_sizes_key,
            tensor_of_abc_ptrs,
            tensor_of_sfasfb_ptrs,
        )

    compiled_func(
        tensor_of_problem_sizes,
        tensor_of_strides_abc,
        tensor_of_abc_ptrs,
        tensor_of_sfasfb_ptrs,
        tensor_of_tensormap,
        total_num_clusters,
    )

    return [abc_tensors[i][2] for i in range(num_groups)]


































#slightly less perf

# Global cache for compiled kernels
_compiled_kernel_cache = {}
# Cache for per-shape metadata tensors
_metadata_cache = {}
# Cache for per-data pointer tensors (keyed by id(data))
_data_ptr_cache = {}
_DATA_PTR_CACHE_MAX = 64


def compile_kernel(problem_sizes: List[Tuple[int, int, int, int]]):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    """
    global _compiled_kernel_cache

    num_groups = len(problem_sizes)
    problem_sizes_tuple = tuple(tuple(ps) for ps in problem_sizes)
    cache_key = (num_groups, problem_sizes_tuple)

    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    grouped_blockscaled_gemm = Sm100GroupedBlockScaledGemmKernel(
        SF_VEC_SIZE,
        MMA_TILER_MN,
        CLUSTER_SHAPE_MN,
    )

    max_active_clusters = _MAX_ACTIVE_CLUSTERS

    cta_tile_shape_mn = [MMA_TILER_MN[0], MMA_TILER_MN[1]]
    cluster_tile_shape_mn = tuple(
        x * y for x, y in zip(cta_tile_shape_mn, CLUSTER_SHAPE_MN)
    )
    total_num_clusters = 0
    for m, n, _, _ in problem_sizes:
        num_clusters_mn = tuple(
            (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
        )
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    problem_sizes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_groups, 4), stride_order=(1, 0)
    )
    strides_abc_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_groups, 3, 2), stride_order=(2, 1, 0)
    )
    ptrs_abc_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (num_groups, 3), stride_order=(1, 0)
    )
    ptrs_sfasfb_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (num_groups, 2), stride_order=(1, 0)
    )
    tensormap_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (
            SM_COUNT,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        ),
        stride_order=(2, 1, 0),
    )
    total_num_clusters = cutlass.Int32(total_num_clusters)

    compiled_func = cute.compile(
        grouped_blockscaled_gemm,
        num_groups,
        problem_sizes_fake,
        strides_abc_fake,
        ptrs_abc_fake,
        ptrs_sfasfb_fake,
        tensormap_fake,
        total_num_clusters,
        max_active_clusters,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


def custom_kernel(data: input_t) -> output_t:
    """
    OPTIMIZED implementation with shape-based caching.
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    num_groups = len(problem_sizes)

    compiled_func = compile_kernel(problem_sizes)

    problem_sizes_key = tuple(tuple(ps) for ps in problem_sizes)
    cached_meta = _metadata_cache.get(problem_sizes_key)
    if cached_meta is None:
        tensor_of_problem_sizes = torch.tensor(
            problem_sizes, dtype=torch.int32, device="cuda"
        )
        strides_abc = [[(k, 1), (k, 1), (n, 1)] for _, n, k, _ in problem_sizes]
        tensor_of_strides_abc = torch.tensor(
            strides_abc, dtype=torch.int32, device="cuda"
        )

        cta_tile_shape_mn = [128, MMA_TILER_MN[1]]
        cluster_tile_shape_mn = tuple(
            x * y for x, y in zip(cta_tile_shape_mn, CLUSTER_SHAPE_MN)
        )
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

        sm_count = SM_COUNT
        max_active_clusters = _MAX_ACTIVE_CLUSTERS
        tensormap_shape = (
            sm_count,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        )
        tensor_of_tensormap = torch.empty(
            tensormap_shape, dtype=torch.int64, device="cuda"
        )

        # Pre-allocate pinned CPU buffers for fast H2D transfer
        cpu_abc_ptrs = torch.empty((num_groups, 3), dtype=torch.int64, pin_memory=True)
        cpu_sfasfb_ptrs = torch.empty((num_groups, 2), dtype=torch.int64, pin_memory=True)
        # Pre-allocate GPU buffers to avoid per-call allocation
        gpu_abc_ptrs = torch.empty((num_groups, 3), dtype=torch.int64, device="cuda")
        gpu_sfasfb_ptrs = torch.empty((num_groups, 2), dtype=torch.int64, device="cuda")

        cached_meta = (
            tensor_of_problem_sizes,
            tensor_of_strides_abc,
            tensor_of_tensormap,
            total_num_clusters,
            max_active_clusters,
            cpu_abc_ptrs,
            cpu_sfasfb_ptrs,
            gpu_abc_ptrs,
            gpu_sfasfb_ptrs,
        )
        _metadata_cache[problem_sizes_key] = cached_meta
    else:
        (
            tensor_of_problem_sizes,
            tensor_of_strides_abc,
            tensor_of_tensormap,
            total_num_clusters,
            max_active_clusters,
            cpu_abc_ptrs,
            cpu_sfasfb_ptrs,
            gpu_abc_ptrs,
            gpu_sfasfb_ptrs,
        ) = cached_meta

    # Fill CPU buffers with current pointers (must be done fresh each call)
    for i in range(num_groups):
        abc = abc_tensors[i]
        sf = sfasfb_reordered_tensors[i]
        cpu_abc_ptrs[i, 0] = abc[0].data_ptr()
        cpu_abc_ptrs[i, 1] = abc[1].data_ptr()
        cpu_abc_ptrs[i, 2] = abc[2].data_ptr()
        cpu_sfasfb_ptrs[i, 0] = sf[0].data_ptr()
        cpu_sfasfb_ptrs[i, 1] = sf[1].data_ptr()

    # Async copy from pinned CPU to GPU
    gpu_abc_ptrs.copy_(cpu_abc_ptrs, non_blocking=True)
    gpu_sfasfb_ptrs.copy_(cpu_sfasfb_ptrs, non_blocking=True)
    tensor_of_abc_ptrs = gpu_abc_ptrs
    tensor_of_sfasfb_ptrs = gpu_sfasfb_ptrs

    compiled_func(
        tensor_of_problem_sizes,
        tensor_of_strides_abc,
        tensor_of_abc_ptrs,
        tensor_of_sfasfb_ptrs,
        tensor_of_tensormap,
        total_num_clusters,
    )

    return [abc_tensors[i][2] for i in range(num_groups)]