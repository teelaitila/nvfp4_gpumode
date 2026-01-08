B200 Blockscaled GEMM - The setup

06 Dec, 2025

Introduction

This continues top down analysis of blockscaled GEMM example from CuTeDSL. In this blogpost I analyze the setup of the kernel, i.e. how are layouts calculated, how are MMA ops setup etc. You may find my previous blogpost helpful before diving into this one.

__call__

Initial setup. Last time we analysed in depth how scaled tensors are constructed. TLDR: We determine the atom via datatype and than use tile_to_shape to match the size to their "non scaling factor" counterparts.

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sf_dtype
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)
Let us now take closer look at _setup_attributes function.

From docstring we can get a feeling on how many different parts are handled in this helper function. We will below analyze step by step.

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
For the mma_inst_shape_mn it is simply the attributes from mma_tiler. For the sfb we will slightly modify the instruction shape. In case of 2 CTA invocation of the kernel we adjust the M tiler to half of its original value.

        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )
cute.round_up boils in the above code snipped essentially down to:

def round_up(a: IntTuple, b: IntTuple) -> IntTuple:
    """
    Rounds up elements of a using elements of b.
    """
    ...
    return ((a + b - 1) // b) * b
IMO the docstring is little bit cryptic. In math notation it means 
⌈
a
b
⌉
·
b
. That means we consider how often we can fit 
b
 into 
a
, than round this up to the next integer and use it to rescale 
b
. For example if we had 
b
N
=
128
 we would have 
⌈
b
N
128
⌉
b
=
128
. A different case could be 
b
N
=
192
 we for which we had 
⌈
192
128
⌉
128
=
256
.

Next step is to setup tiled_mma for both. We immediately see that irrespective if we set 2 CTA instruction the SFB will use cute.nvgpu.tcgen05.CtaGroup.ONE as parameter. From above we know that in this case we will also have scaled down the corresponding tile dimension 
b
M
 by a factor of to in order to account for that.

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
make_blockscaled_trivial_tiled_mma indeed constructs a trivial tiled mma. It is nothing special going on here. Based on datatype we select corresponding MMAOp. Note that instruction shape into K mode is fixed at 64 which agrees with table in PTX docs.

@dsl_user_op
def make_blockscaled_trivial_tiled_mma(
    ab_dtype: Type[Numeric],
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    sf_dtype: Type[Numeric],
    sf_vec_size: int,
    cta_group: CtaGroup,
    mma_tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *,
    loc=None,
    ip=None,
) -> cute.TiledMma:
	...
    if ab_dtype in {Float8E4M3FN, Float8E5M2}:
        mma_op = MmaMXF8Op(
            ab_dtype,
            (*mma_tiler_mn, 32),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    elif ab_dtype == Float4E2M1FN:
        if sf_vec_size == 32:
            mma_op = MmaMXF4Op(
                (*mma_tiler_mn, 64),
                cta_group,
                a_source,
            )
        elif sf_vec_size == 16:
            mma_op = MmaMXF4NVF4Op(
                sf_dtype,
                (*mma_tiler_mn, 64),
                cta_group,
                a_source,
            )
        else:
            raise ValueError(f"unsupported sf_vec_size, got {sf_vec_size}")
    else:
        raise TypeError(f"unsupported ab_dtype, got {ab_dtype}")

    return cute.make_tiled_mma(
        cute.make_mma_atom(mma_op, loc=loc, ip=ip), loc=loc, ip=ip
    )
We use tiled_mma to obtain the full mma_tiler. The size of thr_id.shape will be 1 or 2 depending if we use 2 CTA instruction. If we use 2 CTA instruction two CTAs will collaboratively compute MMA for one tile and therefore we will scale the CTA level tile accordingly.

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )
Similar we will rescale cluster layout depending if we use 2 CTA instruction. We furthermore save information if we use multicast for TMA transfer of a/b depending on the cluster dimension in the direction of corresponding Major mode.

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1
In Epilogue we will transfer in tiles. Tiles are determined here:

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])
Let us analyse corresponding helper function to understand how it is determined.

From the signature we see that this is heuristic to choose good epi tile. We perform direct calculation, so we can safely ignore the parameters layout_c and elem_ty_c.

@dsl_user_op
def compute_epilogue_tile_shape(
    cta_tile_shape: cute.Shape,
    use_2cta_instrs: bool,
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    *,
    layout_c: LayoutEnum = None,
    elem_ty_c: Union[Type[Numeric], None] = None,
    loc=None,
    ip=None,
) -> cute.Tile:
    """Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

    :param cta_tile_shape: A tuple or list representing the dimensions of the CTA tile, where
        cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1]
        corresponds to the width (N) of the tile.
    :type cta_tile_shape: cute.Shape
    :param use_2cta_instrs: A flag indicating whether the configuration is for a 2SM setup.
    :type use_2cta_instrs: bool
    :param layout_d: The layout enum of the output tensor D.
    :type layout_d: LayoutEnum
    :param elem_ty_d: The element type of output tensor D.
    :type elem_ty_d: Type[Numeric]
    :param layout_c: The layout enum of the input tensor C. Defaults to None.
    :type layout_c: LayoutEnum, optional
    :param elem_ty_c: The element type for input tensor C. Defaults to None.
    :type elem_ty_c: Union[Type[Numeric], None], optional

    :return: Returns epilog tiler, which is used in subsequent epilog partitions.
    :rtype: cute.Tile

    :raises ValueError: If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.
    """
    cta_m, cta_n = cta_tile_shape[:2]
    (warp_m, warp_n) = (2, 2) if (cta_m == 64 and use_2cta_instrs) else (4, 1)
    disable_source = elem_ty_c == None
    max_bits = (
        elem_ty_d.width if disable_source else max(elem_ty_c.width, elem_ty_d.width)
    )
Note from above

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
and we furthermore have:

self.use_2cta_instrs = mma_tiler_mn[0] == 256
So effectively we will always choose (warp_m, warp_n) = (4,1) for our kernel. disable_source is True so we will calculate max_bits as the number of bits in one element of the d Tensor. Let's consider case where we accumulate into Float16, than we obviously have max_bits = 16.

So for NVFP4 we have:

(warp_m, warp_n) = (4, 1)
max_bits = 16
    dp_full = 32
    tile_m = min(cta_m, dp_full * warp_m)
    n_perf = 0
    if disable_source:
        if max_bits == 4:
            compute_elts = 8192
        else:
            compute_elts = 4096
        n_perf = compute_elts // tile_m
    else:
        if max_bits == 32:
            n_perf = 16 if (cta_m > 64 and cta_n <= 128) else 32
        elif max_bits == 16:
            n_perf = 32 if cta_n <= 128 else 64
        else:
            n_perf = 64
The tile_m = min(128, 32 * 4) = 128, we'll than select n_perf = 4096 // 128 = 32.

    d_is_m_major = layout_d.is_m_major_c()
    c_is_m_major = True if layout_c is None else layout_c.is_m_major_c()

    n_min_d = (
        8 * warp_n
        if d_is_m_major
        else (128 * warp_n if elem_ty_d.width == 6 else 128 // elem_ty_d.width * warp_n)
    )
    n_min_c = (
        8 * warp_n
        if (c_is_m_major or disable_source)
        else (128 * warp_n if elem_ty_c.width == 6 else 128 // elem_ty_c.width * warp_n)
    )
    tile_n = min(cta_n, max(n_perf, n_min_c, n_min_d))
If we use NVFP4 d will be N major. So we set n_min_d = 128 // 16 * 1 = 8. This is number of elements that fits into 128 bit instruction.

tile_n = min(cta_n, max(n_perf, n_min_c, n_min_d)) = n_perf = 32.

tile_n_layout is first set to (n_perf, 1) : (1, cta_n) which than gets coalesced. Note that coalescing will remove the second mode because it won't change the associated layout function so we end up with two layouts:

128:1, 32:1
the corresponding code is

    tile_m_layout = cute.make_layout(tile_m, loc=loc, ip=ip)
    tile_n_layout = cute.make_layout(
        (tile_n // warp_n, warp_n), stride=(1, cta_n // warp_n), loc=loc, ip=ip
    )
    return (tile_m_layout, cute.coalesce(tile_n_layout, loc=loc, ip=ip))
Note that now we derived by hand that the epi_tile for setting of matrices in NVFP4 and Float16 will get selected to be 128:1, 32:1 irrespective of the mma_tiler_mn or cluster shape we choose. This is the heuristic that is used by default.

Next part is calculation of stages

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )
We see that we provide this function with the tiled_mma, corresponding tiler, the datatypes, tiler for the epilogue, information about the scaling factors and finally the SMEM capacity for our problem.

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
Let us quickly review make_smem_layout_{a|b|sfa|sfb}

The docstring essentially describes what we are doing:

tiled_mma.partition_shape_A perform partitioning, i.e. composing with TV Layout for MMA and than being able to slice into it to get work for each thread
get_smem_layout_atom_ab is determining of corresponding Swizzle mode to avoid bank conflicts.
make_smem_layout_atom will create appropriate atom based on swizzling mode.
We'll than can append the number of stages. Here we just append 1 stage because the number of stages will be calculated later.
At the end we use tile_to_shape to cover our target shape with the above created Swizzle atom
@dsl_user_op
def make_smem_layout_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    is_k_major=None,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps with:

    1. Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    :param tiled_mma: The tiled MMA used to partition tensor A
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape
    :type mma_tiler_mnk: cute.cute.Tile
    :param a_dtype: The element type for tensor A
    :type a_dtype: Type[Numeric]
    :param num_stages: The number of pipeline stages for tensor A
    :type num_stages: int

    :return: SMEM layout for tensor A
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = (tiled_mma.op.a_major_mode == OperandMajorMode.K) if is_k_major is None else is_k_major
    a_major_mode = OperandMajorMode.K if is_k_major else OperandMajorMode.MN
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip), loc=loc, ip=ip
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    smem_layout_atom_kind = get_smem_layout_atom_ab(
        a_major_mode, a_dtype, a_smem_shape_mn_k, loc=loc, ip=ip
    )
    a_smem_layout_atom = make_smem_layout_atom(
        smem_layout_atom_kind, a_dtype, loc=loc, ip=ip
    )

    a_smem_shape = cute.append(a_smem_shape, num_stages, loc=loc, ip=ip)
    order = (2, 1, 3) if not is_k_major else (1, 2, 3)
    return tile_to_mma_shape(
        a_smem_layout_atom, a_smem_shape, order=order, loc=loc, ip=ip
    )
B is similar but uses the partition_B function of tiled_mma.

For the scaled factors we have the Blockscaled atoms and use tile_to_shape to cover whole tile with it. We furthermore calculate MMA_Inst_Shape_K based on inst_shape_k which we saw already above. At the end we append stages.

@dsl_user_op
def make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFA based on:

    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """
    # (CTA_Tile_Shape_M, MMA_Tile_Shape_K)
    sfa_tile_shape = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[2],
    )

    # ((Atom_M, Rest_M),(Atom_K, Rest_K))
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfa_tile_shape,
        (2, 1),
    )

    mma_tile_inst_k = 4
    # (CTA_Tile_Shape_M, MMA_Inst_Shape_K)
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (1, mma_tile_inst_k))
    # ((Atom_Inst_M, Atom_Inst_K), MMA_M, MMA_K))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)

    atom_m = 128
    tiler_inst = ((atom_m, sf_vec_size),)
    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)

    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged
The code for sfb is similar.

Similar to what we do for a and b we construct SMEM layout for epi tile.

@dsl_user_op
def make_smem_layout_epi(
    epi_dtype: Type[Numeric],
    epi_layout: LayoutEnum,
    epi_tile: cute.Tile,
    epi_stage: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps:

    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensor's majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    :param epi_dtype: The element type for the epilog tensor.
    :type epi_dtype: Type[Numeric]
    :param epi_layout: The layout enum for the epilog tensor.
    :type epi_layout: LayoutEnum
    :param epi_tile: The epilogue tile shape.
    :type epi_tile: cute.cute.Tile
    :param epi_stage: The stage of the epilog tensor.
    :type epi_stage: int

    :return: SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    epilog_shape = cute.product_each(
        cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip
    )

    smem_atom_kind = get_smem_layout_atom_epi(
        epi_layout, epi_dtype, epi_tile, loc=loc, ip=ip
    )
    c_smem_layout_atom = make_smem_layout_atom(
        smem_atom_kind, epi_dtype, loc=loc, ip=ip
    )

    epilog_shape = cute.append(epilog_shape, epi_stage, loc=loc, ip=ip)
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        epilog_shape,
        order=((1, 0, 2) if not epi_layout.is_n_major_c() else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )

    return epi_smem_layout_staged
TMA always transfer 1 stage, so the above calculated "1 stage layouts" can be used to determine number of bytes transferred in one stage. The stages are chosen such that we will utilize maximum of SMEM.

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage
Recalculate SMEM layouts (same as above) but this time provide correct number of stages.

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
Further kernel settings.

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols

        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n
This concludes our discussion of the long _setup_attributes function and we return back to __call__.

Here we construct the tiled_mma. We use separate tiled_mma for sfb. See the above discussion for the parameters. Note the atom_thr_size is either 1 or 2 depending if we use 2 CTA instruction.

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
Setup of the TMA units based on the SMEM layouts. Under the hood this helper function will figure out appropriate PTX instruction for TMA transfer based on the kernel settings.

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
Afterwards same is done for SFA and SFB, however for SFB we will provide the tiled_mma_sfb to the construction function. The bN = 192 case deserves some extra attention. See the code.

Determine size of TMA loads per atom and scale by number of atoms. Also sets up the TMA for the accumulator. All based on "one stage" layouts, because TMA transfers always one stage at a time. Use TileScheduler to figure out correct grid. Potentially we will write another post on the TileScheduler on its own to keep this blogpost to a reasonable size.

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )
Afterwards there is only setting up SharedStorage left and we launch the kernel.

Conclusion

I hope this blogpost helped to understand setup of various setup steps we make before launching the kernel. As we saw it is far from trivial but can be understood when analysed in small steps in Top Down approach. If you like to exchange ideas I am happy to connect on Linkedin. If you want to perform experiments on the B200 yourself I recommend Verda for good access to B200 GPUs.