Scale Tensor construction in CuTeDSL

03 Dec, 2025

In Blackwell kernels for NVFP4 we need to associate the 8bit scale tensors with the correct layout. In this blogpost I give a brief analysis of the mathematical interpretation of these scale factors in an easy to follow way. An interesting side fact that I noticed when analyzing the construction was the similarity to Swizzling where we obtain a larger Layout by covering it with small Swizzle atoms. This shows the generality of the CuTe Layout algebra!

The code

The signature of the blockscaled GEMM kernel looks as follows:

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
Block scaled layout needs some extra attention to be brought into right format:

sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
	a_tensor.shape, self.sf_vec_size
)
sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

# ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
	b_tensor.shape, self.sf_vec_size
)
sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
Let us understand what happens here:

@dsl_user_op
def tile_atom_to_shape_SF(
    Shape: cute.Shape,
    sf_vec_size: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    A helper function to get dynamic SFA/SFB layout by filling dynamic A/B shape to the scale factor atom layout.

    :param Shape: The shape of the A/B tensor
    :param sf_vec_size: Scale factor vector size

    :return: The layout of the SFA/SFB tensor
    :rtype: cute.Layout
    """
    # ((Atom_MN, Rest_MN),(Atom_K, Rest_K),RestL)
    sf_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1, 3)
    )
    return sf_layout
Where the BlockScaledBasicChunk is:

@dataclass(frozen=True)
class BlockScaledBasicChunk:
    """
    The basic scale factor atom layout decided by tcgen05 BlockScaled MMA Ops.

    This class represents the fixed layout pattern for scale factors used in
    tcgen05 BlockScaled MMA Ops. The layout is determined by the
    instruction specification and cannot be modified.
    See `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x>`.
    """

    sf_vec_size: int
    major_mode: OperandMajorMode = OperandMajorMode.K
    _layout: cute.Layout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.major_mode == OperandMajorMode.K:
            # K-major layout: (AtomMN, AtomK)
            atom_shape = ((32, 4), (self.sf_vec_size, 4))
            atom_stride = ((16, 4), (0, 1))
        else:
            # MN-major layout: (AtomK, AtomMN)
            atom_shape = ((self.sf_vec_size, 4), (32, 4))
            atom_stride = ((0, 1), (16, 4))

        object.__setattr__(
            self, "_layout", cute.make_layout(atom_shape, stride=atom_stride)
        )

    @property
    def layout(self) -> cute.Layout:
        """
        Get the layout for this block scaled chunk.

        :return: The layout representing the scale factor atom
        :rtype: cute.Layout
        """
        return self._layout
The math

For example for NVFP4 we will have sf_vec_size = 16 and we will always have K-Major Layout. This will result in Layout

(
(
32
,
4
)
,
(
16
,
4
)
)
:
(
(
16
,
4
)
,
(
0
,
1
)
)
 for the Atom.

We can interpret this Atom as follows:

Take a fixed row. For this row we will have the Layout 
(
16
,
4
)
:
(
0
,
1
)
. We have 
16
 times the same value, than increase by 
1
 and have another value. So we have 4 unique values per row.
The stride for the first mode is 
(
16
,
4
)
 which is equal to the shape for the second mode. We therefore move in a K-Major fashion through the matrix.
To summarize we have 4 unique scaling vectors per row in the atom and the elements are layed out such that we have the scaling vectors for the first row consecutive, than for the second row etc.

If you like visuals you could draw the matrix in this way

Pasted image 20251203183004

and continue the pattern until you reach 
32
*
4
*
16
*
4
−
1
=
8191
.

We'll than use

# ((Atom_MN, Rest_MN),(Atom_K, Rest_K),RestL)
sf_layout = cute.tile_to_shape(
	BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1, 3)
)
to obtain our final layout. For NVFP4 we have Atom_MN = (32,2) and Atom_K = (16,4). The last argument is the order in which we lay out the atom across the target, i.e. we first repeat the atom until we reach the end into the K mode, than we will repeat this process for the M mode and finally we cover the L mode with the Atom.

Note that this process is somehow conceptually similar to what we do in swizzling. You may look my blogpost on swizzle to see the connection yourself.

The examples

Let us look at some examples of scale factors for for given A tensor:

a_tensor = (128,64,1):(64,1,8192)
size(a_tensor) = 8192
sfa_tensor = (((32,4),1),((16,4),1),(1,1)):(((16,4),512),((0,1),512),(0,512))
size(sfa_tensor) = 8192
---
a_tensor = (128,128,1):(128,1,16384)
size(a_tensor) = 16384
sfa_tensor = (((32,4),1),((16,4),2),(1,1)):(((16,4),1024),((0,1),512),(0,1024))
size(sfa_tensor) = 16384
---
a_tensor = (256,64,1):(64,1,16384)
size(a_tensor) = 16384
sfa_tensor = (((32,4),2),((16,4),1),(1,1)):(((16,4),512),((0,1),512),(0,1024))
size(sfa_tensor) = 16384
---
a_tensor = (256,128,1):(128,1,32768)
size(a_tensor) = 32768
sfa_tensor = (((32,4),2),((16,4),2),(1,1)):(((16,4),1024),((0,1),512),(0,2048))
size(sfa_tensor) = 32768
The first thing we notice is that we have the size(a_tensor) = size(sfa_tensor). That is because we used tile_to_shape which takes care the entire target shape is covered with atoms in the order we specify in its argument.

We can also deduce that the shape of the tensor is 
(
(
(
32
,
4
)
,
M
32
·
4
)
,
(
(
16
,
4
)
,
K
16
·
4
)
,
(
(
1
,
1
)
,
L
1
·
1
)
, which is closely connected to the above fact because it ensures that the size along each mode is the same as for the target shape and from here of course the above fact follows immediately.

The beauty of tile_to_shape is that it automatically calculates the correct strides for the tile covering of atoms we desire.

Conclusion

I hope this blogpost helps readers to understand better how the blockscaled layouts are constructed. Feel free to connect with me on Linkedin. If you like to try out Blackwell programming yourself you may checkout Verda which provides convenient developer experience for Blackwell kernels.