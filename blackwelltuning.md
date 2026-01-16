1.4.1.1. Occupancy

The maximum number of concurrent warps per SM is 64 for compute capability 10.0 and 48 for compute capability 12.0. Other factors influencing warp occupancy are:

The register file size is 64K 32-bit registers per SM.
The maximum number of registers per thread is 255.
The maximum number of thread blocks per SM is 32 for devices of compute capability 10.0 and 12.0.
For devices of compute capability 10.0 shared memory capacity per SM is 228 KB. For devices of compute capability 12.0, shared memory capacity per SM is 128KB.
For devices of compute capability 10.0 the maximum shared memory per thread block is 227 KB. For devices of compute capability 12.0 the maximum shared memory per thread block is 99 KB.
For applications using Thread Block Clusters, it is always recommended to compute the occupancy using cudaOccupancyMaxActiveClusters and launch cluster-based kernels accordingly.

1.4.1.2. Thread Block Clusters

NVIDIA Hopper Architecture added a new optional level of hierarchy, Thread Block Clusters, that allows for further possibilities when parallelizing applications. Thread block clusters are supported by Blackwell GPUs as well. A thread block can read from, write to, and perform atomics in shared memory of other thread blocks within its cluster. This is known as Distributed Shared Memory. As demonstrated in the CUDA C++ Programming Guide, there are applications that cannot fit required data within shared memory and must use global memory instead. Distributed shared memory can act as an intermediate step between these two options.

Distributed Shared Memory can be used by an SM simultaneously with L2 cache accesses. This can benefit applications that need to communicate data between SMs by utilizing the combined bandwidth of both distributed shared memory and L2.

In order to achieve best performance for accesses to Distributed Shared Memory, access patterns to those described in the CUDA C++ Best Practices Guide for Global Memory should be used. Specifically, accesses to Distributed Shared Memory should be coalesced and aligned to 32-byte segments, if possible. Access patterns with non-unit stride should be avoided if possible, which can be achieved by using local shared memory, similar to what is shown in the CUDA C++ Best Practices Guide for Shared Memory.

The maximum portable cluster size supported is 8; however, NVIDIA Blackwell B200 GPU allows for a nonportable cluster size of 16 by opting in. Launching a kernel with a nonportable cluster size requires setting the cudaFuncAttributeNonPortableClusterSizeAllowed function attribute. Using larger cluster sizes may reduce the maximum number of active blocks across the GPU (refer to Occupancy).

for gemvkernel not necessarily all applicable to bigger shapes:
since the solutions are public, i can share some thoughts from my solution
i tried ld.global, cp.async, and cp.async.bulk, but the more sophisticated ones are not faster (didn't test cp.async.bulk.tensor much)
one of the most important tricks is cache modifier. use .cs / .L2::evict_last / .L1::no_allocate for A. that rules out cp.async since there is a PTX/SASS bug (illegal instruction encountered) on B200 when using cp.async with cache policy.
from my testing, the naive ld.global is still the fastest
other useful tricks: use fp16x2 math (since PTX only supports E2M1->FP16). I think a lot of ppl got this. there is also fma.rn.fp32.fp16 that can do FP16 FMA w/ FP32 accumulation on B200. B200 also has ld.global.v8.b32 (256-bit load per thread)
overall, this problem requires high occupancy i.e. an SM runs multiple threadblocks at the same time -> issue a lot of loads to saturate the mem BW. this requires some care around register usage. i noticed @sam had -maxrregcount=32 but i didn't have much luck trying to tune this value lol
I think this is also why things like cp.async.bulk / pipelining / persistent kernel don't really help much