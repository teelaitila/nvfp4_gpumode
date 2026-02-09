Debugging deadlocks in warp-specialized GEMM kernels with CUDA-GDB
Feb 2, 2026

While writing warp specialized GEMMs for Blackwell with CUDA + PTX, it has come to my attention that there are some great resources on efficient kernel designs for this architecture, but a lack of resources on debugging the often cryptic or even completely opaque CUDA errors that arise during the kernel development process.

Classic “illegal memory access” errors are relatively straight-forward to resolve with compute-sanitizer. However, in kernels with a wide range of in-line PTX instructions, and asynchronous pipelines with complex synchronization patterns, these other CUDA errors became the bane of my existence:

CUDA error: an illegal instruction was encountered - which instruction? In some cases, I have many inline PTX instructions, SMEM descriptors, instruction descriptors, etc., that could all result in illegal instruction errors. How can I quickly isolate the specific instruction that was illegal, and determine why it is illegal?
Last but not least: deadlocks. There are so many things that could cause this in warp specialized kernels on Blackwell with a producer warp, consumer warp, epilogue warp group, double buffered TMEM, persistent scheduling, and complex synchronization to orchestrate it all safely… how can we narrow down the issue efficiently?
This post includes some quick notes on using cuda-gdb to accelerate debugging each of these issues that I have picked up along the way, which hopefully some others will find useful as well.

Each section includes useful cuda-gdb commands, along with an example debugging of real issues I encountered while writing a warp-specialized persistent GEMM kernel with 2 CTA tcgen05.mma instructions for sm100. The kernel code for this is on Github here. For background on these concepts, I recommend this excellent post tcgen05 for dummies which was the inspiration for this kernel.

Table of Contents:

Illegal instruction
Disassembling the instruction
Examining instruction operands
Deadlocks
Finding the Hung Thread
Navigating the Call Stack
Investigating the Consumer Warp
Inspecting Local Variables
Investigating the Producer Warp
Narrowing Down the Issue
The Fix
Prerequisite: make sure to compile CUDA kernels with -G (debug info for device code) and -g (debug info for host code)!

Illegal instruction
TL;DR

Run your code with cuda-gdb <executable> or cuda-gdb --args <program> (e.g. cuda-gdb --args pytest test.py)
Once the interactive session opens, use "r" to run the program.
(observe crash)
cuda-gdb catches the halt and it shows "triggered at <PC address>" in the terminal session
Use command disas <PC address>, +16:
This disassembles the 16 bytes after the program counter address into human readable SASS instruction output, showing the instruction that caused the crash. 16 bytes is the width of an instruction on the sm100 architecture.
Below is a real example where I used this:

Disassembling the instruction

disas <PC address>, +16
... 
UTMALDG.3D [UR8][UR4]  
This is a 3d TMA async load from GMEM to SMEM. UR8 and UR4 are uniform registers, meaning unlike regular registers R0, R1, etc., they are shared across all threads in a warp.

Why is this instruction illegal? Let’s inspect the operands:

Examining instruction operands

info register UR8 UR4
... 
UR8  0x460 
UR4 was a large value that looked like the 64-bit generic address of the Tensormap used for the TMA load. UR8 was a small value that looked like the 32-bit SMEM shared address of the destination.

Since I am using CU_TENSOR_MAP_SWIZZLE_128B in my Tensormap for the TMA load, the destination SMEM address must be 1024-byte aligned.

Converting hex to decimal, we see 0x460 = 1120 is not 1024-byte aligned! This narrows down the issue to how we calculate the SMEM offset for each A/B tile in SMEM, and we’re able to more quickly resolve the issue from there.

Deadlocks
TL;DR

Run your code with cuda-gdb <executable> or cuda-gdb --args <program> (e.g. cuda-gdb --args pytest test.py)
Once the interactive session opens, use "r" to run the program.
Once you hit the hang, hit Ctrl+C to send a SIGINT signal, which will kill the python thread and yield back control to cuda-gdb.
cuda-gdb session should now be in the thread context of a hung thread.
Use cuda thread (x, y, z) to switch to another thread context as necessary (e.g. thread in producer vs consumer warps)
Navigate the call stack with up or down to figure out where that is in its execution lifetime, etc.
Use print <variable> to print variables.
Combining these tools, you should be able to narrow down the issue and accelerate the debugging process.
Other useful cuda-gdb commands for this include:

set cuda printf_flushing on - flush printf calls when a GPU breakpoint is hit, to be sure you can see them.
set cuda break_on_launch application - break on the first instruction of the kernel, allowing you to set additional breakpoints in the command line and more
break <file>:<line> if <condition> - set a conditional breakpoint for a particular thread (e.g., threadIdx.x == 0 && block_id > start_block_id)
cuda info clusters - info on threadblock clusters (on Hopper and later)
cuda info barriers - info on barrier objects
Below is a concrete example of how this helped me debug a deadlock:

Optional context: kernel design of warp-specialized persistent GEMM kernel with 2 CTA tcgen05.mma

To get the most out of the debugging steps below, it’s best to have some context about what it is we are debugging.

It is difficult to capture all the details in a readable diagram, but below is a rough sketch of the Blackwell GEMM kernel design in question, including the synchronization logic:

2cta-mma

Again, for more details on tcgen05 PTX instruction semantics and Blackwell GEMM kernel design, I refer you to gaunerst’s blog.

Problem: A deadlock has occured after transforming a previous, working iteration of this kernel into persistent kernel, which launches one CTA per SM, each with a large SMEM queue, where it will reside on that SMEM chugging through multiple output tiles rather than only computing one. With warp-specialization and double buffered TMEM usage, this should help hide both load latency and store latency. However, we need to solve this deadlock first!

Finding the Hung Thread

We run CUDA GDB. When it hangs, hit Ctrl+C, which will halt on one of the hung threads. Note the block ID, warp ID, and thread ID in the output. For warp-specialized kernels, this is critical:

[New Thread 0x7ffe301c1640 (LWP 19512)]
[New Thread 0x7ffe30bc2640 (LWP 19513)]
...

^C
Thread 1 "python" received signal SIGINT, Interrupt.
[Switching focus to CUDA kernel 0, grid 4, cluster (0,0,0), cluster dim (2,1,1), block (0,0,0), thread (0,0,0), device 0, sm 142, warp 1, lane 0]
0x00007ffe439bac10 in mbarrier_try_wait_parity (mbar_addr=197728, parity=0)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:134
134             : "r"(mbar_addr), "r"(parity) // inputs
Notice in the output we can see the line number in our source code where it’s hung, which corresponds to an inline PTX instruction mbarrier.try_wait.parity.acquire.cta.shared::cta.b64. This is a non-blocking test of the parity bit of the mbarrier object we are using for synchronization, which we check in a loop until the bit has flipped, signaling completion.

Navigating the Call Stack

Use up to get to a more informative part of the call stack:

(cuda-gdb) up

#1  mbarrier_wait_parity<<<(148,1,1),(192,1,1)>>> (mbar_addr=197728, parity=0)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:141
141       while (!mbarrier_try_wait_parity(mbar_addr, parity)) {
(cuda-gdb) up

#2  ws_gemm_2cta_mma<6, 128, 256, 64, 128, 256, 16><<<(148,1,1),(192,1,1)>>> (a_map=<error reading variable: Cannot access memory at address 0x0>,
    b_map=<error reading variable: Cannot access memory at address 0x80>, C=0x7ffbf4000000, M=4096, N=4096, K=4096)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:538
538                 mbarrier_wait_parity(mma_mbar_addr, mma_parity);
I can now see that for warp 1 (one of the warps in the epilogue warpgroup), it’s hung waiting for the mma_mbar mbarrier that signals the epilogue that the target TMEM buffer is ready for use (i.e., all relevant tcgen05.mma instrucions have completed).

Investigating the Consumer Warp

Let’s see what’s going on with the MMA warp in CTA 0. You can switch thread context like so:

cuda thread (160,0,0)
This switches thread context to the first thread of the MMA warp (warp ID 5), which is the important one.

Optional context

This thread in CTA 0 warp 5 is important because it is the only one we use, and it does the following:

For each BK tile along the K dimension of the given output tile we are computing, we do:
For each A/B tiles, iterate through one swizzle atom at a time (BMx128byte for A, BNx128byte for B).
For each swizzle atom, iterate through MMA tiles for the given MMA_K dimension, issuing async tcgen05.mma instructions with the cta_group::2 modififer for 2 CTA mma, using a combination of SMEM on CTA 0 and CTA 1 (via DSMEM).
As we complete using each SMEM buffer (A/B tiles), use tcgen05.commit multicast with the smem_empty_mbar to signal the producer warp on both CTAs that the buffers can be safely re-used.
After we complete ALL BK tiles, we tcgen05.commit multicast to the mma_mbar to signal epilogue warpgroups on both CTAs that the target TMEM buffer is ready for use.
Again, we switch to the target thread then use up to get to a more useful part of the call stack:

(cuda-gdb) cuda thread (160,0,0)
[Switching focus to CUDA kernel 0, grid 4, cluster (0,0,0), cluster dim (2,1,1), block (0,0,0), thread (160,0,0), device 0, sm 142, warp 6, lane 0]
0x00007ffe439b70e0      134             : "r"(mbar_addr), "r"(parity) // inputs
(cuda-gdb) up
#1  mbarrier_wait_parity<<<(148,1,1),(192,1,1)>>> (mbar_addr=197672, parity=1)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:141
141       while (!mbarrier_try_wait_parity(mbar_addr, parity)) {
(cuda-gdb) up
#2  ws_gemm_2cta_mma<6, 128, 256, 64, 128, 256, 16><<<(148,1,1),(192,1,1)>>> (a_map=<error reading variable: Cannot access memory at address 0x0>,
    b_map=<error reading variable: Cannot access memory at address 0x80>, C=0x7ffbf4000000, M=4096, N=4096, K=4096)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:483
483                     mbarrier_wait_parity(smem_full_mbar_addr + consumer_next_buf * sizeof(uint64_t), smem_full_parity[consumer_next_buf]);
Inspecting Local Variables

Hmm, the consumer warp is hung waiting for the smem_full mbarrier that the producer warps on both CTA 0 and CTA 1 use to signal the consumer warp on CTA 0 that a particular shared memory buffer is ready for use. We can check which iteration this happens on by printing local variables like so:

(cuda-gdb) print block_k_idx
$1 = 59

(cuda-gdb) print num_blocks_k
$2 = 64
Interesting, it doesn’t happen immediately in the main loop - rather, the hang happens on a seemingly random block K index, which defines which BK tile we’re at along K in the main/inner loop when computing the output block.

This gives us useful information, though: that we are successfully processing many BK tiles in the inner loop before hanging, meaning the whole synchronization pipeline is working as expected for part of the lifetime, but hitting a bug at some point.

Investigating the Producer Warp

Why isn’t the consumer receiving the signal from the producer that the next SMEM buffer A/B tiles are ready for use?

Let’s check the producer, warp ID 4, to see if it appears to have sent any signals or not. Once again the first thread in the warp is the important one.

Optional context

The producer warp runs on both CTA 0 and CTA. The first thread in the warp is the important one because it is the only one we use, and does the following:

For each output tile we are computing in the static schedule:
For each BK tile along the K dim of A and B for the given output tile we are computing:
First, wait on the smem_empty mbarrier if we are re-using a buffer (so all iterations except the first QUEUE_SIZE iterations before we wrap around to re-use buffers).
We do a 3D TMA loads with a 128b swizzle to asynchronously move tiles of A/B from global memory into shared memory buffers (queue) in a hierchical layout of “swizzle atoms” (BMx128byte for A, BNx128byte for B).
We use mapa.shared::cluster.u32 to map the CTA 1 smem_full mbarrier to CTA 0 (since CTA 0 runs the 2 CTA tcgen05.mma instruction, and thus must know when A/B tiles on both CTAs are ready).
The producer threads on both CTAs use the cta_group::2 modifier on the cp.async.bulk.tensor instruction, so the smem_full mbarrier we provide it will count thread arrivals and byte arrivals from both CTAs.
Switching to the first thread on the producer warp on CTA 0:

(cuda-gdb) cuda thread (128,0,0)

[Switching focus to CUDA kernel 0, grid 4, cluster (0,0,0), cluster dim (2,1,1), block (0,0,0), thread (128,0,0), device 0, sm 142, warp 5, lane 0]
0x00007ffe439b4a30      137       return static_cast<bool>(wait_complete);
Navigate through the call stack:

(cuda-gdb) up

#1  mbarrier_wait_parity<<<(148,1,1),(192,1,1)>>> (mbar_addr=197720, parity=1)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:141
141       while (!mbarrier_try_wait_parity(mbar_addr, parity)) {
(cuda-gdb) up

#2  ws_gemm_2cta_mma<6, 128, 256, 64, 128, 256, 16><<<(148,1,1),(192,1,1)>>> (a_map=<error reading variable: Cannot access memory at address 0x0>,
    b_map=<error reading variable: Cannot access memory at address 0x80>, C=0x7ffbf4000000, M=4096, N=4096, K=4096)
    at /home/dvm/gemm/blackwell/tcgen05_persistent_2cta_warp_specialized/tcgen05_persistent_2cta_warp_specialized.cu:430
430                         mbarrier_wait_parity(smem_empty_mbar_addr + producer_next_buf * sizeof(uint64_t), smem_empty_parity[producer_next_buf]);
We can see the producer warp is hung waiting for the signal that a shared memory buffer is ready to be re-used. Hmm, this seems to be the start of the hang - the whole pipeline grinds to a halt because the producer isn’t producing anything.

One good next step here that I didn’t do would be to check the producer in CTA 1 as well, as it has to send the mbarrier arrival across DSMEM, which can be tricky to set up properly.

Narrowing Down the Issue

This narrows down the issue to 2 general areas to investigate:

Consumer warp signaling smem_empty mbarrier improperly
Producer warp checking smem_empty mbarrier improperly
To get over the finish line, we do need some old-fashioned code analysis and debugging, but we have drastically narrowed down the possibility space.

In this case, the fact that this synchronization pipeline worked properly with a non-persistent kernel (i.e., no grid-strided loop over blocks) suggested I should look for what could change when iterating to a new block. After tracing through the code, I noticed the condition the producer uses to check the smem_empty mbarrier assumed only 1 output tile per block:

if (block_k_idx >= QUEUE_SIZE)
{
    mbarrier_wait_parity(smem_empty_mbar_addr + producer_next_buf * sizeof(uint64_t), smem_empty_parity[producer_next_buf]);
    smem_empty_parity[producer_next_buf] ^= 1;
}
Critically, block_k_idx resets to 0 for every new block, and we will skip waiting on the smem_empty mbarrier and flipping the parity, which will get our producer and consumer out of sync!

The Fix

The fix is simple: wait on the mbarrier once we start re-using SMEM queue buffers within the first output tile, AND for every iteration of the 2nd+ output tiles.

- if (block_k_idx >= QUEUE_SIZE)
+ if (block_k_idx >= QUEUE_SIZE || bid > start_bid) { ... }
{
    mbarrier_wait_parity(smem_empty_mbar_addr + producer_next_buf * sizeof(uint64_t), smem_empty_parity[producer_next_buf]);
    smem_empty_parity[producer_next_buf] ^= 1;
}
Conclusion
Check out the cuda-gdb docs for more details than the basic workflows I outlined here. I put it off for far too long, and regret the time wasted with AI hallucinations and staring at code over and over!