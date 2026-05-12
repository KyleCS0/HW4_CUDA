# HW4 Experiment Ideas

This file tracks possible optimization experiments for
`hw4_skeleton/kernels/student_kernel.cu`.

Use it as a tuning checklist. After each experiment, record correctness,
GFLOPS by size, and whether the change is worth keeping.

## Baseline

Current baseline:

```text
Naive one-thread-per-output kernel
Block size: 16 x 16
Each thread computes one C[row, col]
No shared memory
```

Record measured results here after running on the target environment.

## Experiment 1: Shared-Memory Tiling

Goal: reduce repeated global memory loads.

Idea:

```text
Each block computes one tile of C.
Threads cooperatively load a tile of A and a tile of B into shared memory.
Each thread computes one output element using the shared tiles.
```

Candidate parameters:

```text
TILE = 16
TILE = 32
```

Expected benefit:

```text
Global memory reuse increases.
Loads from A and B are shared across threads in the block.
```

Risks:

```text
Too much synchronization overhead for small tiles.
Shared-memory bank conflicts.
32 x 32 block means 1024 threads, which is the CUDA block limit and may reduce flexibility.
```

Suggested first version:

```text
TILE = 16
block = 16 x 16
one output per thread
```

## Experiment 2: Larger Output Tile With Fewer Threads

Goal: increase work per block without using too many threads.

Idea:

```text
Use a larger C tile, such as 32 x 32 or 64 x 64.
Each thread computes multiple C elements.
```

Candidate designs:

```text
Block tile: 32 x 32, thread tile: 2 x 2, block: 16 x 16 threads
Block tile: 64 x 64, thread tile: 4 x 4, block: 16 x 16 threads
Block tile: 64 x 32, thread tile: 4 x 2, block: 16 x 16 threads
```

Expected benefit:

```text
More arithmetic per global/shared load.
Accumulators stay in registers.
Better use of each thread.
```

Risks:

```text
Register pressure may reduce occupancy.
Indexing becomes more error-prone.
May need tuning for V100 specifically.
```

## Experiment 3: Register Tiling

Goal: keep multiple partial sums in registers.

Idea:

```text
Each thread owns a small micro-tile of C.
For example, each thread computes 4 rows x 4 columns.
```

Example:

```text
THREAD_TILE_M = 4
THREAD_TILE_N = 4
Each thread has float acc[4][4]
```

Expected benefit:

```text
Reuse loaded A values across several B columns.
Reuse loaded B values across several A rows.
Higher arithmetic intensity.
```

Risks:

```text
More registers per thread.
Can reduce occupancy.
More complicated bounds checks.
```

## Experiment 4: K-Tile Depth Tuning

Goal: tune how much of the K dimension each shared-memory phase covers.

Candidate values:

```text
BK = 8
BK = 16
BK = 32
```

Expected benefit:

```text
Larger BK can improve reuse and reduce loop overhead.
Smaller BK can reduce shared-memory usage and improve occupancy.
```

Measure both correctness and GFLOPS. The best value may differ across GPU
architectures, so final tuning should be checked on the grading V100.

## Experiment 5: Shared-Memory Padding

Goal: reduce shared-memory bank conflicts.

Idea:

```cpp
__shared__ float As[BM][BK + 1];
__shared__ float Bs[BK][BN + 1];
```

Expected benefit:

```text
Avoid repeated accesses mapping to the same shared-memory bank.
Can improve inner-loop throughput.
```

Risks:

```text
Uses slightly more shared memory.
May not help every layout.
Benchmark required.
```

## Experiment 6: Transposed Shared-Memory Layout

Goal: make shared-memory access patterns more favorable.

Idea:

```text
Load A into shared memory transposed, or arrange shared tiles so inner-loop
accesses are contiguous or bank-conflict-light.
```

Expected benefit:

```text
Better shared-memory bandwidth.
Cleaner per-thread register reuse.
```

Risks:

```text
Easy to introduce indexing mistakes.
Needs careful comments and verification.
```

## Experiment 7: Vectorized Global Loads

Goal: reduce global load instruction count and improve memory bandwidth.

Idea:

```text
Use float4 loads when reading A and B from global memory into shared memory.
```

Useful because tested sizes are multiples of 128:

```text
128, 256, 512, 1024, 2048, 4096
```

Expected benefit:

```text
Fewer load instructions.
Better memory coalescing.
Can improve bandwidth utilization.
```

Risks:

```text
Requires alignment.
Boundary handling becomes more delicate.
May complicate code before the main tiling strategy is stable.
```

## Experiment 8: Multiple Specialized Kernels

Goal: optimize separately for small and large sizes.

Idea:

```text
In runStudent, choose kernels based on M/N/K.
Use a simple kernel for small sizes.
Use a heavier tiled/register-tiled kernel for 1024+ or 2048+.
```

Possible split:

```text
M <= 256: simple or small-tile kernel
M >= 512: optimized tiled kernel
```

Expected benefit:

```text
Avoid overhead-heavy kernels on small matrices.
Maximize 4096 performance for grading.
```

Risks:

```text
More code paths to verify.
Final grade depends heavily on 4096, so avoid over-investing in tiny sizes.
```

## Experiment 9: Launch Bounds And Occupancy

Goal: improve compiler scheduling and occupancy.

Idea:

```cpp
__launch_bounds__(THREADS_PER_BLOCK)
```

Also inspect register usage from `nvcc` if possible.

Expected benefit:

```text
May improve occupancy or prevent excessive register allocation.
```

Risks:

```text
Can make performance worse if constraints are too strict.
Use only after the kernel structure is stable.
```

## Experiment 10: Loop Unrolling

Goal: reduce loop overhead and improve instruction scheduling.

Idea:

```cpp
#pragma unroll
for (int k = 0; k < BK; k++) {
    ...
}
```

Expected benefit:

```text
Better instruction-level parallelism.
Less loop-control overhead.
```

Risks:

```text
May increase register pressure or code size.
```

## Measurement Template

Copy this table after each experiment.

```text
Experiment:
Description:
Build command:
GPU:
Correctness:

Size    GFLOPS
128
256
512
1024
2048
4096

Keep? yes/no
Notes:
```

## Recommended Order

1. Shared-memory tiled kernel with one output per thread.
2. Tune `TILE = 16` vs `TILE = 32`.
3. Move to a `2 x 2` or `4 x 4` per-thread tile.
4. Tune `BM`, `BN`, and `BK`.
5. Add shared-memory padding.
6. Try vectorized loads.
7. Consider specialized kernels only after the main large-size kernel is solid.
