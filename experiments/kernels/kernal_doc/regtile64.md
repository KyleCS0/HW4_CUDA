# Regtile64 Kernel Notes

This document records the current active optimized kernel:

```text
experiments/kernels/student_kernel_regtile64.cu
hw4_skeleton/kernels/student_kernel.cu
deliverable/kernels/student_kernel.cu
```

Only `hw4_skeleton/kernels/student_kernel.cu` is used when benchmarking, and
only `deliverable/kernels/student_kernel.cu` is intended for submission staging.

## Core Idea

The naive kernel assigns one CUDA thread to one output element:

```text
one thread -> one C[row, col]
```

That is simple, but inefficient because many threads repeatedly load the same
values from global memory.

The `regtile64` kernel instead makes each thread compute a small tile of output
values:

```text
one thread -> 4 x 4 C values
```

Each CUDA block computes a larger output tile:

```text
one block -> 64 x 64 C tile
```

The kernel stages chunks of `A` and `B` through shared memory, then reuses those
values many times while accumulating the output tile in registers.

## Key Design

Constants:

```cpp
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define BLOCK_X 16
#define BLOCK_Y 16
```

Meaning:

| Name | Meaning |
| --- | --- |
| `BM` | output tile height per block |
| `BN` | output tile width per block |
| `BK` | depth of each K tile loaded into shared memory |
| `TM` | number of output rows computed by each thread |
| `TN` | number of output columns computed by each thread |
| `BLOCK_X` | number of threads in x dimension |
| `BLOCK_Y` | number of threads in y dimension |

The block has:

```text
16 x 16 = 256 threads
```

Each thread computes:

```text
4 x 4 = 16 output values
```

So each block computes:

```text
256 threads * 16 outputs/thread = 4096 outputs
```

which matches:

```text
64 x 64 = 4096 output values
```

## Shared Memory Layout

The kernel uses:

```cpp
__shared__ float As[BM][BK];
__shared__ float Bs[BK][BN];
```

For each K tile:

```text
As stores a 64 x 16 tile from A
Bs stores a 16 x 64 tile from B
```

Together, these tiles are enough to update one `64 x 64` tile of `C` for a
16-wide slice of the K dimension.

The full matrix multiplication loops over K like this:

```cpp
for (int k0 = 0; k0 < K; k0 += BK) {
    load A and B tiles into shared memory
    __syncthreads()
    compute using the shared tiles
    __syncthreads()
}
```

## Register Accumulation

Each thread owns a `4 x 4` register tile:

```cpp
float acc[TM][TN];
```

These values are initialized to zero and accumulate partial dot products across
all K tiles.

Inside each `BK` tile, the thread reads:

```cpp
float aFrag[TM];
float bFrag[TN];
```

Then it updates all 16 local accumulators:

```cpp
acc[i][j] += aFrag[i] * bFrag[j];
```

This is the main speedup: one loaded `A` value is reused across 4 columns, and
one loaded `B` value is reused across 4 rows.

## Thread-To-Output Mapping

The block origin is:

```cpp
int blockRow = blockIdx.y * BM;
int blockCol = blockIdx.x * BN;
```

Each thread starts at:

```cpp
int rowBase = blockRow + ty * TM;
int colBase = blockCol + tx * TN;
```

So thread `(tx, ty)` computes:

```text
C[rowBase + 0, colBase + 0..3]
C[rowBase + 1, colBase + 0..3]
C[rowBase + 2, colBase + 0..3]
C[rowBase + 3, colBase + 0..3]
```

This gives a regular 2D partition of the `64 x 64` output tile.

## Why It Is Faster

The kernel improves performance mainly through:

1. Shared-memory reuse.
   - Tiles of `A` and `B` are loaded once from global memory.
   - Many threads reuse those values from shared memory.

2. Register tiling.
   - Each thread computes 16 outputs, not just one.
   - Partial sums stay in registers until the final write.

3. Higher arithmetic intensity.
   - More floating-point work is done per global-memory load.
   - This better matches how GEMM should use the GPU.

4. Coalesced global access for important paths.
   - Adjacent threads tend to load nearby values and write nearby output columns.

## Correctness Behavior

The kernel still implements the required formula:

```text
C = alpha * (A @ B) + beta * C
```

The final write is:

```cpp
C[idx] = alpha * acc[i][j] + beta * C[idx];
```

Bounds checks are kept around global reads and writes:

```cpp
globalRow < M
globalCol < K or N
row < M
col < N
```

The homework sizes are all multiples of 64, but the checks make the kernel more
robust and easier to reason about.

## Current Benchmark Record

Local benchmark command used:

```sh
make CUDA_ARCH=86
./main
```

Correctness:

```text
Passed all six sizes in the harness.
```

Measured performance:

| Size | GFLOPS |
| ---: | ---: |
| 128 | 301.495 |
| 256 | 1407.485 |
| 512 | 5994.003 |
| 1024 | 8335.987 |
| 2048 | 11172.693 |
| 4096 | 11020.381 |

The final grade is measured on the V100 target, so this table is useful for
local comparison but not a guarantee of grading performance.

## Possible Further Improvements

1. Tune tile sizes for V100.
   - Try `BM x BN = 64 x 128`.
   - Try `BM x BN = 128 x 64`.
   - Keep `BLOCK_X x BLOCK_Y = 16 x 16` or try different thread layouts.

2. Tune `BK`.
   - Current `BK = 16`.
   - Try `BK = 8` and `BK = 32`.
   - Larger `BK` can reduce loop overhead but may increase shared-memory pressure.

3. Add shared-memory padding.
   - Example:

   ```cpp
   __shared__ float As[BM][BK + 1];
   __shared__ float Bs[BK][BN + 1];
   ```

   - This may reduce bank conflicts depending on the access pattern.

4. Improve global load vectorization.
   - Use `float4` loads where alignment is guaranteed.
   - This can reduce load instruction count.
   - Add this only after the scalar-load version is stable.

5. Specialize for small sizes.
   - `regtile64` is designed for large matrices.
   - Smaller sizes may perform better with a simpler kernel.
   - Since grading emphasizes `4096`, this is lower priority.

6. Inspect register usage.
   - Use compiler output options to check registers per thread.
   - If register pressure is too high, reduce `TM`, `TN`, or adjust unrolling.

7. Try launch bounds.
   - `__launch_bounds__(256)` may help guide compiler resource usage.
   - Benchmark carefully; it can also hurt performance.

## Report Summary Text

A concise report explanation could be:

```text
The final kernel uses a 64 x 64 block tile and a 16-wide K tile. Each block has
16 x 16 threads, and each thread computes a 4 x 4 micro-tile of C in registers.
For each K tile, the block cooperatively loads a 64 x 16 tile of A and a 16 x 64
tile of B into shared memory. Threads then reuse these shared values to update
their register accumulators. This increases arithmetic intensity compared with
the naive one-output-per-thread kernel and reduces repeated global-memory
traffic.
```
