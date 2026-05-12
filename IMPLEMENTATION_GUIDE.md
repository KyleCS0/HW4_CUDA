# HW4 Implementation Guide

This is the working guide for finishing HW4: CUDA Matmul Optimization.

## Assignment Goal

Implement an optimized CUDA SGEMM kernel for:

```text
C = alpha * (A @ B) + beta * C
```

The matrices are row-major. `A` is `M x K`, `B` is `K x N`, and `C` is
`M x N`. In this homework, all tests use square matrices where:

```text
M = N = K
```

The tested sizes are:

```text
128, 256, 512, 1024, 2048, 4096
```

The only source file that should be edited is:

```text
hw4_skeleton/kernels/student_kernel.cu
```

Do not modify the harness, Makefile, headers, cuBLAS reference code, or stdout
format. The TA will copy only `student_kernel.cu` into the official skeleton.

## Hard Rules

1. Do not call cuBLAS, cuDNN, CUTLASS, or any vendor GEMM library.
2. Do not modify files outside `kernels/student_kernel.cu` for the submitted solution.
3. Keep the `runStudent(...)` signature unchanged.
4. Correctness must pass for all six sizes with tolerance `1e-2`.
5. If any size fails correctness, performance tier and competition rank become 0.

## Performance Target

The main performance score is based on the `4096 x 4096` case.

Thresholds:

| Tier | GFLOPS at 4096 | Homework points from performance tier |
| --- | ---: | ---: |
| T1 | >= 200 | 20 / 40 |
| T2 | >= 2050 | 25 / 40 |
| T3 | >= 3700 | 30 / 40 |
| T4 | >= 7600 | 37 / 40 |
| T5 | >= 9500 | 40 / 40 |

These thresholds are cumulative. Reaching T4 means the tier score is 37/40, not
20 + 25 + 30 + 37.

The benchmark computes:

```text
GFLOPS = 2 * M * N * K * repeat_times / elapsed_time / 1e9
```

The factor `2` comes from one multiply and one add per inner-loop step.

## Build And Run

On the target cluster:

```sh
module load cuda
make
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 1 ./main
```

The default CUDA architecture is `sm_70`, matching the V100 grading GPU.

## Development Tasks

1. Establish a simple correct kernel.
   - One thread computes one `C[row, col]`.
   - Use row-major indexing:
     - `A[row * K + k]`
     - `B[k * N + col]`
     - `C[row * N + col]`
   - Apply `alpha` and `beta` exactly.

2. Add global-memory coalescing.
   - Map adjacent threads to adjacent output columns.
   - Make stores to `C` contiguous across each warp.
   - Prefer block shapes where `threadIdx.x` corresponds to columns.

3. Add shared-memory tiling.
   - Load one tile of `A` and one tile of `B` per block.
   - Reuse tile data across many multiply-adds.
   - Synchronize after loading and after consuming each tile.

4. Increase arithmetic intensity.
   - Have each thread compute more than one output element.
   - Start with a small per-thread tile, then tune.
   - Keep accumulator values in registers.

5. Reduce shared-memory bank conflicts.
   - Consider padding shared arrays.
   - Consider storing the `A` tile transposed in shared memory if it improves
     access patterns.

6. Add vectorized loads where safe.
   - Use `float4` loads/stores only when alignment and bounds are correct.
   - Since test sizes are multiples of 128, main paths can assume clean tiles,
     but correctness should still be protected if using general indexing.

7. Tune parameters.
   - Try block tiles such as `64x64`, `64x128`, or `128x64`.
   - Try `BK` values such as `8`, `16`, or `32`.
   - Balance occupancy, register count, shared memory, and instruction reuse.

## Principles

Correctness comes first. A fast wrong kernel gets no performance or rank points.

Keep each optimization measurable. After every meaningful change, record GFLOPS
for all sizes, especially `4096`.

Prefer simple, stable indexing over clever code until the baseline is correct.
Most CUDA SGEMM bugs come from off-by-one tile bounds, wrong row-major indexing,
or forgetting the `beta * C` term.

Avoid changing the external interface. All tuning should happen inside
`student_kernel.cu` through kernel code, launch configuration, constants, helper
device functions, or templates.

Tune for V100. The grading GPU is compute capability 7.0, so use optimizations
that fit Volta behavior: enough occupancy, coalesced memory traffic, shared
memory reuse, and register-level accumulation.

## Suggested Implementation Path

Start with a known-correct naive kernel, even if it is slow. Then move in small
steps:

1. Naive one-thread-per-output kernel.
2. Coalesced one-thread-per-output kernel with a reasonable `16x16` or `32x8`
   block.
3. Shared-memory tiled kernel, one output per thread.
4. Register-tiled kernel, multiple outputs per thread.
5. Shared-memory layout improvements and padding.
6. Vectorized global loads.
7. Tune tile sizes and compare results.

Do not jump directly to a complex final kernel. It will be harder to debug and
harder to explain in the report.

## Measurement Notes

The harness runs:

1. cuBLAS reference once.
2. Student kernel once for verification and warm-up.
3. Student kernel 50 timed iterations.

The timed loop repeatedly updates `C`, so the kernel must correctly implement
the full formula with the current input `C` every time:

```text
C = alpha * A * B + beta * C
```

After each matrix size, the harness resets `dC` back to the cuBLAS reference
output before moving to the next size.

## Report Checklist

The report should be short, around two pages.

Include:

1. Optimization steps in the order added.
2. A small table showing GFLOPS after each important step.
3. Final kernel design:
   - block tile size
   - thread tile size
   - `BK` / K-tile depth
   - shared-memory layout
   - register accumulation strategy
   - synchronization strategy
   - vectorization, if used
4. Things tried that did not help.
5. Optional GFLOPS scaling plot over all six sizes.

## Submission Checklist

Zip root must contain:

```text
kernels/student_kernel.cu
Team_<Team Number>_HW4_report.pdf
```

The zip filename must be:

```text
Team_<Team Number>_HW4.zip
```

Before submission:

1. Rebuild from clean state if possible.
2. Run `./main` or the required `srun` command.
3. Confirm all six sizes pass verification.
4. Save the final stdout numbers for the report.
5. Check that no forbidden library calls exist in `student_kernel.cu`.
