# Optimization Trajectory Summary

Concise view of the measured optimization path from `optimization_log.md`, `perf_history.csv`, and the current `student_kernel.cu`.

## Current State

- Live kernel: **Iteration 6**, `cp.async` double-buffered shared-memory/register-tiled SGEMM.
- Best measured 4096 result: **16,279.588 GFLOPS**.
- Baseline 4096 result: **2,229.507 GFLOPS**.
- Net 4096 improvement: **+630.2%**.
- Status: **CONTINUE**.

## Attempt Timeline

| Iter | Attempt | Status | 4096 GFLOPS | Change | Takeaway |
|------|---------|--------|-------------|--------|----------|
| 0 | Naive 16x16 output kernel | Baseline | 2,229.507 | - | No data reuse; fully memory-bound. |
| 1 | Shared memory + 8x8 register tile | Kept | 15,003.237 | +573.0% | Major jump from global-load reuse. |
| 2 | Shared-memory padding | Reverted | 10,268.238 | -31.6% | Padding hurt `Bs` access/alignment behavior. |
| 3 | Transposed `B` tile in shared memory | Reverted | 5,818.605 | -61.2% | Created worse shared-memory bank conflicts. |
| 4 | `float4` global loads | Reverted | 11,109.519 | -26.0% | Wider loads likely reduced ILP and added overhead. |
| 5 | `BK=32` | Kept | 15,491.828 | +3.2% | Fewer syncs gave a small gain. |
| 6 | `cp.async` double buffering, `BK=16` | Kept | 16,279.588 | +5.1% | Hid part of global-load latency. |

## Kept Kernel Progression

Each cell shows `GFLOPS (change vs previous kept version)`.

| Size | Baseline | Iter 1 | Iter 5 | Iter 6 |
|------|----------|--------|--------|--------|
| 128 | 677.445 | 107.985 (-84.1%) | 112.292 (+4.0%) | 127.658 (+13.7%) |
| 256 | 1,529.473 | 519.066 (-66.1%) | 543.585 (+4.7%) | 638.550 (+17.5%) |
| 512 | 2,018.501 | 2,300.795 (+14.0%) | 2,420.144 (+5.2%) | 2,918.447 (+20.6%) |
| 1024 | 2,112.658 | 7,874.280 (+272.7%) | 8,249.400 (+4.8%) | 11,764.115 (+42.6%) |
| 2048 | 2,173.543 | 11,872.569 (+446.2%) | 12,277.643 (+3.4%) | 12,764.594 (+4.0%) |
| 4096 | 2,229.507 | 15,003.237 (+573.0%) | 15,491.828 (+3.3%) | 16,279.588 (+5.1%) |

## Evolution

The decisive improvement was iteration 1: shared-memory tiling plus 8x8 register tiling raised 4096 performance from **2.23 TFLOPS** to **15.00 TFLOPS**. Small sizes regressed because the large 128x128 tile has high setup overhead, but the target size improved massively.

Iterations 2-4 explored memory-layout and load-width ideas. All were reverted because they reduced 4096 performance, mostly by worsening shared-memory behavior or scheduling efficiency.

Iteration 5 increased `BK` to 32 and gave a small gain by reducing synchronization frequency. Iteration 6 then moved to `cp.async` double buffering with `BK=16`, improving further by overlapping global-to-shared copies with compute.

## Current Kernel Shape

- Tile: `BM=128`, `BN=128`, `BK=16`.
- Per-thread output: `TM=8`, `TN=8`.
- Block: 256 threads.
- Shared memory: double-buffered `As[2][BM][BK]` and `Bs[2][BK][BN]`.
- Async copy: inline PTX `cp.async.ca.shared.global`.

## Reverted Ideas

| Idea | Result |
|------|--------|
| Shared-memory padding | Regressed to 10,268.238 GFLOPS. |
| Transposed shared `B` tile | Regressed to 5,818.605 GFLOPS. |
| `float4` global loads | Regressed to 11,109.519 GFLOPS. |

## Next Step

Continue from iteration 6. Test one focused shared-memory or register-layout change at a time, and keep using the build, benchmark, and rollback protocol from `AGENT_INSTRUCTIONS.md`.
