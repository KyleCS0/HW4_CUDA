# Kernel Optimization Log

## Baseline

Naive one-thread-per-output kernel, 16×16 block, no shared memory.

| Size | GFLOPS |
|------|--------|
| 128  | 677.445 |
| 256  | 1529.473 |
| 512  | 2018.501 |
| 1024 | 2112.658 |
| 2048 | 2173.543 |
| 4096 | 2229.507 |

### Notes for Next Iteration
- Plateaus at ~2229 GFLOPS, ~6.5% of 34100 peak. Fully memory-bound with zero reuse.
- First direction: smem tiling + register tiling (BM=128, BN=128, BK=16, TM=8, TN=8), arithmetic intensity ≈ 32 FLOPs/byte.
- Expected 5-10x gain from eliminating redundant global loads.

## Iteration 1 — Smem + Register Tiling

### Plan
- **What**: Implement shared-memory tiling with register tiling: BM=128, BN=128, BK=16, TM=8, TN=8. 256 threads per block, each computes 8×8 output elements.
- **Why**: Baseline has zero reuse. This raises arithmetic intensity from ~0 to ~32 FLOPs/byte (= BM/4), just below the 37.4 crossover. Each tile loads A/B once into smem and reuses 128 times.
- **Prediction**: 4096 GFLOPS improves by ~5–8× (target ~12,000–18,000 GFLOPS).
- **Risk**: Register pressure (80 regs/thread with TM=TN=8); potential bank conflicts on As column access.
- **Rejected**: BM=64/TM=4 (intensity only 16, clearly memory-bound); plain smem tiling without register tiles (block limited to 32×32 = 1024 threads, intensity = 32/4 = 8).

### Result
**Status**: KEPT
**Correctness**: PASS
**Δ 4096**: +573%

| Size | GFLOPS |
|------|--------|
| 128  | 107.985 |
| 256  | 519.066 |
| 512  | 2300.795 |
| 1024 | 7874.280 |
| 2048 | 11872.569 |
| 4096 | 15003.237 |

### Notes for Next Iteration
- 4096 at 15,003 GFLOPS (~44% of 34,100 peak). T5 target is ~18,000. Gap is ~3,000 GFLOPS.
- Small sizes (128, 256) severely regressed due to 128×128 tile overhead on tiny grids — worth revisiting but not the priority metric.
- At 44% efficiency, still likely memory-bandwidth limited (intensity ~32 < 37.4 crossover). Increasing BK would raise intensity but also smem pressure.
- Top candidates: (a) shared-memory padding (+1 col on As) to eliminate bank conflicts on column reads, (b) vectorized float4 loads to improve global memory bandwidth, (c) cp.async double-buffering to hide global load latency.
- Bank conflict analysis: As[threadRow*TM+m][k] — threads in a warp with different threadRow hit rows 0,8,16,... in same column k → addresses differ by 128 floats = 512 bytes → bank (row*16+k)%32 = same bank for rows 0,8,16,... (16*8=128, 128%32=0). 2-way conflicts. Padding As to [BM][BK+1] fixes this.
- Try smem padding next as it's low-risk and targets confirmed bank conflict pattern.

## Iteration 2 — Shared Memory Padding

### Plan
- **What**: Pad As to [BM][BK+1] and Bs to [BK][BN+1].
- **Why**: As column reads (As[threadRow*TM+m][k]) access every 8th row → stride 16 → same bank every 8 threads. +1 padding breaks this alignment.
- **Prediction**: 4096 GFLOPS improves ~5–15% from reduced bank conflict stalls.
- **Risk**: Minimal — correctness unaffected, smem increases by ~0.5 KB.
- **Rejected**: Transposing A tile (more complex, similar benefit) — try only if padding insufficient.

### Result
**Status**: REVERTED
**Correctness**: PASS
**Δ 4096**: -31.6% (regression: 15003 → 10268)

| Size | GFLOPS |
|------|--------|
| 128  | 104.660 |
| 256  | 501.414 |
| 512  | 2221.418 |
| 1024 | 7589.148 |
| 2048 | 8044.779 |
| 4096 | 10268.238 |

### Notes for Next Iteration
- Padding Bs[BK][BN+1] caused a consistent ~32% regression at large sizes. Likely cause: Bs row stride = 129 (odd) changes memory alignment in ways that hurt L1/shared bandwidth, or BN+1=129 padding negatively impacts the load coalescing pattern.
- As[BM][BK+1] padding alone might be safe; Bs padding is the culprit. Could try padding only As.
- Better direction: transpose Bs in shared memory (store as Bs[BN][BK]) to eliminate the 4-way bank conflict on column reads. This is a bigger change but addresses the root issue.
- Alternative: vectorized float4 loads to better utilize global memory bandwidth. Load A and B in 128-bit chunks.
- Next plan: try float4 vectorized loads — this is well-tested and avoids smem layout complexity.

## Iteration 3 — Transposed B Tile in Smem

### Plan
- **What**: Store B tile transposed in smem as BsT[BN][BK], load B[gRow][gCol] → BsT[c][r]. Access BsT[threadCol*TN+n][k].
- **Why**: Current Bs[k][threadCol*TN+n] has 4-way bank conflicts (stride 8 with BN=128 means 4 threads share each bank). Transposed access gives threadCol*TN+n as the fast index → consecutive across threads → conflict-free.
- **Prediction**: 4096 GFLOPS improves ~10–20%.
- **Risk**: Must be careful about the loading: the global B load is row-major, so transposing in smem means a different index mapping.
- **Rejected**: float4 loads (BN=128 would need to be divisible by 4 and the stride must be aligned — worth trying after correctness is confirmed here).

### Result
**Status**: REVERTED
**Correctness**: PASS
**Δ 4096**: -61% (15003 → 5819)

| Size | GFLOPS |
|------|--------|
| 128  | 58.278 |
| 256  | 254.813 |
| 512  | 1071.276 |
| 1024 | 4125.947 |
| 2048 | 4259.596 |
| 4096 | 5818.605 |

### Notes for Next Iteration
- Transposed Bs[BN][BK] caused 16-way bank conflicts on inner-loop access BsT[threadCol*TN+n][k]: all 16 threadCol values map to addresses with stride BK=16, which cycles through banks in groups of 2 → 16-way conflict.
- Bs bank conflict: root cause is that Bs[k][threadCol*TN+n] with BN=128 has stride-8 pattern → 4-way conflicts (32 banks / 8 stride = 4 threads per bank). Not catastrophic (~4x slowdown per smem access).
- Better next step: float4 vectorized global loads. At 48% of peak bandwidth (437 GB/s), 2x vectorization could help significantly. BM*BK=2048 and BK*BN=2048 floats → 512 float4s each, 2 per thread with 256 threads.
- After float4, if still bandwidth-limited, consider cp.async double-buffering.

## Iteration 4 — Float4 Vectorized Global Loads

### Plan
- **What**: Replace scalar global loads with float4 for both A and B tiles.
- **Why**: Effective bandwidth is ~437 GB/s (48% of 912 GB/s peak). Float4 loads issue wider memory transactions, improving coalescing and throughput.
- **Prediction**: 4096 GFLOPS improves ~10–20% (16,000–18,000 GFLOPS).
- **Risk**: Addresses must be 16-byte aligned; blockCol = blockIdx.x * BN = multiples of 128 ✓; bk = multiples of BK=16 ✓. Boundary sizes (e.g., 512 ≠ multiple of 4 × 4) need guards.
- **Rejected**: BK increase (no arithmetic intensity gain); Bs padding only (4-way conflict doesn't dominate at current 44% efficiency).

### Result
**Status**: REVERTED
**Correctness**: PASS
**Δ 4096**: -26% (15003 → 11110)

### Notes for Next Iteration
- Float4 loads slower — likely because 8 independent 32-bit loads give more ILP to the warp scheduler than 2 float4 loads, allowing better latency hiding.
- The boundary-check branching in the float4 path may also add overhead even if never taken.
- Current bottleneck seems not to be raw bandwidth but latency: 15003 GFLOPS = 44% of peak at 437 GB/s = 48% bandwidth. Latency hiding is the gap.
- Best remaining option: cp.async double-buffering. Load next tile's data while computing current tile — hides ~200-400 cycle global load latency.
- Simpler alternative to try first: increase BK to 32. Halves __syncthreads() overhead, doubles per-iteration compute, same occupancy (smem grows to 32 KB but register limit already caps at 3 blocks/SM).

## Iteration 5 — Increase BK to 32

### Plan
- **What**: Change BK from 16 to 32.
- **Why**: Doubles compute per K-tile (128*128*32 FMAs vs 128*128*16) while keeping the same number of global loads. Halves __syncthreads() count per K dimension. Smem grows from 16 KB to 32 KB per block; register limit already caps at 3 blocks/SM, so occupancy unchanged. More compute between syncs gives warp scheduler more room to hide memory latency.
- **Prediction**: 4096 GFLOPS improves ~5–15%.
- **Risk**: Smem doubles but still fits (3 blocks × 32 KB = 96 KB < 100 KB). No correctness risk.
- **Rejected**: cp.async double-buffering (try next if this insufficient — more complex).

### Result
**Status**: KEPT
**Correctness**: PASS
**Δ 4096**: +3.2%

| Size | GFLOPS |
|------|--------|
| 128  | 112.292 |
| 256  | 543.585 |
| 512  | 2420.144 |
| 1024 | 8249.400 |
| 2048 | 12277.643 |
| 4096 | 15491.828 |

### Notes for Next Iteration
- +3.2% from BK=32: halved sync overhead, doubled per-tile compute. Still at 45% of peak.
- Smem = 32 KB/block, 3 blocks/SM (register-limited), 24 warps. 50% occupancy.
- Bandwidth: 451 GB/s effective = 49% of 912 peak. Both compute and bandwidth at ~45-49%, suggesting latency is the true bottleneck — loads stall warps at __syncthreads().
- Best next step: cp.async double-buffering. With BK=16 + 2 buffers = 32 KB smem (same as BK=32 single buffer), loads of tile T+1 can fully overlap with compute of tile T. Expected gain: ~10-20%.
- After cp.async: bank conflict fix for Bs (4-way conflict) is still viable.

## Iteration 6 — cp.async Double-Buffering

### Plan
- **What**: Double-buffer As/Bs with BK=16; use PTX cp.async.ca to issue async global→smem copies while computing the previous tile.
- **Why**: Current stall: all warps wait at __syncthreads() until tile loads complete (~180 ns latency per load). cp.async moves loads of tile T+1 into the background during tile T's compute, hiding that latency. BK=16 + 2 buffers = 32 KB smem = same occupancy as BK=32 single buffer.
- **Prediction**: 4096 GFLOPS improves ~10–20% (17,000–18,500 GFLOPS).
- **Risk**: PTX inline assembly; must correctly manage pipeline stages and __syncthreads boundaries.
- **Rejected**: BK=32 + double buffer (smem = 64 KB, drops to 1 block/SM, kills occupancy).

### Result
**Status**: KEPT
**Correctness**: PASS
**Δ 4096**: +5.1%

| Size | GFLOPS |
|------|--------|
| 128  | 127.658 |
| 256  | 638.550 |
| 512  | 2918.447 |
| 1024 | 11764.115 |
| 2048 | 12764.594 |
| 4096 | 16279.588 |

### Notes for Next Iteration
- cp.async +5.1%: latency hiding from prefetching next tile during current tile's compute helped.
- 4096 now at 47.7% of 34,100 peak. Still well below T5 (~18,000 GFLOPS needed).
- Smem bank conflicts are now the primary bottleneck. Analysis:
  - Bs[k][threadCol*TN+n] with TN=8, 16 threadCol values: stride 8, 32 banks → 4-way conflict.
  - 2048 Bs reads/k-step × 4-way = ~256 effective cycles. FMAs = ~193 cycles/k-step. Smem dominates.
- Fix: "column-strided" Bs access: bReg[n] = Bs[cur][k][threadCol + n*(BN/TN)] instead of Bs[cur][k][threadCol*TN+n].
  - Thread x=0..15 accesses columns x, x+16, ..., x+112 for n=0..7. Stride=1 within warp → conflict-free!
  - C writeback also changes: gc = blockCol + threadCol + n*(BN/TN).
  - Global loads and smem stores in load phase unchanged (already conflict-free).
- Expected gain: 15–25% since smem reads are currently the compute bottleneck.

## Iteration 7 — Column-Strided Bs Access (Bank Conflict Fix)

### Plan
- **What**: Change Bs inner-loop access from `Bs[cur][k][threadCol*TN+n]` to `Bs[cur][k][threadCol + n*(BN/TN)]` and adjust C writeback accordingly.
- **Why**: Current access has thread x reading column x*8+n (stride TN=8). GCD(8,32)=8 → 4 threads share each bank (4-way conflict). New pattern: thread x reads column x+n*16 (stride 1 within warp). For fixed n, threads 0..15 access columns n*16..n*16+15 → 16 consecutive banks → conflict-free.
- **Prediction**: 4096 GFLOPS improves ~15–25% (18,700–20,300 GFLOPS).
- **Risk**: C write layout changes (gc = blockCol + threadCol + n*(BN/TN) vs blockCol + threadCol*TN + n). Both are still coalesced since consecutive x access consecutive columns per n. No correctness risk.
- **Rejected**: Padding Bs (tried: regressed); transposing Bs (tried: 16-way conflicts). This approach requires zero smem layout change.
