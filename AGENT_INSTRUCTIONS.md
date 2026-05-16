# Autonomous GPU Kernel Optimizer — Agent Instructions

You are acting as a GPU performance engineer. Optimize
`hw4_skeleton/kernels/student_kernel.cu` iteratively until further progress
is not worth the effort, then stop. One meaningful change per iteration.

**This is a protocol, not a guideline. Every numbered step is mandatory.**
Do not skip, reorder, or combine steps. Do not proceed to the next step until
the current one is fully complete.

---

## Context

**Problem**: SGEMM — `C = alpha * (A @ B) + beta * C`, row-major, square.
**Test sizes**: 128, 256, 512, 1024, 2048, 4096.
**Hardware**: NVIDIA RTX 3080 Ti, sm_86, compute capability 8.6 (Ampere).
**Primary metric**: GFLOPS at size 4096.
**Grading tiers** (V100 reference targets — your 3080 Ti numbers will be higher):

| Tier | V100 GFLOPS | Rough 3080 Ti equivalent |
|------|-------------|--------------------------|
| T1   | ≥ 200       | ≥ 400                    |
| T2   | ≥ 2050      | ≥ 4000                   |
| T3   | ≥ 3700      | ≥ 7000                   |
| T4   | ≥ 7600      | ≥ 14000                  |
| T5   | ≥ 9500      | ≥ 18000                  |

These 3080 Ti equivalents are rough estimates based on relative peak bandwidth.
Treat them as directional only; the stop criterion is % improvement, not an
absolute GFLOPS threshold.

---

## Hard Rules — Never Violate

1. Only edit `hw4_skeleton/kernels/student_kernel.cu`.
2. Do not call cuBLAS, cuDNN, CUTLASS, or any vendor GEMM library.
3. The `runStudent(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)` signature must not change.
4. Correctness must pass for all six sizes. A wrong kernel earns 0 performance points.
5. Do not modify the harness, Makefile, or any other file.

## Anti-Cheat Rules — These Are Equally Hard

- **Never record numbers you did not just observe from running `./main 1`.**
  Run the benchmark, read the output, copy the numbers. No estimating, no carrying forward old numbers from memory.
- **Never hardcode outputs for specific sizes** (e.g., `if (M == 4096) return precomputed_C`).
- **Do not fake a rollback.** If instructions say restore the backup, actually run the `cp` command.
- **Do not claim a step is complete unless you executed it.** If build failed, do not log "build: success."
- **Do not proceed past Step 6 (build) until the build succeeds** or the rollback path is taken.
- **Do not proceed past Step 7 (run) until you have read the actual output** of `./main 1`.

---

## State Files

| File | Purpose |
|------|---------|
| `optimization_log.md` | Full journal. Read at start of every iteration. |
| `perf_history.csv` | One row per iteration. Append only. |
| `agent_status.txt` | `CONTINUE` or `STOP: <reason>`. Overwrite each iter. |
| `hw4_skeleton/kernels/student_kernel_backup.cu` | Rollback target. Overwrite before every edit. |

---

## The Iteration Protocol

### Step 1 — Read State (MANDATORY BEFORE ANYTHING ELSE)

Read `optimization_log.md` in full.
Read `perf_history.csv` in full.

Extract and hold in mind:
- Current iteration number N.
- The "Notes for Next Iteration" section from the most recent completed iteration.
- The 4096 GFLOPS for the last 3 completed iterations (or however many exist).
- Every optimization that was tried and reverted, and why.
- Every optimization that is still active in the kernel.

If neither file exists → this is Iteration 0 (Bootstrap). See the Bootstrap section below.

---

### Step 2 — Stop-or-Continue Check

**Stop** (write `STOP: <reason>` to `agent_status.txt`, log a final summary, then halt) if **any**:
- The last 2 completed iterations each improved 4096 GFLOPS by less than 3%.
- 3 consecutive iterations ended in compile failure or correctness failure without net progress.
- You have genuinely exhausted all viable directions and cannot form a hypothesis with expected gain > 5%.

Otherwise continue.

---

### Step 3 — Engineering Analysis and Plan (The Most Important Step)

This step is the core of good engineering. Take your time here. Write before you code.

**3a. Diagnose the current bottleneck.**

Use the numbers and the RTX 3080 Ti hardware facts below to reason about what is limiting performance.

RTX 3080 Ti hardware reference (Ampere, sm_86):
- Peak FP32 throughput: ~34,100 GFLOPS.
- GDDR6X bandwidth: ~912 GB/s.
- Arithmetic intensity crossover: ~37.4 FLOPs/byte (= 34100 / 912). Above this, compute-bound; below, memory-bound. This is a much higher bar than Volta — you need large tiles and deep register reuse to become compute-bound.
- Shared memory per SM: up to 100 KB configurable (Ampere raised the limit from 48 KB). Default is still 48 KB; to use more, the kernel must call `cudaFuncSetAttribute` or use `__launch_bounds__` with dynamic smem. For simplicity, keep tiles within 48 KB unless you explicitly configure otherwise.
- L1 cache + shared memory: 128 KB per SM total; the split is configurable.
- L2 cache: 6 MB.
- Registers per SM: 65,536.
- SM count: 80.
- Max threads per block: 1024. Warp size: 32.
- Occupancy rule of thumb: aim for ≥ 50% (≥ 8 active warps per SM). On Ampere, the hardware can hide latency better than Volta, but high occupancy still matters.
- **Ampere-exclusive**: `cp.async` (PTX async copy from global to shared) enables software pipelining without sync barriers between load and compute. This is the key technique for double-buffering on sm_80+.

Diagnostics from the number patterns:

| Pattern | Likely cause |
|---------|-------------|
| GFLOPS scales roughly linearly with size | Memory-bound; not enough reuse |
| GFLOPS plateaus at large sizes but << 34,000 | Compute-bound or occupancy-limited |
| 4096 GFLOPS much higher than 2048 | Good tile efficiency at large size; small sizes have launch overhead |
| 4096 GFLOPS lower than 2048 | Register spill or shared-memory pressure at large tile |
| Little gain from tiling already applied | Bank conflicts, or tile too small, or register file pressure limiting occupancy |
| Verification failure on boundary sizes | Off-by-one in tile bounds or K-loop edge case |

Estimate the current kernel's arithmetic intensity: roughly `(2 * BM * BN * BK) / (4 * (BM * BK + BK * BN))` FLOPs/byte. If this is well below 37.4, you are memory-bound and more reuse (larger tiles, register tiling) is the right direction. On the 3080 Ti, even a well-tiled kernel with intensity ~20 is still memory-bound — you need large register tiles (TM=TN=8 or larger) to approach compute bound.

**3b. Review past attempts and their outcomes.**

Read the "Notes for Next Iteration" from the previous iteration entry.
For each direction that was tried and reverted, state explicitly why it failed and whether it might work differently now (e.g., a parameter change that was too aggressive might work at a smaller value).

**3c. Select one optimization and justify it.**

State:
- **What you will change** (one sentence).
- **Why this addresses the diagnosed bottleneck** (one or two sentences).
- **Quantitative prediction**: "I expect 4096 GFLOPS to improve by approximately X% because Y."
- **Risk**: What could go wrong (register spill, bank conflict, correctness bug).
- **What you considered and rejected**: Name at least one alternative and why you didn't choose it.

**3d. Write the plan to `optimization_log.md`.**

Append a new section header `## Iteration N — <title>` and a `### Plan` subsection with your analysis from 3a–3c. **Write this before touching any code.**

---

### Step 4 — Backup

```bash
cp hw4_skeleton/kernels/student_kernel.cu hw4_skeleton/kernels/student_kernel_backup.cu
```

Confirm the copy succeeded before proceeding.

---

### Step 5 — Implement

Edit `hw4_skeleton/kernels/student_kernel.cu`.

Rules:
- Change exactly what the plan in Step 3 described. Do not add unplanned changes.
- Keep code readable. Future iterations build on this.
- Do not add comments that describe what the code does. Comments are only for non-obvious constraints or invariants.

---

### Step 6 — Build

```bash
cd hw4_skeleton && make clean && make CUDA_ARCH=86 2>&1
```

- If build succeeds: proceed to Step 7.
- If build fails: fix the error and retry **once**.
- If build fails a second time: **restore the backup** (`cp kernels/student_kernel_backup.cu kernels/student_kernel.cu`), log `Build failed: <error summary>` under `### Result` in `optimization_log.md`, increment the failure counter, go back to Step 2.

---

### Step 7 — Run Benchmark

```bash
cd hw4_skeleton && ./main 1 2>&1
```

Read the full output. Parse every size line:
```
Running size: NNN...   avg time: X.XXXXXXs, performance: YYY.YYY GFLOPS
```

- If the output contains `verification failed`: **restore the backup**, log `Correctness failure at size X. Likely cause: <your best guess about the indexing/bounds bug>`. Go back to Step 2.
- If the binary crashes or hangs: restore backup, log it, go back to Step 2.

---

### Step 8 — Evaluate

Compare the new 4096 GFLOPS to the previous iteration's 4096 GFLOPS.
Compute `delta_pct = (new - old) / old * 100`.

- If `delta_pct < 0` (regression): **restore backup**. The change hurt. Keep it as "tried+reverted" in the log.
- If `delta_pct >= 0`: keep the change. Even a 1% gain is fine to keep.

Also check all six sizes. A regression on small sizes is acceptable (note it) but a > 10% regression on 2048 should be flagged.

---

### Step 9 — Update Logs

**9a. Append to `optimization_log.md`** under `### Result`:

```
**Status**: KEPT / REVERTED
**Correctness**: PASS / FAIL
**Δ 4096**: +X.X%

| Size | GFLOPS |
|------|--------|
| 128  |        |
| 256  |        |
| 512  |        |
| 1024 |        |
| 2048 |        |
| 4096 |        |
```

Then append a `### Notes for Next Iteration` subsection. This is required. Write:
- What the numbers suggest about the remaining bottleneck (use the diagnostic table from Step 3a).
- Any parameter that looks suboptimal (e.g., "BK=16 seems too small given the 2048→4096 scaling suggests we are still memory-bandwidth limited at large sizes").
- Ideas you considered but didn't apply this iteration, and whether they are still viable.
- Anything surprising in the result that changes your mental model.

Keep this section to 4–6 bullet points. Be specific and technical. This note is the primary input for the next iteration's Step 3.

**9b. Append one row to `perf_history.csv`**:
```
N,<title>,128_gflops,256_gflops,512_gflops,1024_gflops,2048_gflops,4096_gflops,delta_pct,kept
```

---

### Step 10 — Signal

Write to `agent_status.txt`:
- `CONTINUE` if proceeding.
- `STOP: <one-line reason>` if stopping.

---

## Bootstrap (Iteration 0)

Only run this if `optimization_log.md` does not exist.

1. Read `hw4_skeleton/kernels/student_kernel.cu`.
2. Run `cd hw4_skeleton && make clean && make CUDA_ARCH=86 && ./main 1 2>&1`. Record the exact output.
3. Create `optimization_log.md`:
   ```
   # Kernel Optimization Log

   ## Baseline

   Naive one-thread-per-output kernel, 16×16 block, no shared memory.

   | Size | GFLOPS |
   |------|--------|
   ...

   ### Notes for Next Iteration
   - <bottleneck diagnosis of the baseline>
   - <first recommended direction>
   ```
4. Create `perf_history.csv` with header and row 0 (kept=baseline).
5. Write `CONTINUE` to `agent_status.txt`.
6. **Immediately proceed to Iteration 1 in this same session.** Do not wait for the next loop tick.

---

## RTX 3080 Ti SGEMM Optimization Roadmap

Use this as a menu, not a script. Choose based on diagnosis.

| Priority | Technique | Expected gain | When to apply |
|----------|-----------|---------------|---------------|
| 1 | Shared-memory tiling (BM×BN×BK) | Large | Always first if not done |
| 2 | Register tiling (TM×TN per thread) | Large | After smem tiling |
| 3 | Tune BM, BN, BK | Medium | After register tiling works |
| 4 | Shared-memory padding (+1 column) | Small–medium | When bank conflicts suspected |
| 5 | Vectorized global loads (float4) | Small–medium | When memory-bound at large sizes |
| 6 | `__launch_bounds__` | Small | When occupancy is the limiter |
| 7 | `#pragma unroll` on inner K loop | Small | When register tiling is in place |
| 8 | Transposed A tile in smem | Small | When A access pattern causes conflicts |
| 9 | Double-buffering with `cp.async` | Medium–large | Ampere-only; pipeline global→smem loads with compute. Try after all above are stable. |
| 10 | Size-dispatched kernels | Small | Only if small sizes (≤256) drag score |

Good starting parameters for Ampere (3080 Ti):
- `BM=128, BN=128, BK=16, TM=8, TN=8` (256-thread block) — high arithmetic intensity (~37 FLOPs/byte), targets the compute-bound regime on 3080 Ti.
- `BM=64, BN=64, BK=16, TM=4, TN=4` (256-thread block) — lower register pressure, easier to get correct first.
- `BM=128, BN=128, BK=8, TM=8, TN=8` (256-thread block) — slightly less smem pressure than BK=16, similar register count.
- Avoid `BM=32, BN=32` — arithmetic intensity too low for the 3080 Ti's high crossover point.
- On Ampere, smem bank width is 4 bytes, 32 banks (same as Volta). Padding rules are unchanged.
- The `cp.async` path (Technique 9) is an Ampere-native instruction (`sm_80+`). It lets you issue async loads while the previous tile's compute is in flight, hiding ~80% of global memory latency when implemented correctly.

---

## Log Format Rules

- Section headers: `## Iteration N — <title>` (title = 3–5 words, e.g., "Register Tile 8×8").
- Subsections: `### Plan`, `### Result`, `### Notes for Next Iteration`.
- No prose paragraphs in `### Result`. Tables and bullet points only.
- `### Plan` may have prose but keep it under 10 lines.
- Do not repeat information already in the header row of the CSV.
