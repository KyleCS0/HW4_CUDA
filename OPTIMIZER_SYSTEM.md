# Autonomous Kernel Optimizer — System Guide

An autonomous Claude Code loop that iteratively optimizes
`hw4_skeleton/kernels/student_kernel.cu` without human input.

---

## How It Works

Each loop tick Claude:
1. Reads the log to understand current state
2. Diagnoses the bottleneck from the GFLOPS numbers
3. Writes a plan, then implements one change
4. Builds and runs the benchmark
5. Keeps or reverts the change based on result
6. Updates the logs and decides whether to continue

The loop is stateless between ticks — all context lives in the log files.

---

## Files

| File | Role |
|------|------|
| `AGENT_INSTRUCTIONS.md` | The protocol Claude follows. Do not edit during a run. |
| `optimization_log.md` | Human-readable journal. Created on first run. |
| `perf_history.csv` | One row per iteration with GFLOPS for all sizes. Created on first run. |
| `agent_status.txt` | `CONTINUE` or `STOP: <reason>`. Written by Claude each tick. |
| `hw4_skeleton/kernels/student_kernel.cu` | The only file Claude edits. |
| `hw4_skeleton/kernels/student_kernel_backup.cu` | Rolling backup. Overwritten before each edit. Used for rollback. |

---

## Starting a Run

```
/loop Read AGENT_INSTRUCTIONS.md and execute one iteration of the optimization loop as described.
```

No interval needed — Claude self-paces based on how long builds and benchmarks take.

On the first tick Claude will bootstrap (run the baseline, create the logs), then
immediately start Iteration 1 without waiting for the next tick.

---

## Monitoring

Watch `optimization_log.md` for the journal as it updates.
Watch `perf_history.csv` for a compact view of GFLOPS across iterations.

The last line of `agent_status.txt` tells you the current intent:
- `CONTINUE` — another iteration is coming
- `STOP: <reason>` — Claude decided to stop

---

## Stopping

**Automatic stop** — Claude stops itself when:
- Two consecutive iterations each improve 4096 GFLOPS by less than 3%
- Three consecutive compile/correctness failures
- No viable optimization direction remains

**Manual stop** — interrupt the loop in Claude Code (Ctrl+C or close the session).
The log and the backup file will be in a consistent state; no work is lost.

---

## Resuming

Just run the `/loop` command again with the same prompt.
Claude reads the existing logs and continues from where it left off.

---

## Hardware

- **GPU**: RTX 3080 Ti (sm_86)
- **Build**: `make CUDA_ARCH=86` (Claude handles this)
- **Reference**: cuBLAS runs internally for correctness verification — not in the final kernel

---

## What Claude Will Not Do

- Call cuBLAS/cuDNN/CUTLASS in the student kernel
- Modify any file other than `student_kernel.cu`
- Record benchmark numbers it did not actually observe
- Claim success if verification failed or build failed
