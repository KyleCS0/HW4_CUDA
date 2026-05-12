# PP HW4 Workspace

## Structure

- `hw4_skeleton/` - main build/test project. Edit and benchmark here.
- `hw4_skeleton/kernels/student_kernel.cu` - active kernel used by `./main`.
- `deliverable/` - staging folder for final zip contents.
- `deliverable/kernels/student_kernel.cu` - copy of the current submission kernel.
- `experiments/` - saved experimental kernels and benchmark notes.
- `kernal_doc/` - notes explaining the current optimized kernel.
- `IMPLEMENTATION_GUIDE.md` - assignment rules and working plan.
- `EXPERIMENT_IDEAS.md` - possible optimization directions.

## Run Locally

```sh
cd hw4_skeleton
make clean
make CUDA_ARCH=86
./main
```

Use `CUDA_ARCH=86` for the local machine. For the grading V100 target, use:

```sh
cd hw4_skeleton
make clean
make
```

## Submit

Final zip should contain:

```text
kernels/student_kernel.cu
Team_<Team Number>_HW4_report.pdf
```

Only `student_kernel.cu` should matter for the code submission.
