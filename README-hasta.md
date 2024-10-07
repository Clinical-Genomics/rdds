# Hasta

## Submitting Multiple Jobs To Hasta

```bash
export SIF_IMAGE_PATH=tmp/devenv/[IMAGE_VERSION]
for dir in `find -name "[REGEX]"`; do
  sbatch --test-only job.slurm [PYTHON_CMD];
done
```

## Hasta GPUs
This is the current native Hasta `gpu-compute-0-0`, `gpu-compute-0-1` node set-up:  
`Tesla V100-PCIE-32GB`
running on a kernel version
```
Linux version 3.10.0-1160.53.1.el7.x86_64
(mockbuild@kbuilder.bsys.centos.org)
(gcc version 4.8.5 20150623 (Red Hat 4.8.5-44) (GCC) )
#1 SMP Fri Jan 14 13:59:45 UTC 2022
```

```raw
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.32.00    Driver Version: 455.32.00    CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   28C    P0    33W / 250W |      0MiB / 32510MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
