#!/bin/bash
#SBATCH --job-name=fsdp_8gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/fsdp_8gpu.out
#SBATCH --error=logs/fsdp_8gpu.err
#SBATCH --time=00:10:00
#SBATCH --account=crs01
#SBATCH --reservation=pytorch
#SBATCH --no-requeue
#SBATCH --exclusive

export DISTRIBUTED_INITIALIZATION_METHOD=SLURM
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export MPICH_GPU_SUPPORT_ENABLED=0
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=64

srun -ul --environment=./edf.toml bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    python main.py --method fsdp --epochs 10 --batch_size 256
"
