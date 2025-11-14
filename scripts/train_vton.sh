#! /bin/bash

#PBS -q gen_S
#PBS -A LOOKY
#PBS -l elapstim_req=24:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -b 12
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.8/gcc11.4.0-cuda12.8.1

module load openmpi/$NQSV_MPI_VER

cd $PBS_O_WORKDIR

mpirun ${NQSV_MPIOPTS} -np 12 -npernode 1 \
  --merge-stderr-to-stdout \
  -output-filename logs/ \
  bash -c '
  read -r MASTER_ADDR < "$PBS_NODEFILE"
  export MASTER_ADDR
  export MASTER_PORT=9901

  export RANK="${OMPI_COMM_WORLD_RANK}"
  export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE}"
  export LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"

  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  export HF_HOME="/work/LOOKY/yushin/.cache/huggingface"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_HUB_CACHE="$HF_HOME/hub"

  export WANDB_DATA_DIR="/work/LOOKY/yushin/.local/share/wandb"

  source .venv/bin/activate

  python scripts/train_vton.py \
    --output_dir results \
    --mixed_precision bf16 \
    --report_to wandb \
    --seed 42 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 8 \
    --learning_rate 3e-5 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 500 \
    --max_train_steps 30000 \
    --checkpointing_steps 1000 \
    --weighting_scheme logit_normal \
    --logit_mean 0.0 \
    --logit_std 1.0 \
    --mode_scale 1.29
'