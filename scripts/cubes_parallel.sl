#!/bin/bash

#SBATCH --job-name=cubes
#SBATCH --output=logs/cubes/%x_%j.log
#SBATCH --error=logs/cubes/%x_%j.log
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=/home/park.jungy/.slurm/gpu_exclude_nodelist
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-00:00:00

# Hyperparameters
TRANSITION=("TransitionGNN" "TransitionGCN")
SEED=(1001 1002 1003 1004 1005)

# Setup
DATE=$(date +"%y%m%d-%H%M%S")
LOG_DIR="logs/cubes/${DATE}"
mkdir -p "$LOG_DIR"
wandb login "$WANDB_API_KEY"
pip install -e .

if [ -z "$SLURM_JOB_ID" ]; then
  PARALLEL="parallel --delay 1 -j 2 --joblog ${LOG_DIR}/parallel.log"
else
  PARALLEL="parallel --delay 1 -j ${SLURM_NTASKS} --joblog ${LOG_DIR}/${SLURM_JOB_ID}_parallel.log"
  # Srun
  SRUN="srun --output=${LOG_DIR}/%x_%J.log --error=${LOG_DIR}/%x_%J.log --nodes=1 --ntasks=1 --gres=gpu:1"
fi

CMD=(python -u sen/train.py
  --wandb_project "sen_cubes_gcn_mpnn"
  --dataset_path "dataset/cubes/Cubes_train1000.h5"
  --embedding CubesEmbedding
  --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=32
  --encoder CubesEncoder
  --encoder_kwargs input_shape=[5,50,50] output_shape=2 hidden_dim=[512,512] num_objects=5
  --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=512 num_objects=5
  --device "cuda"
  --model Model
  --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True
  --device "cuda"
  --epochs 100
  --train_batch_size 1024
)

echo "Log dir: $LOG_DIR"
if [ -z "$SLURM_JOB_ID" ]; then
  $PARALLEL --rpl '{%gpu} 1 $_=(slot() - 1) % 4' \
    "CUDA_VISIBLE_DEVICES={%gpu} nohup ${CMD[*]} \
    --train_seed {1} \
    --transition {2} \
    --exp_name {2}_seed{1} \
    &> ${LOG_DIR}/{2}_seed{1}.log" \
    ::: "${SEED[@]}" \
    ::: "${TRANSITION[@]}" 
else
  $PARALLEL "$SRUN ${CMD[*]} \
    --train_seed {1} \
    --transition {2} \
    --exp_name {2}_seed{1}" \
    ::: "${SEED[@]}" \
    ::: "${TRANSITION[@]}" 
fi
