#!/bin/bash

#SBATCH --job-name=cubes
#SBATCH --output=logs/cubes/%x_%J.log
#SBATCH --error=logs/cubes/%x_%J.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=08:00:00

# Setup
LOG_DIR=logs/cubes
mkdir -p $LOG_DIR
wandb login "$WANDB_API_KEY"
pip install -e .

CMD=(python -u sen/train.py
  --wandb_project "sen_cubes"
  --dataset_path "dataset/cubes/Cubes_train1000.h5"
  --train_seed 42

  # Ours
  --embedding CubesEmbedding
  --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=32
  --encoder CubesEncoder_C4
  --encoder_kwargs input_shape=[5,50,50] output_dim=2 hidden_dim=[256,256] num_objects=5
  --transition TransitionGNN_C4
  --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=256 num_objects=5
  --exp_name "ours"

  # Nonequivariant
  # --embedding CubesEmbedding
  # --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=32
  # --encoder CubesEncoder
  # --encoder_kwargs input_shape=[5,50,50] output_shape=2 hidden_dim=[512,512] num_objects=5
  # --transition TransitionGNN
  # --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=512 num_objects=5
  # --exp_name "non-equivariant"

  # Fully equivariant
  # --embedding CubesEmbedding_E2
  # --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=16 group_order=4 out_feat_type=trivial
  # --encoder CubesEncoder_C4
  # --encoder_kwargs input_shape=[5,50,50] output_dim=2 hidden_dim=[256,256] num_objects=5
  # --transition TransitionGNN_C4
  # --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=256 num_objects=5
  # --exp_name "fully-equivariant"

  --model Model
  --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True
  --device "cuda"
  --epochs 100
  --train_batch_size 1024
)

echo "[Executing command]" "${CMD[@]}"
# If running from SLURM
if [ -n "$SLURM_JOB_ID" ]; then
  "${CMD[@]}"
else
  # "${CMD[@]}"
  DATE=$(date +"%y%m%d-%H%M%S")
  nohup "${CMD[@]}" &> "${LOG_DIR}/${DATE}.log" &
fi
