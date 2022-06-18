#!/bin/bash

#SBATCH --job-name=decoder
#SBATCH --output=logs/decoder/%x_%J.log
#SBATCH --error=logs/decoder/%x_%J.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=08:00:00

# Setup
LOG_DIR=logs/decoder
mkdir -p $LOG_DIR
wandb login "$WANDB_API_KEY"
pip install -e .

CMD=(python -u sen/train_decoder.py
  # # Reacher
  # --wandb_project_model "sen_reacher_d4"
  # --wandb_id_model "1c09t624"
  # --wandb_project "sen_reacher_d4_decoder"
  # --decoder Decoder_ReacherD4
  # --decoder_kwargs input_shape=[4,8] output_dim=6
  # --train_seed 2002
  # --exp_name "redo_mse_dilation_seed2002"
  # --train_batch_size 128
  # --device="cuda"

  # Cubes
  --wandb_project_model "sen_cubes"
  --wandb_id_model "258i3i3m"
  --wandb_project "sen_cubes_decoder"
  --decoder Decoder_CubesC4
  --decoder_kwargs input_shape=[4,2] num_objects=5 output_shape=[3,50,50]
  --train_seed 2003
  --exp_name "redo_mse_seed2003"
  --train_batch_size 512
  --device="cuda:1"
  --decoder_checkpoint "wandb/run-20211004_100419-258i3i3m/files/decoder.pt"

  --epochs 100
  --loss_fn "mse"
)

echo "[Executing command]" "${CMD[@]}"
"${CMD[@]}"
