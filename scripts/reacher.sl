#!/bin/bash

#SBATCH --job-name=reacher
#SBATCH --output=logs/reacher/%x_%J.log
#SBATCH --error=logs/reacher/%x_%J.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=08:00:00

# Setup
LOG_DIR=logs/reacher
mkdir -p $LOG_DIR
wandb login "$WANDB_API_KEY"
pip install -e .

CMD=(python -u sen/train.py
  --wandb_project "sen_reacher"
  --dataset_path "dataset/reacher/ReacherFixedGoal-v0_train2000.h5"
  --train_seed 42

  # Ours
  --embedding ReacherEmbedding
  --embedding_kwargs input_dim=6 output_dim=8 hidden_dim=[32,32,32,32,32,32]
  --encoder ReacherEncoder_D4
  --encoder_kwargs input_dim=8 output_dim=4 group_order=4 hidden_dim=[181,181,181,181,181]
  --transition Transition_D4
  --transition_kwargs obs_dim=4 action_dim=2 group_order=4 hidden_dim=[181,181]
  --exp_name "ours"

  # Nonequivariant
  # --embedding ReacherEmbedding
  # --embedding_kwargs input_dim=6 output_dim=8 hidden_dim=[32,32,32,32,32,32]
  # --encoder ReacherEncoder
  # --encoder_kwargs input_dim=8 output_dim=4 hidden_dim=[512,512,512,512,512]
  # --transition Transition_Non
  # --transition_kwargs obs_dim=4 action_dim=2 hidden_dim=[512,512]
  # --exp_name "non-equivariant"

  # Fully equivariant
  # --embedding ReacherEmbedding_D4
  # --embedding_kwargs input_dim=6 output_dim=8 group_order=4 hidden_dim=[6,6,6,6,6,6]
  # --encoder ReacherEncoder_D4
  # --encoder_kwargs input_dim=8 output_dim=4 group_order=4 hidden_dim=[181,181,181,181,181] in_feat_type="regular"
  # --transition Transition_D4
  # --transition_kwargs obs_dim=4 action_dim=2 group_order=4 hidden_dim=[181,181]
  # --exp_name "fully-equivariant"

  --model Model
  --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True
  --device="cuda"
  --fp16=true
  --epochs 100
  --train_batch_size 256
)


echo "[Executing command]" "${CMD[@]}"
# If running from SLURM
if [ -n "$SLURM_JOB_ID" ]; then
  "${CMD[@]}"
else
  "${CMD[@]}"
  # DATE=$(date +"%y%m%d-%H%M%S")
  # nohup "${CMD[@]}" &> "${LOG_DIR}/${DATE}.log" &
fi
