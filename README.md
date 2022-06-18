# Learning Symmetric Representations for Equivariant World Models

This code implements Symmetric Embedding Networks (SENs) is for the ICML 2022 paper [Learning Symmetric Representations for Equivariant World Models](https://arxiv.org/abs/2204.11371) by Jung Yeon Park, Ondrej Biza, Linfeng Zhao, Jan Willem van de Meent, and Robin Walters.

## Reference

If you find this work useful, please cite:

```bibtex
@inproceedings{park2022sen, 
  title={Learning Symmetric Representations for Equivariant World Model}, 
  author={Jung Yeon Park, Ondrej Biza, Linfeng Zhao, Jan Willem van de Meent, Robin Walters}
  booktitle={International Conference on Machine Learning}, 
  year={2022}, 
  url={https://arxiv.org/abs/2204.11371} 
}
```

## Installation

* Python (tested on 3.7)
* Mujoco
* For all dependencies, see requirements.txt

To install, clone this repository and then install dependencies and this package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

This repository contains code for the 3D blocks / cubes and Reacher experiments. For the cubes datasets and the nonequivariant model architectures, we base our code from <https://github.com/tkipf/c-swm>. The `wandb` library is used to track experiments.

### 1. 3D Blocks / Cubes

#### 1.1 Generate datasets

The following commands create the original datasets and also the 90 degree rotated (counter-clockwise) versions. The rotated datasets are only used for the evaluation metrics. This may take a while.

```python
python sen/data/gen_cubes.py --env_id BlockPushing-v0 --env_kwargs render_type=cubes --env_timelimit 100 --save_path dataset/cubes/Cubes_train1000.h5 --rot_save_path dataset/cubes/Cubes_rot90_train1000.h5 --num_episodes 1000 --seed 1
python sen/data/gen_cubes.py --env_id BlockPushing-v0 --env_kwargs render_type=cubes --env_timelimit 10 --save_path dataset/cubes/Cubes_eval10000.h5 --rot_save_path dataset/cubes/Cubes_rot90_eval10000.h5 --num_episodes 10000 --seed 2
```

#### 1.2 Training

Use scripts provided in `scripts/` directory or via command-line. You may need backslashes for the brackets on `zsh`.

Ours

```python
python sen/train.py \
  --wandb_project "sen_cubes" \
  --embedding CubesEmbedding --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=32 \
  --encoder CubesEncoder_C4 --encoder_kwargs input_shape=[5,50,50] output_dim=2 hidden_dim=[256,256] num_objects=5 \
  --transition TransitionGNN_C4 --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=256 num_objects=5 \
  --model Model --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True \
  --dataset_path "dataset/cubes/Cubes_train1000.h5" \
  --device "cuda" \
  --epochs 100 \
  --train_seed 42 \
  --train_batch_size 1024 \
  --exp_name "ours"
```

Non-equivariant

```python
python sen/train.py \
  --wandb_project "sen_cubes" \
  --embedding CubesEmbedding --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=32 \
  --encoder CubesEncoder --encoder_kwargs input_shape=[5,50,50] output_shape=2 hidden_dim=[512,512] num_objects=5 \
  --transition TransitionGNN --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=512 num_objects=5 \
  --model Model --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True \
  --dataset_path "dataset/cubes/Cubes_train1000.h5" \
  --device "cuda" \
  --epochs 100 \
  --train_seed 42 \
  --train_batch_size 1024 \
  --exp_name "non-equivariant"
```

Fully equivariant

```python
python sen/train.py \
  --wandb_project "sen_cubes" \
  --embedding CubesEmbedding_E2 --embedding_kwargs input_dim=3 output_dim=5 hidden_dim=16 group_order=4 out_feat_type=trivial \
  --encoder CubesEncoder_C4 --encoder_kwargs input_shape=[5,50,50] output_dim=2 hidden_dim=[256,256] num_objects=5 \
  --transition TransitionGNN_C4 --transition_kwargs obs_dim=2 action_dim=4 hidden_dim=256 num_objects=5 \
  --model Model --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True \
  --dataset_path "dataset/cubes/Cubes_train1000.h5" \
  --device "cuda" \
  --epochs 100 \
  --train_seed 42 \
  --train_batch_size 1024 \
  --exp_name "fully-equivariant"
```

#### 1.3 Evaluation

```python
python sen/eval/eval_objects.py \
  --wandb_project "sen_cubes" \
  --eval_dataset_path "dataset/cubes/Cubes_eval10000.h5" \
  --rot_eval_dataset_path "dataset/cubes/Cubes_rot90_eval10000.h5" \
  --eval_n_episodes 10000 \
  --device "cpu" \
  --wandb_id <wandb run ID>
```

### 2. Reacher

### 2.1 Generate datasets

```python
python sen/data/gen_reacher.py --env_id ReacherFixedGoal-v0 --rot90=false --env_timelimit 10 --save_path dataset/reacher/ReacherFixedGoal-v0_train2000.h5 --num_episodes 2000 --seed 1
python sen/data/gen_reacher.py --env_id ReacherFixedGoal-v0 --rot90=false --env_timelimit 10 --save_path dataset/reacher/ReacherFixedGoal-v0_eval1000.h5 --num_episodes 1000 --seed 2
```

To generate 90 degree rotated versions of the above datasets, run the commands below. The sample indices exactly correspond to the samples in the training datasets as long as the seeds match.

```python
python sen/data/gen_reacher.py --env_id ReacherFixedGoal-v0 --rot90=true --env_timelimit 10 --save_path dataset/reacher/ReacherFixedGoal-v0_rot90_train2000.h5 --num_episodes 2000 --seed 1
python sen/data/gen_reacher.py --env_id ReacherFixedGoal-v0 --rot90=true --env_timelimit 10 --save_path dataset/reacher/ReacherFixedGoal-v0_rot90_eval1000.h5 --num_episodes 1000 --seed 2
```

### 2.2 Training

Ours

```python
python sen/train.py \
  --wandb_project "sen_reacher" \
  --embedding ReacherEmbedding --embedding_kwargs input_dim=6 output_dim=8 hidden_dim=[32,32,32,32,32,32] \
  --encoder ReacherEncoder_D4 --encoder_kwargs input_dim=8 output_dim=4 group_order=4 hidden_dim=[181,181,181,181,181] \
  --transition Transition_D4 --transition_kwargs obs_dim=4 action_dim=2 group_order=4 hidden_dim=[181,181] \
  --model Model --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True \
  --dataset_path "dataset/reacher/ReacherFixedGoal-v0_train2000.h5" \
  --device "cuda" \
  --epochs 100 \
  --train_seed 42 \
  --train_batch_size 256 \
  --fp16=true \
  --exp_name "ours"
```

Non-equivariant

```python
python sen/train.py \
  --wandb_project "sen_reacher" \
  --embedding ReacherEmbedding --embedding_kwargs input_dim=6 output_dim=8 hidden_dim=[32,32,32,32,32,32] \
  --encoder ReacherEncoder --encoder_kwargs input_dim=8 output_dim=4 hidden_dim=[512,512,512,512,512] \
  --transition Transition --transition_kwargs obs_dim=4 action_dim=2 hidden_dim=[512,512] \
  --model Model --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True \
  --dataset_path "dataset/reacher/ReacherFixedGoal-v0_train2000.h5" \
  --device "cuda" \
  --epochs 100 \
  --train_seed 42 \
  --train_batch_size 256 \
  --fp16=true \
  --exp_name "non-equivariant"
```

Fully-equivariant

```python
python sen/train.py \
  --wandb_project "sen_reacher" \
  --embedding ReacherEmbedding_D4 --embedding_kwargs input_dim=6 output_dim=8 group_order=4 hidden_dim=[6,6,6,6,6,6] \
  --encoder ReacherEncoder_D4 --encoder_kwargs input_dim=8 output_dim=4 group_order=4 hidden_dim=[181,181,181,181,181] in_feat_type="regular" \
  --transition Transition_D4 --transition_kwargs obs_dim=4 action_dim=2 group_order=4 hidden_dim=[181,181] \
  --model Model --model_kwargs hinge=1 sigma=0.5 n_neg=1 pred_delta=True \
  --dataset_path "dataset/reacher/ReacherFixedGoal-v0_train2000.h5" \
  --device "cuda" \
  --epochs 100 \
  --train_seed 42 \
  --train_batch_size 256 \
  --fp16=true \
  --exp_name "fully-equivariant"
```

### 2.3 Evaluation

```python
python sen/eval/eval_reacher.py \
  --wandb_project "sen_reacher" \
  --eval_dataset_path "dataset/reacher/ReacherFixedGoal-v0_eval1000.h5" \
  --rot_eval_dataset_path "dataset/reacher/ReacherFixedGoal-v0_rot90_eval1000.h5" \
  --eval_n_episodes 1000 \
  --device "cpu" \
  --wandb_id <wandb run ID>
```

#### Limited action datasets

These commands create the cubes datasets with no left actions. We change the env_timelimit to 1 to prevent a biased dataset where most states are obstructed later in the episode. We train on this limited action dataset and evaluate on the test dataset with all actions.

```python
python -u sen/data/gen_cubes.py --env_id BlockPushing-v0 --env_kwargs render_type=cubes --env_timelimit 1 --save_path dataset/cubes/Cubes_URD_train100000.h5 --rot_save_path dataset/cubes/Cubes_uponly_rot90_train100000.h5 --actions 0 1 2 4 5 6 8 9 10 12 13 14 16 17 18 --num_episodes 100000 --seed 1
```

#### Limited actions datasets

```python
python sen/data/gen_reacher.py --env_id ReacherFixedGoal-v0 --reacher_positive --rot90=false --env_timelimit 10 --save_path dataset/reacher/ReacherFixedGoal-v0_pos2nd_train2000.h5 --num_episodes 2000 --seed 1
```

To generate 90 degree rotated versions of the above dataset, run:

```python
python sen/data/gen_reacher.py --env_id ReacherFixedGoal-v0 --reacher_positive --rot90=true --env_timelimit 10 --save_path dataset/reacher/ReacherFixedGoal-v0_pos2nd_rot90_train2000.h5 --num_episodes 2000 --seed 1
```
