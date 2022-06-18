#!/bin/bash

# Setup
LOG_DIR=logs/gen_reacher
mkdir -p $LOG_DIR
pip install -e .

ENVS=("ReacherFixedGoal-v0")
SAVE_FOLDER=("reacher")
ROT=("false" "true")
ROT_NAME=("_" "_rot90_")
MODE=("train" "eval")
NUM_EPISODES=(2000 1000)
SEED=(1 2)

PARALLEL="parallel --delay 0.2 -j 4 --joblog ${LOG_DIR}/parallel.log"

CMD=(python sen/data/gen_reacher.py)

$PARALLEL "${CMD[*]} --env_id {1} --rot90 {3} --save_path dataset/{2}/{1}{4}{5}{6}.h5 --num_episodes {6} --seed {7} &> ${LOG_DIR}/{1}{4}{5}{6}.out" \
  ::: "${ENVS[@]}" \
  :::+ "${SAVE_FOLDER[@]}" \
  ::: "${ROT[@]}" \
  :::+ "${ROT_NAME[@]}" \
  ::: "${MODE[@]}" \
  :::+ "${NUM_EPISODES[@]}" \
  :::+ "${SEED[@]}"
