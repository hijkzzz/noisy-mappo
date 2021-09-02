#!/bin/sh
env="StarCraft2"
map=$1
algo="nv-ippo"
exp="rnn" 
seed_max=$2

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do {
    seed=$((seed+1))
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python onpolicy/scripts/train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --use_value_active_masks --use_eval \
        --ppo_epoch 5 --use_value_noise --sigma 10 --reset_interval -1 # -1 means noise_reset_interval == infinity (set to 100 also works)
    sleep 3s
} &
done
wait
