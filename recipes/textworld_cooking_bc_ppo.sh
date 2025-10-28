#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1

# BC-PPO Recipe for TextWorld Cooking Task
# This script demonstrates how to run BC-PPO (Bias-Corrected PPO) 
# instead of standard PPO by using the bc_ppo_trainer configuration.

# DATA/TASK CONFIG
scratch_dir="$SCRATCH"
env_name="textworld"
task_prefix="tw_dense"
instance_id_start=50001
instance_id_end=53000
hf_data_repo="PEARLS-Lab/meow-tea-taro-dataset"
hf_instances_dir="$scratch_dir/$env_name/$task_prefix/instances"
hf_train_data_dir="$scratch_dir/$env_name/$task_prefix/multiturn_rl_data/3000_train_data"
local_instances_dir="$scratch_dir/local/$hf_instances_dir"
local_train_data_dir="$scratch_dir/local/$hf_train_data_dir"
local_parquet_dir="$scratch_dir/local/train_parquet"
reward_method="dense"

# MODEL CONFIG
hf_actor_repo_id=""
hf_actor_model_path=""
hf_critic_repo_id=""
hf_critic_model_path=""
actor_model_path="$scratch_dir/local/model/actor"
critic_model_path="$scratch_dir/local/model/critic"
base_model="Qwen/Qwen2.5-7B-Instruct"

# AGENTIC CONFIG
# env_name=... # from above
is_multiturn=True
is_async=False
max_iter=24
reward_density=$reward_method
reward_type="verified"
reward_manager="agentic_verified"
rollout_name="vllm_agentic"
rollout_mode=$( [ "$is_async" = "True" ] && echo "async" || echo "sync" ) # Set 'async' if is_async=True, else 'sync'.

# BC-PPO ALGORITHM CONFIG
# BC-PPO uses 'bc_gae' advantage estimator with bias correction enabled
adv_estimator=bc_gae
gamma=1.0
bias_correction=True      # BC-PPO: Enable bias correction
bias_decay=0.95           # BC-PPO: Decay rate for bias estimate EMA

use_kl_loss=False # Whether to use KL loss in objective. True for GRPO.
use_kl_in_reward=True # Whether to use KL divergence in reward calculation.
kl_coef=0.01
clip_ratio=0.2

# TRAINING CONFIG
rollout_temp=0.7
val_rollout_temp=0.4
train_batch_size=256
ppo_mini_batch_size=256
max_epochs=10
rollout_batch_size_per_device=32
critic_warmup=0
val_before_train=False
save_freq=1

# CRITIC CONFIG
critic_lr=5e-6
critic_update_epochs=1

# LOGGING CONFIG
project_name="${env_name}_${task_prefix}_${reward_method}_${adv_estimator}" # TODO (optional). WandB project name.
experiment_name="${env_name}_${task_prefix}_${reward_method}_${adv_estimator}_kl${kl_coef}_actor${actor_lr}_critic${critic_lr}_bs${train_batch_size}_ep${num_epochs}_seed${instance_id_start}" # TODO (optional). WandB experiment name.
logger="console,wandb"

# HARDWARE CONFIG
nnodes=1
n_gpus_per_node=4
colocate_critic_reward=True
colocate_actor_ref=True

# ====================================================================
# DOWNLOAD DATA
# ====================================================================
echo "Downloading training data from HuggingFace..."
python3 -m huggingface_hub.commands.huggingface_cli download \
    --repo-type dataset \
    --local-dir $local_train_data_dir \
    "$hf_data_repo" "$hf_train_data_dir"

echo "Downloading task instances from HuggingFace..."
python3 -m huggingface_hub.commands.huggingface_cli download \
    --repo-type dataset \
    --local-dir $local_instances_dir \
    "$hf_data_repo" "$hf_instances_dir"

# ====================================================================
# PREPARE MODEL CHECKPOINT
# ====================================================================
# Load from HF or local path
if [ -n "$hf_actor_repo_id" ] && [ -n "$hf_actor_model_path" ]; then
    echo "Downloading actor model from HuggingFace..."
    python3 -m huggingface_hub.commands.huggingface_cli download \
        --local-dir $actor_model_path \
        "$hf_actor_repo_id" "$hf_actor_model_path"
else
    echo "Using base model as initial actor: $base_model"
    actor_model_path=$base_model
fi

if [ -n "$hf_critic_repo_id" ] && [ -n "$hf_critic_model_path" ]; then
    echo "Downloading critic model from HuggingFace..."
    python3 -m huggingface_hub.commands.huggingface_cli download \
        --local-dir $critic_model_path \
        "$hf_critic_repo_id" "$hf_critic_model_path"
else
    echo "Using base model as initial critic: $base_model"
    critic_model_path=$base_model
fi

# ====================================================================
# RUN BC-PPO TRAINING
# ====================================================================
echo "Starting BC-PPO training..."

# Note: We use main_bc_ppo.py instead of main_ppo.py
# This automatically loads bc_ppo_trainer.yaml configuration
python3 -m meow_tea_train.verl.trainer.main_bc_ppo \
    actor_rollout_ref.model.path=$actor_model_path \
    critic.model.path=$critic_model_path \
    data.train_files=$local_train_data_dir \
    data.val_files=$local_train_data_dir \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.gamma=$gamma \
    algorithm.bias_correction=$bias_correction \
    algorithm.bias_decay=$bias_decay \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_kwargs.clip_ratio=$clip_ratio \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.temperature=$rollout_temp \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.batch_size=$rollout_batch_size_per_device \
    critic.optim.lr=$critic_lr \
    critic.ppo_micro_batch_size=$ppo_mini_batch_size \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_kwargs.num_epochs=$critic_update_epochs \
    reward_model.enable=False \
    reward_model.reward_manager=$reward_manager \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=[$logger] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.total_epochs=$max_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=1 \
    trainer.val_before_train=$val_before_train \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$train_batch_size \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    agentic.enable=$is_multiturn \
    agentic.environment=$env_name \
    agentic.max_iter=$max_iter \
    agentic.instance_id_start=$instance_id_start \
    agentic.instance_id_end=$instance_id_end \
    agentic.local_instances_dir=$local_instances_dir \
    agentic.reward_density=$reward_density \
    agentic.reward_type=$reward_type \
    agentic.rollout_temp=$rollout_temp \
    agentic.val_rollout_temp=$val_rollout_temp

echo "BC-PPO training completed!"
