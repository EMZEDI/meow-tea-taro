#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1

# BC-PPO Recipe for TextWorld Cooking Task
# This script demonstrates how to run BC-PPO (Bias-Corrected PPO) 
# instead of standard PPO by using the bc_ppo_trainer configuration.

# DATA/TASK CONFIG
scratch_dir="$SCRATCH/.cache/huggingface/meow_tea_train"
export HF_HOME="$scratch_dir"
export HF_HUB_CACHE="$scratch_dir"
export HF_DATASETS_CACHE="$scratch_dir"
env_name="textworld"
task_prefix="tw_dense"
instance_id_start=50001
instance_id_end=53000
hf_data_repo="PEARLS-Lab/meow-tea-taro-dataset"
hf_instances_dir="$env_name/$task_prefix/instances"
hf_train_data_dir="$env_name/$task_prefix/multiturn_rl_data/3000_train_data"
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
logger="['console','wandb']"

# HARDWARE CONFIG
nnodes=1
n_gpus_per_node=1
colocate_critic_reward=True
colocate_actor_ref=True

# Step 1: Process RL data
echo "Processing multiturn RL data for tasks ${env_name}-${task_prefix} ${instance_id_start}-${instance_id_end}"
python3 -m meow_tea_train.agentic_utils.data_process.rl_data_processor \
    --env_name "$env_name" \
    --task_prefix "$task_prefix" \
    --instance_id_range "$instance_id_start" "$instance_id_end" \
    --hf_data_repo "$hf_data_repo" \
    --hf_instances_dir "$hf_instances_dir" \
    --hf_train_data_dir "$hf_train_data_dir" \
    --local_instances_dir "$local_instances_dir" \
    --local_train_data_dir "$local_train_data_dir" \
    --local_parquet_dir "$local_parquet_dir" \
    --reward_method "$reward_method"

# Step 2: Load models
echo "Loading models..."
# Check if actor model is specified
if [ -n "$hf_actor_repo_id" ]; then
    # If specified, download from HF path if available
    if [ -z "$hf_actor_model_path" ]; then
        # Download entire repo if path is empty/None
        hf download $hf_actor_repo_id --local-dir $actor_model_path
    else
        # Download specific path and flatten
        hf download $hf_actor_repo_id --include="${hf_actor_model_path}/*" --local-dir $actor_model_path
        mv $actor_model_path/$hf_actor_model_path/* $actor_model_path/
        rm -rf $actor_model_path/$hf_actor_model_path
        rm -rf $actor_model_path/.cache
    fi
else
    # Otherwise, use base model (from HF)
    actor_model_path=$base_model
fi

# Check if critic model is specified
if [ -n "$hf_critic_repo_id" ]; then
    # If specified, download from HF path if available
    if [ -z "$hf_critic_model_path" ]; then
        # Download entire repo if path is empty/None
        hf download $hf_critic_repo_id --local-dir $critic_model_path
    else
        # Download specific path and flatten
        hf download $hf_critic_repo_id --include="${hf_critic_model_path}/*" --local-dir $critic_model_path
        mv $critic_model_path/$hf_critic_model_path/* $critic_model_path/
        rm -rf $critic_model_path/$hf_critic_model_path
        rm -rf $critic_model_path/.cache
    fi
else
    # Otherwise, use base model (from HF)
    critic_model_path=$base_model
fi

# ====================================================================
# RUN BC-PPO TRAINING
# ====================================================================
echo "Starting BC-PPO training..."

# Note: We use main_bc_ppo.py instead of main_ppo.py
# This automatically loads bc_ppo_trainer.yaml configuration
source .venv/bin/activate
python3 -m meow_tea_train.verl.trainer.main_bc_ppo \
    data.train_files="$local_parquet_dir/train.parquet" \
    data.val_files="$local_parquet_dir/validation.parquet" \
    data.return_raw_chat=True \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.train_batch_size=$train_batch_size \
    data.dataloader_num_workers=4 \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.bias_correction=$bias_correction \
    algorithm.bias_decay=$bias_decay \
    algorithm.gamma=$gamma \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    agentic.environment.name=$env_name \
    agentic.environment.is_multiturn=$is_multiturn \
    agentic.environment.is_async=$is_async \
    agentic.environment.max_iter=$max_iter \
    agentic.reward.density=$reward_density \
    agentic.reward.type=$reward_type \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.clip_ratio=$clip_ratio \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    +actor_rollout_ref.rollout.agentic='${agentic}' \
    actor_rollout_ref.rollout.temperature=$rollout_temp \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_rollout_temp \
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.use_dynamic_bsz=True \
    reward_model.reward_manager=$reward_manager \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.validation_data_dir="local/val_results" \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.val_before_train=True \
    trainer.hf_kwargs.save_hf_repo_id=$save_hf_repo_id \
    trainer.hf_kwargs.resume_wandb_logs=$resume_wandb_logs \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$num_epochs $@

echo "BC-PPO training completed!"
