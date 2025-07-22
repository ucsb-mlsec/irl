set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false

policy_val_path=$HOME/irl/data/validation.parquet

# train_path=$HOME/irl/data/s1k_train_filter.parquet
# demo_path=$HOME/irl/data/claude_s1k_demo_filter.parquet

# train_path=$HOME/irl/data/prime_correct_only_train.parquet
# demo_path=$HOME/irl/data/prime_correct_only_expert_demo.parquet

# train_path=$HOME/irl/data/prime_train.parquet
# demo_path=$HOME/irl/data/prime_expert_demo.parquet

train_path=$HOME/irl/data/prime_train.parquet
demo_path=$HOME/irl/data/prime_expert_demo.parquet

train_files="['$train_path']"
expert_files="['$demo_path']"
test_files="['$policy_val_path']"

model_path=Qwen/Qwen2.5-3B-Instruct

python -m recipe.irl.main_irl \
    data.policy_train_files="$train_files" \
    data.expert_demo_files="$expert_files" \
    data.policy_val_files="$test_files" \
    data.train_batch_size=128 \
    data.shuffle=True \
    data.val_batch_size=6312 \
    data.max_prompt_length=1536 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=False \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.1 \
    reward_model.model.path=$model_path \
    reward_model.model.use_remove_padding=False \
    reward_model.mini_batch_size=32 \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.rm_epochs=1 \
    reward_model.policy_reward_weight=1 \
    reward_model.model.optim.lr=3e-8 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='irl' \
    trainer.experiment_name='Qwen2.5-3B_bs128_ppo1_lr3e-8_correct_clip_10.0' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=4 \
    trainer.max_ckpt_to_keep=1 \
    trainer.total_epochs=10 $@ > Qwen2.5-3B_bs128_ppo1_lr3e-8_correct_clip_10.0.log
