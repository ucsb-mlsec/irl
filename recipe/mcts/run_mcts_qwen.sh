set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_API_KEY='c2fef355c334fecafdb3d2d32f0e8c07b8349538'

policy_val_path=$HOME/irl/data/validation.parquet

train_path=$HOME/irl/data/prime_train.parquet
demo_path=$HOME/irl/data/prime_expert_demo.parquet

train_files="['$train_path']"
expert_files="['$demo_path']"
test_files="['$policy_val_path']"

ENTROPY_COEFF=0.001

model_path=Qwen/Qwen2.5-3B-Instruct
reward_model_path=peiyi9979/math-shepherd-mistral-7b-prm
exp_name=mcts_math

nohup python3 -m recipe.mcts.main_mcts \
    data.policy_train_files="$train_files" \
    data.expert_demo_files="$expert_files" \
    data.policy_val_files="$test_files" \
    data.shuffle=True \
    data.train_batch_size=128 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1536 \
    data.max_response_length=3100 \
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
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.model.path=$reward_model_path \
    reward_model.model.use_remove_padding=False \
    reward_model.mini_batch_size=32 \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.rm_epochs=1 \
    reward_model.policy_reward_weight=1 \
    reward_model.model.optim.lr=3e-8 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    trainer.val_before_train=True \
    trainer.resume_mode="disable" \
    trainer.resume_from_path="/scr/xian/rloo/global_step_94" \
    trainer.logger=['console','wandb'] \
    trainer.project_name='iclr_mcts' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.max_ckpt_to_keep=1 \
    trainer.default_local_dir="/scr/xian/mcts_math"\
    trainer.total_epochs=2 \
    > ./logs/${exp_name}.log 2>&1 &