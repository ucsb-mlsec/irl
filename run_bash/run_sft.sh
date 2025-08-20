set -x

export TOKENIZERS_PARALLELISM=True
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=6,7
# export RANK=0
# export WORLD_SIZE=1
# export LOCAL_RANK=0
# export MASTER_ADDR=localhost
# export MASTER_PORT=12355

data_path=$HOME/irl/data/prime_expert_demo.parquet
data_files="['$data_path']"
model_path=Qwen/Qwen2.5-3B-Instruct

nohup torchrun --nproc_per_node=2 --nnodes=1 \
  -m recipe.sft.main_sft \
  ulysses_sequence_parallel_size=1 \
  data.data_files="$data_files" \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=2 \
  data.val_batch_size=4 \
  data.max_prompt_length=1500 \
  data.max_response_length=3000 \
  data.truncation=right \
  data.train_split_ratio=0.98 \
  model.partial_pretrain=$model_path \
  model.enable_gradient_checkpointing=false \
  optim.lr=1e-6 \
  optim.warmup_steps_ratio=0.1 \
  optim.clip_grad=1 \
  trainer.total_epochs=1 \
  trainer.val_freqs=400 \
  trainer.logger='["console"]' \
  trainer.project_name=irl \
  trainer.experiment_name=Qwen2.5-3B-sft \
  trainer.default_local_dir=/home/henrygwb/irl/checkpoints/Qwen2.5-3B-sft \
  trainer.default_hdfs_dir=/home/henrygwb/irl/checkpoints/Qwen2.5-3B-sft > sft_log 2>&1 &


# torchrun --nproc_per_node=2 --nnodes=1 \
#   -m recipe.sft.main_sft \
#   ulysses_sequence_parallel_size=1 \
#   data.data_files="$data_files" \
#   data.train_batch_size=64 \
#   data.micro_batch_size_per_gpu=2 \
#   data.val_batch_size=4 \
#   data.max_prompt_length=1500 \
#   data.max_response_length=3000 \
#   data.truncation=right \
#   data.train_split_ratio=0.98 \
#   model.partial_pretrain=$model_path \
#   model.enable_gradient_checkpointing=false \
#   optim.lr=1e-6 \
#   optim.warmup_steps_ratio=0.1 \
#   optim.clip_grad=1 \
#   trainer.total_epochs=1 \
#   trainer.val_freqs=10 \
#   trainer.logger='["console"]' \
#   trainer.project_name=irl \
#   trainer.experiment_name=Qwen2.5-3B-sft \
#   trainer.default_local_dir=/home/henrygwb/irl/checkpoints/Qwen2.5-3B-sft \
#   trainer.default_hdfs_dir=/home/henrygwb/irl/checkpoints/Qwen2.5-3B-sft
