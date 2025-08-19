#!/usr/bin/env python3
"""
FSDP SFT Trainer
"""

import os
import logging
import asyncio
import re
import sys
import statistics
import uuid
from copy import deepcopy
from pprint import pprint
from collections import defaultdict
from omegaconf import OmegaConf, open_dict
from contextlib import nullcontext

import torch
import numpy as np
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

from peft import LoraConfig, TaskType, get_peft_model # for lora

from verl.utils.dataset.rl_dataset import RLHFDataset
from ..irl.dataset import IRLDataset, collate_fn



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

from concurrent.futures import ProcessPoolExecutor
from functools import partial


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_info=None,
                                       num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDP_SFT_Trainer(object):

    def __init__(self, config, device_mesh, ulysses_device_mesh):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        # manage fsdp and ulysses sharding
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh) 

        # build tokenizer 
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True) # download model to the local path
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)

        # normalize dp size
        self._normalize_batch_size()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1) # default to 1
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False) # whether to remove padding tokens
        # ensure only one process prints
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')

        self._build_dataloader()
        self._build_model_optimizer(local_model_path) # self.fsdp_model, self.optimizer

    def _normalize_batch_size(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        # global batch size must be divisible by dp size
        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        # the batch size per GPU must be divisible by micro batch size per GPU
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_model_optimizer(self, local_model_path):
        """
        Build the model and optimizer.

        Procedure: 
        1. Load model config and model; 
        2. Config monkey patch for Ulysses sequence parallelism and remove padding optimizations; 
        3. Apply Liger kernel for optimization; 
        4. Wrap model with FSDP and CPU offloading; 
        5. Initialize optimizer.
        """

        # log_gpu_memory_usage('Before model allocation', logger=logger)

        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=False) # load model config
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # Create a context manager for initializing weights (for memory optimization)
        # Don't use meta tensor if tie_word_embeddings is enabled
        # Config.tie_word_embeddings means that the word embeddings and output layer shares the same weight
            # more efficient and similar performance 
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path, # model path in huggingface repo
                                                                               config=config,
                                                                               torch_dtype=torch.float32,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=self.config.model.trust_remote_code)

            # Apply_monkey_patch: modifies existing HuggingFace transformer models at runtime to support Ulysses sequence parallelism and remove padding optimizations
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel: optimization of transformer operations, loss, activations
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,           # Type of task (Causal Language Modeling)
                    'r': self.config.model.lora_rank,          # Rank of adaptation (e.g., 8, 16, 64)
                    'lora_alpha': self.config.model.lora_alpha, # LoRA scaling parameter
                    'target_modules': convert_to_regular_types(self.config.model.target_modules), # Which layers to adapt
                    'bias': "none"                             # Don't adapt bias terms
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        # Save only partial activations (checkpoints) during forward pass; the discarded activations will be recomputed during backward pass
        # Trade off between memory and computation
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        # log_gpu_memory_usage('After model allocation', logger=logger)

        if self.config.model.get('mixed_precision', True):
            mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, # model parameters
                                            reduce_dtype=torch.float32, # gradients
                                            buffer_dtype=torch.float32) # buffers (layernorm stats)
        else:
            mixed_precision = None

        # auto_wrap_policy: controls the granularity of sharding; layer-based or size-based 
        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy, 
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)

        # CPU offloading: move parameters to CPU when not in use to save GPU memory
        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        # log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        # log_gpu_memory_usage('After initialize optimizer', logger=logger)

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        self.steps_per_epoch = len(self.train_dataloader) # num_of_batch
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
    
        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch (batches) {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=self.total_steps)


    def _build_dataloader(self):
        """
        Load data and build dataloaders.
        1. Construct datasets;
        2. Configure ulysses sequence (local rank and mesh size)
        3. Configure data samplers (for distributed sampling) and loaders (load each batch based on the sampler)
        """

        config = self.config
        # Build dataset
        self.train_dataset = IRLDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.truncation, 
            filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False)
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.truncation, 
            filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False)
        )
        
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size, # per gpu batch size
                                           sampler=self.train_sampler,
                                           num_workers=config.data.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=False,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.val_batch_size,
                                         sampler=self.val_sampler,
                                         num_workers=config.data.num_workers,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=collate_fn) # merge individual element into a batch

        # print(f'Size of training dataloader: {len(self.train_dataloader)}')
        # print(f'Size of testing dataloader: {len(self.val_dataloader)}')

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch['input_ids'].cuda() # [batch_size, seqlen+response_len]
        response_ids = batch['responses'].cuda() # [batch_size, response_len]
        attention_mask = batch['attention_mask'].cuda() # [batch_size, seqlen]
        position_ids = batch['position_ids'].cuda() # [batch_size, seqlen]
        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[:,self.config.data.max_prompt_length:] = attention_mask[:,self.config.data.max_prompt_length:]
        loss_mask = loss_mask.cuda()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids[:, 1:].contiguous() 
                    output = self.fsdp_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             position_ids=position_ids,
                                             use_cache=False) # next token for all the tokens in the input; same size as the input but need to remove the last token 
                    logits = output.logits

                    shift_logits = logits[..., :-1, :].contiguous() # make sure the matrix stores continuous in memory
                    shift_labels = labels.contiguous()
                    loss_mask = loss_mask[:, 1:].contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    loss_mask = loss_mask.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)  # to make sure the data is on the same device; when model is distributed into different devices
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss * loss_mask.to(loss.device)
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler
                    # TODO: debug this code
                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                               attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        use_cache=False)

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    # Gather and unpad for sequence parallelism
                    loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                    # This is the loss collected from all ulysses ranks
                    full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                                          indices=indices,
                                          batch=batch_size,
                                          seqlen=seqlen)
                    full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    full_loss = full_loss.reshape(-1)
                    loss_mask = loss_mask.to(full_loss.device)
                    loss = full_loss * loss_mask

                valid_token_this_rank = torch.sum(loss_mask)

                if self.config.data.balance_dp_token:
                    torch.distributed.all_reduce(valid_token_this_rank)
                    dp_size = self.ulysses_device_mesh.size('dp') if use_sp else torch.distributed.get_world_size()
                else:
                    dp_size = 1

                loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

                if do_backward:
                    loss.backward()
                return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

    
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG) # sync loss across all gpus

        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}


    def validation_loss(self):
        self.fsdp_model.eval()

        for val_batch in self.val_dataloader:
            input_ids = val_batch['input_ids'].cuda() # [batch_size, seqlen+response_len]
        response_ids = val_batch['responses'].cuda() # [batch_size, response_len]
        attention_mask = batch['attention_mask'].cuda() # [batch_size, seqlen]
        position_ids = batch['position_ids'].cuda() # [batch_size, seqlen]
        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[:,self.config.data.max_prompt_length:] = attention_mask[:,self.config.data.max_prompt_length:]
        loss_mask = loss_mask.cuda()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids[:, 1:].contiguous() 
                    output = self.fsdp_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             position_ids=position_ids,
                                             use_cache=False) # next token for all the tokens in the input; same size as the input but need to remove the last token 
                    logits = output.logits

                    shift_logits = logits[..., :-1, :].contiguous() # make sure the matrix stores continuous in memory
                    shift_labels = labels.contiguous()
                    loss_mask = loss_mask[:, 1:].contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    loss_mask = loss_mask.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)  # to make sure the data is on the same device; when model is distributed into different devices
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss * loss_mask.to(loss.device)
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler
                    # TODO: debug this code
                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                               attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        use_cache=False)

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    # Gather and unpad for sequence parallelism
                    loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                    # This is the loss collected from all ulysses ranks
                    full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                                          indices=indices,
                                          batch=batch_size,
                                          seqlen=seqlen)
                    full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    full_loss = full_loss.reshape(-1)
                    loss_mask = loss_mask.to(full_loss.device)
                    loss = full_loss * loss_mask

                valid_token_this_rank = torch.sum(loss_mask)

                if self.config.data.balance_dp_token:
                    torch.distributed.all_reduce(valid_token_this_rank)
                    dp_size = self.ulysses_device_mesh.size('dp') if use_sp else torch.distributed.get_world_size()
                else:
                    dp_size = 1

                loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

                if do_backward:
                    loss.backward()
                return loss



    def validation_acc(self):

        """
            Batch: 'input_ids', 'attention_mask', 'position_ids', 'data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids', 'raw_prompt', 'index'])
            # input_ids: [val_batch_size, input_seqlen]
            # 'reward_model' seems how to compte the whether the answer is correct
            output keys ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids'])) # 'input_ids', 'attention_mask', 'position_ids' connection of prompts and responses
            attention_mask: 1 means it is a valid token, 0 means it is a padding token
        """

        sample_scores = []
        data_source_lst = []

        self.fsdp_model.eval()

        for val_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(val_data)
            with torch.no_grad(), FSDP.summon_full_params(self.fsdp_model, writeback=False): # pull all model paramters to the current gpu
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    input_ids = test_batch.batch['input_ids'].cuda()
                    # input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                    attention_mask = test_batch.batch['attention_mask'].cuda()
                    position_ids = test_batch.batch['position_ids'].cuda()

                    output_ids = self.fsdp_model.module.generate(
                        input_ids=input_ids.to(torch.long),
                        attention_mask=attention_mask.to(torch.long),
                        max_new_tokens=self.config.data.max_response_length,
                        do_sample=False,  # Greedy for validation
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        temperature=1.0,
                        use_cache=True
                    )
                            
            # evaluate using reward_function
            sequences_str = [self.tokenizer.batch_decode(ids[self.config.data.max_prompt_length:], skip_special_tokens=True) for ids in output_ids]
            ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in test_batch]
            data_sources = test_batch.non_tensor_batch['data_source']
            extra_info = test_batch.non_tensor_batch.get('extra_info', None)

            assert len(sequences_str) == len(ground_truth) == len(data_sources)

            self.compute_score = _default_compute_score
            # score: a list of whether each answer is correct or not 0 or 1
            try:
                scores = asyncio.run(
                    parallel_compute_score_async(self.compute_score,
                                                sequences_str,
                                                ground_truth,
                                                data_sources,
                                                extra_info=extra_info,
                                                num_processes=64))
            except asyncio.TimeoutError as e:
                print('Global timeout in reward computing! Setting all as 0.')
                scores = [0. for _ in range(len(sequences_str))]
            except Exception as e:
                print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
                scores = [0. for _ in range(len(sequences_str))]

            sample_scores.extend(scores)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(scores)))

        data_sources = np.concatenate(data_source_lst, axis=0)

        dataset_totals = defaultdict(int)
        dataset_correct = defaultdict(int)

        # Process each example
        for score, dataset in zip(sample_scores, data_sources):
            # Convert score to float to ensure it's numeric
            score_float = float(score)
            dataset_totals[dataset] += 1
            dataset_correct[dataset] += score_float

        # Calculate accuracy for each dataset
        metric_dict = {}
        for dataset in dataset_totals:
            accuracy = dataset_correct[dataset] / dataset_totals[dataset]
            metric_dict[f"val_acc/{dataset}"] = accuracy
        
        # metric_dict["val_acc/overall"] = sum(dataset_correct.values()) / sum(dataset_totals.values())
        metric_dict['val_acc/overall'] = sum(metric_dict.values()) / len(metric_dict)
        
        return metric_dict

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank() # for print and logging purposes

        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        global_step = 0

        for epoch in range(self.config.trainer.total_epochs):
            print(f"Starting epoch {epoch+1}/{self.config.trainer.total_epochs}")
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    print(f"Step {global_step} out of {len(self.train_dataloader)} steps")
                    tracking.log(data=metric, step=global_step)

                # for early exit validation
                # if global_step%self.config.trainer.val_freqs == 0:
                #     metric_dict = self.validation()
                #     if rank == 0:
                #         print('=====================================================================')
                #         print('=====================================================================')
                #         print(metric_dict)
                #         tracking.log(data=metric_dict, step=global_step)
                #     torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)

        return


# save and load model
# change val as loss
# do testing
