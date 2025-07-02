# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""
import itertools
from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

from .irl_core_algos import compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelIRLRewardModel']


class DataParallelIRLRewardModel:

    def __init__(self, config, reward_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Reward model use_remove_padding={self.use_remove_padding}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch, response_length):
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        input_ids = micro_batch['input_ids']
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch['attention_mask']
        position_ids = micro_batch['position_ids']

        max_positions = micro_batch['attention_mask'][:, -response_length:].sum(-1)

        # TODO: remove padding and uylsses pad and slice inputs are not compatible yet
        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                print("using ulysses pad and slice inputs")
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            rm_output_logits = self.reward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False
            )

            if self.ulysses_sequence_parallel_size > 1:
                rm_output_logits = gather_outpus_and_unpad(rm_output_logits, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            
            rm_output_logits = pad_input(
                hidden_states=rm_output_logits.unsqueeze(-1),
                indices=indices,
                batch=batch_size,
                seqlen=seqlen
            ).squeeze(-1)[:, -response_length - 1:-1]

        else:
            rm_output_logits = self.reward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )

        # q = rm_output_logits[:, -num_actions:]

        # # trim unnecessary logprobs here
        # for i in range(micro_batch['input_ids'].shape[0]):
        #     q[i, max_positions[i]:] = 0

        # return q, max_positions

        # Create a mask to avoid in-place operations
        mask = torch.ones_like(rm_output_logits[:, -response_length:])
        for i in range(micro_batch['input_ids'].shape[0]):
            mask[i, max_positions[i]:] = 0
            
        # Apply mask via multiplication rather than in-place assignment
        q = rm_output_logits[:, -response_length:] * mask

        return q, max_positions

        # q = rm_output_logits[:, -num_actions:].clone()  # Clone here to avoid in-place issues
        
        # # Create a mask for valid positions
        # mask = torch.ones_like(q, dtype=torch.bool)
        # for i in range(micro_batch['input_ids'].shape[0]):
        #     mask[i, max_positions[i]:] = False
        
        # # Apply the mask without in-place operations
        # masked_q = q * mask.float()
        
        # return masked_q, max_positions

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm of the reward model is not finite: {grad_norm}, skipping the update")
            self.reward_optimizer.zero_grad()
        else:
            self.reward_optimizer.step()
        
        return grad_norm

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        micro_batch_size = self.config.micro_batch_size_per_gpu
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        response_length = data.batch['responses'].shape[-1]

        if self.config.use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, _ = self._forward_micro_batch(micro_batch, response_length)
            rm_scores_lst.append(rm_score)
        
        rm_scores = torch.concat(rm_scores_lst, dim=0)

        if self.config.use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return rm_scores

    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        # select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'is_expert', 'old_log_probs']
        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'is_expert', 'old_log_probs', 'labels']

        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        print(f"dataloader size: {len(dataloader)}")

        # expert_losses = []
        # policy_losses = []
        # importance_scores = []
        # grad_norms = []

        # first update the reward model
        loss = 0
        for epoch in range(self.config.rm_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                # split batch into micro_batches
                # TODO: dynamic bsz may not be compatible yet
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu
                
                self.reward_module.zero_grad()

                is_expert = mini_batch['is_expert']
                
                cur_idx = 0
                for micro_batch in micro_batches:
                    cur_idx += len(micro_batch)

                    question_id = cur_idx // 8
                    # hard code the batch size to 8
                    cur_is_expert = is_expert[question_id:question_id+8]
                    expert_num = torch.sum(cur_is_expert).item()
                    policy_num = torch.sum(torch.logical_not(cur_is_expert)).item()

                    micro_batch = micro_batch.cuda()

                    response_ids = micro_batch['responses']
                    response_length = response_ids.shape[-1]
                    bs = response_ids.shape[0]

                    # Forward pass to get rewards
                    rm_score, max_positions = self._forward_micro_batch(micro_batch, response_length)
                    print(f"Epoch {epoch}, batch {batch_idx}, bs: {bs}, response_length: {response_length}")

                    labels = micro_batch['labels']
                    expert_mask = micro_batch['is_expert']
                    policy_mask = torch.logical_not(expert_mask)

                    # Calculate the total reward for each trajectory
                    # Sum along sequence dimension to get total reward per sample
                    trajectory_rewards = rm_score.sum(dim=-1, keepdim=True)
                    normalized_trajectory_rewards = trajectory_rewards / max_positions.unsqueeze(-1).float()

                    # # Calculate importance scores for policy samples
                    # if policy_mask.sum() > 0:
                    #     policy_log_probs = micro_batch["old_log_probs"][policy_mask]

                    #     # Create a mask to avoid in-place operations
                    #     mask = torch.ones_like(policy_log_probs)
                    #     for i in range(micro_batch['input_ids'].shape[0]):
                    #         mask[i, max_positions[i]:] = 0
                            
                    #     # Apply mask via multiplication rather than in-place assignment
                    #     policy_log_probs = policy_log_probs[:, -response_length:] * mask
                        
                    #     policy_log_probs = torch.sum(policy_log_probs, dim=-1, keepdim=True) / max_positions[policy_mask].unsqueeze(-1).float()

                    #     policy_rewards = normalized_trajectory_rewards[policy_mask]

                    #     epsilon = 1e-10
                    #     # normalize the importance score here to avoid near zero importance score
                    #     # actually the importance score decays with larger reasoning length
                    #     # if normalized, the importance score will be nearly equal to 1
                    #     traj_importance = torch.exp((policy_rewards - (policy_log_probs + epsilon)))
                    # else:
                    #     traj_importance = torch.ones_like(trajectory_rewards[policy_mask])

                    # # Calculate the loss
                    # loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    #     normalized_trajectory_rewards.squeeze(-1),
                    #     labels.float(),
                    #     reduction='mean'
                    # )

                    if policy_mask.sum() > 0:
                        loss = normalized_trajectory_rewards / policy_num
                    
                    if expert_mask.sum() > 0:
                        loss = -normalized_trajectory_rewards / expert_num

                    print(f"Epoch {epoch}, batch {batch_idx}, labels: {labels}, trajectory_rewards: {trajectory_rewards}, normalized_trajectory_rewards: {normalized_trajectory_rewards}, loss: {loss.item()}")

                    # # Separate expert and policy rewards
                    # expert_rewards = normalized_trajectory_rewards[expert_mask]
                    # policy_rewards = normalized_trajectory_rewards[policy_mask]

                    print("*"*20)
                    if expert_mask.sum() > 0:               
                        print(f"Epoch {epoch}, batch {batch_idx}, expert rewards: {normalized_trajectory_rewards}")
                        text = self.reward_module.tokenizer.batch_decode(micro_batch["responses"][expert_mask], skip_special_tokens=True)
                        print(f"Epoch {epoch}, batch {batch_idx}, expert text: {text[0][:100]}")
                    else:
                        print(f"Epoch {epoch}, batch {batch_idx}, policy rewards: {normalized_trajectory_rewards}")
                        text = self.reward_module.tokenizer.batch_decode(micro_batch["responses"][policy_mask], skip_special_tokens=True)
                        print(f"Epoch {epoch}, batch {batch_idx}, policy text: {text[0][:100]}")
                    
                    print("*"*20)
                                    
                    # weighted_policy_rewards = policy_rewards * traj_importance
                    # # weighted_policy_rewards = policy_rewards

                    # # print("weighted_policy_rewards")
                    # # print(weighted_policy_rewards)

                    # # For experts: we want to maximize rewards, and the optimal reward is 1
                    # expert_loss = 1 - torch.mean(expert_rewards) if expert_rewards.numel() > 0 else 0
                    
                    # beta = 1 / 3

                    # policy_loss = torch.sum(weighted_policy_rewards) if weighted_policy_rewards.numel() > 0 else 0
                    
                    # # expert_loss = expert_loss / expert_num if expert_num > 0 else 0
                    # # policy_loss = policy_loss / policy_num if policy_num > 0 else 0
                    
                    # loss = (expert_loss + beta * policy_loss) / bs
                    # # loss = (expert_loss + policy_loss) / bs

                    # print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss.item()}")
                    # if weighted_policy_rewards.numel() > 0:
                    #     print(f"Epoch {epoch}, batch {batch_idx}, policy_loss: {policy_loss.item()}")
                    # if expert_rewards.numel() > 0:
                    #     print(f"Epoch {epoch}, batch {batch_idx}, expert_loss: {expert_loss.item()}")

                    # # Record metrics
                    # if expert_rewards.numel() > 0:
                    #     expert_losses.append(expert_loss.detach().item())
                    # if weighted_policy_rewards.numel() > 0:
                    #     policy_losses.append(policy_loss.detach().item())
                    # if traj_importance.numel() > 0:
                    #     importance_scores.append(traj_importance.detach().mean().item())

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = loss * (len(micro_batch) / self.config.ppo_mini_batch_size)
                    else:
                        loss = loss / self.gradient_accumulation
                    
                    loss.backward()

                self._optimizer_step()
                # grad_norm = self._optimizer_step()
                # grad_norms.append(grad_norm.detach().item())

        # # Add counts to metrics to help with proper synchronization
        # metrics['rm/expert_loss'] = expert_losses
        # metrics['rm/policy_loss'] = policy_losses
        # metrics['rm/importance_score'] = importance_scores
        # metrics['rm/grad_norm'] = grad_norms
        
        self.reward_optimizer.zero_grad()

        return {}