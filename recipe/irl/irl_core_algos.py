
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

import torch
import verl
import verl.utils.torch_functional as verl_F


# TODO: implement more ways of doing advantage
"""
PPO (GAE):
    - Compute one-step TD residuals: \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    - Compute advantage: A_t = \sum_l=0^{\T-t-1} (\gamma\lambda)^l \delta_{t+l}
REINFORCE:
    - Compute Gain first: G_t = \sum_t \gamma^{t} r_t
    - advantage A_t = G_t - b_t, b_t = 0
RLOO:
    - Compute Gain first: G_t = \sum_t \gamma^{t} r_t
    - advantage A_t^{k} = [G_t^{k} - b_t^{k}], b_t^{k} = 1/(K-1) * \sum_{k' \neq k} G_t^{k'}
GRPO:
    - Compute Gain first: G_t = \sum_t \gamma^{t} r_t
    - advantage A_t^{k} =  [G_t^{k} - b_t^{k}]/\sigma, b_t^{k} = 1/(K-1) * \sum_{k' \neq k} G_t^{k'}, \sigma = std(G_t)
"""
def masked_rloo(reward_tensor_original, mask_tensor, n_samples, gamma=1.0):

    reward_tensor = reward_tensor_original.clone() # n_responses (n_samples(n_rollout) * n_inputs) x seq_len
    reward_tensor[~mask_tensor] = 0

    if gamma == 1.0:
        returns = reward_tensor.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1]) # G_t
    else:
        returns = torch.zeros_like(reward_tensors)
        running_return = 0
        for t in reversed(range(reward_tensor.size(1))):
            running_return = reward_tensor[:, t] + gamma * running_return
            returns[:, t] = running_return

    all_adjusted_returns = []

    for i in range(0, returns.shape[0], n_samples):
        # --- Get the chunk for this calculation ---
        chunk_returns = returns[i:i + n_samples] # n_samples x seq_len; G_t^{k} k = 1, 2, ..., n_samples
        chunk_mask = mask_tensor[i:i + n_samples].float()

        # --- Calculate a per-timestep baseline ---
        # Sum of returns from all samples in the chunk (for active tokens)
        sum_of_chunk_returns = (chunk_returns * chunk_mask).sum(dim=0) # sum_k G_t^{k} -> seq_len
        # Number of active samples at each timestep in the chunk
        num_active_samples = chunk_mask.sum(dim=0) # seq_len

        # Sum of returns from OTHER samples
        sum_of_other_returns = sum_of_chunk_returns - (chunk_returns * chunk_mask) # \sum_{k' \neq k} G_t^{k'}
        num_other_active = num_active_samples - chunk_mask # K-1
        
        # Clamp denominator to 1 to avoid division by zero
        # (if a token is active where all others are padded)
        denominator = num_other_active.clamp(min=1)

        baseline = sum_of_other_returns / denominator # b_t^{k}
        
        adjusted_returns = chunk_returns - baseline # A_t^{k}
        all_adjusted_returns.append(adjusted_returns)

    final_returns = torch.cat(all_adjusted_returns, dim=0)
    final_returns = final_returns * mask_tensor

    return final_returns

def masked_grpo(reward_tensor_original, mask_tensor, n_samples, gamma=1.0):

    def masked_std(x, mask, eps=1e-6):
        mask = mask.float()

        cnt  = mask.sum(dim=0)                       
        cnt_clamped = cnt.clamp(min=1)                
        mean = (x * mask).sum(0) / cnt_clamped 

        sq_diff = ((x - mean) * mask).pow(2)

        var  = sq_diff.sum(dim=0) / (cnt - 1).clamp(min=1)
        return var.sqrt().clamp_min(eps)               # σ ≥ eps

    reward_tensor = reward_tensor_original.clone()
    reward_tensor[~mask_tensor] = 0
    
    if gamma == 1.0:
        returns = reward_tensor.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1]) # G_t
    else:
        returns = torch.zeros_like(reward_tensor)
        running_return = 0
        for t in reversed(range(reward_tensor.size(1))):
            running_return = reward_tensor[:, t] + gamma * running_return
            returns[:, t] = running_return

    all_adjusted_returns = []

    for i in range(0, returns.shape[0], n_samples):
        # --- Get the chunk for this calculation ---
        chunk_returns = returns[i:i + n_samples]
        chunk_mask = mask_tensor[i:i + n_samples].float()

        # --- Calculate a per-timestep baseline ---
        # Sum of returns from all samples in the chunk (for active tokens)
        sum_of_chunk_returns = (chunk_returns * chunk_mask).sum(dim=0)
        # Number of active samples at each timestep in the chunk
        num_active_samples = chunk_mask.sum(dim=0)

        # Sum of returns from OTHER samples
        sum_of_other_returns = sum_of_chunk_returns - (chunk_returns * chunk_mask)
        num_other_active = num_active_samples - chunk_mask
        
        # Clamp denominator to 1 to avoid division by zero
        # (if a token is active where all others are padded)
        denominator = num_other_active.clamp(min=1)

        baseline = sum_of_other_returns / denominator
        
        adjusted_returns = chunk_returns - baseline

        chunk_std = masked_std(chunk_returns, chunk_mask, eps=1e-6) # std(G_t^k) -> seq_len
        adjusted_returns = adjusted_returns / chunk_std 

        all_adjusted_returns.append(adjusted_returns)

    final_returns = torch.cat(all_adjusted_returns, dim=0) # n_responses x seq_len
    final_returns = final_returns * mask_tensor

    return final_returns


def masked_gae(token_level_rewards, values, response_mask, gamma, lam):
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns



def compute_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):

    reward_tensors = []

    if config.algorithm.adv_estimator == 'rloo':
        masked_adv = masked_rloo
    elif config.algorithm.adv_estimator == 'grpo':
        masked_adv = masked_grpo
    elif config.algorithm.adv_estimator == 'gae':
        masked_adv = masked_gae
    else:
        raise NotImplementedError

    with torch.no_grad():

        if config.algorithm.adv_estimator == 'gae':
            # token-level rewards
            reward_tensor = data.batch['rm_scores'] * config.algorithm.reward_dpo_coef
            reward_mask = response_mask
            valid_response_length = reward_mask.sum(-1)
            if config.algorithm.reward_gt_coef != 0:
                reward_tensor[
                    torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                    valid_response_length - 1] += data.batch['labels'] * config.algorithm.reward_gt_coef
            
            # normalize the rewards
            reward_tensor[~reward_mask] = 0.0
            reverse_cumsum = torch.cumsum(reward_tensor.flip(dims=[1]),dim=-1).flip(dims=[1])
            reward_tensor = reward_tensor/(reverse_cumsum.abs().max()+1e-6)

            advantages, returns = masked_gae(reward_tensor, data.batch['values'], reward_mask, gamma=1.0, lam=1.0)
            return advantages, returns

        else:
            if 'rm_scores' in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.:
                reward_tensor = data.batch['rm_scores']
                reward_mask = response_mask.bool()
                # normalized the reward
                reward_tensor[~reward_mask] = 0.0
                reverse_cumsum = torch.cumsum(reward_tensor.flip(dims=[1]),dim=-1).flip(dims=[1])
                reward_tensor = reward_tensor/(reverse_cumsum.abs().max()+1e-6)
                reward_tensors.append(masked_adv(reward_tensor, reward_mask, n_samples, gamma=1.0) * config.algorithm.reward_dpo_coef)
            
            if 'labels' in data.batch.keys() and config.algorithm.reward_gt_coef != 0.:
                reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
                reward_mask = response_mask.bool()
                valid_response_length = reward_mask.sum(-1)
                reward_tensor[
                    torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                    valid_response_length - 1] = data.batch['labels']
                reward_tensors.append(masked_adv(reward_tensor, reward_mask, n_samples, gamma=1.0) * config.algorithm.reward_gt_coef)

            advantages = sum(reward_tensors)
            advantages = verl_F.masked_whiten(advantages, response_mask)

            return advantages, advantages
