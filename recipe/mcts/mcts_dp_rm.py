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
import torch.nn.functional as F


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from verl.utils import hf_tokenizer

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelMCTSRewardModel']


class DataParallelMCTSRewardModel:

    def __init__(self, config, tokenizer, reward_module: nn.Module):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_module = reward_module
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

        # hard coding
        self.policy_tokenizer = hf_tokenizer('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=config.model.get('trust_remote_code', False))

        # process reward 
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902


    def _forward_micro_batch(self, micro_batch, response_length):
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        input_ids = micro_batch['input_ids']
        max_positions = micro_batch['attention_mask'][:, -response_length:].sum(-1)

        q_ids = input_ids[:, :-response_length]
        a_ids = input_ids[:, -response_length:]

        # decode the sentence first
        q_texts = [self.policy_tokenizer.decode(q_id, skip_special_tokens=True) for q_id in q_ids]
        a_texts = [self.policy_tokenizer.decode(a_id, skip_special_tokens=True) for a_id in a_ids]

        # add step_tag before a_texts
        a_texts = [ans.replace("\n\n", " ки\n\n") + ' ки' for ans in a_texts]

        ret = torch.zeros_like(input_ids[:, -response_length:]).to(torch.float)

        for b_idx, (question, output, a_id_row) in enumerate(zip(q_texts, a_texts, a_ids)):
            input_for_prm = f"{question} {output}"
            input_id = torch.tensor([self.tokenizer.encode(input_for_prm)])
            logits = self.reward_module(input_id).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            step_scores = scores[input_id == self.step_tag_id]
            # fill the last step first
            ret[b_idx, max_positions[b_idx] - 1] = step_scores[-1]
            step_scores = step_scores[:-1]
            if len(step_scores) != 0:
                a_id_list = a_id_row.tolist()
                n = max_positions[b_idx]
                newline_positions = []
                for i in range(n):
                    if '\n\n' in self.policy_tokenizer.decode(a_id_list[i]):
                        newline_positions.append(i)
                # --- Fill ret with step scores ---
                for score, pos in zip(step_scores, newline_positions):
                    if pos < n:
                        ret[b_idx, pos] = score
        return ret
 
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
                rm_score = self._forward_micro_batch(micro_batch, response_length)
            rm_scores_lst.append(rm_score)
        
        rm_scores = torch.concat(rm_scores_lst, dim=0)

        if self.config.use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]
        
        return rm_scores