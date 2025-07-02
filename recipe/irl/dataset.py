# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
import json
from typing import List, Union, Optional, Callable
import copy
import datasets
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import torch.nn.functional as F

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)

def tokenize_and_postprocess_whole_chat(prompt: str,
                                        complete_chat: str,
                                        tokenizer: PreTrainedTokenizer,
                                        max_prompt_length: int,
                                        max_response_length: int,
                                        pad_token_id: int,
                                        truncation='error'):
    
    assert truncation in ['left', 'right', 'error']

    prompt_input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
    prompt_attention_mask = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['attention_mask']

    response_input_ids = tokenizer(complete_chat, return_tensors='pt', add_special_tokens=False)['input_ids']
    response_attention_mask = tokenizer(complete_chat, return_tensors='pt', add_special_tokens=False)['attention_mask']

    response_input_ids = response_input_ids[:, prompt_input_ids.shape[-1]:]
    response_attention_mask = response_attention_mask[:, prompt_attention_mask.shape[-1]:]

    prompt_sequence_length = prompt_input_ids.shape[-1]
    response_sequence_length = response_input_ids.shape[-1]
    
    if prompt_sequence_length < max_prompt_length:
        prompt_input_ids = pad_sequence_to_length(prompt_input_ids,
                                                 max_seq_len=max_prompt_length,
                                                 pad_token_id=pad_token_id,
                                                 left_pad=True)
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask,
                                                       max_seq_len=max_prompt_length,
                                                       pad_token_id=0,
                                                       left_pad=True)
    elif prompt_sequence_length > max_prompt_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            prompt_input_ids = prompt_input_ids[:, -max_prompt_length:]
            prompt_attention_mask = prompt_attention_mask[:, -max_prompt_length:]
        elif truncation == 'right':
            prompt_input_ids = prompt_input_ids[:, :max_prompt_length]
            prompt_attention_mask = prompt_attention_mask[:, :max_prompt_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{prompt_sequence_length=} is larger than {max_prompt_length=}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    if response_sequence_length < max_response_length:
        response_input_ids = pad_sequence_to_length(response_input_ids,
                                                    max_seq_len=max_response_length,
                                                    pad_token_id=pad_token_id,
                                                    left_pad=False)
        response_attention_mask = pad_sequence_to_length(response_attention_mask,
                                                         max_seq_len=max_response_length,
                                                         pad_token_id=0,
                                                         left_pad=False)
    elif response_sequence_length > max_response_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            response_input_ids = response_input_ids[:, -max_response_length:]
            response_attention_mask = response_attention_mask[:, -max_response_length:]
        elif truncation == 'right':
            response_input_ids = response_input_ids[:, :max_response_length]
            response_attention_mask = response_attention_mask[:, :max_response_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{response_sequence_length=} is larger than {max_response_length=}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')
    
    # concatenate prompt and response
    input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

    return input_ids, attention_mask



def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class IRLDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key: str = 'prompt',
                 image_key: str = 'images',
                 max_prompt_length: int = 1024,
                 max_response_length: int = 8192,
                 filter_prompts=True,
                 cache_dir: str = '~/.cache/verl/rlhf',
                 chat_template_func: Optional[Callable] = None,
                 return_raw_chat: bool = False,
                 truncation: str = 'error',
                 filter_overlong_prompts: bool = False,
                 num_workers: Optional[int] = None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() // 4)
        else:
            self.num_workers = min(num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
                               ) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        chat = row_dict.pop(self.prompt_key)

        chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if "response" in row_dict.keys():
            complete_chat = row_dict.pop("complete_chat")
            overall_chat_template = self.tokenizer.apply_chat_template(complete_chat, add_generation_prompt=True, tokenize=False)
            input_ids, attention_mask = tokenize_and_postprocess_whole_chat(
                prompt=chat_template,
                complete_chat=overall_chat_template,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                truncation=self.truncation
            )
        else:            
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=chat_template, 
                tokenizer=self.tokenizer, 
                max_length=self.max_prompt_length, 
                pad_token_id=self.tokenizer.pad_token_id, 
                left_pad=True, 
                truncation=self.truncation
            )

        position_ids = compute_position_id_with_mask(attention_mask)

        if "response" in row_dict.keys():
            raw_response = row_dict.pop("response")
            response_ids, _ = verl_F.tokenize_and_postprocess_data(prompt=raw_response, tokenizer=self.tokenizer, max_length=self.max_response_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=False, truncation=self.truncation)
            row_dict['responses'] = response_ids[0]
            is_expert = row_dict.pop("is_expert")
            # row_dict['is_expert'] = torch.tensor(1, dtype=torch.bool)
            # row_dict['labels'] = torch.tensor(1)
            row_dict['is_expert'] = torch.tensor(is_expert, dtype=torch.bool)
            row_dict['labels'] = torch.tensor(1) if is_expert else torch.tensor(0)
            # row_dict["old_log_probs"] = torch.zeros(self.max_response_length, dtype=torch.float16)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
