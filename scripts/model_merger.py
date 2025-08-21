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

from typing import List, Tuple, Dict
import re
import os
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor
from torch.distributed._tensor import DTensor, Shard, Placement
from safetensors.torch import load_file
from verl.utils import hf_tokenizer

# Running script:
#   python scripts/model_merger.py     --backend fsdp     --hf_model_path Qwen/Qwen2.5-3B-Instruct     --local_dir /scr/xian/rloo/best_model/actor     --target_dir /scr/xian/rloo/final_model
parser = argparse.ArgumentParser()
parser.add_argument('--backend', type = str, required=True, help="The backend of the model")
parser.add_argument('--tie-word-embedding', action='store_true', help="Whether to tie word embedding weights")
parser.add_argument('--is-value-model', action='store_true', help="Whether the model loaded as value model")
parser.add_argument('--hf_model_path', type = str, required=True, help="The path for the huggingface model")
parser.add_argument('--local_dir', type = str, required=True, help="The path for your saved model. For megatron, point to the base dir of model, rng, optimizer checkpoints, commonly be `config.default_local_dir/global_step_\{global_step\}`.")
parser.add_argument('--target_dir', required=False, default="tmp", type = str, help="The path for the target model")
parser.add_argument("--hf_upload_path", default=False, type = str, help="The path of the huggingface repo to upload")
parser.add_argument("--test", action="store_true", help="test correctness of hf_model")
parser.add_argument("--test_hf_dir", type = str, required=False, help="test correctness of hf_model, , with hf_model in checkpoint.contents")
args = parser.parse_args()
os.makedirs(args.target_dir, exist_ok=True)
if args.test:
    assert args.test_hf_dir is not None, f'You must run verl save checkpoint first, with hf_model in checkpoint.contents, and provide the directory here'

def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(hf_path):
    # Push to hugging face
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
    api.upload_folder(
        folder_path=hf_path,
        repo_id=args.hf_upload_path,
        repo_type="model"
    )
    

def convert_fsdp_checkpoints_to_hfmodels():
    local_dir = args.local_dir

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)  
            break  
    assert world_size, "No model file with the proper format"
        
    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'), map_location='cpu')
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor)
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    assert mesh_dim_names in (
        ('fsdp',),
    ), f'Unsupported mesh_dim_names {mesh_dim_names}'

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:
                print("-"*30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == 'dp':
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print('Writing to local disk')
    if args.target_dir is None:
        hf_path = os.path.join(local_dir, 'huggingface')
    else:
        hf_path = args.target_dir
    config = AutoConfig.from_pretrained(args.hf_model_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)
    tokenizer = hf_tokenizer(args.hf_model_path)
    tokenizer.save_pretrained(hf_path)



    del state_dict
    del model
    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path)


def get_tp_pp_rank_from_sharded_dir(sharded_dir):
    match = re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir)
    tp_rank = int(match.group(1))
    pp_rank = int(match.group(2))
    return tp_rank, pp_rank

def check_megatron_checkpoint_path(model_path):
    sharded_dirs = sorted(os.listdir(model_path))
    tp_size = 0
    pp_size = 0
    for sharded_dir in sharded_dirs:
        match = re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir)
        assert match, f"Invalid sharded dir {sharded_dir}"
        assert f"model.pt" in os.listdir(os.path.join(model_path, sharded_dir)), f"model.pt not found in {sharded_dir}"
        tp_rank = int(match.group(1))
        pp_rank = int(match.group(2))
        if tp_size < tp_rank + 1:
            tp_size = tp_rank + 1
        if pp_size < pp_rank + 1:
            pp_size = pp_rank + 1
    return sharded_dirs, tp_size, pp_size
    

def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if "layers" in megatron_name:  # deal with decoder layers
            megatron_name = megatron_name.replace("decoder", "model")
            megatron_name_list = megatron_name.split(".")
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name
            
if __name__ == '__main__':
    if args.backend == "fsdp":
        convert_fsdp_checkpoints_to_hfmodels()
    else:
        raise NotImplementedError(f"{args.backend} not supported")
