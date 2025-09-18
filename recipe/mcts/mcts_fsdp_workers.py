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
import copy
import logging
import os
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.reward_model.base import BasePPORewardModel
from verl.utils.model import LambdaLayer, print_model_size, squeeze
from verl.utils.torch_dtypes import PrecisionType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch import optim
from transformers import AutoConfig, AutoModelForCausalLM


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class MCTSRewardModelWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
            assert self.config.mini_batch_size % self.config.micro_batch_size_per_gpu == 0

    def _build_reward_model_and_optimizer(self, config):
        # the following line is necessary

        local_path = copy_local_path_from_hdfs(config.model.path)

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))
        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)


        trust_remote_code = False
        reward_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        init_context = get_init_weight_context_manager(use_meta_tensor=not reward_model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            reward_module  = AutoModelForCausalLM.from_pretrained(local_path,
                                                            torch_dtype=torch_dtype,
                                                            config=reward_model_config,
                                                            attn_implementation='flash_attention_2',
                                                            trust_remote_code=trust_remote_code)
            
            if config.model.get('use_remove_padding', False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            if config.model.get('enable_gradient_checkpointing', False):
                reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        
        if self.rank == 0:
            print_model_size(reward_module)

        self.reward_model_config = reward_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before reward model FSDP', logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(reward_module,
                             param_init_fn=init_fn,
                             use_orig_params=True,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)

        log_gpu_memory_usage('After reward FSDP', logger=None)
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from .mcts_dp_rm import DataParallelMCTSRewardModel
        self.reward_module = self._build_reward_model_and_optimizer(config=self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        
        local_path = copy_local_path_from_hdfs(self.config.model.path)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        self.rm = DataParallelMCTSRewardModel(config=self.config, tokenizer=tokenizer, reward_module=self.reward_module)

        self.flops_counter = FlopsCounter(self.reward_model_config)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        micro_batch_size = self.config.micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            rm_scores = self.rm.compute_rm_score(data=data)
            output = DataProto.from_dict(tensors={'rm_scores': rm_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        
        return output
