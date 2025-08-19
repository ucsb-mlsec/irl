#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT).
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.distributed import initialize_global_process_group
from .sft_trainer import FSDP_SFT_Trainer

# os.environ.update({
#     "CUDA_VISIBLE_DEVICES": "7",
#     "VLLM_ATTENTION_BACKEND": "XFORMERS", 
#     "WANDB_API_KEY": "0da9605b7e93d6ada3221d8f5aa9df8c96a5406e",
#     "TOKENIZERS_PARALLELISM": "false",
#     "NCCL_DEBUG": "WARN",
#     "RANK": "0",
#     "WORLD_SIZE": "1", 
#     "LOCAL_RANK": "0",
#     "MASTER_ADDR": "localhost",
#     "MASTER_PORT": "12355"
# })

@hydra.main(config_path="config", config_name="sft_config", version_base=None) # it will read the default config from the sft_config.yaml file
def main(config: DictConfig) -> None:

    OmegaConf.resolve(config)  

    # No need to define worker and roles as only one model is involved

    # Initialize the distrubted training environment
    # local_rank: The GPU ID on the current machine/node
    # rank: The global rank across all machines/nodes
    # world_size: Total number of GPUs across all machines/nodes
    local_rank, rank, world_size = initialize_global_process_group()

    # Create a device_mesh for model sharding based on the mash_shape 
    # Here, we create a 1d mesh for linear arrangement of GPUs [GPU0, GPU1]    # FSDP will use this mesh to model sharding
    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))

    # Set up the device mesh for data paralelism and long sequence
    # config.ulysses_sequence_parallel_size: the num of GPUs to handle one sequence
    # dp_size: number of parallel GPUs
    # create a 2d mesh for data parallelism and sequence parallelism
    # dp: data parallelism; sp: sequence parallelism
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDP_SFT_Trainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()

if __name__ == "__main__":
    main()

