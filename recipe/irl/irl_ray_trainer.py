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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
import uuid
from copy import deepcopy
from pprint import pprint
from collections import defaultdict
import torch.distributed as dist

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.ray_trainer import Role, WorkerType, ResourcePoolManager, reduce_metrics
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.profiler.performance import simple_timer
from verl.trainer.ppo.core_algos import agg_loss

from .dataset import IRLDataset, collate_fn
from . import irl_core_algos
from .utils import cal_outcome_reward


def compute_advantage(data: DataProto, adv_estimator, config):
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    advantages, returns = irl_core_algos.compute_advantage_return(data, response_mask, config.actor_rollout_ref.rollout.n, config)
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    return data


def compute_data_metrics(batch):
    max_response_length = batch.batch['responses'].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    metrics = {
        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


class RayIRLTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 device_name="cuda"):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=device_name,
        )
        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        else:
            self.use_critic = False

    def _validate_config(self):
        super()._validate_config()
        # TODO: Additional config checks can be added here
        config = self.config

    def _create_dataloader(self, *args, **kwargs):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # building the policy train dataset
        self.policy_train_dataset = IRLDataset(
            parquet_files=self.config.data.policy_train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False)
        )
        
        if self.config.data.shuffle:
            print("Using random sampler for policy train dataset")
            train_dataloader_generator = torch.Generator()
            seed = self.config.data.get('seed')
            if seed is not None:
                train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=self.policy_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.policy_train_dataset)

        self.policy_train_dataloader = DataLoader(
            dataset=self.policy_train_dataset, 
            batch_size=int(self.config.data.train_batch_size), 
            drop_last=True, 
            collate_fn=collate_fn, 
            sampler=sampler
        )

        # building the expert dataset
        self.expert_demo_dataset = IRLDataset(
            parquet_files=self.config.data.expert_demo_files,
            tokenizer=self.tokenizer, 
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='right',
            filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False)
        )

        if self.config.data.shuffle:
            print("Using random sampler for expert demo dataset")
            expert_demo_dataloader_generator = torch.Generator()
            seed = self.config.data.get('seed')
            if seed is not None:
                train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=self.expert_demo_dataset, generator=expert_demo_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.expert_demo_dataset)

        self.expert_demo_dataloader = DataLoader(
            dataset=self.expert_demo_dataset, 
            batch_size=int(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n), 
            drop_last=True, 
            collate_fn=collate_fn
        )

        # building the val dataset
        self.policy_val_dataset = RLHFDataset(
            data_files=self.config.data.policy_val_files,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )
        
        self.val_dataloader = DataLoader(
            dataset=self.policy_val_dataset,
            batch_size=len(self.policy_val_dataset),
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        print(f'Size of policy train dataloader: {len(self.policy_train_dataloader)}')
        print(f'Size of expert dataloader: {len(self.expert_demo_dataloader)}')
        print(f'Size of policy val dataloader: {len(self.val_dataloader)}')
        # print("="*50)
        # print(self.policy_train_dataset[0])
        # print("="*50)
        # print(self.policy_val_dataset[0])
        # print("="*50)
        # print(self.expert_demo_dataset[0])

        # exit(0)

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.policy_train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _save_checkpoint(self, is_best=False):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        if is_best:
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, 'best_model')
        else:
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f'global_step_{self.global_steps}')

        if not os.path.exists(local_global_step_folder):
            os.makedirs(local_global_step_folder, exist_ok=True)

        print(f'local_global_step_folder: {local_global_step_folder}')

        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps)
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps)

        if self.use_rm:
            reward_local_path = os.path.join(local_global_step_folder, 'reward')
            reward_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'reward')
            self.rm_wg.save_checkpoint(reward_local_path,
                                       reward_remote_path,
                                       self.global_steps)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.policy_train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        
        print(f'Load from checkpoint folder: {global_step_folder}')
        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, "critic")
        reward_path = os.path.join(global_step_folder, 'reward')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
         # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        # load rm
        if self.use_rm:
            self.rm_wg.load_checkpoint(reward_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            scores = self.val_reward_fn(test_batch, return_dict=False)
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

    def fit(self):
        """
        Training loop with alternating reward and policy training by epoch:
        For each epoch:
        1. Train the reward model using all batches
        2. Train the policy model using all batches with the updated reward model
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()
        val_metrics = self._validate()
        logger.log(data=val_metrics, step=self.global_steps)

        # We start from step 1
        self.global_steps += 1
        best_val_acc = 0.0
                
        for epoch in range(self.config.trainer.total_epochs):
            print(f"Epoch {epoch+1}/{self.config.trainer.total_epochs}")
            
            # Phase 1: Generate policy samples to combine with expert data
            print("=" * 50)
            print(f"Phase 1: Generating policy samples for reward model training")
            print("=" * 50)

            for policy_batch_dict, expert_batch_dict in zip(self.policy_train_dataloader, self.expert_demo_dataloader):
                metrics = {}
                timing_raw = {}

                with simple_timer('policy_model_rollout', timing_raw):
                    # generate policy samples
                    policy_batch = DataProto.from_single_dict(policy_batch_dict)
                    # pop those keys for generation
                    gen_batch = policy_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)

                    policy_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(policy_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    policy_batch = policy_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    policy_batch = policy_batch.union(gen_batch_output)

                    policy_log_prob = self.actor_rollout_wg.compute_log_prob(policy_batch)
                    policy_batch = policy_batch.union(policy_log_prob)

                    scores = self.reward_fn.verify(policy_batch)
                    expert_flags = torch.tensor(scores) > 0.99
                    policy_batch.batch['labels'] = torch.tensor(scores)
                    policy_batch.batch['is_expert'] = expert_flags
                

                filter_reorder_index = self.filter_and_downsample(scores, policy_batch)
                policy_batch.reorder(filter_reorder_index[:int(len(policy_batch) // 2)])
                
                # load expert samples
                expert_batch = DataProto.from_single_dict(expert_batch_dict)
                expert_batch.reorder(filter_reorder_index[:int(len(expert_batch) // 2)])


                expert_log_prob = self.actor_rollout_wg.compute_log_prob(expert_batch)
                expert_batch = expert_batch.union(expert_log_prob)

                policy_batch.non_tensor_batch = {}
                expert_batch.non_tensor_batch = {}
                policy_batch.pop(batch_keys=['prompts'])


                policy_correct = torch.sum(policy_batch.batch['is_expert']).item()
                policy_count = len(policy_batch.batch['is_expert'])
                policy_accuracy = policy_correct / policy_count

                expert_correct = torch.sum(expert_batch.batch['is_expert']).item()
                expert_count = len(expert_batch.batch['is_expert'])
                expert_accuracy = expert_correct / expert_count

                logger.log(data={'train_accuracy': {
                    '/before_update_policy_accuracy': policy_accuracy,
                    '/before_update_expert_accuracy': expert_accuracy,
                }}, step=self.global_steps)


                n_samples = policy_count

                # Generate policy indices (0, 1, 2, ..., n_samples-1)
                policy_indices = torch.arange(n_samples)

                # Generate expert indices (n_samples, n_samples+1, ..., 2*n_samples-1)
                expert_indices = torch.arange(n_samples, 2*n_samples)

                reorder_index = torch.zeros(2*n_samples, dtype=torch.long)
                reorder_index[0::2] = policy_indices
                reorder_index[1::2] = expert_indices

                # reorder the policy and expert batches
                batch = DataProto.concat([policy_batch, expert_batch])
                batch.reorder(reorder_index)

                print("batch shape: ", batch.batch["responses"].shape)
                
                with simple_timer('train_reward_model', timing_raw):
                    self.rm_wg.update_rm(batch)
                
                policy_batch.meta_info['global_token_num'] = torch.sum(policy_batch.batch['attention_mask'], dim=-1).tolist()

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data={'reward_model_training': metrics}, step=self.global_steps)

                resp_metric = compute_data_metrics(batch=batch)
                logger.log(data={"Response info": resp_metric}, step=self.global_steps)

                # This was your interleaved index (e.g., [0, 4, 1, 5, 2, 6, 3, 7] for n_samples=4)
                reorder_index = torch.zeros(2*n_samples, dtype=torch.long)
                reorder_index[0::2] = torch.arange(n_samples)  # Policy indices in even positions
                reorder_index[1::2] = torch.arange(n_samples, 2*n_samples)  # Expert indices in odd positions

                # To get the inverse index, we need to figure out where each original index ends up
                inverse_reorder_index = torch.empty_like(reorder_index)
                inverse_reorder_index[reorder_index] = torch.arange(len(reorder_index))

                # reorder the batch to match the original order, we can only use rollouts generated by the policy model to train
                batch.reorder(inverse_reorder_index)

                print("=" * 50)
                print("Phase 2: Training policy model based on the updated reward model")
                print("=" * 50)

                # Compute reward scores after policy update and sync them across processes
                rm_scores = self.rm_wg.compute_rm_score(batch).batch['rm_scores']

                reward_metric = self.reward_model_metrics(batch, rm_scores)
                logger.log(data=reward_metric, step=self.global_steps)

                policy_rollout_rm_scores = rm_scores[:len(rm_scores)//2]
                policy_rollout_rm_scores = DataProto.from_dict(tensors={'rm_scores': policy_rollout_rm_scores})
                policy_batch = policy_batch.union(policy_rollout_rm_scores)

                policy_batch.batch['response_mask'] = compute_response_mask(policy_batch)

                # log entropy information
                entropys = policy_batch.batch["entropys"]
                response_masks = policy_batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                policy_batch.batch.pop("entropys")


                if self.use_reference_policy:
                    # compute reference log_prob
                    with simple_timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(policy_batch)
                        policy_batch = policy_batch.union(ref_log_prob)
                
                # compute values
                if self.use_critic:
                    with simple_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(policy_batch)
                        policy_batch = policy_batch.union(values)

                with simple_timer('update_policy_model', timing_raw):
                    # Skip generation since we already have the samples
                    # We already have the complete policy batch with generated responses
                    policy_batch = compute_advantage(policy_batch, adv_estimator=self.config.algorithm.adv_estimator, config=self.config)
                    advantages = policy_batch.batch['advantages']

                    min_adv = torch.min(advantages)
                    max_adv = torch.max(advantages)
                    mean_adv = torch.mean(advantages)
                    print(f"Advantages: min={min_adv:.4f}, max={max_adv:.4f}, mean={mean_adv:.4f}")

                    logger.log(data={'policy_training': {
                        'advantages/min': min_adv.item(),
                        'advantages/max': max_adv.item(),
                        'advantages/mean': mean_adv.item(),
                    }}, step=self.global_steps)
                
                # update critic
                if self.use_critic:
                    with simple_timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(policy_batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with simple_timer('update_actor', timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(policy_batch)                        
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                
                # Collect metrics
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data={'policy_training': metrics}, step=self.global_steps)
                # Update global step
                self.global_steps += 1
                
                # Validation and checkpointing logic
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    logger.log(data=val_metrics, step=self.global_steps)
                
                    cur_val_acc = val_metrics.get('val_acc/overall', 0.0)
                    if cur_val_acc > best_val_acc:  
                        best_val_acc = cur_val_acc
                        print(f"Best validation accuracy so far: {best_val_acc}")
                        # Save the best model
                        self._save_checkpoint(is_best=True)

                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()
                
                if self.global_steps >= self.total_training_steps:
                    return
        # Final checkpoint at the end of training
        self._save_checkpoint()

    def reward_model_metrics(self, batch, rm_scores):
        # Get policy and expert masks
        batch_size = int(rm_scores.shape[0] // 2)

        expert_mask = batch.batch['is_expert']         
        policy_mask = torch.logical_not(expert_mask)
        
        # Calculate effective response length
        all_response_length = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[-1]:].sum(-1)

        # Create a mask tensor of the same shape as policy_rm_scores
        response_mask = torch.zeros_like(rm_scores)

        # Fill the mask with 1s for valid response tokens based on all_response_length
        for i in range(2 * batch_size):
            response_len = all_response_length[i]
            response_mask[i, :response_len] = 1.0

        scores = torch.sum(rm_scores * response_mask, dim=-1)

        expert_score = torch.mean(scores[expert_mask])
        policy_score = torch.mean(scores[policy_mask])
        
        normalized_score = scores / all_response_length
        normalized_expert_score = torch.mean(normalized_score[expert_mask])
        normalized_policy_score = torch.mean(normalized_score[policy_mask])

        return {
                'reward_model': {
                    'expert_score': expert_score, 
                    'policy_score': policy_score,
                    "diff_score": policy_score - expert_score, 
                    'normalized_expert_score': normalized_expert_score,
                    'normalized_policy_score': normalized_policy_score,
                    "normalized_diff_score": normalized_policy_score - normalized_expert_score,
                }
        }

    def filter_and_downsample(self, scores, batch: DataProto):
        """
        downsample the batch according to oversample_factor
        samples passing the filters will be prioritized
        """
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        reward_matrix = torch.tensor(scores).reshape(-1, n_samples)

        filter_mask = torch.ones((reward_matrix.shape[0]), dtype=torch.bool)

        if self.config.data.filter_accuracy:
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            filter_mask[(acc_tensor > self.config.data.accuracy_upper_bound) |
                        (acc_tensor < self.config.data.accuracy_lower_bound)] = False

        if self.config.data.filter_truncate:
            length_matrix = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[-1]:].sum(dim=-1).reshape(
                -1, n_samples)
            length_tensor = torch.max(length_matrix, dim=-1)[0]
            filter_mask[length_tensor >= self.config.data.max_response_length - 1] = False

        reorder_index = torch.argsort(filter_mask, descending=True)
        reorder_index = (reorder_index.unsqueeze(-1) * n_samples + torch.arange(0, n_samples).unsqueeze(0)).view(-1)

        return reorder_index