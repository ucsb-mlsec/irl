#!/usr/bin/env python3
"""
Testing for PRIME data
"""

import asyncio
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from concurrent.futures import ProcessPoolExecutor
from functools import partial


LOCAL_MODEL_PATH = "/home/henrygwb/irl/checkpoints/Qwen3-sft/global_step_20" # "Qwen/Qwen2.5-3B-Instruct"
# default_local_dir: "/home/henrygwb/irl/checkpoints"  # Output directory for models and logs
# default_hdfs_dir: "/home/henrygwb/irl/checkpoints"  # HDFS output directory for models and logs
DATA_FILE = "/home/henrygwb/irl/data/validation.parquet"
VAL_BATCH_SIZE = 4  # Validation batch size
MAX_PROMPT_LENGTH = 1000  # Maximum length for prompts
MAX_RESPONSE_LENGTH = 30  # Maximum length for responses
TRUNCATE = "error"  # truncate original input if longer than max_prompt_length; left: prompt_input_ids[:, -max_prompt_length:]; right prompt_input_ids[:, :max_prompt_length]
FILTER_OVERLONG_PROMPTS = False  # Filter out prompts that are too long
NUM_WORKERS = 4  # Number of data loading workers
NUM_PROCESSES = 8  # Number of processes for computating scores
RETURN_RAW_CHAT = False #  whether to include raw data in the loaded data
PROMPT_KEY = "prompt"  # Key containing the conversation prompt
TEMPERATURE = 1  # Sampling temperature for response generation


class MATH_Evaluator(object):

    def __init__(self, model_path, trust_remote_code=False):

        self.local_model_path = copy_to_local(src=model_path, verbose=True)  # download model to the local path
        self.tokenizer = hf_tokenizer(self.local_model_path, trust_remote_code=trust_remote_code)
        self.trust_remote_code = trust_remote_code

    @staticmethod
    async def _single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
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


    async def _parallel_compute_score_async(self,
                                            evaluation_func,
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
                self._single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)
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

    def eval_single(self,
                    data_file, 
                    batch_size,
                    prompt_key,
                    max_prompt_length,
                    max_response_length,
                    return_raw_chat,
                    truncation,
                    num_workers,
                    temperature,
                    filter_overlong_prompts,
                    num_processes):

        """
            Batch: 'input_ids', 'attention_mask', 'position_ids', 'data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids', 'raw_prompt', 'index'])
            # input_ids: [val_batch_size, input_seqlen]
            # 'reward_model' seems how to compte the whether the answer is correct
            output keys ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids'])) # 'input_ids', 'attention_mask', 'position_ids' connection of prompts and responses
            attention_mask: 1 means it is a valid token, 0 means it is a padding token
        """

        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, # model path in huggingface repo
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation='flash_attention_2',
                                                          trust_remote_code=self.trust_remote_code)

        self.val_dataset = RLHFDataset(
            parquet_files=data_file,
            tokenizer=self.tokenizer,
            prompt_key=prompt_key,
            max_prompt_length=max_prompt_length,
            filter_prompts=True,
            return_raw_chat=return_raw_chat,
            truncation=truncation, 
            filter_overlong_prompts=filter_overlong_prompts,
        )
                
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=batch_size,
                                         sampler=SequentialSampler(self.val_dataset),
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=collate_fn) # merge individual element into a batch


        sample_scores = []
        data_source_lst = []

        self.model.eval().cuda()

        for val_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(val_data)
            with torch.no_grad(): 
                input_ids = test_batch.batch['input_ids'].cuda()
                # input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                attention_mask = test_batch.batch['attention_mask'].cuda()

                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_response_length,
                    do_sample=False,  # Greedy for validation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    use_cache=True
                )
                            
            # evaluate using reward_function
            sequences_str = [self.tokenizer.batch_decode(ids[max_prompt_length:], skip_special_tokens=True) for ids in output_ids]
            ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in test_batch]
            data_sources = test_batch.non_tensor_batch['data_source']
            extra_info = test_batch.non_tensor_batch.get('extra_info', None)

            assert len(sequences_str) == len(ground_truth) == len(data_sources)

            self.compute_score = _default_compute_score
            # score: a list of whether each answer is correct or not 0 or 1
            try:
                scores = asyncio.run(
                    self._parallel_compute_score_async(self.compute_score,
                                                       sequences_str,
                                                       ground_truth,
                                                       data_sources,
                                                       extra_info=extra_info,
                                                       num_processes=num_processes))
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


if __name__ == "__main__":
    evaluator = MATH_Evaluator(model_path=LOCAL_MODEL_PATH, trust_remote_code=True)
    evaluator.eval_single(
        data_file=DATA_FILE,
        batch_size=VAL_BATCH_SIZE,
        prompt_key=PROMPT_KEY,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_response_length=MAX_RESPONSE_LENGTH,
        return_raw_chat=RETURN_RAW_CHAT,
        truncation=TRUNCATE,
        num_workers=NUM_WORKERS,
        filter_overlong_prompts=FILTER_OVERLONG_PROMPTS,
        temperature=TEMPERATURE,
        num_processes=NUM_PROCESSES,
    )