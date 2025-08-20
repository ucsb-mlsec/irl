#!/usr/bin/env python3
"""
Testing for PRIME data
"""
import os

import psutil
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["VLLM_ATTENTION_BACKEND"]="FLASH_ATTN"
os.environ["TOKENIZERS_PARALLELISM"]="False"

import asyncio
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, SequentialSampler
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from vllm import LLM, SamplingParams
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

# LOCAL_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct" 
LOCAL_MODEL_PATH = "/home/henrygwb/irl/checkpoints/Qwen2.5-3B-sft/global_step_10" # "Qwen/Qwen2.5-3B-Instruct"
# default_local_dir: "/home/henrygwb/irl/checkpoints"  # Output directory for models and logs
# default_hdfs_dir: "/home/henrygwb/irl/checkpoints"  # HDFS output directory for models and logs
DATA_FILE = "/home/henrygwb/irl/data/validation.parquet"
VAL_BATCH_SIZE = -1  # Validation batch size
MAX_PROMPT_LENGTH = 1500  # Maximum length for prompts
MAX_RESPONSE_LENGTH = 3000  # Maximum length for responses
TRUNCATE = "error"  # truncate original input if longer than max_prompt_length; left: prompt_input_ids[:, -max_prompt_length:]; right prompt_input_ids[:, :max_prompt_length]
FILTER_OVERLONG_PROMPTS = False  # Filter out prompts that are too long
NUM_WORKERS = 4  # Number of data loading workers
NUM_PROCESSES = 64  # Number of processes for computating scores
RETURN_RAW_CHAT = False #  whether to include raw data in the loaded data
PROMPT_KEY = "prompt"  # Key containing the conversation prompt
NUM_GPUS = 2 # number of GPUs for parallel with vllm
TEMPERATURE = 1  # Sampling temperature for response generation
N_SAMPLE = 1 # num_samples to decode
TOP_K = -1 # top_k token
TOP_P = 1.0 # top # tokens based on probablity 
USE_VLLM = True


class MATH_Evaluator(object):

    def __init__(self, model_path, trust_remote_code=False):

        self.local_model_path = copy_to_local(src=model_path, verbose=True)  # download model to the local path
        self.tokenizer = hf_tokenizer(self.local_model_path, trust_remote_code=trust_remote_code)
        self.trust_remote_code = trust_remote_code

    @staticmethod
    async def _single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
        loop = asyncio.get_running_loop() # asyncio event loop - the engine that coordinates all async operations 
        # The loop takes a synchronous function (_default_compute_score) and runs it in a separate process, while making it awaitable in the async context.

        # This is what happens:
        # 1. Main thread: async function calls _single_compute_score
        # 2. Event loop: Takes sync function and submits it to ProcessPoolExecutor  
        # 3. Separate process: Runs the actual scoring function
        # 4. Event loop: Waits for result and returns it to main thread

        try:
            # Ensure process_completion is called properly
            result = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        partial(evaluation_func, task, completion, reference, task_extra_info)  # Ensure synchronous
                    ),
                    timeout=timeout
                    )
            return result
        except asyncio.TimeoutError:
            print(f"Timeout occurred for completion: {completion}")
            return None  # Default value for timed-out rows
        except Exception as e:
            print(f"Error processing completion: {completion[:10]}, Error: {e}")
            return None  # Default value for failed rows

    # kill the process.... after running, seems seeing the num of process increasing
    async def _parallel_compute_score_async(self,
                                            evaluation_func,
                                            completions,
                                            references,
                                            tasks,
                                            extra_info=None,
                                            num_processes=64):
        scores = []
        main_process = psutil.Process(os.getpid())

        # children = len(main_process.children(recursive=True))
        # print(f"Processes before ProcessPoolExecutor: {children}")
    
        with ProcessPoolExecutor(max_workers=num_processes) as executor: # automatically clean up the 64 sub-processes created here
            if extra_info is None:
                extra_info = [None] * len(tasks)
            # Create tasks for all rows
            tasks_async = [
                self._single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)
                for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
            ]
            # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
            results = await asyncio.gather(*tasks_async, return_exceptions=False)

            # children = len(main_process.children(recursive=True))
            # print(f"Processes before ProcessPoolExecutor: {children}")

            # for _, proc in executor._processes.items():
            #     proc.kill()

        # children = len(main_process.children(recursive=True))
        # print(f"Processes before ProcessPoolExecutor: {children}")

        # Process results
        for result in results:
            if isinstance(result, Exception) or result is None:
                # Handle failed or timed-out tasks
                scores.append(0.0)
            elif isinstance(result, (int, float, bool)):
                scores.append(float(result))
            else:
                scores.append(float(result[0]))
        return scores

    def eval(self,
             data_file, 
             batch_size,
             prompt_key,
             max_prompt_length,
             max_response_length,
             return_raw_chat,
             truncation,
             num_workers,
             filter_overlong_prompts,
             num_processes,
             num_gpus=1,
             temperature=1,
             n_sample=1,
             top_k=-1,
             top_p=1.0,
             use_vllm=True):

        """
            Batch: 'input_ids', 'attention_mask', 'position_ids', 'data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids', 'raw_prompt', 'index'])
            # input_ids: [val_batch_size, input_seqlen]
            # 'reward_model' seems how to compte the whether the answer is correct
            output keys ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids'])) # 'input_ids', 'attention_mask', 'position_ids' connection of prompts and responses
            attention_mask: 1 means it is a valid token, 0 means it is a padding token
        """

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
                
        # vllm does not need distributed sampler
        if VAL_BATCH_SIZE == -1:
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                            batch_size=len(self.val_dataset),
                                            sampler=SequentialSampler(self.val_dataset),
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=True,
                                            collate_fn=collate_fn) # merge individual element into a batch
        else:
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                            batch_size=VAL_BATCH_SIZE,
                                            sampler=SequentialSampler(self.val_dataset),
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=True,
                                            collate_fn=collate_fn) # merge individual element into a batch

        if use_vllm: 
            # check attention_backend
            self.model = LLM(
                model=self.local_model_path,
                tokenizer=self.local_model_path,
                trust_remote_code=self.trust_remote_code,
                tensor_parallel_size=num_gpus,  # Num of GPUs for parallel decoding
                dtype='bfloat16')

            sampling_params = SamplingParams(
                n=n_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_response_length
            )   # typically no need to specify padding tokens
        else:  
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, # model path in huggingface repo
                                                            torch_dtype=torch.bfloat16,
                                                            attn_implementation='flash_attention_2',
                                                            trust_remote_code=self.trust_remote_code)
            self.model.eval().cuda()

        sample_scores = []
        data_source_lst = []

        batch = 0
        for val_data in self.val_dataloader:
            print(f"Processing batch {batch} out of {len(self.val_dataloader)}")
            test_batch = DataProto.from_single_dict(val_data)
            with torch.no_grad(): 
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                attention_mask = test_batch.batch['attention_mask']

                if use_vllm:
                    # Generate with vLLM - much faster batch processing!
                    outputs = self.model.generate(input_texts, sampling_params=sampling_params)
                    # input_ids = [ids for ids in input_dis]
                    # outputs = self.model.generate(prompts=input_ids, sampling_params=sampling_params)
                    sequences_str = [output.outputs[0].text for output in outputs]

                else:
                    output_ids = self.model.generate(
                        input_ids=input_ids.cuda(),
                        attention_mask=attention_mask.cuda(),
                        max_new_tokens=max_response_length,
                        do_sample=False,  # Greedy for validation
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        temperature=temperature,
                        use_cache=True
                    )
                    sequences_str = [self.tokenizer.batch_decode(ids[max_prompt_length:], skip_special_tokens=True) for ids in output_ids]
                            
            # evaluate using reward_function
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
            batch += 1
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
    metric_dict = evaluator.eval(
        data_file=DATA_FILE,
        batch_size=VAL_BATCH_SIZE,
        prompt_key=PROMPT_KEY,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_response_length=MAX_RESPONSE_LENGTH,
        return_raw_chat=RETURN_RAW_CHAT,
        truncation=TRUNCATE,
        num_workers=NUM_WORKERS,
        filter_overlong_prompts=FILTER_OVERLONG_PROMPTS,
        num_processes=NUM_PROCESSES,
        num_gpus=NUM_GPUS,
        temperature=TEMPERATURE,
        n_sample=N_SAMPLE,
        top_k=TOP_K,
        top_p=TOP_P,
        use_vllm=USE_VLLM
    )
    print(metric_dict)
