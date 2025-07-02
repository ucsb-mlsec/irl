import pandas as pd
from transformers import AutoTokenizer

# List of files to process
# files = [
#     '/home/kaijie/irl/data/claude_s1k_demo.parquet',
#     '/home/kaijie/irl/data/s1k_train.parquet',
# ]

# # Process each file
# for file_path in files:
#     # Read the parquet file
#     df = pd.read_parquet(file_path)
    
#     # Rename the 'chat' column to 'prompt'
#     df = df.rename(columns={'chat': 'prompt'})
    
#     # Create a new file name with '_updated' suffix
#     new_file_path = file_path.replace('.parquet', '_updated.parquet')
    
#     # Write the modified dataframe back to a new parquet file
#     df.to_parquet(new_file_path)
    
#     print(f"Processed {file_path} -> {new_file_path}")
#     print(f"Columns in new file: {df.columns.tolist()}")


# file = '/home/kaijie/irl/data/claude_s1k_demo.parquet'

# df = pd.read_parquet(file)

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# # print 0-th row
# all_lengths = []
# for i in range(len(df)):
#     response = df.iloc[i]['response']
#     response_ids = tokenizer(response, return_tensors="pt").input_ids[0]
#     length = len(response_ids)
#     all_lengths.append(length)

# print(max(all_lengths))
# print(min(all_lengths))
# print(sum(all_lengths) / len(all_lengths))

# import torch
# from transformers import AutoConfig
# from recipe.irl.rm import RewardModule
# from recipe.irl.dataset import IRLDataset

# local_path = "Qwen/Qwen2.5-0.5B-Instruct"
# trust_remote_code = False
# reward_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
# reward_model_config.num_labels = 1

# reward_module = RewardModule(
#     base_model=local_path,
#     torch_dtype="auto",
#     config=reward_model_config,
#     trust_remote_code=trust_remote_code,
# )

# dataset = IRLDataset(parquet_files="/home/kaijie/irl/data/claude_s1k_demo.parquet", tokenizer=reward_module.tokenizer, prompt_key="prompt", max_prompt_length=1024, max_response_length=16000, filter_prompts=True, return_raw_chat=False, truncation='error', filter_overlong_prompts=False)

# for d in dataset:
#     input_ids = d['input_ids'].to(reward_module.device).unsqueeze(0)
#     attention_mask = d['attention_mask'].to(reward_module.device).unsqueeze(0)
#     response_ids = d['responses'].to(reward_module.device).unsqueeze(0)
#     position_ids = d['position_ids'].to(reward_module.device).unsqueeze(0)
    
#     response = reward_module.tokenizer.decode(response_ids[0, :10], skip_special_tokens=False)
    
#     num_actions = response_ids.shape[-1]
#     max_positions = attention_mask[:, -num_actions:].sum(-1)

#     print(num_actions)
#     print(max_positions)

#     rm_output_logits = reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)

#     mask = torch.ones_like(rm_output_logits[:, -num_actions:])
#     for i in range(input_ids.shape[0]):
#         mask[max_positions[i]:] = 0
        
#     # Apply mask via multiplication rather than in-place assignment
#     q = rm_output_logits[:, -num_actions:]
#     partial_input_ids = input_ids[:, -num_actions:]
#     text = reward_module.tokenizer.decode(partial_input_ids[0, :10], skip_special_tokens=False)
#     print(q.shape)
#     print("="*50)
#     print(q[0, :10])
#     print(q[0, -10:])
#     q = q * mask
#     print(q[0, :10])
#     print(q[0, -10:])
#     print("="*50)
#     print(response)
#     print(text)

#     exit()



# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import torch.nn.functional as F
# from transformers import PreTrainedTokenizer


# def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
#     """
#     pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
#     input shape: [bs, seq_length]
#     output shape: [bs, max_seq_length]
#     (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
#     """
#     if tensors.shape[-1] >= max_seq_len:
#         return tensors
#     pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
#     return F.pad(tensors, pad_tuple, 'constant', pad_token_id)

# def tokenize_and_postprocess_whole_chat(prompt: str,
#                                         overall_chat: str,
#                                         tokenizer: PreTrainedTokenizer,
#                                         max_prompt_length: int,
#                                         max_response_length: int,
#                                         pad_token_id: int,
#                                         left_pad=True,
#                                         truncation='error'):
    
#     assert truncation in ['left', 'right', 'error']

#     prompt_input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
#     prompt_attention_mask = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['attention_mask']
    
#     print(prompt_input_ids)
#     print(prompt_attention_mask)

#     print("prompt")
#     prompt_text = tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=False)
#     print(prompt_text)

#     response_input_ids = tokenizer(overall_chat, return_tensors='pt', add_special_tokens=False)['input_ids']
#     response_attention_mask = tokenizer(overall_chat, return_tensors='pt', add_special_tokens=False)['attention_mask']

#     print("overall chat")
#     overall_chat_text = tokenizer.batch_decode(response_input_ids, skip_special_tokens=False)
#     print(overall_chat_text)

#     response_input_ids = response_input_ids[:, prompt_input_ids.shape[-1]:]
#     response_attention_mask = response_attention_mask[:, prompt_attention_mask.shape[-1]:]

#     print("after slicing, response")
#     response_text = tokenizer.batch_decode(response_input_ids, skip_special_tokens=False)
#     print(response_text)


#     prompt_sequence_length = prompt_input_ids.shape[-1]
#     response_sequence_length = response_input_ids.shape[-1]
    
#     if prompt_sequence_length < max_prompt_length:
#         prompt_input_ids = pad_sequence_to_length(prompt_input_ids,
#                                                  max_seq_len=max_prompt_length,
#                                                  pad_token_id=pad_token_id,
#                                                  left_pad=True)
#         prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask,
#                                                        max_seq_len=max_prompt_length,
#                                                        pad_token_id=0,
#                                                        left_pad=True)
#     elif prompt_sequence_length > max_prompt_length:
#         if truncation == 'left':
#             # actually, left truncation may not be reasonable
#             prompt_input_ids = prompt_input_ids[:, -max_prompt_length:]
#             prompt_attention_mask = prompt_attention_mask[:, -max_prompt_length:]
#         elif truncation == 'right':
#             prompt_input_ids = prompt_input_ids[:, :max_prompt_length]
#             prompt_attention_mask = prompt_attention_mask[:, :max_prompt_length]
#         elif truncation == 'error':
#             raise NotImplementedError(f'{prompt_sequence_length=} is larger than {max_prompt_length=}')
#         else:
#             raise NotImplementedError(f'Unknown truncation method {truncation}')

#     if response_sequence_length < max_response_length:
#         response_input_ids = pad_sequence_to_length(response_input_ids,
#                                                     max_seq_len=max_response_length,
#                                                     pad_token_id=pad_token_id,
#                                                     left_pad=False)
#         response_attention_mask = pad_sequence_to_length(response_attention_mask,
#                                                          max_seq_len=max_response_length,
#                                                          pad_token_id=0,
#                                                          left_pad=False)
#     elif response_sequence_length > max_response_length:
#         if truncation == 'left':
#             # actually, left truncation may not be reasonable
#             response_input_ids = response_input_ids[:, -max_response_length:]
#             response_attention_mask = response_attention_mask[:, -max_response_length:]
#         elif truncation == 'right':
#             response_input_ids = response_input_ids[:, :max_response_length]
#             response_attention_mask = response_attention_mask[:, :max_response_length]
#         elif truncation == 'error':
#             raise NotImplementedError(f'{response_sequence_length=} is larger than {max_response_length=}')
#         else:
#             raise NotImplementedError(f'Unknown truncation method {truncation}')
    
#     # concatenate prompt and response
#     input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=-1)
#     attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

#     return input_ids, attention_mask




# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)

# chat = [
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm fine, thank you! How can I assist you today?"}
# ]

# text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

# # print("Tokenized chat:")
# # print(text)

# prompt_only_chat = [
#     {"role": "user", "content": "Hello, how are you?"}
# ]

# prompt_only_text = tokenizer.apply_chat_template(prompt_only_chat, add_generation_prompt=True, tokenize=False)

# # print("Tokenized prompt only:")
# # print(prompt_only_text)

# # response = "I'm fine, thank you! How can I assist you today?"
# input_ids, attention_mask = tokenize_and_postprocess_whole_chat(
#     prompt=prompt_only_text,
#     overall_chat=text,
#     tokenizer=tokenizer,
#     max_prompt_length=128,
#     max_response_length=64,
#     pad_token_id=tokenizer.pad_token_id,
#     left_pad=True,
#     truncation='error'
# )

# print("Input IDs:")
# print(input_ids)
# print("Attention Mask:")
# print(attention_mask)

import json
import os
from datasets import load_dataset, Dataset

json_files = [
    "/scratch/yuzhou/irl/openai_expert_demo_prime_0_400.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_400_800.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_800_1200.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_1200_1600.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_1600_2000.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_2000_2300.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_2300_2600.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_2600_2900.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_2900_3200.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_3200_3500.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_3500_3800.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_3800_4100.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_4100_4400.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_4400_4700.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_4700_5000.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_5000_5500.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_5500_6000.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_6000_6500.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_6500_7000.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_7000_7500.json",
    "/scratch/yuzhou/irl/openai_expert_demo_prime_7500_8000.json",
]

combined_data = []

for f in json_files:
    with open(f, "r") as file:
        data = json.load(file)
    
    combined_data.extend(data)
    print(f"Loaded {len(data)} records from {f}")

raw_dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data")["train"]

raw_dataset = raw_dataset.select(range(0, len(combined_data)))

all_data = []
all_corrected_data = []
partial_corrected_data = []
partial_corrected_data_aug_2 = []
for idx, d in enumerate(combined_data):

    question = d['question']
    raw_d = raw_dataset[idx]
    if raw_d['prompt'][-1]['content'] != question:
        print(f"Question mismatch at index {idx}: {question} != {raw_d['prompt'][-1]['content']}")
        exit()

    data_source = raw_d['data_source']

    responses = d['responses']
    d["data_source"] = data_source

    if len(responses) < 4:
        print(f"Not enough responses at index {idx}: {len(responses)}")
        continue

    correct_responses = []
    incorrect_responses = []

    for response in responses:
        if response['correct']:
            correct_responses.append(response)
        else:
            incorrect_responses.append(response)
    
    if len(correct_responses) >= 4:
        all_corrected_d = {}
        all_corrected_d['question'] = d['question']
        all_corrected_d['answer'] = d['answer']
        all_corrected_d['responses'] = correct_responses[:4]
        all_corrected_d['data_source'] = data_source
        all_corrected_data.append(all_corrected_d)
    
    if len(correct_responses) > 0:
        partial_corrected_d = {}
        partial_corrected_d['question'] = d['question']
        partial_corrected_d['answer'] = d['answer']
        partial_corrected_d['data_source'] = data_source

        if len(correct_responses) >= 4:
            partial_corrected_d['responses'] = correct_responses[:4]
        else:
            partial_corrected_d['responses'] = correct_responses + incorrect_responses[:4-(len(correct_responses))]

        partial_corrected_data.append(partial_corrected_d)

        if len(correct_responses) >= 2:
            
            partial_corrected_data_aug_2.append(partial_corrected_d)

    if len(correct_responses) >= 4:
        all_d = {}
        all_d['question'] = d['question']
        all_d['answer'] = d['answer']
        all_d['responses'] = correct_responses[:4]
        all_d['data_source'] = data_source
        all_data.append(all_d)
    else:
        all_d = {}
        all_d['question'] = d['question']
        all_d['answer'] = d['answer']
        all_d['responses'] = correct_responses + incorrect_responses[:4-(len(correct_responses))]
        all_d['data_source'] = data_source
        all_data.append(all_d)


print(f"Total records: {len(all_data)}")
print(f"Total records with all correct responses: {len(all_corrected_data)}")
print(f"Total records with partial correct responses: {len(partial_corrected_data)}")
print(f"Total records with partial correct responses (augmented): {len(partial_corrected_data_aug_2)}")


def merge(data, name):
    prompt_only_data = []
    expert_demo_data = []

    for d in data:

        data_source = d["data_source"]
        prompt = d["question"]
        answer = d["answer"]

        prompt_only = [
            {"role": "user", "content": prompt}
        ]

        prompt_only_data.append({"prompt": prompt_only, "answer": answer, "data_source": data_source})

        for raw_response in d["responses"]:
            

            response = raw_response["thinking"] + raw_response["text"]

            prompt_response = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]


            expert_demo_data.append(
                {
                    "prompt": prompt_only, 
                    "complete_chat": prompt_response, 
                    "response": response, 
                    "is_expert": raw_response["correct"],
                    "answer": answer, 
                    "data_source": data_source
                }
            )
    
        # Convert the list to a Dataset object
    train_dataset = Dataset.from_list(prompt_only_data)
    expert_demo_dataset = Dataset.from_list(expert_demo_data)

    # Save the datasets to json files
    with open(f"{name}_train.json", "w") as file:
        json.dump(prompt_only_data, file, indent=4)
    with open(f"{name}_expert_demo.json", "w") as file:
        json.dump(expert_demo_data, file, indent=4)

    train_dataset.to_parquet(f"{name}_train.parquet")
    expert_demo_dataset.to_parquet(f"{name}_expert_demo.parquet")


merge(all_data, "prime")
merge(all_corrected_data, "prime_correct_only")
merge(partial_corrected_data, "prime_partial_correct")
# merge(partial_corrected_data_aug_2, "prime_partial_correct_aug")