# Adapt from https://github.com/hendrycks/math/blob/main/modeling/evaluate_gpt3.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import traceback
import openai
import argparse
import numpy as np
import operator
import json
import tqdm
import pandas as pd
from utils.util import clean_numbers, last_boxed_only, last_boxed_only_string
from utils.math_equivalence import is_equiv
from utils.grader import math_equal
from collections import defaultdict
import re
import math
from transformers import AutoTokenizer
# add recipe path
sys.path.append(os.path.abspath(".."))
from recipe.irl.rm import RewardModule

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
import torch



def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def get_rm_score(question, answer):
    system_prompt = open("system_prompt.md").read()
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
        {"role": "assistant", "content": answer},
    ]
    msg_context = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    # Format and tokenize the conversations
    model_device = next(rm.parameters()).device
    msg_formatted = rm_tokenizer.apply_chat_template(msg, tokenize=False)
    msg_tokenized = rm_tokenizer(msg_formatted, return_tensors="pt").to(model_device)
    # Get start index of the answer generation
    context_formatted = rm_tokenizer.apply_chat_template(msg_context, tokenize=False, add_generation_prompt=True)
    context_tokenized = rm_tokenizer(context_formatted, return_tensors="pt")
    start_index = context_tokenized.input_ids.shape[1]
    # Get the reward scores
    with torch.no_grad():
        score = rm(**msg_tokenized)
    # Do average pooling over the response tokens
    response_score = score[0, start_index:].mean().item()
    return response_score

def generate_sample_batch(question_list, temperature=0.0, n=1):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(max_tokens=8192,
                                    temperature=temperature,
                                    n=n,
                                    stop=["\n###\nProblem: ", "<|eot_id|>"],)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = [output.outputs[0].text for output in outputs]
    if n == 1:
        return completions
    else:
        all_completions = []
        for output in outputs:
            question_completions = [completion.text for completion in output.outputs]
            all_completions.append(question_completions)
        
        tts_completions = []
        for i, question in enumerate(question_list):
            group = {}
            # best_of_n_max_score = -1
            # best_of_n = None
            for pred in all_completions[i]:
                # first parse the answer for this prediction
                is_matched, model_output = match_answer(pred)
                pred_choice = model_output.strip("The final answer is ").strip(". I hope it is correct.")
                # the group is a dictionary that is used to do self-consistency
                if pred_choice not in group:
                    group[pred_choice] = []

                score = get_rm_score(question, pred)

                group[pred_choice].append((score, pred))

            self_consistent_max_score = float("-inf")
            self_consistent = None
            for _, scores in group.items():
                cur_score = np.mean([score for score, _ in scores])
                if cur_score > self_consistent_max_score:
                    self_consistent_max_score = cur_score
                    single_max_score = float("-inf")
                    for score, pred in scores:
                        if score > single_max_score:
                            single_max_score = score
                            self_consistent = pred
            
            tts_completions.append(self_consistent)
            # tts_completions.append(best_of_n)
        return tts_completions


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    ans_marker = 'The answer is: '
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]
            
    ans_marker = 'answer:\n'
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]

    ans_marker = 'answer: '
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    # Grade
    return is_matched, response


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def make_conv_hf(question, tokenizer):
    system_prompt = open("system_prompt.md").read()
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat


def run(args, max=-1):
    all_problems = read_jsonl_file(os.path.join(args.data_dir, "aimo-validation-aime.jsonl"))
    # only keep problems in 2024
    for problem_data in all_problems:
        url = problem_data["url"]
        if "2024_AIME_I_Problems" in url or "2024_AIME_II_Problems" in url:
            continue
        else:
            all_problems.remove(problem_data)
    print("reading problems done!")
    if args.test_time_scaling:
        # load reward model
        global rm
        global rm_tokenizer
        rm = RewardModule.from_pretrained(
            args.reward_model, 
            base_model='Qwen/Qwen3-4B-Base',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        rm_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base') # hardcoded for now

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    completions = generate_sample_batch([make_conv_hf(problem_data["question"], tokenizer) for problem_data in all_problems], args.temperature, args.n)

    tmp_data = []
    for problem_data, model_output in zip(all_problems, completions):
        problem_data["completion"] = model_output
        tmp_data.append(problem_data)
    write_jsonl_file(os.path.join(args.save_dir, "completions.jsonl"), tmp_data)

    total = len(all_problems)
    correct = 0
    save_data = []
    for problem_data, model_output in zip(all_problems, completions):
        answer = str(problem_data["answer"])
        answer = answer.lstrip('0') 
        problem_data["completion"] = model_output
        is_matched, model_output = match_answer(model_output)
        model_output = model_output.strip("The final answer is ").strip(". I hope it is correct.")
        try:
            if "\pi" in model_output or "\pi" in answer:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
                equiv = any(equivs)
            else:
                equiv = math_equal(model_output, answer, timeout=True)
        except:
            equiv = False

        if equiv:
            correct += 1
        problem_data["success"] = equiv
        save_data.append(problem_data)

    print("##########AIME")
    print(f"total: {total}, success: {correct}, rate: {correct / total}")
    comp_name = []
    for line in save_data:
        comp_name.append(line["url"].split("/")[-2])
    comp_name = list(set(comp_name))
    dic = {}
    for line in comp_name:
        dic[line] = {}
        dic[line]["total"] = 0
        dic[line]["success"] = 0
    for line in save_data:
        dic[line["url"].split("/")[-2]]["total"] += 1
        if line["success"]:
            dic[line["url"].split("/")[-2]]["success"] += 1
    print(json.dumps(dic, indent=4))
    aime2024_total = 30
    aime2024_success = dic["2024_AIME_II_Problems"]["success"] + dic["2024_AIME_I_Problems"]["success"]
    print("##########AIME2024")
    print(f"total: {aime2024_total}, success: {aime2024_success}, rate: {aime2024_success / aime2024_total}")

    output_file = os.path.join(args.save_dir, "results_total.txt")
    with open(output_file, "w+") as f:
        f.write(f"AIME ALL-total: {total}, success: {correct}, rate: {correct / total}")
        f.write(f"\n\nAIME2024-total: {aime2024_total}, success: {aime2024_success}, rate: {aime2024_success / aime2024_total}")
    output_file = os.path.join(args.save_dir, "results_split.txt")
    with open(output_file, "w+") as f:
        f.write(json.dumps(dic, indent=4))
    write_jsonl_file(os.path.join(args.save_dir, "results.jsonl"), save_data)
    import pandas as pd
    df = pd.DataFrame(save_data)
    df.to_excel(os.path.join(args.save_dir, "results.xlsx"), index=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="")
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    # test time scaling options
    parser.add_argument("--test_time_scaling", type=bool, default=True)
    parser.add_argument("--reward_model", type=str, default="/scr/xian/IRL_Qwen3_4B_Base/global_step_75/reward_model")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()
    run(args)
