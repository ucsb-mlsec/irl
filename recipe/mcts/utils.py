from openai import OpenAI
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

import os
from openai import AzureOpenAI

endpoint = "https://kaiji-m9tecg6e-eastus2.cognitiveservices.azure.com/"
model_name = "o4-mini"
deployment = "o4-mini"

subscription_key = "1zxYvu2omMujmxYDcJBXXGVDx4zybATukNjq6KePBVDUDddE9MI6JQQJ99BDACHYHv6XJ3w3AAAAACOGodZq"
api_version = "2025-03-01-preview"


def process_sample(sample):

    # Create a new client for each thread to avoid concurrency issues
    thread_client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )


    response = sample["text"]
    question = sample["question"]
    ground_truth = sample["answer"]

    input_texts = f"""
Act as a judge to evaluate the correctness of the following response to the question.

### Question:
{question}

### Ground Truth:
{ground_truth}

### Response:
{response}

Please answer with <<<Yes>>> if the response is correct, otherwise answer with <<<No>>>.
"""

    result = get_response(thread_client, input_texts)
    
    # Extract the answer from the result
    match = re.search(r'<<<(Yes|No)>>>', result, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = "No answer found"

    if answer.lower() == "yes":
        return True
    else:
        return False

def get_response(client, prompt):
    # Call the OpenAI API to get a response
    response = client.responses.create(
        model="o4-mini",
        reasoning={"effort": "low"},
        input=[
            {
                "role": "user",
                "content": prompt
            },
        ]
    )
    return response.output_text

def cal_outcome_reward(tokenizer, batch):
    
    # Initialize empty lists for results
    results = []

    max_workers = 8

    bs = len(batch)
    
    data = []
    
    for idx in range(bs):
        sample = batch.batch[idx]
        question = tokenizer.decode(sample["prompts"], skip_special_tokens=True)
        question = question.split("user")[-1].strip()
        answer = tokenizer.decode(sample["answers"], skip_special_tokens=True)
        response = tokenizer.decode(sample["responses"], skip_special_tokens=True)
        data.append({
            "question": question,
            "answer": answer,
            "text": response
        })
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_sample = {
            executor.submit(process_sample, sample): sample
            for sample in data
        }
        
        # Process results as they complete
        for future in as_completed(future_to_sample):
            try:
                result = future.result()
                sample = future_to_sample[future]
                
                results.append(result)
                
            except Exception as e:
                sample = future_to_sample[future]
                print(f"Error processing sample {sample['question']}: {str(e)}")

    return results