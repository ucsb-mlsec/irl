import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

class PolicyModel():
    def __init__(self, model_name, hidden_size=1024, device="cuda"):
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, question, traj, top_k=5):
        """
        Get the top-k tokens for the next position in the sequence.
        
        Args:
            inputs (str): Input text
            temperature (float): Temperature for sampling (0 for greedy)
            top_k (int): Number of top tokens to return
            
        Returns:
            list: Top-k tokens with their probabilities
        """
        conversation = [
            {
                "role": "system", 
                "content": "You are a helpful assistant."
            },
            {
                "role": "user", 
                "content": question
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=True
        )

        if len(traj) > 0:
            traj_text = self.tokenizer.decode(traj)
            formatted_prompt += traj_text
        
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device).input_ids
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last position
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k tokens and their probabilities
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            
            # Convert to Python lists
            topk_indices = topk_indices[0].tolist()
            topk_probs = topk_probs[0]
            
            # token_lists = [[idx] for idx in topk_indices]
            # results = self.tokenizer.batch_decode(token_lists, skip_special_tokens=False)
            # probs = topk_probs
            # return results, probs
            return  topk_indices, topk_probs
             

class RewardModel():
    def __init__(self, encode_model, hidden_size=1024, device="cuda"):
        self.device = device
        
        self.encoder = AutoModelForCausalLM.from_pretrained(encode_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(encode_model)

        text = "Hello, world!"
        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        # Forward pass with output_hidden_states=True
        with torch.no_grad():
            outputs = self.encoder(**inputs, output_hidden_states=True)

        # Get the last hidden state from the last layer
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        self.input_size = last_hidden_state.shape[-1]
        self.hidden_size = hidden_size
        
        self.reward = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(device)

    def calculate_sampled_state_actions(self, question, traj, actions, batch_size=4):
        # Ensure actions is a list
        if not isinstance(actions, list):
            actions = [actions]
        
        # Process in batches to avoid OOM
        all_rewards = []
        
        # Process actions in chunks of batch_size
        for i in range(0, len(actions), batch_size):
            # Get the current batch of actions
            batch_actions = actions[i:i+batch_size]
            
            texts = []
            # Create separate sequences for each state+action pair
            conversation = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=True
            )

            for action in batch_actions:
                new_traj = copy.deepcopy(traj)
                new_traj.append(action)
                traj_action_text = self.tokenizer.decode(new_traj)
                texts.append(formatted_prompt+traj_action_text)


            # texts = [formatted_prompt+traj+action for action in batch_actions]
            
            # Tokenize all sequences in the current batch
            batch_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    output_hidden_states=True
                )
            
            # Extract features from the last token of each sequence
            last_hidden_states = outputs.hidden_states[-1]
            
            features = []
            for j, seq_len in enumerate(batch_inputs.attention_mask.sum(dim=1)):
                # Get the last non-padding token
                features.append(last_hidden_states[j, seq_len-1])
            
            # Stack features
            stacked_features = torch.stack(features)

            # print(stacked_features)
            
            # Compute rewards for this batch
            batch_rewards = self.reward(stacked_features)
            all_rewards.append(batch_rewards)
        
        # Concatenate all batch results
        if len(all_rewards) > 1:
            return torch.cat(all_rewards, dim=0)
        else:
            return all_rewards[0]
    
    def calculate_trajectory(self, questions, trajectories, batch_size=4):
        """
        Calculate rewards for multiple question-trajectory pairs with true parallel batch processing.
        
        Args:
            questions (list): List of question/initial state strings.
            trajectories (list): List of trajectory strings corresponding to each question.
            batch_size (int): Number of pairs to process in each batch.
            
        Returns:
            list: List of reward tensors for each question-trajectory pair.
        """
        all_rewards = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_trajectories = trajectories[i:i+batch_size]
            
            # Get question token lengths
            state_lengths = []
            combined_texts = []
            
            # Prepare combined texts and track question lengths
            for question, trajectory in zip(batch_questions, batch_trajectories):
        
                conversation = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ]


                inputs = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False,
                    add_generation_prompt=True
                )

                state_tokens = self.tokenizer(inputs, return_tensors="pt")
                
                state_lengths.append(len(state_tokens.input_ids[0]))
                
                conversation.append({"role": "assistant", "content": trajectory})

                # Combine question and trajectory
                complete_text = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False,
                    add_generation_prompt=False
                )

                combined_texts.append(complete_text)
            
            # Tokenize the entire batch at once
            batch_inputs = self.tokenizer(combined_texts, padding=True, return_tensors="pt").to(self.device)
            
            # Single forward pass for the entire batch
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    output_hidden_states=True
                )
                
                # Get hidden states for the entire batch
                last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # Extract trajectory features for each example
            batch_features = []
            
            for idx, state_len in enumerate(state_lengths):
                # Get sequence length using attention mask
                seq_len = batch_inputs.attention_mask[idx].sum().item()
                
                # Extract trajectory features (tokens after the question)
                features = last_hidden_states[idx, state_len:seq_len]
                batch_features.append(features)
            
            # Process rewards for each example
            batch_rewards = []
            for features in batch_features:
                rewards = self.reward(features).squeeze()
                batch_rewards.append(rewards)
            
            all_rewards.extend(batch_rewards)
        
        return all_rewards




if __name__ == "__main__":
    # reward_model = RewardModel("Qwen/Qwen2.5-7B-Instruct")
    
    # # Test data
    # state = "hello world"
    # a1 = "!"
    # a2 = "i am"
    # a3 = "claude."
    
    # actions = [a1, a2, a3]
    
    # # Method 1: Using your optimized implementation with custom masking
    # print("Running the optimized forward method...")
    # feature_all = reward_model.calculate_sampled_state_actions(state, actions)
    
    # # Method 2: Calculate each embedding separately as reference
    # print("Running separate forward passes for reference...")
    # feature_2 = reward_model.calculate_sampled_state_actions(state, a2)
    # feature_3 = reward_model.calculate_sampled_state_actions(state, a3)

    # # Compare the results
    # print("Comparing results...")
    # print("Features for all actions:")
    # print(feature_all)
    # print("Features for action 2:")
    # print(feature_2)
    # print("Features for action 3:")
    # print(feature_3)

    # questions = ["hello world", "how are you"]
    # trajectories = ["!i am", "doing today?"]
    # reward = reward_model.calculate_trajectory(questions, trajectories)
    # print(reward)

    policy_model = PolicyModel("Qwen/Qwen2.5-7B-Instruct")
    results = policy_model("hello my name is qwen, i am developed by", top_k=10)
    print(results)
    print(len(results))
    
