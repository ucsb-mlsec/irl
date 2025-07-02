import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup
import logging

from llm import PolicyModel, RewardModel
from dataset import prepare_dataset


class IRLDataset(Dataset):
    def __init__(self, questions, expert_trajectories, sampled_trajectories):
        self.questions = questions
        self.expert_trajectories = expert_trajectories
        self.sampled_trajectories = sampled_trajectories
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "expert_trajectory": self.expert_trajectories[idx],
            "sampled_trajectory": self.sampled_trajectories[idx]
        }


class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning for LLMs"""
    def __init__(self, reward_model, policy_model):
        self.reward_model = reward_model
        self.policy_model = policy_model
        
    def sample_trajectory(self, dataset, temperature=0.7, top_k=5, max_new_tokens=1024, sample_batch_size=4, beta=1):
        """
        Sample trajectories from the policy model based on rewards converted to probabilities.
        
        Args:
            dataset: List of dictionaries with 'question' keys
            temperature: Temperature for reward softmax (0 = greedy, higher = more random)
            top_k: Only consider top k tokens by policy model
            max_new_tokens: Maximum trajectory length
            sample_batch_size: Batch size for reward calculation
            
        Returns:
            List of sampled trajectories
        """
        sampled_trajectories = []
        
        for data in dataset:
            traj = []
            question = data["question"]
            
            for i in range(max_new_tokens):
                # Get top actions from policy model
                sampled_actions, policy_probs = self.policy_model(question, traj, top_k=top_k)
                
                # Calculate rewards for all actions
                rewards = self.reward_model.calculate_sampled_state_actions(question, traj, sampled_actions, batch_size=sample_batch_size)
                
                logits = (rewards / max(0.01, temperature)).squeeze()  # Apply temperature scaling

                # print(policy_probs)
                # print(rewards)

                logits = logits * policy_probs.to(logits.device)  # Combine with policy probabilities

                # normalize
                probs = logits / torch.sum(logits)

                # probs = F.softmax(logits, dim=0).squeeze()
                
                # print(probs)
                
                # Sample based on probabilities
                selected_idx = torch.multinomial(probs, 1, replacement=True)[0].item()
                
                # Get the selected action
                selected_action = sampled_actions[selected_idx]
                
                # print(probs)
                # print(sampled_actions)
                # print(selected_action)
                # print(self.policy_model.tokenizer.eos_token)
                
                traj.append(selected_action)

                print(f"Step {i} traj: {self.policy_model.tokenizer.decode(traj)}")
                
                # Check for EOS token
                if self.policy_model.tokenizer.decode(selected_action) == self.policy_model.tokenizer.eos_token:
                # if selected_action == self.policy_model.tokenizer.eos_token:
                    break
                    
                del rewards, logits, probs, selected_idx, selected_action
                torch.cuda.empty_cache()
            
            sampled_trajectories.append(traj)
        
        return sampled_trajectories
    
    def train(self, dataset, beta=1, lr=1e-4, encoder_lr=1e-6, train_encoder=False,
            train_batch_size=4, num_epochs=5, reward_batch_size=4, 
            temperature=0.7, top_k=5, max_new_tokens=4096, 
            validation_split=0.1, save_path="irl_reward_model.pt",
            checkpoint_dir="checkpoints", save_interval=200,
            log_interval=10, resume_from=None, max_grad_norm=1.0):
        """
        Train the reward model using Maximum Entropy IRL with flexible options
        for updating the encoder or just the reward head.
        
        Args:
            dataset: List of dictionaries with 'question' and 'response' keys
            beta: Temperature parameter for MaxEnt IRL
            lr: Learning rate for the reward head
            encoder_lr: Learning rate for the encoder (if trained)
            train_encoder: Whether to train the encoder backbone
            train_batch_size: Number of examples per batch for training
            num_epochs: Number of training epochs
            reward_batch_size: Batch size for reward calculation
            temperature: Temperature for policy sampling
            top_k: Number of top tokens to sample from policy
            max_new_tokens: Maximum trajectory length when sampling
            validation_split: Fraction of data to use for validation
            save_path: Path to save the best model
            checkpoint_dir: Directory to save checkpoints
            save_interval: How often to save model checkpoints (in steps)
            log_interval: How often to log training progress (in steps)
            resume_from: Path to checkpoint to resume from
            max_grad_norm: Maximum gradient norm for gradient clipping
            
        Returns:
            Best validation loss
        """

        
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )
        logger = logging.getLogger(__name__)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Configure parameter groups for optimizer
        parameters = []
        
        # Add encoder parameters (if training)
        if train_encoder:
            logger.info("Will train the encoder backbone")
            parameters.append({
                "params": self.reward_model.encoder.parameters(),
                "lr": encoder_lr
            })
            # Make sure encoder requires grad
            for param in self.reward_model.encoder.parameters():
                param.requires_grad = True
        else:
            logger.info("Freezing encoder backbone")
            # Freeze encoder parameters
            for param in self.reward_model.encoder.parameters():
                param.requires_grad = False
        
        # Add reward head parameters
        parameters.append({
            "params": self.reward_model.reward.parameters(),
            "lr": lr
        })
        
        # Set up optimizer
        optimizer = optim.AdamW(parameters, weight_decay=0.01)
        
        # Split dataset into train and validation
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Create random indices for train/val split
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Prepare questions and expert trajectories
        questions = [data["question"] for data in dataset]
        expert_trajectories = [data["response"] for data in dataset]
        
        # Create train/val data
        train_questions = [questions[i] for i in train_indices]
        train_expert_trajectories = [expert_trajectories[i] for i in train_indices]
        val_questions = [questions[i] for i in val_indices]
        val_expert_trajectories = [expert_trajectories[i] for i in val_indices]
        
        # Calculate total training steps for scheduler
        total_steps = (len(train_indices) // train_batch_size) * num_epochs
        
        # Set up learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(500, int(0.1 * total_steps)),
            num_training_steps=total_steps
        )
        
        # Resume from checkpoint if provided
        global_step = 0
        best_val_loss = float('inf')
        
        if resume_from:
            logger.info(f"Loading checkpoint from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.reward_model.device)
            
            # Load reward head
            self.reward_model.reward.load_state_dict(checkpoint["reward_head"])
            
            # Load encoder if present in checkpoint and we're training it
            if "encoder" in checkpoint and train_encoder:
                self.reward_model.encoder.load_state_dict(checkpoint["encoder"])
            
            # Load optimizer and scheduler state
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and checkpoint["scheduler"]:
                scheduler.load_state_dict(checkpoint["scheduler"])
            
            # Restore step and best validation loss
            global_step = checkpoint.get("step", 0)
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            logger.info(f"Resuming from step {global_step} with validation loss {best_val_loss:.4f}")
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Sample trajectories using current reward model
            logger.info("Sampling trajectories...")
            train_subset = [dataset[i] for i in train_indices]
            train_sampled_trajectories = self.sample_trajectory(
                train_subset, 
                temperature=temperature, 
                top_k=top_k, 
                max_new_tokens=max_new_tokens, 
                sample_batch_size=reward_batch_size
            )
            
            # Create dataset and dataloader for training
            train_dataset = IRLDataset(train_questions, train_expert_trajectories, train_sampled_trajectories)
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
            
            # Training
            self.reward_model.reward.train()
            if train_encoder:
                self.reward_model.encoder.train()
                
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                batch_questions = batch["question"]
                batch_expert_trajectories = batch["expert_trajectory"]
                batch_sampled_trajectories = batch["sampled_trajectory"]
                
                # Calculate rewards for expert trajectories
                expert_rewards = self.reward_model.calculate_trajectory(
                    batch_questions, 
                    batch_expert_trajectories, 
                    batch_size=reward_batch_size
                )
                
                # Calculate rewards for sampled trajectories
                sampled_rewards = self.reward_model.calculate_trajectory(
                    batch_questions, 
                    batch_sampled_trajectories, 
                    batch_size=reward_batch_size
                )
                
                # Calculate MaxEnt IRL loss
                batch_loss = 0

                expert_total = torch.sum(exp_reward)
                sampled_total = torch.sum(sample_reward)
                    
                # MaxEnt IRL loss: maximize expert reward and minimize sampled reward
                # This follows the MaxEnt IRL objective: maximize P(expert | reward) - P(sampled | reward)
                batch_loss = -beta * (expert_total - sampled_total)
                
                # Average loss over batch
                batch_loss = batch_loss / len(batch_questions)
                
                # Backpropagation
                optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.reward.parameters(), 
                    max_grad_norm
                )
                if train_encoder:
                    torch.nn.utils.clip_grad_norm_(
                        self.reward_model.encoder.parameters(), 
                        max_grad_norm
                    )
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
                global_step += 1
                
                # Logging
                if global_step % log_interval == 0:
                    logger.info(f"Step {global_step}, Loss: {batch_loss.item():.4f}")
                
                # Save checkpoint
                if global_step % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"reward_model_step_{global_step}.pt")
                    
                    # Create state dict with different components
                    checkpoint = {
                        "reward_head": self.reward_model.reward.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": global_step,
                        "best_val_loss": best_val_loss,
                        "train_config": {
                            "train_encoder": train_encoder
                        }
                    }
                    
                    # Add encoder weights if we're training it
                    if train_encoder:
                        checkpoint["encoder"] = self.reward_model.encoder.state_dict()
                    
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            avg_train_loss = epoch_loss / num_batches
            logger.info(f"Training loss: {avg_train_loss:.4f}")
            
            # Validation
            logger.info("Running validation...")
            self.reward_model.reward.eval()
            if train_encoder:
                self.reward_model.encoder.eval()
                
            # Sample trajectories for validation
            val_subset = [dataset[i] for i in val_indices]
            val_sampled_trajectories = self.sample_trajectory(
                val_subset, 
                temperature=temperature, 
                top_k=top_k, 
                max_new_tokens=max_new_tokens, 
                sample_batch_size=reward_batch_size
            )
            
            # Create validation dataset and dataloader
            val_dataset = IRLDataset(val_questions, val_expert_trajectories, val_sampled_trajectories)
            val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
            
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    batch_questions = batch["question"]
                    batch_expert_trajectories = batch["expert_trajectory"]
                    batch_sampled_trajectories = batch["sampled_trajectory"]
                    
                    # Calculate rewards for expert trajectories
                    expert_rewards = self.reward_model.calculate_trajectory(
                        batch_questions, 
                        batch_expert_trajectories, 
                        batch_size=reward_batch_size
                    )
                    
                    # Calculate rewards for sampled trajectories
                    sampled_rewards = self.reward_model.calculate_trajectory(
                        batch_questions, 
                        batch_sampled_trajectories, 
                        batch_size=reward_batch_size
                    )
                    
                    # Calculate MaxEnt IRL loss
                    batch_loss = 0
                    for exp_reward, sample_reward in zip(expert_rewards, sampled_rewards):
                        expert_total = torch.sum(exp_reward)
                        sampled_total = torch.sum(sample_reward)
                        loss = -beta * (expert_total - sampled_total)
                        batch_loss += loss
                    
                    batch_loss = batch_loss / len(batch_questions)
                    val_loss += batch_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save best model
                best_checkpoint = {
                    "reward_head": self.reward_model.reward.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_config": {
                        "train_encoder": train_encoder
                    }
                }
                
                if train_encoder:
                    best_checkpoint["encoder"] = self.reward_model.encoder.state_dict()
                    
                torch.save(best_checkpoint, save_path)
                logger.info(f"Saved best model to {save_path}")
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return best_val_loss



if __name__ == "__main__":
    policy_model = PolicyModel("Qwen/Qwen2.5-7B-Instruct", device="cuda:0")
    reward_model = RewardModel("Qwen/Qwen2.5-7B-Instruct", device="cuda:1")
    maxent_irl = MaxEntIRL(reward_model, policy_model)

    # Prepare dataset
    dataset = prepare_dataset("s1k")

    import time
    start = time.time()
    train_subset = dataset[:3]
    train_sampled_trajectories = maxent_irl.sample_trajectory(
        train_subset, 
        temperature=0.7, 
        top_k=5, 
        max_new_tokens=4096, 
        sample_batch_size=4
    )

    print(f"Time taken: {time.time() - start:.2f}s")

    sampled_dataset = []
    for i, data in enumerate(train_subset):
        sampled_dataset.append({
            "question": data["question"],
            "response": data["response"],
            "sampled_response": train_sampled_trajectories[i],
            "answer": data["answer"]
        })
    
    import json
    with open("sampled_dataset.json", "w") as f:
        json.dump(sampled_dataset, f, indent=4)
    


