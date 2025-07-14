import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class RewardModule(nn.Module):
    def __init__(self, base_model, torch_dtype, config, trust_remote_code, ffn_only=False, ffn_hidden_size=1024, device="cuda"):
        super().__init__()
        self.device = device
        
        self.encoder = AutoModelForCausalLM.from_pretrained(base_model,
                                                            torch_dtype=torch_dtype,
                                                            config=config,
                                                            attn_implementation='flash_attention_2',
                                                            trust_remote_code=trust_remote_code)
        
        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        self.input_size = self.config.hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # TODO: architecture of the reward model can be changed more automatically
        self.ffn = nn.Sequential(
            nn.Linear(self.input_size, self.ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.ffn_hidden_size, 1),
            nn.Tanh()
            # nn.Sigmoid()
        )

        if ffn_only:
            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enables gradient checkpointing for the encoder model only.
        
        Args:
            gradient_checkpointing_kwargs (dict, optional): Additional keyword arguments 
                for gradient checkpointing. Defaults to None.
        """
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        
        # Enable gradient checkpointing only on the base encoder model
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
        # Set a flag to track that gradient checkpointing is enabled
        self.is_gradient_checkpointing = True

    def __call__(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        """
        Process inputs through the model and calculate rewards.
        
        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor, optional): Attention mask for inputs.
            position_ids (torch.Tensor, optional): Position ids for inputs.
            use_cache (bool, optional): Whether to use the cache for faster inference.
            
        Returns:
            torch.Tensor: Reward scores for the inputs.
        """
        # Forward pass through the encoder to get hidden states
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_hidden_states=True
        )
        
        # Get the last hidden state from the last layer
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Process all tokens through the FFN
        rewards = self.ffn(last_hidden_states)  # [batch_size, seq_len, 1]
        
        # Squeeze the last dimension to get [batch_size, seq_len]
        rewards = rewards.squeeze(-1)
        
        return rewards
