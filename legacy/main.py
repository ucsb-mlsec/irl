from irl import MaxEntIRL
from llm import PolicyModel, RewardModel
from dataset import prepare_dataset
    

def main():
    # Load models
    policy_model = PolicyModel("Qwen/Qwen2.5-7B-Instruct", device="cuda:0")
    reward_model = RewardModel("Qwen/Qwen2.5-7B-Instruct", device="cuda:1")
    maxent_irl = MaxEntIRL(reward_model, policy_model)

    # Prepare dataset
    dataset = prepare_dataset("s1k")

    # Train reward model
    best_val_loss = maxent_irl.train(
        dataset,
        beta=1,
        lr=1e-4,
        encoder_lr=1e-6,
        train_encoder=False,
        train_batch_size=4,
        num_epochs=5,
        reward_batch_size=4,
        temperature=0.7,
        top_k=5,
        max_new_tokens=4096,
        validation_split=0.1,
        save_path="irl_reward_model.pt",
        checkpoint_dir="checkpoints",
        save_interval=200,
        log_interval=10,
        resume_from=None,
        max_grad_norm=1.0
    )
    print(f"Best validation loss: {best_val_loss:.4f}")



if __name__ == "__main__":
    main()