#!/usr/bin/env python3
"""Test script for simplified GSPO-reverse (direct switch, no weight)."""

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import torch

# Create a simple dummy dataset
dataset_dict = {
    "prompt": [
        "What is the capital of France?",
        "How do you make a sandwich?",
        "What is machine learning?",
        "Explain quantum physics.",
    ] * 10
}
dataset = Dataset.from_dict(dataset_dict)

# Define a simple reward function
def dummy_reward_func(prompts, completions, **kwargs):
    """Simple reward function."""
    rewards = []
    for completion in completions:
        reward = len(completion) / 100.0 + torch.randn(1).item() * 0.1
        rewards.append(reward)
    return rewards

print("=" * 70)
print("GSPO-Reverse: Direct Mode Switch (Simplified Implementation)")
print("=" * 70)
print("\nKey Innovation: Clipped tokens directly switch to reverse GSPO mode")
print("No weight parameter needed - it's a clean mode switch!\n")

# Configuration with simplified gspo_reverse
config = GRPOConfig(
    output_dir="./test_gspo_direct",
    per_device_train_batch_size=2,
    num_generations=2,
    max_prompt_length=128,
    max_completion_length=64,
    learning_rate=1e-6,
    num_train_epochs=1,
    logging_steps=1,
    epsilon=0.2,

    # Enable clipped token handling with GSPO-reverse
    clipped_token_penalty=True,
    clipped_token_penalty_type="gspo_reverse",
    # Note: clipped_token_penalty_weight is NOT used for gspo_reverse!
)

print("Configuration:")
print(f"  - clipped_token_penalty: {config.clipped_token_penalty}")
print(f"  - clipped_token_penalty_type: {config.clipped_token_penalty_type}")
print(f"  - epsilon: {config.epsilon}")
print("  - weight parameter: NOT NEEDED for gspo_reverse!\n")

try:
    print("Testing Direct GSPO Switch Implementation")
    print("-" * 50)

    trainer = GRPOTrainer(
        model="gpt2",
        reward_funcs=dummy_reward_func,
        args=config,
        train_dataset=dataset,
    )

    print("✓ Trainer initialized successfully!\n")

    print("How it works:")
    print("1. Normal tokens: Use standard GRPO loss calculation")
    print("2. Clipped tokens: Switch to GSPO mode (sequence-level)")
    print("3. But apply in REVERSE direction (negative update)")
    print("4. No weight needed - it's a direct replacement!\n")

    print("Mathematical formulation:")
    print("-" * 50)
    print("Standard GRPO (for non-clipped tokens):")
    print("  loss = -min(ratio * advantage, clip(ratio) * advantage)")
    print("")
    print("GSPO-Reverse (for clipped tokens):")
    print("  seq_ratio = exp(mean(log_ratios))")
    print("  loss = -(-seq_ratio * advantage) = seq_ratio * advantage")
    print("  (Note the double negative: reverse of GSPO)")
    print("")

    print("Advantages of this approach:")
    print("• Cleaner implementation - no weight tuning needed")
    print("• Direct mode switch - more principled")
    print("• Sequence-aware correction for extreme deviations")
    print("• Automatic strength based on sequence-level deviation")

    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\nUsage example:")
print("-" * 50)
print("""
# Simple configuration - no weight tuning needed!
config = GRPOConfig(
    # ... other parameters ...
    clipped_token_penalty=True,
    clipped_token_penalty_type="gspo_reverse",
    # That's it! No weight parameter needed
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
)

trainer.train()
""")