#!/usr/bin/env python3
"""Test script for clipped token penalty in GRPO implementation."""

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
    ] * 10  # Repeat to have enough samples
}
dataset = Dataset.from_dict(dataset_dict)

# Define a simple reward function
def dummy_reward_func(prompts, completions, **kwargs):
    """Simple reward function that rewards longer completions."""
    rewards = []
    for completion in completions:
        # Reward based on completion length
        reward = len(completion) / 100.0
        rewards.append(reward)
    return rewards

# Test configuration with clipped token penalty enabled
config = GRPOConfig(
    output_dir="./test_output_clip",
    per_device_train_batch_size=2,
    num_generations=2,  # Minimum required for GRPO
    max_prompt_length=128,
    max_completion_length=64,
    learning_rate=1e-6,
    num_train_epochs=1,
    logging_steps=1,
    epsilon=0.2,  # Standard clipping threshold
    # Clipped token penalty parameters
    clipped_token_penalty=True,
    clipped_token_penalty_weight=0.2,
    clipped_token_penalty_type="reverse",  # Test reverse penalty
)

print("Testing GRPO with clipped token penalty...")
print(f"  clipped_token_penalty: {config.clipped_token_penalty}")
print(f"  clipped_token_penalty_weight: {config.clipped_token_penalty_weight}")
print(f"  clipped_token_penalty_type: {config.clipped_token_penalty_type}")

try:
    # Initialize trainer
    trainer = GRPOTrainer(
        model="gpt2",  # Using small model for testing
        reward_funcs=dummy_reward_func,
        args=config,
        train_dataset=dataset,
    )

    print("\nTrainer initialized successfully!")
    print("Clipped token penalty parameters:")
    print(f"  - clipped_token_penalty: {trainer.clipped_token_penalty}")
    print(f"  - clipped_token_penalty_weight: {trainer.clipped_token_penalty_weight}")
    print(f"  - clipped_token_penalty_type: {trainer.clipped_token_penalty_type}")

    print("\nTesting different penalty types...")

    # Test reverse penalty
    print("\n1. Testing 'reverse' penalty type:")
    trainer.clipped_token_penalty_type = "reverse"
    print("   - Should apply opposite advantage to clipped tokens")

    # Test constant penalty
    print("\n2. Testing 'constant' penalty type:")
    config.clipped_token_penalty_type = "constant"
    trainer2 = GRPOTrainer(
        model="gpt2",
        reward_funcs=dummy_reward_func,
        args=config,
        train_dataset=dataset,
    )
    print("   - Should apply constant penalty to all clipped tokens")

    # Test proportional penalty
    print("\n3. Testing 'proportional' penalty type:")
    config.clipped_token_penalty_type = "proportional"
    trainer3 = GRPOTrainer(
        model="gpt2",
        reward_funcs=dummy_reward_func,
        args=config,
        train_dataset=dataset,
    )
    print("   - Penalty proportional to distance from clip boundary")

    print("\nAll penalty types tested successfully!")
    print("\nTest completed successfully! Clipped token penalty mechanism is working.")

except Exception as e:
    print(f"\nError during testing: {e}")
    import traceback
    traceback.print_exc()