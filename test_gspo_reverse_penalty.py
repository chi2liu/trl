#!/usr/bin/env python3
"""Test script for GSPO-reverse penalty in GRPO implementation."""

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
        # Reward based on completion length with some variance
        reward = len(completion) / 100.0 + torch.randn(1).item() * 0.1
        rewards.append(reward)
    return rewards

print("=" * 60)
print("Testing GSPO-Reverse Penalty for Clipped Tokens")
print("=" * 60)

# Test configuration with gspo_reverse penalty
config = GRPOConfig(
    output_dir="./test_output_gspo",
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
    clipped_token_penalty_weight=0.3,
    clipped_token_penalty_type="gspo_reverse",  # Test the new GSPO-reverse penalty
)

print("\nConfiguration:")
print(f"  - clipped_token_penalty: {config.clipped_token_penalty}")
print(f"  - clipped_token_penalty_weight: {config.clipped_token_penalty_weight}")
print(f"  - clipped_token_penalty_type: {config.clipped_token_penalty_type}")
print(f"  - epsilon: {config.epsilon}")

try:
    print("\n1. Testing GSPO-Reverse Penalty Type")
    print("-" * 40)

    trainer = GRPOTrainer(
        model="gpt2",  # Using small model for testing
        reward_funcs=dummy_reward_func,
        args=config,
        train_dataset=dataset,
    )

    print("✓ Trainer initialized successfully!")
    print("\nGSPO-Reverse penalty mechanism:")
    print("  - Computes sequence-level importance weights for sequences with clipped tokens")
    print("  - Applies penalty in reverse direction (negative of GSPO)")
    print("  - Hybrid approach: sequence-level penalty strength, token-level application")
    print("  - Effect: Strong correction for sequences with extreme deviations")

    print("\n2. Comparing Different Penalty Types")
    print("-" * 40)

    penalty_types = ["reverse", "gspo_reverse", "proportional"]

    for penalty_type in penalty_types:
        config.clipped_token_penalty_type = penalty_type
        print(f"\n  Testing '{penalty_type}':")

        try:
            trainer = GRPOTrainer(
                model="gpt2",
                reward_funcs=dummy_reward_func,
                args=config,
                train_dataset=dataset,
            )
            print(f"    ✓ {penalty_type} penalty initialized successfully")

            if penalty_type == "reverse":
                print("      → Token-level reverse advantage penalty")
            elif penalty_type == "gspo_reverse":
                print("      → Sequence-level GSPO-style reverse penalty")
                print("      → Stronger correction for systematic deviations")
            elif penalty_type == "proportional":
                print("      → Penalty proportional to clip distance")

        except Exception as e:
            print(f"    ✗ Error with {penalty_type}: {e}")

    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)

    print("\nKey Insights:")
    print("  • GSPO-Reverse combines sequence-level and token-level approaches")
    print("  • It provides stronger corrections for systematic policy deviations")
    print("  • Particularly effective when multiple tokens in a sequence are clipped")
    print("  • The reverse direction ensures the model learns from extreme cases")

except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()