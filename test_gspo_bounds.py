#!/usr/bin/env python3
"""Test script for GSPO-specific clipping bounds."""

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import torch

# Create test dataset
dataset_dict = {
    "prompt": [
        "Test prompt 1",
        "Test prompt 2",
        "Test prompt 3",
        "Test prompt 4",
    ] * 10
}
dataset = Dataset.from_dict(dataset_dict)

def dummy_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = len(completion) / 100.0 + torch.randn(1).item() * 0.1
        rewards.append(reward)
    return rewards

print("=" * 70)
print("Testing GSPO-Specific Clipping Bounds")
print("=" * 70)

# Test 1: Default GSPO bounds (same as token-level)
print("\n1. Test with default GSPO bounds (same as token-level)")
print("-" * 50)

config1 = GRPOConfig(
    output_dir="./test_gspo_bounds_default",
    per_device_train_batch_size=2,
    num_generations=2,
    max_prompt_length=128,
    max_completion_length=64,
    learning_rate=1e-6,
    epsilon=0.2,  # Token-level: [0.8, 1.2]
    # gspo_epsilon_low and gspo_epsilon_high will default to epsilon
    clipped_token_penalty=True,
    clipped_token_penalty_gspo=True,
)

print(f"Token-level bounds: [{1-config1.epsilon:.2f}, {1+config1.epsilon:.2f}]")
print(f"GSPO bounds (defaulted): [{1-config1.epsilon:.2f}, {1+config1.epsilon:.2f}]")
print("→ Same bounds for both levels\n")

# Test 2: Wider GSPO bounds (more flexible)
print("2. Test with wider GSPO bounds (more flexibility)")
print("-" * 50)

config2 = GRPOConfig(
    output_dir="./test_gspo_bounds_wider",
    per_device_train_batch_size=2,
    num_generations=2,
    max_prompt_length=128,
    max_completion_length=64,
    learning_rate=1e-6,
    epsilon=0.2,  # Token-level: [0.8, 1.2]
    gspo_epsilon_low=0.4,   # GSPO: [0.6, 1.6]
    gspo_epsilon_high=0.6,  # Wider bounds for sequence-level
    clipped_token_penalty=True,
    clipped_token_penalty_gspo=True,
)

print(f"Token-level bounds: [{1-config2.epsilon:.2f}, {1+config2.epsilon:.2f}]")
print(f"GSPO bounds (wider): [{1-config2.gspo_epsilon_low:.2f}, {1+config2.gspo_epsilon_high:.2f}]")
print("→ GSPO has more flexibility for sequence-level corrections\n")

# Test 3: Narrower GSPO bounds (stricter)
print("3. Test with narrower GSPO bounds (stricter control)")
print("-" * 50)

config3 = GRPOConfig(
    output_dir="./test_gspo_bounds_narrower",
    per_device_train_batch_size=2,
    num_generations=2,
    max_prompt_length=128,
    max_completion_length=64,
    learning_rate=1e-6,
    epsilon=0.2,  # Token-level: [0.8, 1.2]
    gspo_epsilon_low=0.1,   # GSPO: [0.9, 1.1]
    gspo_epsilon_high=0.1,  # Tighter bounds for sequence-level
    clipped_token_penalty=True,
    clipped_token_penalty_gspo=True,
)

print(f"Token-level bounds: [{1-config3.epsilon:.2f}, {1+config3.epsilon:.2f}]")
print(f"GSPO bounds (narrower): [{1-config3.gspo_epsilon_low:.2f}, {1+config3.gspo_epsilon_high:.2f}]")
print("→ GSPO is more constrained than token-level\n")

# Test 4: Asymmetric GSPO bounds
print("4. Test with asymmetric GSPO bounds")
print("-" * 50)

config4 = GRPOConfig(
    output_dir="./test_gspo_bounds_asymmetric",
    per_device_train_batch_size=2,
    num_generations=2,
    max_prompt_length=128,
    max_completion_length=64,
    learning_rate=1e-6,
    epsilon=0.2,  # Token-level: [0.8, 1.2]
    gspo_epsilon_low=0.3,   # More tolerance for low confidence
    gspo_epsilon_high=0.1,  # Less tolerance for over-confidence
    clipped_token_penalty=True,
    clipped_token_penalty_gspo=True,
)

print(f"Token-level bounds: [{1-config4.epsilon:.2f}, {1+config4.epsilon:.2f}]")
print(f"GSPO bounds (asymmetric): [{1-config4.gspo_epsilon_low:.2f}, {1+config4.gspo_epsilon_high:.2f}]")
print("→ Different tolerance for under vs over-confidence\n")

# Verify trainer initialization
print("=" * 50)
print("Verifying Trainer Initialization")
print("=" * 50)

try:
    trainer = GRPOTrainer(
        model="gpt2",
        reward_funcs=dummy_reward_func,
        args=config2,  # Use config2 for testing
        train_dataset=dataset,
    )

    print("\n✓ Trainer initialized successfully!")
    print(f"  Token epsilon: {trainer.epsilon_low}, {trainer.epsilon_high}")
    print(f"  GSPO epsilon: {trainer.gspo_epsilon_low}, {trainer.gspo_epsilon_high}")

    # Check that the values are set correctly
    assert hasattr(trainer, 'gspo_epsilon_low'), "Missing gspo_epsilon_low"
    assert hasattr(trainer, 'gspo_epsilon_high'), "Missing gspo_epsilon_high"
    assert trainer.gspo_epsilon_low == config2.gspo_epsilon_low, "gspo_epsilon_low mismatch"
    assert trainer.gspo_epsilon_high == config2.gspo_epsilon_high, "gspo_epsilon_high mismatch"

    print("\n✓ All GSPO bounds parameters verified!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Summary: GSPO Bounds Configuration")
print("=" * 70)
print("""
Use Cases:

1. Default (gspo_epsilon = None):
   - GSPO uses same bounds as token-level
   - Simple and consistent

2. Wider GSPO bounds:
   - Allow more flexibility at sequence level
   - Good when token-level is too restrictive
   - Example: gspo_epsilon_low=0.4, gspo_epsilon_high=0.6

3. Narrower GSPO bounds:
   - Stricter control at sequence level
   - Prevent extreme sequence-level corrections
   - Example: gspo_epsilon_low=0.1, gspo_epsilon_high=0.1

4. Asymmetric bounds:
   - Different tolerance for under vs over-confidence
   - Fine-tuned control based on your model's behavior
   - Example: more tolerance for uncertainty, less for overconfidence
""")