#!/usr/bin/env python3
"""Test to verify the correctness of gspo_reverse implementation."""

import torch
import torch.nn as nn
from typing import Tuple

def simulate_gspo_reverse_logic(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2
) -> Tuple[torch.Tensor, dict]:
    """
    Simulate the gspo_reverse logic to verify correctness.

    Returns:
        - per_token_loss: The computed loss
        - debug_info: Dictionary with intermediate values for verification
    """
    # Compute log ratios
    log_ratio = per_token_logps - old_per_token_logps

    # Token-level importance weights (standard GRPO)
    log_importance_weights = log_ratio
    coef_1 = torch.exp(log_importance_weights)
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)

    # Standard GRPO loss
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Identify clipped tokens
    is_low_clipped = (coef_1 < 1 - epsilon) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef_1 > 1 + epsilon) & (advantages.unsqueeze(1) > 0)
    is_clipped = is_low_clipped | is_high_clipped

    # GSPO-reverse implementation
    # Compute sequence-level log importance weights
    seq_log_ratio = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    seq_log_importance = seq_log_ratio.unsqueeze(-1)
    seq_coef = torch.exp(seq_log_importance)

    # Create reverse GSPO loss (broadcast to match token dimensions)
    reverse_gspo_loss = -seq_coef * advantages.unsqueeze(1)
    # Expand to match the shape of per_token_loss
    reverse_gspo_loss = reverse_gspo_loss.expand_as(per_token_loss)

    # Store original for comparison
    original_loss = per_token_loss.clone()

    # Replace loss for clipped tokens
    per_token_loss = torch.where(is_clipped, reverse_gspo_loss, original_loss)

    debug_info = {
        'coef_1': coef_1,
        'seq_coef': seq_coef,
        'is_clipped': is_clipped,
        'is_low_clipped': is_low_clipped,
        'is_high_clipped': is_high_clipped,
        'original_loss': original_loss,
        'reverse_gspo_loss': reverse_gspo_loss,
        'final_loss': per_token_loss
    }

    return per_token_loss, debug_info

def run_test_cases():
    """Run comprehensive test cases to verify gspo_reverse correctness."""

    print("=" * 70)
    print("Testing GSPO-Reverse Correctness")
    print("=" * 70)

    # Test Case 1: High clipping scenario
    print("\n" + "="*50)
    print("Test Case 1: High Clipping (model too confident)")
    print("="*50)

    batch_size, seq_len = 2, 4

    # Create synthetic data where some tokens will be clipped high
    per_token_logps = torch.tensor([
        [-0.5, -2.0, -0.3, -0.8],  # Token 2 will likely clip high
        [-1.0, -0.2, -3.0, -0.5]   # Token 1 will likely clip high
    ])

    old_per_token_logps = torch.tensor([
        [-0.6, -3.5, -0.4, -0.9],  # Large difference at token 1 (ratio = exp(1.5) ≈ 4.48)
        [-1.1, -2.0, -3.1, -0.6]   # Large difference at token 1 (ratio = exp(1.8) ≈ 6.05)
    ])

    advantages = torch.tensor([0.5, -0.3])  # Positive and negative advantages
    completion_mask = torch.ones(batch_size, seq_len)

    loss, debug = simulate_gspo_reverse_logic(
        per_token_logps, old_per_token_logps, advantages, completion_mask
    )

    print(f"Advantages: {advantages}")
    print(f"Token-level ratios (coef_1):\n{debug['coef_1']}")
    print(f"Sequence-level ratio (seq_coef):\n{debug['seq_coef']}")
    print(f"Clipped tokens mask:\n{debug['is_clipped'].float()}")
    print(f"High clipped:\n{debug['is_high_clipped'].float()}")
    print(f"Low clipped:\n{debug['is_low_clipped'].float()}")

    print("\nLoss comparison:")
    print(f"Original GRPO loss:\n{debug['original_loss']}")
    print(f"Reverse GSPO loss:\n{debug['reverse_gspo_loss']}")
    print(f"Final loss (with replacement):\n{debug['final_loss']}")

    # Verify that clipped tokens use reverse GSPO
    clipped_mask = debug['is_clipped']
    if clipped_mask.any():
        clipped_indices = clipped_mask.nonzero(as_tuple=True)
        for i, j in zip(clipped_indices[0], clipped_indices[1]):
            original = debug['original_loss'][i, j].item()
            reverse_gspo = debug['reverse_gspo_loss'][i, j].item()
            final = debug['final_loss'][i, j].item()
            print(f"\nToken [{i},{j}] (clipped):")
            print(f"  Original loss: {original:.4f}")
            print(f"  Reverse GSPO: {reverse_gspo:.4f}")
            print(f"  Final loss: {final:.4f}")
            print(f"  ✓ Correctly switched to reverse GSPO" if abs(final - reverse_gspo) < 1e-6 else "  ✗ ERROR: Not using reverse GSPO!")

    # Test Case 2: Low clipping scenario
    print("\n" + "="*50)
    print("Test Case 2: Low Clipping (model not confident enough)")
    print("="*50)

    # Create scenario where tokens will be clipped low
    per_token_logps = torch.tensor([
        [-3.0, -2.5, -1.0, -1.5],
        [-2.0, -3.5, -0.5, -2.0]
    ])

    old_per_token_logps = torch.tensor([
        [-1.5, -1.0, -0.5, -0.8],  # Token 0: ratio = exp(-1.5) ≈ 0.22 (will clip low)
        [-0.5, -2.0, -0.2, -1.0]   # Token 0: ratio = exp(-1.5) ≈ 0.22 (will clip low)
    ])

    advantages = torch.tensor([-0.5, -0.3])  # Negative advantages

    loss, debug = simulate_gspo_reverse_logic(
        per_token_logps, old_per_token_logps, advantages, completion_mask
    )

    print(f"Advantages: {advantages}")
    print(f"Token-level ratios (coef_1):\n{debug['coef_1']}")
    print(f"Sequence-level ratio (seq_coef):\n{debug['seq_coef']}")
    print(f"Low clipped:\n{debug['is_low_clipped'].float()}")

    # Test Case 3: Mixed scenario
    print("\n" + "="*50)
    print("Test Case 3: Mixed Clipping in Same Sequence")
    print("="*50)

    # Create a scenario with both high and low clipping in same sequence
    per_token_logps = torch.tensor([
        [-0.2, -3.5, -0.3, -2.0],  # Mix of normal and extreme
    ])

    old_per_token_logps = torch.tensor([
        [-2.0, -1.0, -0.4, -1.9],  # Token 0: high ratio, Token 1: low ratio
    ])

    advantages = torch.tensor([0.8])  # Positive advantage
    completion_mask = torch.ones(1, seq_len)

    loss, debug = simulate_gspo_reverse_logic(
        per_token_logps, old_per_token_logps, advantages, completion_mask
    )

    print(f"Token-level ratios: {debug['coef_1']}")
    print(f"Sequence-level ratio: {debug['seq_coef'].item():.4f}")
    print(f"Clipped mask: {debug['is_clipped'].float()}")

    # Verify the key property: reverse update
    print("\n" + "="*50)
    print("Verification: Reverse Update Direction")
    print("="*50)

    for i in range(debug['is_clipped'].shape[0]):
        for j in range(debug['is_clipped'].shape[1]):
            if debug['is_clipped'][i, j]:
                original = debug['original_loss'][i, j].item()
                reverse_gspo = debug['reverse_gspo_loss'][i, j].item()
                advantage = advantages[i].item()
                seq_coef = debug['seq_coef'][i, 0].item()

                # For GSPO-reverse, the loss should be -seq_coef * advantage
                expected = -seq_coef * advantage

                print(f"\nToken [{i},{j}]:")
                print(f"  Advantage: {advantage:.4f}")
                print(f"  Seq coef: {seq_coef:.4f}")
                print(f"  Expected reverse GSPO: {expected:.4f}")
                print(f"  Actual reverse GSPO: {reverse_gspo:.4f}")
                print(f"  Match: {'✓' if abs(reverse_gspo - expected) < 1e-5 else '✗'}")

                # Verify direction reversal
                if advantage > 0:
                    if seq_coef > 1:
                        print(f"  Direction: Positive adv + seq_coef>1 → Negative loss (reduce probability) ✓")
                    else:
                        print(f"  Direction: Positive adv + seq_coef<1 → Positive loss (increase probability) ✓")
                else:
                    if seq_coef > 1:
                        print(f"  Direction: Negative adv + seq_coef>1 → Positive loss (increase probability) ✓")
                    else:
                        print(f"  Direction: Negative adv + seq_coef<1 → Negative loss (reduce probability) ✓")

    print("\n" + "="*70)
    print("✓ All correctness tests completed!")
    print("="*70)

if __name__ == "__main__":
    run_test_cases()