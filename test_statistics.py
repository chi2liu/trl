#!/usr/bin/env python3
"""Test script to verify the new statistics for GSPO-reverse."""

import torch
import torch.nn.functional as F
from typing import Dict

def simulate_gspo_statistics(batch_size=4, seq_len=10, epsilon=0.2):
    """Simulate GSPO-reverse statistics calculation."""

    print("=" * 70)
    print("Testing GSPO-Reverse Statistics")
    print("=" * 70)
    print(f"\nSimulation parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Epsilon: {epsilon}")
    print()

    # Create synthetic data
    # Simulate log ratios with some extreme values
    log_ratio = torch.randn(batch_size, seq_len) * 0.5
    # Add some extreme values to ensure clipping
    log_ratio[0, 2] = 1.5  # Will create high ratio
    log_ratio[1, 0] = -1.5  # Will create low ratio
    log_ratio[2, 5] = 2.0  # Very high ratio
    log_ratio[3, 3] = -2.0  # Very low ratio

    # Create advantages (mix of positive and negative)
    advantages = torch.tensor([0.5, -0.3, 0.8, -0.6])

    # Completion mask (all valid for simplicity)
    completion_mask = torch.ones(batch_size, seq_len)

    # Compute token-level coefficients
    coef_1 = torch.exp(log_ratio)

    print("Token-level ratios (coef_1):")
    print(f"  Min: {coef_1.min():.4f}")
    print(f"  Max: {coef_1.max():.4f}")
    print(f"  Mean: {coef_1.mean():.4f}")
    print()

    # Identify clipped tokens
    epsilon_low = epsilon
    epsilon_high = epsilon
    is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_clipped = is_low_clipped | is_high_clipped

    # Compute sequence-level coefficients (GSPO style)
    seq_log_ratio = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    seq_coef = torch.exp(seq_log_ratio)

    print("Sequence-level ratios (seq_coef):")
    for i, (coef, adv) in enumerate(zip(seq_coef, advantages)):
        print(f"  Sequence {i}: {coef:.4f} (advantage: {adv:.2f})")
    print()

    # Calculate statistics
    print("=" * 50)
    print("Statistics Calculation")
    print("=" * 50)

    # 1. Percentage of tokens that are clipped
    total_tokens = completion_mask.sum()
    clipped_tokens = (is_clipped.float() * completion_mask).sum()
    clipped_percentage = (clipped_tokens / total_tokens.clamp(min=1.0)) * 100.0

    print(f"\n1. Clipping Statistics:")
    print(f"   Total tokens: {total_tokens.item():.0f}")
    print(f"   Clipped tokens: {clipped_tokens.item():.0f}")
    print(f"   Clipped percentage: {clipped_percentage.item():.2f}%")

    # Breakdown by type
    low_clipped_tokens = (is_low_clipped.float() * completion_mask).sum()
    high_clipped_tokens = (is_high_clipped.float() * completion_mask).sum()
    low_clipped_pct = (low_clipped_tokens / total_tokens.clamp(min=1.0)) * 100.0
    high_clipped_pct = (high_clipped_tokens / total_tokens.clamp(min=1.0)) * 100.0

    print(f"   - Low clipped: {low_clipped_pct.item():.2f}%")
    print(f"   - High clipped: {high_clipped_pct.item():.2f}%")

    # 2. Percentage that switch to GSPO
    gspo_switched_percentage = 100.0 if clipped_tokens > 0 else 0.0
    print(f"\n2. GSPO Switch Statistics:")
    print(f"   Switched to GSPO: {gspo_switched_percentage:.2f}%")
    print(f"   (Note: All clipped tokens switch to GSPO-reverse)")

    # 3. Check if GSPO tokens would still be clipped
    seq_coef_expanded = seq_coef.unsqueeze(1).expand_as(coef_1)
    would_be_clipped_low = (seq_coef_expanded < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
    would_be_clipped_high = (seq_coef_expanded > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
    would_be_clipped = would_be_clipped_low | would_be_clipped_high

    gspo_tokens = is_clipped.float() * completion_mask
    gspo_still_clipped = (would_be_clipped.float() * is_clipped.float() * completion_mask).sum()
    gspo_reclip_percentage = (gspo_still_clipped / gspo_tokens.sum().clamp(min=1.0)) * 100.0 if clipped_tokens > 0 else 0.0

    print(f"\n3. GSPO Re-clipping Statistics:")
    print(f"   GSPO tokens that would still be clipped: {gspo_reclip_percentage:.2f}%")
    print(f"   (This shows if sequence-level ratio is also extreme)")

    # 4. Distribution of seq_coef values
    seq_coef_lt_08 = (seq_coef < 0.8).float().mean() * 100.0
    seq_coef_08_12 = ((seq_coef >= 0.8) & (seq_coef <= 1.2)).float().mean() * 100.0
    seq_coef_gt_12 = (seq_coef > 1.2).float().mean() * 100.0

    print(f"\n4. Sequence Coefficient Distribution:")
    print(f"   seq_coef < 0.8: {seq_coef_lt_08.item():.2f}%")
    print(f"   0.8 <= seq_coef <= 1.2: {seq_coef_08_12.item():.2f}%")
    print(f"   seq_coef > 1.2: {seq_coef_gt_12.item():.2f}%")

    # Detailed analysis
    print("\n" + "=" * 50)
    print("Detailed Analysis")
    print("=" * 50)

    # Show which tokens are clipped and why
    print("\nClipped tokens details:")
    for batch_idx in range(batch_size):
        clipped_positions = is_clipped[batch_idx].nonzero(as_tuple=True)[0]
        if len(clipped_positions) > 0:
            print(f"\nBatch {batch_idx} (advantage={advantages[batch_idx]:.2f}):")
            for pos in clipped_positions:
                token_ratio = coef_1[batch_idx, pos].item()
                clip_type = "low" if is_low_clipped[batch_idx, pos] else "high"
                print(f"  Token {pos}: ratio={token_ratio:.4f} ({clip_type} clipped)")

            # Show sequence-level correction
            print(f"  → Sequence ratio: {seq_coef[batch_idx].item():.4f}")

            # Check if sequence ratio would also be clipped
            seq_would_clip = would_be_clipped[batch_idx].any()
            if seq_would_clip:
                print(f"  → WARNING: Sequence ratio would also be clipped!")
            else:
                print(f"  → Good: Sequence ratio is within bounds")

    # Summary insights
    print("\n" + "=" * 70)
    print("Summary Insights")
    print("=" * 70)

    if clipped_percentage > 20:
        print("⚠ High clipping rate detected (>20%). Consider adjusting epsilon.")
    elif clipped_percentage < 5:
        print("✓ Low clipping rate (<5%). Model predictions are mostly within bounds.")
    else:
        print("✓ Moderate clipping rate. This is typical during training.")

    if gspo_reclip_percentage > 50:
        print("⚠ Many GSPO tokens would still be clipped. Systematic policy deviation detected.")
    elif gspo_reclip_percentage < 10:
        print("✓ Few GSPO tokens would be re-clipped. Sequence-level correction is effective.")
    else:
        print("✓ Moderate re-clipping rate. GSPO-reverse is providing correction.")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return {
        'clipped_pct': clipped_percentage.item(),
        'low_clipped_pct': low_clipped_pct.item(),
        'high_clipped_pct': high_clipped_pct.item(),
        'gspo_switched_pct': gspo_switched_percentage,
        'gspo_reclipped_pct': gspo_reclip_percentage,
        'seq_coef_mean': seq_coef.mean().item(),
        'seq_coef_std': seq_coef.std().item(),
    }

if __name__ == "__main__":
    # Run test with different scenarios
    print("\nScenario 1: Normal training")
    stats1 = simulate_gspo_statistics(batch_size=4, seq_len=10, epsilon=0.2)

    print("\n\n" + "="*70)
    print("\nScenario 2: Tighter clipping (smaller epsilon)")
    stats2 = simulate_gspo_statistics(batch_size=4, seq_len=10, epsilon=0.1)

    print("\n\nComparison:")
    print(f"Epsilon 0.2: {stats1['clipped_pct']:.2f}% clipped")
    print(f"Epsilon 0.1: {stats2['clipped_pct']:.2f}% clipped")