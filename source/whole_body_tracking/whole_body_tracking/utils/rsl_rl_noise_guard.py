from __future__ import annotations

import functools

import torch


def sanitize_scalar_policy_std(
    policy,
    *,
    min_std: float = 1.0e-6,
    reset_std: float = 1.0,
) -> bool:
    """Ensure scalar RSL-RL policy std stays finite and strictly positive.

    RSL-RL's default ActorCritic can parameterize exploration noise as a direct
    trainable ``std`` tensor. Under some optimizer states this tensor can cross
    zero, which later crashes ``torch.distributions.Normal`` sampling.

    Returns ``True`` when the tensor was modified.
    """
    if getattr(policy, "noise_std_type", None) != "scalar" or not hasattr(policy, "std"):
        return False

    std = policy.std.data
    changed = False

    with torch.no_grad():
        nonfinite_mask = ~torch.isfinite(std)
        if torch.any(nonfinite_mask):
            std[nonfinite_mask] = float(reset_std)
            changed = True

        clamped_std = torch.clamp(std, min=float(min_std))
        if not torch.equal(clamped_std, std):
            std.copy_(clamped_std)
            changed = True

    return changed


def install_scalar_std_optimizer_guard(
    optimizer,
    policy,
    *,
    min_std: float = 1.0e-6,
    reset_std: float = 1.0,
) -> bool:
    """Wrap ``optimizer.step()`` so scalar policy std is sanitized every step."""
    if getattr(policy, "noise_std_type", None) != "scalar" or not hasattr(policy, "std"):
        return False

    if getattr(optimizer, "_whole_body_tracking_std_guard_installed", False):
        return True

    original_step = optimizer.step

    @functools.wraps(original_step)
    def guarded_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        sanitize_scalar_policy_std(policy, min_std=min_std, reset_std=reset_std)
        return result

    optimizer.step = guarded_step
    optimizer._whole_body_tracking_std_guard_installed = True
    return True
