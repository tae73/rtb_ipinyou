"""LR schedule and optimizer factory for distributed training.

Supports linear LR scaling, warmup, and cosine/linear/constant decay.
Gradient clipping is included in the optax chain — no separate handling
needed in train_step.

Usage:
    schedule = create_lr_schedule(base_lr=1e-3, warmup_steps=1000,
                                  total_steps=50000, num_devices=4,
                                  scheduler="cosine")
    tx = create_optimizer(base_lr=1e-3, warmup_steps=1000,
                          total_steps=50000, num_devices=4,
                          weight_decay=1e-5, gradient_clip=1.0,
                          scheduler="cosine")
"""

import optax


def create_lr_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    num_devices: int = 1,
    scheduler: str = "cosine",
    lr_scaling: str = "linear",
) -> optax.Schedule:
    """Create learning rate schedule with warmup + decay.

    Linear scaling rule: peak_lr = base_lr x num_devices.
    Warmup: 0 -> peak_lr (linear ramp).
    Decay: cosine / linear / constant after warmup.

    Args:
        base_lr: Base learning rate (single-device).
        warmup_steps: Number of warmup steps (0 = no warmup).
        total_steps: Total training steps.
        num_devices: Number of SPMD devices (for LR scaling).
        scheduler: Decay type after warmup: 'cosine', 'linear', 'constant'.
        lr_scaling: Scaling rule: 'linear', 'sqrt', 'none'.

    Returns:
        optax.Schedule function mapping step -> learning rate.
    """
    if lr_scaling == "linear":
        peak_lr = base_lr * num_devices
    elif lr_scaling == "sqrt":
        peak_lr = base_lr * (num_devices ** 0.5)
    else:
        peak_lr = base_lr

    if warmup_steps <= 0:
        # No warmup — direct decay from peak_lr
        return _create_decay_schedule(peak_lr, total_steps, scheduler)

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )

    decay_steps = max(total_steps - warmup_steps, 1)

    if scheduler == "cosine":
        decay_fn = optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=decay_steps,
        )
    elif scheduler == "linear":
        decay_fn = optax.linear_schedule(
            init_value=peak_lr,
            end_value=0.0,
            transition_steps=decay_steps,
        )
    else:  # constant
        decay_fn = optax.constant_schedule(peak_lr)

    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )


def _create_decay_schedule(
    peak_lr: float,
    total_steps: int,
    scheduler: str,
) -> optax.Schedule:
    """Create decay-only schedule (no warmup)."""
    if scheduler == "cosine":
        return optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=total_steps,
        )
    elif scheduler == "linear":
        return optax.linear_schedule(
            init_value=peak_lr,
            end_value=0.0,
            transition_steps=total_steps,
        )
    else:  # constant
        return optax.constant_schedule(peak_lr)


def create_optimizer(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    num_devices: int = 1,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    scheduler: str = "cosine",
    lr_scaling: str = "linear",
) -> optax.GradientTransformation:
    """Create optimizer with LR schedule, gradient clipping, and weight decay.

    Chain: clip_by_global_norm -> adamw(lr_schedule, weight_decay)

    Args:
        base_lr: Base learning rate.
        warmup_steps: Warmup steps (0 = no warmup).
        total_steps: Total training steps.
        num_devices: SPMD device count (for LR scaling).
        weight_decay: AdamW weight decay coefficient.
        gradient_clip: Global gradient norm clipping threshold (0 = no clip).
        scheduler: LR decay type: 'cosine', 'linear', 'constant'.
        lr_scaling: LR scaling rule: 'linear', 'sqrt', 'none'.

    Returns:
        optax.GradientTransformation (chain of clip + adamw).
    """
    lr_schedule = create_lr_schedule(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        num_devices=num_devices,
        scheduler=scheduler,
        lr_scaling=lr_scaling,
    )

    components = []

    if gradient_clip > 0:
        components.append(optax.clip_by_global_norm(gradient_clip))

    components.append(
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    )

    return optax.chain(*components)
