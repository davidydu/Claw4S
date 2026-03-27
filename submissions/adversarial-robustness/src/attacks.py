"""
Adversarial attack implementations: FGSM and PGD.

References:
- FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.
- PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
"""

import torch
import torch.nn as nn


def fgsm_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                epsilon: float) -> torch.Tensor:
    """Generate adversarial examples using FGSM (Fast Gradient Sign Method).

    Perturbs inputs in the direction of the gradient sign, scaled by epsilon.
    x_adv = x + epsilon * sign(grad_x(loss(model(x), y)))

    Args:
        model: Target model (must be in eval mode externally).
        X: Clean inputs, shape (N, D).
        y: True labels, shape (N,).
        epsilon: Perturbation magnitude (L-inf norm).

    Returns:
        Adversarial examples, shape (N, D).
    """
    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)

    logits = model(X_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    # FGSM perturbation
    grad_sign = X_adv.grad.data.sign()
    X_adv = X_adv.detach() + epsilon * grad_sign

    return X_adv.detach()


def pgd_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
               epsilon: float, step_size: float | None = None,
               n_steps: int = 10) -> torch.Tensor:
    """Generate adversarial examples using PGD (Projected Gradient Descent).

    Iteratively applies FGSM steps and projects back into the epsilon-ball.

    Args:
        model: Target model (must be in eval mode externally).
        X: Clean inputs, shape (N, D).
        y: True labels, shape (N,).
        epsilon: Maximum perturbation magnitude (L-inf norm).
        step_size: Step size per iteration. Defaults to epsilon / 4.
        n_steps: Number of PGD iterations.

    Returns:
        Adversarial examples, shape (N, D).
    """
    if step_size is None:
        step_size = epsilon / 4.0

    model.eval()
    X_adv = X.clone().detach()

    for _ in range(n_steps):
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Gradient step
        grad_sign = X_adv.grad.data.sign()
        X_adv = X_adv.detach() + step_size * grad_sign

        # Project back into L-inf epsilon-ball around original X
        perturbation = X_adv - X
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        X_adv = (X + perturbation).detach()

    return X_adv.detach()


def evaluate_robust(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                    attack_fn, epsilon: float, **attack_kwargs) -> float:
    """Evaluate model accuracy on adversarial examples.

    Args:
        model: Trained model.
        X: Clean test inputs.
        y: True test labels.
        attack_fn: Attack function (fgsm_attack or pgd_attack).
        epsilon: Perturbation magnitude.
        **attack_kwargs: Additional keyword arguments for the attack function.

    Returns:
        Robust accuracy as a float in [0, 1].
    """
    model.eval()
    X_adv = attack_fn(model, X, y, epsilon=epsilon, **attack_kwargs)

    with torch.no_grad():
        logits = model(X_adv)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()

    return accuracy


EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.5]
