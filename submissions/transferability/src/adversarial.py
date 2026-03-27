"""FGSM adversarial example generation and transfer evaluation."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.models import MLP


def fgsm_attack(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.3,
) -> torch.Tensor:
    """Generate FGSM adversarial examples.

    Applies the Fast Gradient Sign Method (Goodfellow et al., 2015):
        X_adv = X + epsilon * sign(grad_X(loss))

    Args:
        model: Source model (must be in eval mode externally).
        X: Input tensor of shape (N, D), float32.
        y: True labels of shape (N,), int64.
        epsilon: Perturbation magnitude.

    Returns:
        Adversarial examples X_adv of same shape as X.
    """
    model.eval()
    X_input = X.clone().detach().requires_grad_(True)
    logits = model(X_input)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    if X_input.grad is None:
        raise RuntimeError("Gradient computation failed: X_input.grad is None")

    perturbation = epsilon * X_input.grad.sign()
    X_adv = X_input.detach() + perturbation
    return X_adv


def evaluate_clean_accuracy(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute clean accuracy of model on (X, y).

    Args:
        model: Classifier in eval mode.
        X: Input tensor.
        y: True labels.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def compute_transfer_rate(
    source_model: MLP,
    target_model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.3,
) -> dict:
    """Compute adversarial transfer rate from source to target.

    Generates FGSM adversarial examples on source_model and evaluates
    what fraction of them also fool target_model.

    Only considers samples that source correctly classifies on clean data
    AND that source adversarials successfully fool the source model.
    Transfer rate = fraction of those successful adversarials that also
    fool the target model.

    Args:
        source_model: Model used to generate adversarial examples.
        target_model: Model used to evaluate transferability.
        X: Clean input tensor.
        y: True labels.
        epsilon: FGSM perturbation magnitude.

    Returns:
        Dict with:
            - 'transfer_rate': fraction of source-adversarials that fool target
            - 'source_clean_acc': source accuracy on clean data
            - 'target_clean_acc': target accuracy on clean data
            - 'source_adv_acc': source accuracy on adversarial data
            - 'target_adv_acc': target accuracy on adversarial data
            - 'n_successful_source_advs': number of adversarials that fool source
    """
    source_model.eval()
    target_model.eval()

    # Clean accuracies
    source_clean_acc = evaluate_clean_accuracy(source_model, X, y)
    target_clean_acc = evaluate_clean_accuracy(target_model, X, y)

    # Generate adversarial examples on source
    X_adv = fgsm_attack(source_model, X, y, epsilon=epsilon)

    # Source predictions on clean and adversarial
    with torch.no_grad():
        source_clean_preds = source_model(X).argmax(dim=1)
        source_adv_preds = source_model(X_adv).argmax(dim=1)
        target_adv_preds = target_model(X_adv).argmax(dim=1)

    # Source accuracy on adversarial data
    source_adv_acc = (source_adv_preds == y).float().mean().item()
    target_adv_acc = (target_adv_preds == y).float().mean().item()

    # Identify samples where source was correct on clean but wrong on adversarial
    source_correct_clean = source_clean_preds == y
    source_fooled = source_adv_preds != y
    successful_advs = source_correct_clean & source_fooled

    n_successful = successful_advs.sum().item()

    if n_successful == 0:
        transfer_rate = 0.0
    else:
        # Of those successful adversarials, how many also fool the target?
        target_fooled_on_successful = (target_adv_preds[successful_advs] != y[successful_advs])
        transfer_rate = target_fooled_on_successful.float().mean().item()

    return {
        "transfer_rate": transfer_rate,
        "source_clean_acc": source_clean_acc,
        "target_clean_acc": target_clean_acc,
        "source_adv_acc": source_adv_acc,
        "target_adv_acc": target_adv_acc,
        "n_successful_source_advs": int(n_successful),
    }
