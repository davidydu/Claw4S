"""Shadow-model membership inference attack (Shokri et al. 2017).

Attack strategy:
  1. Train N shadow models that mimic the target model's behavior.
     Each shadow model has known member/non-member splits.
  2. For each shadow model, collect prediction vectors (softmax outputs)
     for members (label=1) and non-members (label=0).
  3. Train a binary attack classifier on these (prediction_vector, label) pairs.
  4. Apply the attack classifier to the target model's predictions to
     infer membership.

Attack features: softmax probability vector from the model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import generate_gaussian_clusters, make_member_nonmember_split
from src.dp_sgd import DPConfig
from src.model import MLP
from src.train import train_model


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC from true labels and predicted scores.

    Uses the trapezoidal rule on the ROC curve.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_score: Predicted scores/probabilities.

    Returns:
        ROC AUC value in [0, 1].
    """
    # Sort by decreasing score
    desc = np.argsort(-y_score)
    y_true_sorted = y_true[desc]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined; return random

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoidal rule
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr

    return float(auc)


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy in [0, 1].
    """
    return float(np.mean(y_true == y_pred))


def get_attack_features(
    model: MLP,
    dataset: TensorDataset,
) -> np.ndarray:
    """Extract rich features for the membership inference attack.

    Features per sample (concatenated):
      - Softmax probability vector (num_classes values)
      - Max confidence (1 value)
      - Prediction entropy (1 value)
      - Cross-entropy loss on true label (1 value)
      - Whether prediction is correct (1 value)

    Total features = num_classes + 4.

    Args:
        model: Trained model.
        dataset: Input dataset.

    Returns:
        Array of shape (n_samples, num_classes + 4).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_features = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)

            # Max confidence
            max_conf = probs.max(dim=1).values.unsqueeze(1)

            # Prediction entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)

            # Cross-entropy loss on true label
            loss_per_sample = -log_probs[torch.arange(len(batch_y)), batch_y].unsqueeze(1)

            # Correctness
            correct = (probs.argmax(dim=1) == batch_y).float().unsqueeze(1)

            features = torch.cat([probs, max_conf, entropy, loss_per_sample, correct], dim=1)
            all_features.append(features.numpy())

    return np.concatenate(all_features, axis=0)


def train_shadow_models(
    n_shadows: int = 3,
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    hidden_dim: int = 128,
    cluster_std: float = 2.5,
    dp_config: DPConfig | None = None,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 0.1,
    base_seed: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Train shadow models and collect attack training data.

    Each shadow model is trained on a fresh random dataset with a known
    member/non-member split. We collect prediction vectors for both
    members and non-members.

    Args:
        n_shadows: Number of shadow models.
        n_samples: Samples per shadow dataset.
        n_features: Feature dimensionality.
        n_classes: Number of classes.
        hidden_dim: Hidden layer width.
        cluster_std: Gaussian cluster std for data generation.
        dp_config: DP-SGD config to match target model training.
        epochs: Training epochs per shadow model.
        batch_size: Batch size.
        lr: Learning rate.
        base_seed: Base seed (each shadow uses base_seed + i).

    Returns:
        attack_X: Feature matrix for attack classifier, shape (N, n_classes).
            Each row is a softmax prediction vector.
        attack_y: Labels, shape (N,). 1 = member, 0 = non-member.
    """
    all_X = []
    all_y = []

    for i in range(n_shadows):
        shadow_seed = base_seed + i

        # Generate fresh data for this shadow
        X, y = generate_gaussian_clusters(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            cluster_std=cluster_std,
            seed=shadow_seed,
        )
        member_ds, nonmember_ds = make_member_nonmember_split(
            X, y, member_ratio=0.5, seed=shadow_seed
        )

        # Train shadow model with same config as target
        shadow_model, _ = train_model(
            train_dataset=member_ds,
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_classes=n_classes,
            dp_config=dp_config,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=shadow_seed + 10000,
        )

        # Collect predictions for members (label=1)
        member_preds = get_attack_features(shadow_model, member_ds)
        all_X.append(member_preds)
        all_y.append(np.ones(len(member_preds), dtype=np.int64))

        # Collect predictions for non-members (label=0)
        nonmember_preds = get_attack_features(shadow_model, nonmember_ds)
        all_X.append(nonmember_preds)
        all_y.append(np.zeros(len(nonmember_preds), dtype=np.int64))

    attack_X = np.concatenate(all_X, axis=0)
    attack_y = np.concatenate(all_y, axis=0)
    return attack_X, attack_y


class AttackClassifier(nn.Module):
    """Binary classifier for membership inference.

    Takes enriched feature vectors (softmax + confidence + entropy + loss +
    correctness) as input and outputs a membership probability.
    """

    def __init__(self, input_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_attack_classifier(
    attack_X: np.ndarray,
    attack_y: np.ndarray,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 42,
) -> AttackClassifier:
    """Train the attack classifier on shadow model data.

    Args:
        attack_X: Prediction vectors from shadow models.
        attack_y: Membership labels (1=member, 0=non-member).
        epochs: Training epochs.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        Trained attack classifier.
    """
    torch.manual_seed(seed)
    input_dim = attack_X.shape[1]
    clf = AttackClassifier(input_dim=input_dim)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tensor = torch.tensor(attack_X, dtype=torch.float32)
    y_tensor = torch.tensor(attack_y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    clf.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = clf(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()

    return clf


def run_attack(
    attack_clf: AttackClassifier,
    target_model: MLP,
    member_dataset: TensorDataset,
    nonmember_dataset: TensorDataset,
) -> dict:
    """Run membership inference attack against a target model.

    Args:
        attack_clf: Trained attack classifier.
        target_model: Target model to attack.
        member_dataset: Known members of the target model's training set.
        nonmember_dataset: Known non-members.

    Returns:
        Dictionary with attack metrics:
            - attack_auc: ROC-AUC of the attack
            - attack_accuracy: Accuracy of the attack
            - n_members: Number of member samples
            - n_nonmembers: Number of non-member samples
    """
    attack_clf.eval()
    target_model.eval()

    # Get prediction vectors from target model
    member_preds = get_attack_features(target_model, member_dataset)
    nonmember_preds = get_attack_features(target_model, nonmember_dataset)

    # Combine
    X_attack = np.concatenate([member_preds, nonmember_preds], axis=0)
    y_true = np.concatenate([
        np.ones(len(member_preds)),
        np.zeros(len(nonmember_preds)),
    ])

    # Run attack classifier
    with torch.no_grad():
        X_tensor = torch.tensor(X_attack, dtype=torch.float32)
        attack_logits = attack_clf(X_tensor).numpy()
        attack_probs = 1.0 / (1.0 + np.exp(-attack_logits))  # sigmoid
        attack_preds = (attack_probs >= 0.5).astype(np.int64)

    auc = _roc_auc_score(y_true, attack_probs)
    acc = _accuracy_score(y_true, attack_preds)

    return {
        "attack_auc": float(auc),
        "attack_accuracy": float(acc),
        "n_members": int(len(member_preds)),
        "n_nonmembers": int(len(nonmember_preds)),
    }
