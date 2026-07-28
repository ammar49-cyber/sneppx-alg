"""Knowledge distillation utilities for model compression."""

from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
import copy

from .tensor import Tensor
from .advanced_ops import softmax, log_softmax, kl_divergence
from .pruning import magnitude_prune

# =========================================================================
# Distillation Loss Functions
# =========================================================================


def kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
    labels: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Knowledge distillation loss combining hard and soft targets.

    L = alpha * CE(student_logits, labels) + (1-alpha) * T^2 * KL(teacher || student)

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Softmax temperature
        alpha: Weight for hard target CE loss (1-alpha for soft targets)
        labels: Ground truth labels (optional, if None use only distillation)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    from .advanced_ops import softmax, log_softmax

    s_logits = np.asarray(student_logits.data, dtype=np.float64)
    t_logits = np.asarray(teacher_logits.data, dtype=np.float64)

    # Soft targets with temperature
    t_soft = softmax(t_logits / temperature, axis=-1)
    s_log_soft = log_softmax(s_logits / temperature, axis=-1)

    # KL divergence
    kl = np.sum(t_soft * (np.log(t_soft + 1e-12) - s_log_soft), axis=-1)
    kl_loss = np.mean(kl) * (temperature**2)

    if labels is not None:
        # Hard target CE loss
        labels_np = np.asarray(labels.data, dtype=np.int64)
        s_soft = softmax(s_logits, axis=-1)
        ce_loss = -np.mean(np.log(s_soft[np.arange(len(labels_np)), labels_np] + 1e-12))
        loss = alpha * ce_loss + (1 - alpha) * kl_loss * (temperature**2)
    else:
        loss = kl_loss * (temperature**2)

    if reduction == "sum":
        loss = loss * student_logits.shape[0]

    return Tensor.from_numpy(np.array(loss, dtype=np.float32))


def attention_transfer_loss(
    student_attn: List[Tensor],
    teacher_attn: List[Tensor],
    beta: float = 1.0,
) -> Tensor:
    """Attention transfer loss (AT) - matches attention maps.

    L_AT = beta * sum ||A_s - A_t||_F^2
    """
    if len(student_attn) != len(teacher_attn):
        raise ValueError("Number of attention layers must match")

    total_loss = 0.0
    for s_attn, t_attn in zip(student_attn, teacher_attn):
        s = np.asarray(s_attn.data, dtype=np.float64)
        t = np.asarray(t_attn.data, dtype=np.float64)
        total_loss += np.mean((s - t) ** 2)

    return Tensor.from_numpy(
        np.array(total_loss * student_attn[0].shape[0], dtype=np.float32)
    )


def feature_matching_loss(
    student_features: List[Tensor],
    teacher_features: List[Tensor],
    weights: Optional[List[float]] = None,
) -> Tensor:
    """Feature matching loss (FitNets style).

    L_FM = sum ||F_s - F_t||_2^2
    """
    if len(student_features) != len(teacher_features):
        raise ValueError("Number of feature layers must match")

    if weights is None:
        weights = [1.0] * len(student_features)

    total_loss = 0.0
    for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
        s = np.asarray(s_feat.data, dtype=np.float64)
        t = np.asarray(t_feat.data, dtype=np.float64)
        total_loss += weights[i] * np.mean((s - t) ** 2)

    return Tensor.from_numpy(np.array(total_loss, dtype=np.float32))


def correlation_congruence_loss(
    student_feat: Tensor,
    teacher_feat: Tensor,
) -> Tensor:
    """Correlation Congruence (CC) loss - matches feature correlations.

    L_CC = ||C_s - C_t||_F^2 where C = F^T F / N
    """
    s = np.asarray(student_feat.data, dtype=np.float64)  # [N, C, H, W] or [N, D]
    t = np.asarray(teacher_feat.data, dtype=np.float64)

    # Flatten spatial dims
    if s.ndim == 4:
        N, C, H, W = s.shape
        s = s.reshape(N, C, -1).transpose(0, 2, 1)  # [N, HW, C]
        t = t.reshape(N, C, -1).transpose(0, 2, 1)

    # Correlation matrices
    c_s = np.matmul(s.transpose(0, 2, 1), s) / s.shape[1]
    c_t = np.matmul(t.transpose(0, 2, 1), t) / t.shape[1]

    loss = np.mean((c_s - c_t) ** 2)
    return Tensor.from_numpy(np.array(loss, dtype=np.float32))


def hint_loss(
    student_hint: Tensor,
    teacher_hint: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Hint-based training loss (FitNets)."""
    s = np.asarray(student_hint.data, dtype=np.float64)
    t = np.asarray(teacher_hint.data, dtype=np.float64)
    loss = beta * np.mean((s - t) ** 2)
    return Tensor.from_numpy(np.array(loss, dtype=np.float32))


def crd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    negative_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Contrastive Representation Distillation (CRD) loss.

    Uses NCE (Noise Contrastive Estimation) with negative samples.
    """
    s = np.asarray(student_logits.data, dtype=np.float64)
    t = np.asarray(teacher_logits.data, dtype=np.float64)
    n = np.asarray(negative_logits.data, dtype=np.float64)

    # Positive similarity
    pos = np.sum(s * t, axis=-1) / student_logits.data.shape[-1]

    # Negative similarities
    neg = np.mean(
        np.sum(s[:, None, :] * n, axis=-1) / student_logits.data.shape[-1], axis=1
    )

    # NCE loss
    log_prob = np.log(
        np.exp(pos) / (np.exp(pos) + np.sum(np.exp(neg), axis=-1) + 1e-12)
    )
    loss = -np.mean(log_prob)

    return Tensor.from_numpy(np.array(loss, dtype=np.float32))


# =========================================================================
# Multi-Teacher Distillation
# =========================================================================


def multi_teacher_distillation_loss(
    student_logits: Tensor,
    teacher_logits_list: List[Tensor],
    weights: Optional[List[float]] = None,
    temperature: float = 4.0,
) -> Tensor:
    """Multi-teacher distillation loss with weighted ensemble.

    L = sum w_i * KL(teacher_i || student)
    """
    if weights is None:
        weights = [1.0 / len(teacher_logits_list)] * len(teacher_logits_list)

    total_loss = 0.0
    for w, t_logits in zip(weights, teacher_logits_list):
        total_loss += (
            w
            * kd_loss(student_logits, t_logits, temperature=temperature, alpha=0.0).data
        )

    return Tensor.from_numpy(np.array(total_loss, dtype=np.float32))


def ensemble_teacher_distillation(
    student_logits: Tensor,
    teacher_logits_list: List[Tensor],
    temperature: float = 4.0,
    method: str = "mean",
) -> Tensor:
    """Distill from ensemble of teachers."""
    if method == "mean":
        # Average teacher logits
        avg_teacher = sum(teacher_logits_list) / len(teacher_logits_list)
        return kd_loss(student_logits, avg_teacher, temperature=temperature, alpha=0.0)
    elif method == "vote":
        # Hard voting
        preds = [np.argmax(np.asarray(t.data), axis=-1) for t in teacher_logits_list]
        voted = np.stack(preds).mean(axis=0)  # soft vote
        # Convert back to logits-like
        voted_logits = np.eye(student_logits.shape[-1])[voted.astype(int)]
        return kd_loss(
            student_logits,
            Tensor.from_numpy(voted_logits),
            temperature=temperature,
            alpha=0.0,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# =========================================================================
# Self-Distillation
# =========================================================================


def self_distillation_loss(
    logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Self-distillation using dropout perturbations.

    L_self = KL(p || p') where p' is from different dropout mask
    """
    # Would need two forward passes with different dropout
    # This is a placeholder for the actual implementation
    raise NotImplementedError("Requires two forward passes with different dropout")


# =========================================================================
# Online Distillation
# =========================================================================


class OnlineDistillation:
    """Online distillation with multiple branches/heads."""

    def __init__(
        self,
        num_branches: int = 3,
        temperature: float = 4.0,
        distillation_weight: float = 0.5,
    ):
        self.num_branches = num_branches
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.branch_logits = []

    def compute_loss(
        self,
        branch_logits: List[Tensor],
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute online distillation loss among branches."""
        total_loss = 0.0

        for i in range(self.num_branches):
            for j in range(i + 1, self.num_branches):
                loss = kd_loss(
                    branch_logits[i],
                    branch_logits[j],
                    temperature=self.temperature,
                    alpha=0.0,
                )
                total_loss += loss.data

        # Average pairwise loss
        total_loss /= self.num_branches * (self.num_branches - 1) / 2

        if labels is not None:
            # Add CE loss for each branch
            from .advanced_ops import softmax

            ce_losses = []
            for logits in branch_logits:
                probs = softmax(logits, axis=-1)
                labels_np = np.asarray(labels.data, dtype=np.int64)
                ce = -np.mean(
                    np.log(probs.data[np.arange(len(labels_np)), labels_np] + 1e-12)
                )
                ce_losses.append(ce)
            total_loss += np.mean(ce_losses)

        return Tensor.from_numpy(np.array(total_loss, dtype=np.float32))


# =========================================================================
# Distillation with Pruning
# =========================================================================


class DistillationPruner:
    """Combines pruning with knowledge distillation for model compression."""

    def __init__(
        self,
        target_sparsity: float = 0.5,
        distill_temp: float = 4.0,
        distill_alpha: float = 0.5,
        epochs_per_stage: int = 5,
    ):
        self.target_sparsity = target_sparsity
        self.distill_temp = distill_temp
        self.distill_alpha = distill_alpha
        self.epochs_per_stage = epochs_per_stage

    def compress(
        self,
        student_weights: List[Tensor],
        teacher_weights: List[Tensor],
        train_loader: Any,
        optimizer: Any,
        sparsity_scheduler: Any,
    ) -> List[Tensor]:
        """Iterative pruning with knowledge distillation."""
        current_weights = copy.deepcopy(student_weights)

        for epoch in range(self.epochs_per_stage):
            # Get current sparsity target
            target_sparsity = sparsity_scheduler.get_sparsity()

            # Prune student
            pruned_weights = []
            for i, (s, t) in enumerate(zip(current_weights, teacher_weights)):
                pruned, mask = magnitude_prune(s, target_sparsity)
                current_weights[i] = pruned

            # Knowledge distillation training step
            # (Would need actual training loop here)

            sparsity_scheduler.step()

        return current_weights


# =========================================================================
# Distillation for Specific Architectures
# =========================================================================


def distill_bert(
    student_config: Any,
    teacher_config: Any,
    student_model: Any,
    teacher_model: Any,
    data_loader: Any,
    num_epochs: int = 10,
    temperature: float = 4.0,
    alpha: float = 0.5,
    distillation_type: str = "layer",
) -> Any:
    """BERT-specific distillation.

    Args:
        distillation_type: "layer", "logit", "attention", "embedding"
    """
    # This would implement BERT-specific distillation
    # including layer-wise distillation, attention transfer, etc.
    raise NotImplementedError("Use framework-specific implementation")


def distill_gpt(
    student_model: Any,
    teacher_model: Any,
    data_loader: Any,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Any:
    """GPT-style decoder distillation."""
    # For causal LM, use causal mask in attention
    # and standard logit distillation
    raise NotImplementedError("Use framework-specific implementation")


# =========================================================================
# Export
# =========================================================================

__all__ = [
    # Loss functions
    "kd_loss",
    "attention_transfer_loss",
    "feature_matching_loss",
    "correlation_congruence_loss",
    "hint_loss",
    "crd_loss",
    # Multi-teacher
    "multi_teacher_distillation_loss",
    "ensemble_teacher_distillation",
    # Online distillation
    "OnlineDistillation",
    # Pruning + distillation
    "DistillationPruner",
    "distillation_loss",
    # Architecture-specific
    "distill_bert",
    "distill_gpt",
]
