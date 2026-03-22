import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_roc_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute macro-averaged ROC-AUC, skipping classes with no positive samples."""
    scores = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            scores.append(roc_auc_score(targets[:, i], preds[:, i]))
    return float(np.mean(scores)) if scores else 0.0
