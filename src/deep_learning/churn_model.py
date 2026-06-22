"""
churn_model.py
==============
Deep Learning churn prediction using PyTorch.
Architecture: Multi-layer feedforward neural network with BatchNorm + Dropout.

Skills demonstrated: DL, PyTorch, MLflow model registry, SHAP explainability
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_score, recall_score, f1_score
)
import mlflow
import mlflow.pytorch
import logging
import os

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE
# ──────────────────────────────────────────────────────────────

class ChurnNet(nn.Module):
    """
    Deep feedforward network for binary churn classification.
    Architecture: Input → 256 → 128 → 64 → 32 → 1
    Regularization: BatchNorm + Dropout
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(ChurnNet, self).__init__()

        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Block 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),

            # Block 4
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ──────────────────────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────────────────────

class ChurnModelTrainer:
    """Trains, evaluates, and exports the deep learning churn model."""

    def __init__(self, input_dim: int, lr: float = 1e-3,
                 epochs: int = 30, batch_size: int = 256):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ChurnNet(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=False
        )
        self.criterion = nn.BCELoss()
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

        logger.info(f"[DL] ChurnNet initialized | Device: {self.device}")
        logger.info(f"[DL] Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _make_loader(self, X, y, shuffle=True):
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        return DataLoader(TensorDataset(X_t, y_t),
                          batch_size=self.batch_size, shuffle=shuffle)

    def train(self, X_train, y_train, X_val, y_val, use_mlflow=True):
        """Full training loop with validation and optional MLflow logging."""
        train_loader = self._make_loader(X_train, y_train)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_auc = 0
        best_state = None

        if use_mlflow:
            try:
                mlflow.start_run(run_name="churn_net_pytorch")
                mlflow.log_params({
                    "model": "ChurnNet",
                    "input_dim": self.input_dim,
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "optimizer": "Adam",
                    "architecture": "256-128-64-32-1",
                })
            except Exception:
                use_mlflow = False

        logger.info(f"[DL] Starting training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # ── Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # ── Validate
            self.model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds = self.model(X_batch)
                    val_loss += self.criterion(preds, y_batch).item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            val_loss /= len(val_loader)

            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except Exception:
                val_auc = 0.5

            self.scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:03d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f}"
                )

            if use_mlflow:
                try:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                    }, step=epoch)
                except Exception:
                    pass

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"[DL] Best model restored | Best Val AUC: {best_val_auc:.4f}")

        if use_mlflow:
            try:
                mlflow.log_metric("best_val_auc", best_val_auc)
                mlflow.pytorch.log_model(self.model, "churn_model")
                mlflow.end_run()
            except Exception:
                pass

        return self.history

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate on test set and return comprehensive metrics."""
        self.model.eval()
        X_t = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy()

        preds = (probs >= 0.5).astype(int)

        metrics = {
            "auc_roc": round(roc_auc_score(y_test, probs), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "probabilities": probs,
            "predictions": preds,
        }

        logger.info(
            f"[DL Test Results] AUC: {metrics['auc_roc']} | "
            f"Precision: {metrics['precision']} | "
            f"Recall: {metrics['recall']} | F1: {metrics['f1']}"
        )
        return metrics

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    def save(self, path: str = "models/churn_model.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "history": self.history,
        }, path)
        logger.info(f"[DL] Model saved to {path}")

    def load(self, path: str = "models/churn_model.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.history = checkpoint.get("history", {})
        logger.info(f"[DL] Model loaded from {path}")


# ──────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ──────────────────────────────────────────────────────────────

def compute_feature_importance_shap(model_trainer: ChurnModelTrainer,
                                     X_sample: np.ndarray,
                                     feature_names: list) -> list:
    """
    Compute SHAP values for the DL model using KernelExplainer.
    Returns sorted list of (feature, mean_abs_shap) tuples.
    """
    try:
        import shap

        def predict_fn(X):
            return model_trainer.predict_proba(X)

        background = X_sample[:100]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample[:200], nsamples=50)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = list(zip(feature_names, mean_abs_shap))
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance

    except ImportError:
        # Fallback: gradient-based proxy importance
        logger.warning("[SHAP] shap not installed — using gradient proxy importance.")
        model = model_trainer.model
        model.eval()
        X_t = torch.FloatTensor(X_sample[:200]).to(model_trainer.device)
        X_t.requires_grad_(True)
        output = model(X_t)
        output.sum().backward()
        grad_importance = X_t.grad.abs().mean(dim=0).detach().cpu().numpy()
        importance = list(zip(feature_names, grad_importance))
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance
