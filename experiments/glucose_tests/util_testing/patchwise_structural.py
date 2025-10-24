import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union
from neuralforecast.losses.pytorch import BasePointLoss, _divide_no_nan, _weighted_mean 

class _PSLoss_Corr(nn.Module):
    """
    [cite_start]Computes the Patch-wise Correlation Loss (L_Corr) [cite: 145-147].
    Expects patches as input.
    """

    def __init__(self):
        super(_PSLoss_Corr, self).__init__()

    def forward(
        self,
        y_patches: torch.Tensor,
        y_hat_patches: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> torch.Tensor:
        dim = -1  # Patch dimension P
        mu_y_expanded = torch.mean(y_patches, dim=dim, keepdim=True)
        mu_y_hat_expanded = torch.mean(y_hat_patches, dim=dim, keepdim=True)
        y_centered = y_patches - mu_y_expanded
        y_hat_centered = y_hat_patches - mu_y_hat_expanded
        covariance = torch.mean(y_centered * y_hat_centered, dim=dim)
        std_y = torch.std(y_patches, dim=dim, unbiased=False)
        std_y_hat = torch.std(y_hat_patches, dim=dim, unbiased=False)
        pcc = _divide_no_nan(covariance, (std_y * std_y_hat) + 1e-6)
        loss_corr_patches = 1.0 - pcc
        return _weighted_mean(losses=loss_corr_patches, weights=patch_weights)


class _PSLoss_Var(nn.Module):
    """
    [cite_start]Computes the Patch-wise Variance Loss (L_Var) using KL-Divergence [cite: 163-164].
    Expects patches as input.
    """

    def __init__(self):
        super(_PSLoss_Var, self).__init__()

    def forward(
        self,
        y_patches: torch.Tensor,
        y_hat_patches: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> torch.Tensor:
        dim = -1  # Patch dimension P
        phi_y = F.softmax(y_patches, dim=dim)
        log_phi_y_hat = F.log_softmax(y_hat_patches, dim=dim)
        kl_div_patches = F.kl_div(
            log_phi_y_hat, phi_y, reduction="none", log_target=False
        )
        loss_var_patches = torch.sum(kl_div_patches, dim=dim)
        return _weighted_mean(losses=loss_var_patches, weights=patch_weights)


class _PSLoss_Mean(nn.Module):
    """
    Computes the Patch-wise Mean Loss (L_Mean).
    Expects patches as input.
    """

    def __init__(self):
        super(_PSLoss_Mean, self).__init__()

    def forward(
        self,
        y_patches: torch.Tensor,
        y_hat_patches: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> torch.Tensor:
        dim = -1  # Patch dimension P
        mu_y = torch.mean(y_patches, dim=dim)
        mu_y_hat = torch.mean(y_hat_patches, dim=dim)
        loss_mean_patches = torch.abs(mu_y - mu_y_hat)
        return _weighted_mean(losses=loss_mean_patches, weights=patch_weights)


class PSLoss_GDW(BasePointLoss):
    """
    Computes and stores Patch-wise Structural (PS) Loss components.
    This class is designed to be used with the GDWLossMixin.

    The `__call__` (forward) pass computes L_mse, L_corr, L_var, and L_mean,
    stores them as attributes, and returns L_mse to satisfy the
    Lightning `training_step`. The actual backward pass is orchestrated
    by the GDWLossMixin.
    """

    def __init__(
        self,
        lambda_ps: float = 1.0,
        delta_patch: int = 24,
        horizon_weight=None,
    ):
        super(PSLoss_GDW, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
        self.lambda_ps = lambda_ps
        self.delta_patch = delta_patch
        self.corr_loss = _PSLoss_Corr()
        self.var_loss = _PSLoss_Var()
        self.mean_loss = _PSLoss_Mean()
        self.eps = 1e-8  # Epsilon for numerical stability

        # Placeholders for computed losses and logging
        self.L_mse = None
        self.L_corr = None
        self.L_var = None
        self.L_mean = None
        self.y_detached = None
        self.y_hat_detached = None
        self.alpha_detached = None
        self.beta_detached = None
        self.gamma_detached = None
        self.L_ps_weighted_detached = None

    def _get_patch_length(self, y: torch.Tensor) -> Tuple[int, int]:
        y_avg = torch.mean(y, dim=(0, 2))
        T = y_avg.shape[0]
        if T < 3:
            return max(1, T), max(1, int(np.ceil(T / 2.0)))
        
        y_fft = torch.fft.fft(y_avg)
        amplitudes = torch.abs(y_fft[1 : T // 2 + 1])
        dominant_freq_idx = torch.argmax(amplitudes) + 1
        p = torch.ceil(T / dominant_freq_idx.float())
        P = torch.min(
            p, torch.tensor(self.delta_patch, dtype=p.dtype, device=y.device)
        ).long().item()
        S = torch.ceil(torch.tensor(P / 2.0)).long().item()
        return max(1, P), max(1, S)

    def _unfold_to_patches(
        self, x: torch.Tensor, P: int, S: int
    ) -> torch.Tensor:
        """Converts [B, T, N] -> [B, N_patches, N_series, P]"""
        x_permuted = x.permute(0, 2, 1)  # [B, N_series, T]
        patches = x_permuted.unfold(dimension=2, size=P, step=S)
        return patches.permute(0, 2, 1, 3)  # [B, N_patches, N_series, P]

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        y_insample: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Computes and stores all four loss components.
        Returns L_mse as the "dummy" loss for Lightning's trainer.
        """
        y_hat_mapped = self.domain_map(y_hat)
        weights = self._compute_weights(y=y, mask=mask)

        # 1. Compute L_MSE
        loss_mse_points = (y - y_hat_mapped) ** 2
        self.L_mse = _weighted_mean(losses=loss_mse_points, weights=weights)

        # 2. Get Patches
        P, S = self._get_patch_length(y)
        y_patches = self._unfold_to_patches(y, P, S)
        y_hat_patches = self._unfold_to_patches(y_hat_mapped, P, S)
        weight_patches = self._unfold_to_patches(weights, P, S)
        patch_weights = torch.mean(weight_patches, dim=-1)

        # 3. Compute and store PS Loss Components (as scalars)
        self.L_corr = self.corr_loss(y_patches, y_hat_patches, patch_weights)
        self.L_var = self.var_loss(y_patches, y_hat_patches, patch_weights)
        self.L_mean = self.mean_loss(y_patches, y_hat_patches, patch_weights)

        # [cite_start]4. Store detached tensors needed for scaling factors [cite: 208-216]
        self.y_detached = y.detach()
        self.y_hat_detached = y_hat_mapped.detach()
        
        # 5. Return the "dummy" loss for the training_step
        return self.L_mse

    def _get_grad_norm(self, loss: torch.Tensor, params: list) -> torch.Tensor:
        """Helper to compute grad norm."""
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        grad_vec = torch.cat([g.view(-1) for g in grads if g is not None])
        return torch.norm(grad_vec, 2)

    def compute_gdw_total_loss(self, model_output_layer: nn.Module) -> torch.Tensor:
        """
        Computes the GDW weights and returns the final, weighted loss tensor.
        This is called by the GDWLossMixin during the `backward` hook.
        """
        if self.L_mse is None:
            raise RuntimeError("Loss components not computed. Call forward first.")

        params = list(model_output_layer.parameters())
        if not params:
             raise ValueError("Could not retrieve parameters from model_output_layer. "
                              "Ensure the layer is correct and has parameters.")

        # [cite_start]1. Compute Gradient Norms [cite: 188]
        G_corr = self._get_grad_norm(self.L_corr, params)
        G_var = self._get_grad_norm(self.L_var, params)
        G_mean = self._get_grad_norm(self.L_mean, params)

        # [cite_start]2. Compute GDW Weights $\alpha, \beta, \gamma$ [cite: 199-207]
        G_avg = (G_corr + G_var + G_mean) / 3.0
        alpha = G_avg / (G_corr + self.eps)
        beta = G_avg / (G_var + self.eps)
        gamma = G_avg / (G_mean + self.eps)
        
        # [cite_start]3. Compute Scaling Factors c and v [cite: 208-216]
        y_flat = self.y_detached.view(-1)
        y_hat_flat = self.y_hat_detached.view(-1)
        
        cov_matrix = torch.cov(torch.stack([y_flat, y_hat_flat]))
        sigma_y_y_hat = cov_matrix[0, 1]
        sigma_y_sq = cov_matrix[0, 0]
        sigma_y_hat_sq = cov_matrix[1, 1]
        
        c = (2 * sigma_y_y_hat) / (sigma_y_sq + sigma_y_hat_sq + self.eps)
        v = (2 * torch.sqrt(sigma_y_sq) * torch.sqrt(sigma_y_hat_sq)) / (sigma_y_sq + sigma_y_hat_sq + self.eps)
        
        # Apply scaling factors to gamma
        gamma_scaled = gamma * (1 + c) * (1 + v)

        # 4. Detach weights for the final backward pass
        self.alpha_detached = alpha.detach()
        self.beta_detached = beta.detach()
        self.gamma_detached = gamma_scaled.detach()

        # [cite_start]5. Compute Final Weighted Loss [cite: 175, 218]
        self.L_ps_weighted_detached = (
            self.alpha_detached * self.L_corr
            + self.beta_detached * self.L_var
            + self.gamma_detached * self.L_mean
        )
        
        L_total = self.L_mse + self.lambda_ps * self.L_ps_weighted_detached
        
        return L_total