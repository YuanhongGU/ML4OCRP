"""Neural modules used by the OCRP (Optimal CRISPR Ratio Prediction) pipeline.

This module provides:

* ``DeepKernelGP`` — a deep-kernel Gaussian process: an MLP maps inputs into a
  latent feature space, then an RBF kernel is applied there (ExactGP).
* ``RatioFeatureEncoder`` — a small MLP that maps a scalar ratio to a
  ``d_model``-dimensional embedding, plus an auxiliary score head used only
  during pretraining.
"""

import torch
import torch.nn as nn
import gpytorch


class DeepKernelGP(gpytorch.models.ExactGP):
    """Deep-kernel Gaussian process on top of GPyTorch ExactGP.

    The input is first projected by a two-layer MLP. Mean and covariance are
    then computed in that latent space, so the RBF kernel operates on learned
    features rather than on the raw ratio vector.

    :param train_x: Training inputs of shape ``(N, input_dim)``.
    :type train_x: torch.Tensor
    :param train_y: Training targets of shape ``(N,)``.
    :type train_y: torch.Tensor
    :param likelihood: Gaussian observation model used by ExactGP.
    :type likelihood: gpytorch.likelihoods.GaussianLikelihood
    :param input_dim: Dimensionality of each input vector (ratio + encoder features).
    :type input_dim: int
    :param feat_dim: Dimensionality of the latent space after the MLP extractor.
    :type feat_dim: int
    :param hidden_dim: Width of the feature-extractor hidden layer. Defaults to 32.
        Override from the training CLI with ``--hidden_dim``.
    :type hidden_dim: int
    """

    def __init__(self,
                 train_x    :   torch.Tensor,
                 train_y    :   torch.Tensor,
                 likelihood :   gpytorch.likelihoods.GaussianLikelihood,
                 input_dim  :   int,
                 feat_dim   :   int,
                 hidden_dim :   int = 32):
        super().__init__(train_x, train_y, likelihood)
        # Map raw (ratio, encoder_feat) into a compact latent space for the kernel.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.mean_module = gpytorch.means.ZeroMean()
        # ARD RBF: one length-scale per latent dimension.
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim)
        )

    def forward(self, x):
        """Return the GP posterior over ``x`` in latent feature space.

        :param x: Input tensor of shape ``(N, input_dim)``.
        :type x: torch.Tensor
        :returns: Predictive distribution whose mean/covariance are computed
            from extracted features.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        features = self.feature_extractor(x)
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class RatioFeatureEncoder(nn.Module):
    """MLP encoder from a scalar ratio to a ``d_model``-dimensional embedding.

    ``forward`` also produces an auxiliary scalar score used only while
    pretraining the encoder against observed scores. After pretraining, call
    ``get_feature`` so the frozen embedding can be concatenated with the raw
    ratio as GP input.

    :param d_model: Size of the ratio embedding. Defaults to 16.
        Override from the training CLI with ``--d_model``.
    :type d_model: int
    :param hidden_dim: Width of the first hidden layer. Defaults to 32.
        Override from the training CLI with ``--hidden_dim``.
    :type hidden_dim: int
    """

    def __init__(self, d_model=16, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        # Auxiliary head: the ratio embedding predicts the experimental score during pretraining.
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """Encode ratio and predict score (used during pretraining).

        :param x: Ratio tensor of shape ``(..., 1)``.
        :type x: torch.Tensor
        :returns: ``(feat, score)`` where ``feat`` has last dim ``d_model``
            and ``score`` has last dim 1.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        feat = self.fc2(x2)
        score = self.pred_head(feat)
        return feat, score

    def get_feature(self, x):
        """Return only the ratio embedding (no score head).

        Used after the encoder is frozen, so GP inputs do not depend on the
        auxiliary prediction head.

        :param x: Ratio tensor of shape ``(..., 1)``.
        :type x: torch.Tensor
        :returns: Embedding of shape ``(..., d_model)``.
        :rtype: torch.Tensor
        """
        h = self.relu(self.fc1(x))
        return self.fc2(h)
