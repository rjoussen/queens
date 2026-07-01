#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Gaussian kernel density estimate distribution."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class GaussianKDE(Continuous):
    """Multivariate Gaussian kernel density estimate.

    Samples are expected row-wise, consistent with the other QUEENS distributions. Internally,
    SciPy represents the KDE as a weighted mixture of equally shaped Gaussian kernels.
    """

    @log_init_args
    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        bandwidth: str | float | None = None,
    ) -> None:
        """Initialize a Gaussian kernel density estimate.

        Args:
            samples: Samples used to estimate the density, with one sample per row.
            weights: Non-negative sample weights. They are normalized by SciPy.
            bandwidth: KDE bandwidth method or scalar factor. See ``scipy.stats.gaussian_kde``.
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2:
            raise ValueError("Samples must be a one- or two-dimensional array.")
        if samples.shape[0] < 2:
            raise ValueError("At least two samples are required to fit a Gaussian KDE.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Samples must contain only finite values.")

        weights_array = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        if weights_array is not None:
            if weights_array.size != samples.shape[0]:
                raise ValueError("Number of weights does not match the number of samples.")
            if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0):
                raise ValueError("Weights must be finite and non-negative.")
            if not np.any(weights_array > 0):
                raise ValueError("At least one weight must be positive.")

        self.samples = samples
        self.bandwidth = bandwidth
        self.scipy_kde = gaussian_kde(
            samples.T,
            weights=weights_array,
            bw_method=bandwidth,
        )
        self.weights = self.scipy_kde.weights

        mean = np.average(samples, axis=0, weights=self.weights)
        centered_samples = samples - mean
        sample_covariance = np.einsum(
            "n,ni,nj->ij", self.weights, centered_samples, centered_samples
        )
        covariance = sample_covariance + self.scipy_kde.covariance
        super().__init__(mean=mean, covariance=covariance, dimension=samples.shape[1])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the cumulative distribution function."""
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        lower = np.full(self.dimension, -np.inf)
        return np.array([self.scipy_kde.integrate_box(lower, point) for point in x])

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples from the estimated density."""
        return self.scipy_kde.resample(num_draws).T

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the log probability density."""
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        return self.scipy_kde.logpdf(x.T)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the probability density."""
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        return self.scipy_kde.pdf(x.T)

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the gradient of the log probability density."""
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        differences = x[:, np.newaxis, :] - self.samples[np.newaxis, :, :]
        kernel_logpdf = np.log(self.weights)[np.newaxis, :] - 0.5 * np.einsum(
            "mni,ij,mnj->mn",
            differences,
            self.scipy_kde.inv_cov,
            differences,
        )
        responsibilities = np.exp(kernel_logpdf - logsumexp(kernel_logpdf, axis=1, keepdims=True))
        kernel_gradients = -np.einsum("ij,mnj->mni", self.scipy_kde.inv_cov, differences)
        return np.einsum("mn,mni->mi", responsibilities, kernel_gradients)

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Raise because the inverse CDF is not available for Gaussian KDEs."""
        raise NotImplementedError("PPF not available for Gaussian KDE distributions.")
