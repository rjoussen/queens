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
"""Tests for the Gaussian KDE distribution."""

import numpy as np
import pytest
from scipy.stats import norm

from queens.distributions.gaussian_kde import GaussianKDE


@pytest.fixture(name="distribution")
def fixture_distribution():
    """Create a weighted two-dimensional Gaussian KDE."""
    samples = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5], [2.0, 1.5]])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    return GaussianKDE(samples, weights=weights, bandwidth=0.7)


def test_pdf_and_logpdf(distribution):
    """Match the wrapped SciPy KDE."""
    points = np.array([[-0.5, 0.2], [0.5, 0.7], [1.5, 1.0]])
    np.testing.assert_allclose(distribution.pdf(points), distribution.scipy_kde.pdf(points.T))
    np.testing.assert_allclose(distribution.logpdf(points), distribution.scipy_kde.logpdf(points.T))


def test_mean_and_covariance(distribution):
    """Compute moments of the complete kernel mixture."""
    mean = np.average(distribution.samples, axis=0, weights=distribution.weights)
    centered = distribution.samples - mean
    sample_covariance = np.einsum("n,ni,nj->ij", distribution.weights, centered, centered)
    np.testing.assert_allclose(distribution.mean, mean)
    np.testing.assert_allclose(
        distribution.covariance, sample_covariance + distribution.scipy_kde.covariance
    )


def test_draw(distribution):
    """Draw row-wise samples with the correct dimension."""
    np.random.seed(42)
    samples = distribution.draw(20)
    assert samples.shape == (20, 2)
    assert np.all(np.isfinite(samples))


def test_grad_logpdf(distribution):
    """Match a finite-difference approximation of the score."""
    points = np.array([[-0.5, 0.2], [0.5, 0.7], [1.5, 1.0]])
    epsilon = 1e-6
    numerical_gradient = np.empty_like(points)
    for dimension in range(points.shape[1]):
        offset = np.zeros(points.shape[1])
        offset[dimension] = epsilon
        numerical_gradient[:, dimension] = (
            distribution.logpdf(points + offset) - distribution.logpdf(points - offset)
        ) / (2 * epsilon)
    np.testing.assert_allclose(distribution.grad_logpdf(points), numerical_gradient, rtol=1e-5)


def test_fit_normal_samples():
    """Approximate the density that generated a representative sample."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=1.0, scale=2.0, size=5_000)
    distribution = GaussianKDE(samples)
    points = np.array([-1.0, 1.0, 3.0])
    np.testing.assert_allclose(
        distribution.pdf(points), norm.pdf(points, loc=1.0, scale=2.0), rtol=0.12
    )


def test_cdf():
    """Evaluate a one-dimensional CDF."""
    distribution = GaussianKDE([-1.0, 0.0, 1.0])
    assert distribution.cdf(np.array([-100.0, 0.0, 100.0])) == pytest.approx([0.0, 0.5, 1.0])


def test_ppf(distribution):
    """Reject inverse-CDF evaluation."""
    with pytest.raises(NotImplementedError, match="PPF not available"):
        distribution.ppf(np.array([0.5]))


@pytest.mark.parametrize(
    ("samples", "weights", "message"),
    [
        ([[1.0]], None, "At least two samples"),
        ([[0.0], [1.0]], [1.0], "Number of weights"),
        ([[0.0], [1.0]], [1.0, -1.0], "finite and non-negative"),
        ([[0.0], [1.0]], [0.0, 0.0], "At least one weight"),
    ],
)
def test_invalid_input(samples, weights, message):
    """Reject malformed samples and weights."""
    with pytest.raises(ValueError, match=message):
        GaussianKDE(samples, weights=weights)
