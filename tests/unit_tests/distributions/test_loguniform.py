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
"""Test-module for log-uniform distribution."""

import numpy as np
import pytest

from queens.distributions.loguniform import LogUniform


@pytest.fixture(name="sample_pos_1d", params=[0.5, [-2.0, 0.0, 0.5, 1.0, 3.0]])
def fixture_sample_pos_1d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="lower_bound_1d", scope="module")
def fixture_lower_bound_1d():
    """A possible left bound of interval."""
    return 0.5


@pytest.fixture(name="upper_bound_1d", scope="module")
def fixture_upper_bound_1d():
    """A possible right bound of interval."""
    return 4.0


@pytest.fixture(name="loguniform_1d", scope="module")
def fixture_loguniform_1d(lower_bound_1d, upper_bound_1d):
    """A log-uniform distribution."""
    return LogUniform(lower_bound=lower_bound_1d, upper_bound=upper_bound_1d)


@pytest.fixture(
    name="sample_pos_2d",
    params=[
        [0.25, 0.75],
        [[0.1, 0.5], [0.5, 0.5], [1.0, 2.0], [2.5, 0.1], [5.0, 5.0]],
    ],
)
def fixture_sample_pos_2d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="lower_bound_2d", scope="module")
def fixture_lower_bound_2d():
    """A possible left bound of interval."""
    return np.array([0.5, 0.25])


@pytest.fixture(name="upper_bound_2d", scope="module")
def fixture_upper_bound_2d():
    """A possible right bound of interval."""
    return np.array([4.0, 2.0])


@pytest.fixture(name="loguniform_2d", scope="module")
def fixture_loguniform_2d(lower_bound_2d, upper_bound_2d):
    """A log-uniform distribution."""
    return LogUniform(lower_bound=lower_bound_2d, upper_bound=upper_bound_2d)


def test_init_loguniform_1d(loguniform_1d, lower_bound_1d, upper_bound_1d):
    """Test init method of LogUniform Distribution class."""
    lower_bound_1d = np.array(lower_bound_1d).reshape(1)
    upper_bound_1d = np.array(upper_bound_1d).reshape(1)
    log_width = np.log(upper_bound_1d / lower_bound_1d)
    mean_ref = ((upper_bound_1d - lower_bound_1d) / log_width).reshape(1)
    var_ref = (
        (upper_bound_1d**2 - lower_bound_1d**2) / (2.0 * log_width) - mean_ref**2
    ).reshape(1, 1)

    assert loguniform_1d.dimension == 1
    np.testing.assert_allclose(loguniform_1d.mean, mean_ref)
    np.testing.assert_allclose(loguniform_1d.covariance, var_ref)
    np.testing.assert_equal(loguniform_1d.lower_bound, lower_bound_1d)
    np.testing.assert_equal(loguniform_1d.upper_bound, upper_bound_1d)
    np.testing.assert_equal(loguniform_1d.log_width, log_width)


def test_init_loguniform_1d_wrong_interval(lower_bound_1d):
    """Test init method of LogUniform Distribution class."""
    with pytest.raises(ValueError, match=r"Lower bound must be smaller than upper bound*"):
        upper_bound = lower_bound_1d - 0.1
        return LogUniform(lower_bound=lower_bound_1d, upper_bound=upper_bound)


def test_init_loguniform_1d_nonpositive_bounds(upper_bound_1d):
    """Test that nonpositive support bounds are rejected."""
    with pytest.raises(ValueError, match=r"The parameter 'lower_bound' has to be positive*"):
        LogUniform(lower_bound=-1.0, upper_bound=upper_bound_1d)


def test_cdf_loguniform_1d(loguniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test cdf method of LogUniform distribution class."""
    sample_pos_1d = np.asarray(sample_pos_1d, dtype=float).reshape(-1)
    log_width = np.log(upper_bound_1d / lower_bound_1d)
    ref_sol = np.zeros(sample_pos_1d.shape)
    positive_support = sample_pos_1d > 0
    if np.any(positive_support):
        clipped = np.clip(
            (np.log(sample_pos_1d[positive_support]) - np.log(lower_bound_1d)) / log_width,
            0.0,
            1.0,
        )
        ref_sol[positive_support] = clipped
    np.testing.assert_allclose(loguniform_1d.cdf(sample_pos_1d), ref_sol)


def test_draw_loguniform_1d(loguniform_1d, mocker):
    """Test the draw method of log-uniform distribution."""
    sample = np.asarray([[1.0]])
    mocker.patch("numpy.random.uniform", return_value=np.log(sample))
    draw = loguniform_1d.draw()
    np.testing.assert_equal(draw, sample)


def test_logpdf_loguniform_1d(loguniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test logpdf method of LogUniform distribution class."""
    sample_pos_1d = np.asarray(sample_pos_1d, dtype=float).reshape(-1)
    log_width = np.log(upper_bound_1d / lower_bound_1d)
    ref_sol = np.full(sample_pos_1d.shape, -np.inf)
    within_bounds = (sample_pos_1d >= lower_bound_1d) & (sample_pos_1d <= upper_bound_1d)
    if np.any(within_bounds):
        ref_sol[within_bounds] = -np.log(sample_pos_1d[within_bounds]) - np.log(log_width)
    np.testing.assert_allclose(loguniform_1d.logpdf(sample_pos_1d), ref_sol)


def test_grad_logpdf_loguniform_1d(loguniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test *grad_logpdf* method of LogUniform distribution class."""
    sample_pos_1d = np.asarray(sample_pos_1d, dtype=float).reshape(-1, 1)
    ref_sol = np.full(sample_pos_1d.shape, np.nan)
    within_bounds = (sample_pos_1d[:, 0] >= lower_bound_1d) & (
        sample_pos_1d[:, 0] <= upper_bound_1d
    )
    ref_sol[within_bounds] = -1.0 / sample_pos_1d[within_bounds]
    np.testing.assert_allclose(loguniform_1d.grad_logpdf(sample_pos_1d), ref_sol, equal_nan=True)


def test_pdf_loguniform_1d(loguniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test pdf method of LogUniform distribution class."""
    sample_pos_1d = np.asarray(sample_pos_1d, dtype=float).reshape(-1)
    log_width = np.log(upper_bound_1d / lower_bound_1d)
    ref_sol = np.zeros(sample_pos_1d.shape)
    within_bounds = (sample_pos_1d >= lower_bound_1d) & (sample_pos_1d <= upper_bound_1d)
    if np.any(within_bounds):
        ref_sol[within_bounds] = 1.0 / (sample_pos_1d[within_bounds] * log_width)
    np.testing.assert_allclose(loguniform_1d.pdf(sample_pos_1d), ref_sol)


def test_ppf_loguniform_1d(loguniform_1d, lower_bound_1d, upper_bound_1d):
    """Test ppf method of LogUniform distribution class."""
    quantile = 0.5
    log_width = np.log(upper_bound_1d / lower_bound_1d)
    ref_sol = (lower_bound_1d * np.exp(quantile * log_width)).reshape(-1)
    np.testing.assert_allclose(loguniform_1d.ppf(quantile), ref_sol)


def test_init_loguniform_2d(loguniform_2d, lower_bound_2d, upper_bound_2d):
    """Test init method of LogUniform Distribution class."""
    log_width = np.log(upper_bound_2d / lower_bound_2d)
    mean_ref = (upper_bound_2d - lower_bound_2d) / log_width
    var_ref = np.diag((upper_bound_2d**2 - lower_bound_2d**2) / (2.0 * log_width) - mean_ref**2)

    assert loguniform_2d.dimension == 2
    np.testing.assert_allclose(loguniform_2d.mean, mean_ref)
    np.testing.assert_allclose(loguniform_2d.covariance, var_ref)
    np.testing.assert_equal(loguniform_2d.lower_bound, lower_bound_2d)
    np.testing.assert_equal(loguniform_2d.upper_bound, upper_bound_2d)
    np.testing.assert_equal(loguniform_2d.log_width, log_width)


def test_cdf_loguniform_2d(loguniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test cdf method of LogUniform distribution class."""
    sample_pos_2d = np.asarray(sample_pos_2d, dtype=float).reshape(-1, 2)
    log_width = np.log(upper_bound_2d / lower_bound_2d)
    ref_sol = np.zeros(sample_pos_2d.shape[0])
    positive_support = np.all(sample_pos_2d > 0, axis=1)
    if np.any(positive_support):
        clipped = np.clip(
            (np.log(sample_pos_2d[positive_support]) - np.log(lower_bound_2d)) / log_width,
            0.0,
            1.0,
        )
        ref_sol[positive_support] = np.prod(clipped, axis=1)
    np.testing.assert_allclose(loguniform_2d.cdf(sample_pos_2d), ref_sol)


def test_draw_loguniform_2d(loguniform_2d, mocker):
    """Test the draw method of log-uniform distribution."""
    sample = np.asarray([[1.0, 0.5]])
    mocker.patch("numpy.random.uniform", return_value=np.log(sample))
    draw = loguniform_2d.draw()
    np.testing.assert_equal(draw, sample)


def test_logpdf_loguniform_2d(loguniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test logpdf method of LogUniform distribution class."""
    sample_pos_2d = np.asarray(sample_pos_2d, dtype=float).reshape(-1, 2)
    log_width = np.log(upper_bound_2d / lower_bound_2d)
    ref_sol = np.full(sample_pos_2d.shape[0], -np.inf)
    within_bounds = (
        (sample_pos_2d >= lower_bound_2d).all(axis=1)
        & (sample_pos_2d <= upper_bound_2d).all(axis=1)
        & (sample_pos_2d > 0).all(axis=1)
    )
    if np.any(within_bounds):
        ref_sol[within_bounds] = -np.sum(np.log(sample_pos_2d[within_bounds]), axis=1) - np.sum(
            np.log(log_width)
        )
    np.testing.assert_allclose(loguniform_2d.logpdf(sample_pos_2d), ref_sol)


def test_grad_logpdf_loguniform_2d(loguniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test *grad_logpdf* method of LogUniform distribution class."""
    sample_pos_2d = np.asarray(sample_pos_2d, dtype=float).reshape(-1, 2)
    ref_sol = np.full(sample_pos_2d.shape, np.nan)
    within_bounds = (sample_pos_2d >= lower_bound_2d).all(axis=1) & (
        sample_pos_2d <= upper_bound_2d
    ).all(axis=1)
    ref_sol[within_bounds] = -1.0 / sample_pos_2d[within_bounds]
    np.testing.assert_allclose(loguniform_2d.grad_logpdf(sample_pos_2d), ref_sol, equal_nan=True)


def test_pdf_loguniform_2d(loguniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test pdf method of LogUniform distribution class."""
    sample_pos_2d = np.asarray(sample_pos_2d, dtype=float).reshape(-1, 2)
    log_width = np.log(upper_bound_2d / lower_bound_2d)
    ref_sol = np.zeros(sample_pos_2d.shape[0])
    within_bounds = (
        (sample_pos_2d >= lower_bound_2d).all(axis=1)
        & (sample_pos_2d <= upper_bound_2d).all(axis=1)
        & (sample_pos_2d > 0).all(axis=1)
    )
    if np.any(within_bounds):
        ref_sol[within_bounds] = np.prod(1.0 / (sample_pos_2d[within_bounds] * log_width), axis=1)
    np.testing.assert_allclose(loguniform_2d.pdf(sample_pos_2d), ref_sol)


def test_ppf_loguniform_2d(loguniform_2d):
    """Test ppf method of LogUniform distribution class."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        loguniform_2d.ppf(np.zeros(2))


def test_narrow_loguniform_has_stable_positive_variance():
    """Test variance calculation for a narrow support interval."""
    lower_bound = 1.0
    upper_bound = 1.0 + 1e-9

    distribution = LogUniform(lower_bound, upper_bound)

    expected_variance = (upper_bound - lower_bound) ** 2 / 12.0
    assert distribution.covariance[0, 0] > 0
    np.testing.assert_allclose(distribution.covariance[0, 0], expected_variance, rtol=1e-9)


def test_grad_logpdf_does_not_modify_input(loguniform_1d):
    """Test that evaluating the gradient leaves its input unchanged."""
    sample = np.array([0.0, 1.0])
    sample_before = sample.copy()

    loguniform_1d.grad_logpdf(sample)

    np.testing.assert_array_equal(sample, sample_before)
