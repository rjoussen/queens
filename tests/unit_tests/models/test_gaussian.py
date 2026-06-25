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
"""Unit tests for gaussian likelihood model."""

from collections import namedtuple

import numpy as np
import pytest
from mock import Mock

from queens.distributions.normal import Normal
from queens.models.likelihoods.gaussian import Gaussian


# ---------------- some fixtures ---------------------------------- #
@pytest.fixture(name="dummy_config")
def fixture_dummy_config():
    """A dummy config dictionary."""
    config = {
        "my_lik_model": {
            "type": "gaussian",
            "nugget_noise_variance": 1e-6,
            "forward_model_name": "my_forward_model",
            "noise_type": "fixed_variance",
            "noise_value": 1e-3,
        },
        "my_forward_model": "dummy_model",
    }
    return config


@pytest.fixture(name="my_lik_model")
def fixture_my_lik_model():
    """A dummy likelihood model."""
    forward_model_dummy = namedtuple("forward_model", ["evaluate"])

    class FakeDistr:
        """A fake distribution class."""

        def __init__(self):
            """A fake init method."""
            self.cov = None
            self.covariance = np.eye(1)
            self.precision = np.eye(1)

        def logpdf(self, x):
            """A fake logpdf method."""
            return x**2

        def evaluate(self, x):
            """A fake eval method."""
            return x**2

        def update_covariance(self, x):
            """A fake eval method."""
            self.cov = x
            self.covariance = x
            self.precision = np.linalg.pinv(x)

        def grad_logpdf(self, x):
            """A fake grad logpdf fun."""
            return -2 * np.linalg.norm(x, axis=1, keepdims=True)

    distr_dummy = FakeDistr()

    nugget_noise_variance = 1e-6
    noise_value = 1e-3
    forward_model = forward_model_dummy(lambda x: {"result": x + 1})
    noise_type = "fixed_variance"
    y_obs = np.array([[3.0]])

    gauss_lik_obj = Gaussian(
        forward_model=forward_model,
        noise_type=noise_type,
        noise_value=noise_value,
        nugget_noise_variance=nugget_noise_variance,
        y_obs=y_obs,
    )
    gauss_lik_obj.normal_distribution = distr_dummy
    return gauss_lik_obj


# ----------------- actual unit tests ------------------------------#
def test_init():
    """Test for the init method."""
    nugget_noise_variance = 1e-6
    forward_model = "my_forward_model"
    noise_type = "fixed_variance"
    noise_value = 0.1
    noise_var_iterative_averaging = None
    y_obs = np.array([[3.0]])

    gauss_lik_obj = Gaussian(
        forward_model=forward_model,
        noise_type=noise_type,
        noise_value=noise_value,
        nugget_noise_variance=nugget_noise_variance,
        noise_var_iterative_averaging=noise_var_iterative_averaging,
        y_obs=y_obs,
    )
    assert gauss_lik_obj.nugget_noise_variance == nugget_noise_variance
    assert gauss_lik_obj.forward_model == forward_model
    assert gauss_lik_obj.noise_type == noise_type
    assert gauss_lik_obj.noise_var_iterative_averaging == noise_var_iterative_averaging
    assert isinstance(gauss_lik_obj.normal_distribution, Normal)
    assert gauss_lik_obj.normal_distribution.mean == y_obs
    assert gauss_lik_obj.normal_distribution.covariance == np.eye(y_obs.size) * noise_value

    assert gauss_lik_obj.y_obs == y_obs


def test_evaluate(mocker, my_lik_model):
    """Test for the evaluate method."""
    samples = np.array([[1.0]])
    # test working evaluation
    response = my_lik_model.evaluate(samples)["result"]
    assert response == 4

    # test update of covariance for MAP
    my_lik_model.noise_type = "MAP_abc"
    m1 = mocker.patch("queens.models.likelihoods.gaussian.Gaussian.update_covariance")
    response = my_lik_model.evaluate(samples)["result"]
    m1.assert_called_once_with(np.array([[2.0]]))


def test_update_covariance(my_lik_model):
    """Test for update of covariance."""
    y_model = np.array([[1.0, 7.0]])

    # test MAP jeffrey variance, no averaging
    my_lik_model.noise_type = "MAP_jeffrey_variance"
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[2, 0], [0, 2]])
    # we duck-typed the normal distribution obj and write its argument into an attribute
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)

    # test MAP jeffery variance vector, no averaging
    my_lik_model.noise_type = "MAP_jeffrey_variance_vector"
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[0.8, 0], [0, 3.2]])
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)

    # test other MAP case, no averaging
    my_lik_model.noise_type = "MAP_abcdefg"
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[0.8, -1.6], [-1.6, 3.2]])
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)

    # test other MAP case with averaging
    class DummyClass:
        """Dummy class for testing."""

        def __init__(self):
            """Dummy init method."""
            self.cov = None

        def update_average(self, cov):
            """Dummy method for testing."""
            self.cov = cov
            return cov

    dummy_averaging_obj = DummyClass()
    my_lik_model.noise_var_iterative_averaging = dummy_averaging_obj
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[0.8, -1.6], [-1.6, 3.2]])

    # actual tests
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)
    # also check if method was actually called (we just write the method argument as an attribute
    # of the dummy obj here to make this check
    np.testing.assert_array_equal(my_lik_model.noise_var_iterative_averaging.cov, expected_cov)


def test_has_dynamic_noise(my_lik_model):
    """Test dynamic noise detection."""
    my_lik_model.noise_type = "MAP_abc"
    assert my_lik_model.has_dynamic_noise()

    my_lik_model.noise_type = "best_fit_adaptive_variance"
    assert my_lik_model.has_dynamic_noise()

    my_lik_model.noise_type = "fixed_variance"
    assert not my_lik_model.has_dynamic_noise()


def test_update_covariance_empty_input_is_noop(my_lik_model):
    """Test empty covariance updates are ignored."""
    my_lik_model.noise_type = "best_fit_adaptive_variance"
    my_lik_model.sigma_current = np.array([1.0])
    my_lik_model.normal_distribution.cov = "unchanged"

    my_lik_model.update_covariance(np.empty((0, 1)))

    assert my_lik_model.normal_distribution.cov == "unchanged"


def test_update_covariance_best_fit_variance(my_lik_model):
    """Test best-fit adaptive variance update."""
    my_lik_model.noise_type = "best_fit_adaptive_variance"
    my_lik_model.sigma_factor = 1.0
    my_lik_model.sigma_current = np.array([2.0])
    my_lik_model.normal_distribution.covariance = np.array([[1.0]])
    my_lik_model.normal_distribution.precision = np.array([[1.0]])
    my_lik_model.covariance_update_history = np.empty((0, 1))

    my_lik_model.update_covariance(np.array([[2.0], [2.5]]))

    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, np.array([[0.25]]))


def test_update_covariance_best_fit_variance_uses_best_sample_from_all_outputs(my_lik_model):
    """Test best-fit adaptive variance selects the best sample from all outputs."""
    my_lik_model.noise_type = "best_fit_adaptive_variance"
    my_lik_model.sigma_factor = 1.0
    my_lik_model.sigma_current = np.array([3.0])
    my_lik_model.normal_distribution.covariance = np.array([[1.0]])
    my_lik_model.normal_distribution.precision = np.array([[1.0]])
    my_lik_model.covariance_update_history = np.empty((0, 1))
    my_lik_model.update_covariance(np.array([[4.0]]))
    my_lik_model.update_covariance(np.array([[2.5]]))

    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, np.array([[0.25]]))
    np.testing.assert_array_equal(my_lik_model.covariance_update_history, np.array([[4.0], [2.5]]))


def test_update_covariance_best_fit_variance_vector_uses_best_sample_from_all_outputs(
    my_lik_model,
):
    """Test best-fit adaptive variance vector selects the best sample from all outputs."""
    my_lik_model.noise_type = "best_fit_adaptive_variance_vector"
    my_lik_model.sigma_factor = 1.0
    my_lik_model.sigma_current = np.array([3.0, 3.0])
    my_lik_model.y_obs = np.array([3.0, 3.0])
    my_lik_model.normal_distribution.covariance = np.eye(2)
    my_lik_model.normal_distribution.precision = np.eye(2)
    my_lik_model.covariance_update_history = np.empty((0, 2))

    my_lik_model.update_covariance(np.array([[4.0, 6.0]]))
    my_lik_model.update_covariance(np.array([[2.5, 3.5]]))

    np.testing.assert_array_equal(
        my_lik_model.normal_distribution.cov, np.array([[0.25, 0.0], [0.0, 0.25]])
    )
    np.testing.assert_array_equal(
        my_lik_model.covariance_update_history, np.array([[4.0, 6.0], [2.5, 3.5]])
    )


def test_update_covariance_smallest_mahalanobis_distance_keeps_best_sample(my_lik_model):
    """Test Mahalanobis-based updates reuse the stored best residual if no improvement occurs."""
    my_lik_model.noise_type = "smallest_mahalanobis_distance"
    my_lik_model.sigma_factor = 1.0
    my_lik_model.y_obs = np.array([3.0, 3.0])
    my_lik_model.normal_distribution.covariance = np.eye(2)
    my_lik_model.normal_distribution.precision = np.eye(2)
    my_lik_model.best_mahalanobis_residual = None

    my_lik_model.update_covariance(np.array([[2.5, 3.5]]))
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, np.array([[0.25, 0.0], [0.0, 0.25]]))

    my_lik_model.update_covariance(np.array([[5.0, 5.0]]))
    np.testing.assert_array_equal(my_lik_model.best_mahalanobis_residual, np.array([-0.5, 0.5]))
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, np.array([[0.25, 0.0], [0.0, 0.25]]))


def test_grad(my_lik_model):
    """Test grad method."""
    samples = np.array([[1.0, 2.0], [2.0, 3.0]])
    my_lik_model.evaluate(samples)
    my_lik_model.forward_model = Mock()
    my_lik_model.forward_model.grad = lambda _samples, _upstream: _samples + _upstream
    upstream_gradient = np.array([[6], [7]])
    grad = my_lik_model.grad(samples, upstream_gradient=upstream_gradient)
    expected_grad = np.array([[-42.2666153056, -41.2666153056], [-68.0000000000, -67.0000000000]])
    np.testing.assert_almost_equal(expected_grad, grad)
