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
"""Unit tests for adaptive sampling visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from queens.distributions.normal import Normal
from queens.distributions.uniform import Uniform
from queens.parameters.parameters import Parameters
from queens.visualization.adaptive_sampling_visualization import AdaptiveSamplingVisualization


@pytest.fixture(name="parameters_2d")
def fixture_parameters_2d():
    """Create two scalar parameters."""
    return Parameters(
        x1=Uniform(lower_bound=-3, upper_bound=3), x2=Uniform(lower_bound=-3, upper_bound=3)
    )


@pytest.fixture(name="parameters_3d")
def fixture_parameters_3d():
    """Create three scalar parameters."""
    return Parameters(
        x1=Uniform(lower_bound=-3, upper_bound=3),
        x2=Uniform(lower_bound=-3, upper_bound=3),
        x3=Uniform(lower_bound=-3, upper_bound=3),
    )


def _adaptive_sampling_results(dimension):
    """Create adaptive sampling-like result data."""
    generator = np.random.default_rng(41)
    particles = generator.normal(size=(30, dimension))
    weights = np.linspace(1.0, 2.0, particles.shape[0])
    weights /= np.sum(weights)
    x_train = generator.normal(size=(8, dimension))
    y_train = np.linspace(-10.0, -1.0, x_train.shape[0]).reshape(-1, 1)
    return {
        "x_train": [x_train],
        "x_train_failed": [np.empty((0, dimension))],
        "model_outputs": [np.arange(x_train.shape[0] * 2, dtype=float).reshape(x_train.shape[0], 2)],
        "model_outputs_failed": [np.empty((0, 2))],
        "y_train": [y_train],
        "x_train_new": [generator.normal(size=(3, dimension))],
        "particles": [particles],
        "weights": [weights],
        "log_posterior": [generator.normal(size=particles.shape[0])],
        "cs_div": [np.nan],
    }


def test_plot_marginal_posterior_grid_saves_2d_plot(tmp_path, parameters_2d):
    """Test that the marginal posterior pair grid is saved for 2D results."""
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_2d,
        ground_truth=[0.1, -0.2],
        kde_grid_size=8,
        kde_num_points=20,
        contour_levels=4,
    )
    visualization.prepare(tmp_path)

    visualization.plot(results, iteration=0)

    assert (tmp_path / "adaptive_sampling_iteration_0.png").is_file()


def test_plot_saves_pair_grid_for_three_scalar_parameters(tmp_path, parameters_3d):
    """Test that three scalar parameters are shown with the marginal posterior pair grid."""
    results = _adaptive_sampling_results(dimension=3)
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_3d,
        ground_truth=[0.1, -0.2, 0.3],
        kde_grid_size=8,
        kde_num_points=20,
        contour_levels=4,
    )
    visualization.prepare(tmp_path)

    visualization.plot(results, iteration=0)

    assert (tmp_path / "adaptive_sampling_iteration_0.png").is_file()


def test_plot_marginal_posterior_grid_draws_posterior_on_both_triangles(parameters_2d):
    """Test that off-diagonal plots include the bivariate posterior on both triangles."""
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_2d,
        kde_grid_size=8,
        contour_levels=4,
    )

    pair_grid = visualization._plot_marginal_posterior_grid(results, iteration=0)

    upper_axes = pair_grid.axes[0, 1]
    lower_axes = pair_grid.axes[1, 0]
    assert upper_axes.collections
    assert lower_axes.collections
    plt.close(pair_grid.figure)


def test_non_scalar_parameters_raise_value_error():
    """Test that only scalar parameters are accepted."""
    parameters = Parameters(x=Normal(mean=[0.0, 1.0], covariance=np.eye(2)))

    with pytest.raises(ValueError, match="supports only scalar parameters"):
        AdaptiveSamplingVisualization(parameters=parameters)


def test_adaptive_sampling_prepare_uses_agg_backend(tmp_path, parameters_2d, mocker):
    """Test that adaptive-sampling plotting avoids GUI backends."""
    switch_backend = mocker.patch(
        "queens.visualization.adaptive_sampling_visualization.plt.switch_backend"
    )
    visualization = AdaptiveSamplingVisualization(parameters=parameters_2d)

    visualization.prepare(tmp_path)

    switch_backend.assert_called_once_with("Agg")


def test_map_estimate_is_drawn_on_diag_and_lower_triangle(parameters_2d):
    """Test MAP estimate is shown in diagonal and lower-triangle panels."""
    results = _adaptive_sampling_results(dimension=2)
    results["x_train"][0] = np.array([[0.2, -0.1], [0.8, 0.6]])
    results["y_train"][0] = np.array([[-3.0], [-0.5]])
    results["model_outputs"][0] = np.array([[10.0, 11.0], [20.0, 21.0]])
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_2d,
        plot_map_estimate=True,
        kde_grid_size=8,
        contour_levels=4,
    )

    pair_grid = visualization._plot_marginal_posterior_grid(results, iteration=0)

    diag_lines = pair_grid.axes[0, 0].lines
    lower_collections = pair_grid.axes[1, 0].collections
    assert any(line.get_label() == "MAP estimate" for line in diag_lines)
    assert any(
        getattr(collection, "get_label", lambda: None)() == "MAP estimate"
        for collection in lower_collections
    )
    assert any(
        "MAP model output" in text.get_text() and "[20. 21.]" in text.get_text()
        for text in pair_grid.figure.texts
    )
    plt.close(pair_grid.figure)


def test_map_training_sample_uses_log_prior_and_y_train(parameters_2d):
    """Test MAP training sample is chosen via y_train plus prior log-density."""
    parameters = Parameters(
        x1=Normal(mean=[0.0], covariance=np.array([[1.0]])),
        x2=Normal(mean=[0.0], covariance=np.array([[1.0]])),
    )
    results = _adaptive_sampling_results(dimension=2)
    results["x_train"][0] = np.array([[2.8, 2.8], [0.0, 0.0]])
    results["y_train"][0] = np.array([[0.0], [-0.2]])
    results["model_outputs"][0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    visualization = AdaptiveSamplingVisualization(parameters=parameters, plot_map_estimate=True)

    map_sample, map_model_output = visualization._get_map_sample(results, 0)

    np.testing.assert_array_equal(map_sample, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(map_model_output, np.array([3.0, 4.0]))
