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
        "model_outputs": [np.arange(x_train.shape[0]).reshape(-1, 1)],
        "model_outputs_failed": [np.empty((0, 1))],
        "y_train": [y_train],
        "x_train_new": [generator.normal(size=(3, dimension))],
        "particles": [particles],
        "weights": [weights],
        "log_posterior": [generator.normal(size=particles.shape[0])],
        "cs_div": [np.nan],
    }


def test_plot_marginal_posterior_grid_saves_2d_plot(global_settings, parameters_2d):
    """Test that the marginal posterior pair grid is saved for 2D results."""
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_2d,
        ground_truth=[0.1, -0.2],
        kde_grid_size=8,
        kde_num_points=20,
        contour_levels=4,
    )

    saved_paths = visualization.plot(results, iteration=0, plotting_dir=global_settings.output_dir)

    assert saved_paths["marginal_posterior"].is_file()
    assert saved_paths["marginal_posterior"] == (
        global_settings.output_dir / "adaptive_sampling_marginal_posterior_0.png"
    )


def test_plot_saves_pair_grid_for_three_scalar_parameters(global_settings, parameters_3d):
    """Test that three scalar parameters are shown with the marginal posterior pair grid."""
    results = _adaptive_sampling_results(dimension=3)
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_3d,
        ground_truth=[0.1, -0.2, 0.3],
        kde_grid_size=8,
        kde_num_points=20,
        contour_levels=4,
    )

    saved_paths = visualization.plot(results, iteration=0, plotting_dir=global_settings.output_dir)

    assert saved_paths["marginal_posterior"].is_file()
    assert set(saved_paths) == {"marginal_posterior"}


def test_plot_marginal_posterior_grid_draws_posterior_on_both_triangles(parameters_2d, tmp_path):
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
    pair_grid.figure.clf()


def test_non_scalar_parameters_raise_value_error(tmp_path):
    """Test that only scalar parameters are accepted."""
    parameters = Parameters(x=Normal(mean=[0.0, 1.0], covariance=np.eye(2)))

    with pytest.raises(ValueError, match="supports only scalar parameters"):
        AdaptiveSamplingVisualization(
            parameters=parameters,
        )


def test_negative_iteration_raises_value_error(global_settings, parameters_2d):
    """Test that negative iteration indices are not accepted."""
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(
        parameters=parameters_2d,
    )

    with pytest.raises(ValueError, match="Iteration must be non-negative"):
        visualization.plot(results, iteration=-1, plotting_dir=global_settings.output_dir)


def test_adaptive_sampling_uses_agg_backend(global_settings, parameters_2d, mocker):
    """Test that adaptive-sampling plotting avoids GUI backends."""
    switch_backend = mocker.patch(
        "queens.visualization.adaptive_sampling_visualization.plt.switch_backend"
    )

    AdaptiveSamplingVisualization(
        parameters=parameters_2d,
    )

    switch_backend.assert_called_once_with("Agg")
