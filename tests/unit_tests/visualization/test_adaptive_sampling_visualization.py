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

# pylint: disable=protected-access

import matplotlib.pyplot as plt
import numpy as np
import pytest

from queens.distributions.normal import Normal
from queens.distributions.uniform import Uniform
from queens.parameters.parameters import Parameters
from queens.visualization.adaptive_sampling_visualization import (
    AdaptiveSamplingVisualization,
)


@pytest.fixture(name="parameters_2d")
def fixture_parameters_2d():
    """Create two scalar parameters."""
    return Parameters(
        x1=Uniform(lower_bound=-3, upper_bound=3),
        x2=Uniform(lower_bound=-3, upper_bound=3),
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
        "model_outputs": [
            np.arange(x_train.shape[0] * 2, dtype=float).reshape(x_train.shape[0], 2)
        ],
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
        kde_grid_size=8,
        kde_num_points=20,
        contour_levels=4,
    )
    visualization.plot(results, iteration=0, parameters=parameters_2d, plotting_dir=tmp_path)

    expected_plots = {
        "adaptive_sampling_iteration_0_posterior.svg",
        "adaptive_sampling_iteration_0_parameter_evolution.svg",
    }
    assert expected_plots <= {path.name for path in tmp_path.iterdir()}


def test_plot_saves_pair_grid_for_three_scalar_parameters(tmp_path, parameters_3d):
    """Test the marginal posterior pair grid for three scalar parameters."""
    results = _adaptive_sampling_results(dimension=3)
    visualization = AdaptiveSamplingVisualization(
        kde_grid_size=8,
        kde_num_points=20,
        contour_levels=4,
    )
    visualization.plot(results, iteration=0, parameters=parameters_3d, plotting_dir=tmp_path)

    assert (tmp_path / "adaptive_sampling_iteration_0_posterior.svg").is_file()


def test_plot_marginal_posterior_grid_draws_posterior_on_both_triangles(parameters_2d):
    """Test that off-diagonal plots include the posterior on both triangles."""
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(
        kde_grid_size=8,
        contour_levels=4,
    )

    pair_grid = visualization._plot_marginal_posterior_grid(
        results, iteration=0, parameters=parameters_2d
    )

    upper_axes = pair_grid.axes[0, 1]
    lower_axes = pair_grid.axes[1, 0]
    assert upper_axes.collections
    assert lower_axes.collections
    plt.close(pair_grid.figure)


def test_multidimensional_parameter_is_plotted_without_prior():
    """Test that vector components are plotted without prior marginals."""
    parameters = Parameters(x=Normal(mean=[0.0, 1.0], covariance=np.eye(2)))
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(kde_grid_size=8, contour_levels=4)

    pair_grid = visualization._plot_marginal_posterior_grid(
        results, iteration=0, parameters=parameters
    )

    assert parameters.parameters_keys == ["x_0", "x_1"]
    assert all(
        "Prior Density" not in axes.get_legend_handles_labels()[1] for axes in pair_grid.figure.axes
    )
    plt.close(pair_grid.figure)


def test_pair_grid_axes_keep_parameter_bounds_for_different_scales():
    """Test that diagonal densities do not alter shared parameter axes."""
    parameters = Parameters(x=Normal(mean=[250_000.0, 3_500.0], covariance=np.eye(2)))
    results = _adaptive_sampling_results(dimension=2)
    results["particles"][0][:, 0] += 250_000.0
    results["particles"][0][:, 1] = 3_500.0 + 1e-6 * results["particles"][0][:, 1]
    for key in ("x_train", "x_train_new", "x_train_failed"):
        results[key][0][:, 0] += 250_000.0
        results[key][0][:, 1] += 3_500.0
    visualization = AdaptiveSamplingVisualization(kde_grid_size=8, contour_levels=4)

    pair_grid = visualization._plot_marginal_posterior_grid(
        results, iteration=0, parameters=parameters
    )

    assert pair_grid.axes[1, 0].get_ylim() == pytest.approx(visualization.plot_bounds["x_1"])
    assert pair_grid.axes[0, 1].get_xlim() == pytest.approx(visualization.plot_bounds["x_1"])
    plt.close(pair_grid.figure)


def test_adaptive_sampling_plot_uses_agg_backend(tmp_path, parameters_2d, mocker):
    """Test that adaptive-sampling plotting avoids GUI backends."""
    results = _adaptive_sampling_results(dimension=2)
    switch_backend = mocker.patch(
        "queens.visualization.adaptive_sampling_visualization.plt.switch_backend"
    )
    visualization = AdaptiveSamplingVisualization(kde_grid_size=8, contour_levels=4)

    visualization.plot(results, iteration=0, parameters=parameters_2d, plotting_dir=tmp_path)

    switch_backend.assert_called_once_with("Agg")
    assert not hasattr(visualization, "plotting_dir")


def test_map_estimate_is_drawn_on_diagonal_with_legend_entry(parameters_2d):
    """Test MAP estimate is shown only on diagonal panels and in the legend."""
    results = _adaptive_sampling_results(dimension=2)
    results["log_posterior"][0] = np.linspace(-2.0, 2.0, results["particles"][0].shape[0])
    visualization = AdaptiveSamplingVisualization(
        plot_map_estimate=True,
        kde_grid_size=8,
        contour_levels=4,
    )

    pair_grid = visualization._plot_marginal_posterior_grid(
        results, iteration=0, parameters=parameters_2d
    )

    lower_collections = pair_grid.axes[1, 0].collections
    assert not any(
        getattr(collection, "get_label", lambda: "")() == "MAP estimate"
        for collection in lower_collections
    )
    legend_labels = [text.get_text() for text in pair_grid.figure.legends[0].get_texts()]
    assert "MAP estimate" in legend_labels
    assert "Most likely evaluated sample" in legend_labels
    assert all("\n" not in label for label in legend_labels)
    map_text = next(
        text.get_text()
        for text in pair_grid.figure.texts
        if text.get_text().startswith(r"$\bf{Most\ likely\ evaluated\ sample}$")
    )
    assert "MAP estimate" not in map_text
    assert "Training sample ID:" in map_text
    assert "Model output =" in map_text
    plt.close(pair_grid.figure)


def test_pair_grid_shows_weighted_summary_statistics(parameters_2d):
    """Test that posterior means, medians, and credible intervals are shown."""
    results = _adaptive_sampling_results(dimension=2)
    visualization = AdaptiveSamplingVisualization(kde_grid_size=8, contour_levels=4)

    pair_grid = visualization._plot_marginal_posterior_grid(
        results, iteration=0, parameters=parameters_2d
    )

    diagonal_labels = pair_grid.diag_axes[0].get_legend_handles_labels()[1]
    assert "Posterior mean" not in diagonal_labels
    assert "Posterior median" not in diagonal_labels
    assert "95% credible interval" in diagonal_labels
    lower_labels = pair_grid.axes[1, 0].get_legend_handles_labels()[1]
    assert "Posterior mean" not in lower_labels
    assert "Posterior median" not in lower_labels
    plt.close(pair_grid.figure)


def test_weighted_posterior_summary():
    """Test weighted posterior statistics."""
    particles = np.array([[0.0, 0.0], [2.0, 4.0], [10.0, 20.0]])
    weights = np.array([0.6, 0.3, 0.1])

    summary = AdaptiveSamplingVisualization._posterior_summary(particles, weights)

    np.testing.assert_allclose(summary.mean, np.array([1.6, 3.2]))
    np.testing.assert_allclose(summary.median, np.array([0.0, 0.0]))


def test_parameter_evolution_uses_all_iterations(parameters_2d):
    """Test that evolution includes all requested iterations."""
    results = _adaptive_sampling_results(dimension=2)
    for _, values in list(results.items()):
        values.append(np.copy(values[0]))
    results["particles"][1] += 1.0
    visualization = AdaptiveSamplingVisualization(plot_map_estimate=True)

    figure = visualization._plot_parameter_evolution(results, iteration=1, parameters=parameters_2d)

    assert len(figure.axes) == parameters_2d.num_parameters
    assert all(len(line.get_xdata()) == 2 for line in figure.axes[0].lines)
    evolution_labels = figure.axes[0].get_legend_handles_labels()[1]
    assert "MAP estimate" in evolution_labels
    assert "Most likely evaluated sample" in evolution_labels
    plt.close(figure)


def test_get_map_estimate_uses_particle_log_posterior():
    """Test MAP selection from particle log posteriors."""
    results = _adaptive_sampling_results(dimension=2)
    results["particles"][0] = np.array([[2.8, 2.8], [0.0, 0.0]])
    results["log_posterior"][0] = np.array([-3.0, -0.2])

    map_estimate = AdaptiveSamplingVisualization._get_map_estimate(results, 0)

    np.testing.assert_array_equal(map_estimate.sample, np.array([0.0, 0.0]))


def test_get_most_likely_evaluated_sample_includes_output_and_index():
    """Test metadata for the highest-posterior evaluated sample."""
    results = _adaptive_sampling_results(dimension=2)
    results["y_train"][0] = np.arange(8, dtype=float).reshape(-1, 1)
    evaluated_sample = AdaptiveSamplingVisualization._get_most_likely_evaluated_sample(
        results,
        0,
    )

    np.testing.assert_array_equal(evaluated_sample.sample, results["x_train"][0][-1])
    np.testing.assert_array_equal(evaluated_sample.model_output, np.array([14.0, 15.0]))
    assert evaluated_sample.training_sample_index == 7
