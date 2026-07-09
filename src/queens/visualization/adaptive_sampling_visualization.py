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
"""Visualization utilities for adaptive sampling results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from seaborn.axisgrid import PairGrid
from scipy.stats import gaussian_kde

from queens.distributions.uniform import Uniform
from queens.parameters.parameters import Parameters

AdaptiveSamplingResults = Mapping[str, Sequence[Any]]
Bounds = tuple[float, float]

class AdaptiveSamplingVisualization:
    """Plot adaptive sampling posterior diagnostics for scalar parameters."""

    def __init__(
        self,
        parameters: Parameters,
        kde_grid_size: int = 50,
        kde_num_points: int = 100,
        contour_levels: int = 10,
        plot_map_estimate: bool = False,
    ) -> None:
        """Initialize adaptive sampling visualization.

        Args:
            parameters (Parameters): Scalar QUEENS parameters to visualize.
            kde_grid_size (int): Number of grid points per axis for bivariate KDEs.
            kde_num_points (int): Number of support points for univariate KDEs.
            contour_levels (int): Number of contour levels for bivariate KDEs.
            plot_map_estimate (bool): Whether to highlight the current MAP training sample.
        """
        self._check_scalar_parameters(parameters)
        self.parameters = parameters
        self.plot_bounds: dict[str, Bounds] = {}
        self.kde_grid_size = kde_grid_size
        self.kde_num_points = kde_num_points
        self.contour_levels = contour_levels
        self.plot_map_estimate = plot_map_estimate

    def prepare(self, plotting_dir: Path) -> None:
        """Prepare the visualization environment."""

        plt.switch_backend("Agg") # prevent GUI backends from being used in adaptive-sampling plotting

        self.plotting_dir = plotting_dir
        plotting_dir.mkdir(parents=True, exist_ok=True)
        # delete any existing files in the plotting directory
        for file in plotting_dir.glob("adaptive_sampling_iteration_*.png"):
            file.unlink()

    def plot(
        self, results: AdaptiveSamplingResults, iteration: int
    ):
        """Plot adaptive sampling posterior diagnostics for one iteration."""
        
        pair_grid = self._plot_marginal_posterior_grid(results, iteration)
        
        pair_grid.figure.savefig(self.plotting_dir / f"adaptive_sampling_iteration_{iteration}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_marginal_posterior_grid(
        self, results: AdaptiveSamplingResults, iteration: int
    ) -> PairGrid:
        """Create a pair grid of univariate and bivariate marginal posteriors."""
        data_frame = self._particles_to_dataframe(results, iteration)
        weights = data_frame["weights"].to_numpy()
        contours: list[QuadContourSet] = []
        parameter_names = self.parameters.names
        self.plot_bounds = self._plot_bounds(results, iteration)

        map_sample = None
        if self.plot_map_estimate:
            map_sample = self._get_map_sample(results, iteration)

        pair_grid = sns.PairGrid(data=data_frame, vars=parameter_names, diag_sharey=False)
        pair_grid.figure.set_size_inches(10, 10)

        def plot_diag(x: pd.Series, **_kwargs: Any) -> None:
            self._plot_1d_posterior(x, weights, map_sample)
            if self.plot_map_estimate:
                axes = plt.gca()
                axes.set_ylim(bottom=0)
        

        if len(parameter_names) > 1:
            def plot_offdiag(x: pd.Series, y: pd.Series, **_kwargs: Any) -> None:
                contours.append(self._plot_2d_posterior(x, y, weights))

            def plot_upper(x: pd.Series, y: pd.Series, **_kwargs: Any) -> None:
                self._plot_training_samples(x, y, results, iteration)

            def plot_lower(x: pd.Series, y: pd.Series, **_kwargs: Any) -> None:
                if self.plot_map_estimate:
                    self._plot_map_estimate_2d(x, y, map_sample)

            pair_grid.map_diag(plot_diag)
            pair_grid.map_offdiag(plot_offdiag)
            pair_grid.map_upper(plot_upper)
            pair_grid.map_lower(plot_lower)

        pair_grid.figure.suptitle(
            f"Marginal posterior distributions - iteration {iteration}",
            y=0.995,
            fontsize=14,
            fontweight="bold",
        )
        self._format_pair_grid_figure(pair_grid.figure, contours[0])
        return pair_grid

    def _particles_to_dataframe(
        self, results: AdaptiveSamplingResults, iteration: int
    ) -> pd.DataFrame:
        """Convert posterior particles for one iteration into a weighted DataFrame."""
        particles = np.asarray(results["particles"][iteration], dtype=float)
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)
        if particles.shape[1] != self.parameters.num_parameters:
            raise ValueError("Particle dimension does not match the scalar parameters.")

        data_frame = pd.DataFrame(particles, columns=self.parameters.names)
        weights = np.asarray(results["weights"][iteration], dtype=float).reshape(-1)
        if weights.size != particles.shape[0]:
            raise ValueError("Number of weights does not match the number of particles.")
        data_frame["weights"] = weights
        return data_frame

    def _plot_1d_posterior(
        self,
        x: pd.Series,
        weights: ArrayLike,
        map_sample: np.ndarray | None = None,
        **_kwargs: Any,
    ) -> None:
        """Plot one univariate marginal posterior."""
        axes = plt.gca()
        name = str(x.name)
        values = np.asarray(x, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        bounds = self.plot_bounds[name]
        grid = np.linspace(bounds[0], bounds[1], self.kde_num_points)
        density = gaussian_kde(values, weights=weights)(grid)

        axes.plot(grid, density, label= "Posterior Density" if len(self.parameters.names) == 1 else "Marginal Posterior Density")
        axes.fill_between(grid, density, alpha=0.3)

        prior_density = np.asarray(self.parameters.dict[name].pdf(grid)).reshape(-1)
        axes.plot(grid, prior_density, linestyle=":", label="Prior Density")

        if self.plot_map_estimate and map_sample is not None:
            axes.axvline(
                map_sample[self.parameters.names.index(name)],
                color="grey",
                linestyle="--",
                linewidth=1.2,
                label="MAP estimate (univariate)",
            )

        axes.set_xlim(bounds)
        axes.set_ylim(bottom=0)

    def _plot_2d_posterior(
        self,
        x: pd.Series,
        y: pd.Series,
        weights: ArrayLike,
    ) -> QuadContourSet:
        """Plot one bivariate marginal posterior."""
        axes = plt.gca()
        x_values = np.asarray(x, dtype=float).reshape(-1)
        y_values = np.asarray(y, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        x_bounds = self.plot_bounds[str(x.name)]
        y_bounds = self.plot_bounds[str(y.name)]

        kde = gaussian_kde(np.vstack([x_values, y_values]), weights=weights)
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_bounds[0], x_bounds[1], self.kde_grid_size),
            np.linspace(y_bounds[0], y_bounds[1], self.kde_grid_size),
        )
        density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        contour = axes.contourf(
            grid_x, grid_y, density, levels=self.contour_levels, cmap="plasma"
        )
        axes.set_xlim(x_bounds)
        axes.set_ylim(y_bounds)
        return contour

    def _plot_training_samples(
        self,
        x: pd.Series,
        y: pd.Series,
        results: AdaptiveSamplingResults,
        iteration: int,
    ) -> None:
        """Overlay adaptive training samples in bivariate panels."""
        x_index = self.parameters.names.index(str(x.name))
        y_index = self.parameters.names.index(str(y.name))

        plt.scatter(
            results["x_train"][iteration][:, x_index], results["x_train"][iteration][:, y_index], marker=".", color="gray", label="Training Samples", s=15
        )
        plt.scatter(
            results["x_train_new"][iteration][:, x_index], results["x_train_new"][iteration][:, y_index], marker="+", color="green", linewidth=2, label="Next Training Samples", s=35
        )
        plt.scatter(
            results["x_train_failed"][iteration][:, x_index], results["x_train_failed"][iteration][:, y_index], marker="x", color="red", linewidth=2, label="Failed Training Samples", s=20
        )

    def _plot_map_estimate_2d(
        self, x: pd.Series, y: pd.Series, map_sample: np.ndarray
    ) -> None:
        """Plot the MAP training sample in lower-triangle panels."""
        if not self.plot_map_estimate or map_sample is None:
            return
        x_index = self.parameters.names.index(str(x.name))
        y_index = self.parameters.names.index(str(y.name))
        plt.scatter(
            map_sample[x_index],
            map_sample[y_index],
            marker="o",
            color="white",
            edgecolors="black",
            linewidths=1.2,
            s=45,
            label=f"MAP estimate\n[{"\n ".join(f'{name}={map_sample[i]:.3f}' for i, name in enumerate(self.parameters.names))}]",
        )

    def _format_pair_grid_figure(
        self,
        figure: Figure,
        contour: QuadContourSet | None
    ) -> None:
        """Add figure-level legend and colorbar."""
        legend = self._draw_legend_without_duplicates(
            figure,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        if contour is not None and legend is not None:
            self._add_colorbar_below_legend(figure, legend, contour)

    def _plot_bounds(
        self, results: AdaptiveSamplingResults, iteration: int
    ) -> dict[str, Bounds]:
        """Return plotting bounds determined from posterior particles and evaluated training samples."""
        all_samples = self._all_plot_samples(results, iteration)

        plot_bounds = {}
        for index, name in enumerate(self.parameters.names):
            values = all_samples[:, index]
            finite = values[np.isfinite(values)]

            if finite.size == 0:
                lower, upper = 0.0, 1.0
            else:
                lower = float(np.min(finite))
                upper = float(np.max(finite))

            parameter = self.parameters.dict[name]
            if isinstance(parameter, Uniform):
                lower_bound = np.asarray(parameter.lower_bound, dtype=float).reshape(-1)[0]
                upper_bound = np.asarray(parameter.upper_bound, dtype=float).reshape(-1)[0]
                lower = min(lower, float(lower_bound))
                upper = max(upper, float(upper_bound))

            margin = (
                max(1.0, abs(lower) * 0.1)
                if np.isclose(lower, upper)
                else 0.05 * (upper - lower)
            )
            plot_bounds[name] = (lower - margin, upper + margin)

        return plot_bounds

    def _all_plot_samples(
        self, results: AdaptiveSamplingResults, iteration: int
    ) -> np.ndarray:
        """Return all samples that should be visible in the plots."""
        return np.vstack(
            [
                np.asarray(results[key][iteration], dtype=float).reshape(
                    -1, self.parameters.num_parameters
                )
                for key in ("particles", "x_train", "x_train_new", "x_train_failed")
            ]
        )

    @staticmethod
    def _draw_legend_without_duplicates(figure: Figure, **legend_kwargs: Any) -> Any:
        """Draw a single figure-level legend without duplicate labels."""
        handles, labels = [], []
        for axes in figure.axes:
            axes_handles, axes_labels = axes.get_legend_handles_labels()
            handles.extend(axes_handles)
            labels.extend(axes_labels)

        unique = {}
        for handle, label in zip(handles, labels):
            if label and label not in unique:
                unique[label] = handle

        if unique:
            return figure.legend(unique.values(), unique.keys(), **legend_kwargs)
        return None

    def _add_colorbar_below_legend(
        self,
        figure: Figure,
        legend: Any,
        contour: QuadContourSet,
    ) -> None:
        """Place a horizontal colorbar directly below the figure legend."""
        legend_box = legend.get_window_extent().transformed(figure.transFigure.inverted())
        colorbar_height = 0.03
        colorbar_padding = 0.05
        colorbar_axes = figure.add_axes(
            [
                legend_box.x0,
                max(0.02, legend_box.y0 - colorbar_padding),
                legend_box.width,
                colorbar_height,
            ]
        )
        colorbar = figure.colorbar(contour, orientation="horizontal", cax=colorbar_axes)
        colorbar.set_label("Posterior Density" if len(self.parameters.names) == 2 else "Marginal Posterior Density")
        if len(contour.levels) >= 2:
            colorbar.set_ticks([contour.levels[0], contour.levels[-1]])
            colorbar.set_ticklabels(["Low", "High"])

    @staticmethod
    def _check_scalar_parameters(parameters: Parameters) -> None:
        """Check that all parameters are scalar."""
        for parameter_name, parameter in parameters.dict.items():
            if parameter.dimension != 1:
                raise ValueError(
                    "Adaptive sampling visualization currently supports only scalar parameters. "
                    f"Parameter '{parameter_name}' has dimension {parameter.dimension}."
                )

    def _get_map_sample(
        self, results: AdaptiveSamplingResults, iteration: int
    ) -> np.ndarray:
        """Return the MAP training sample"""

        log_posterior = np.asarray(results["log_posterior"][iteration], dtype=float)

        map_index = np.argmax(log_posterior)
        map_sample = np.asarray(results["particles"][iteration][map_index], dtype=float)

        return map_sample
