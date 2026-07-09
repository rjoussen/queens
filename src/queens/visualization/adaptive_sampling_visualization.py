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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from seaborn.axisgrid import PairGrid

from queens.distributions.gaussian_kde import GaussianKDE
from queens.distributions.uniform import Uniform
from queens.parameters.parameters import Parameters

AdaptiveSamplingResults = Mapping[str, Sequence[Any]]
Bounds = tuple[float, float]


@dataclass
class MAPEstimate:
    """Posterior MAP estimate."""

    sample: np.ndarray


@dataclass
class EvaluatedSample:
    """Highest-likelihood sample among evaluated training points."""

    sample: np.ndarray
    model_output: np.ndarray
    training_sample_index: int


@dataclass
class PosteriorSummary:
    """Weighted summary statistics of posterior particles."""

    mean: np.ndarray
    median: np.ndarray
    credible_interval_lower: np.ndarray
    credible_interval_upper: np.ndarray


class AdaptiveSamplingVisualization:
    """Plot adaptive sampling posterior diagnostics."""

    def __init__(
        self,
        kde_grid_size: int = 50,
        kde_num_points: int = 100,
        contour_levels: int = 10,
        plot_map_estimate: bool = False,
    ) -> None:
        """Initialize adaptive sampling visualization.

        Args:
            kde_grid_size (int): Number of grid points per axis for bivariate KDEs.
            kde_num_points (int): Number of support points for univariate KDEs.
            contour_levels (int): Number of contour levels for bivariate KDEs.
            plot_map_estimate (bool): Whether to highlight and describe the MAP estimates.
        """
        self.plot_bounds: dict[str, Bounds] = {}
        self.kde_grid_size = kde_grid_size
        self.kde_num_points = kde_num_points
        self.contour_levels = contour_levels
        self.plot_map_estimate = plot_map_estimate

    def plot(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
        plotting_dir: Path,
    ) -> None:
        """Plot adaptive sampling posterior diagnostics for one iteration."""
        plt.switch_backend("Agg")
        plotting_dir.mkdir(parents=True, exist_ok=True)
        pair_grid = self._plot_marginal_posterior_grid(results, iteration, parameters)

        pair_grid.figure.savefig(
            plotting_dir / f"adaptive_sampling_iteration_{iteration}_posterior.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(pair_grid.figure)

        evolution_figure = self._plot_parameter_evolution(results, iteration, parameters)
        evolution_figure.savefig(
            plotting_dir / f"adaptive_sampling_iteration_{iteration}_parameter_evolution.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(evolution_figure)

    def _plot_marginal_posterior_grid(
        self, results: AdaptiveSamplingResults, iteration: int, parameters: Parameters
    ) -> PairGrid:
        """Create a pair grid of posterior marginals."""
        data_frame = self._particles_to_dataframe(results, iteration, parameters)
        weights = data_frame["weights"].to_numpy()
        contours: list[QuadContourSet] = []
        parameter_names = parameters.parameters_keys
        self.plot_bounds = self._plot_bounds(results, iteration, parameters)
        posterior_summary = self._posterior_summary(
            data_frame[parameter_names].to_numpy(),
            weights,
        )

        map_estimate = None
        most_likely_evaluated_sample = None
        if self.plot_map_estimate:
            map_estimate = self._get_map_estimate(results, iteration)
            most_likely_evaluated_sample = self._get_most_likely_evaluated_sample(
                results, iteration
            )

        pair_grid = sns.PairGrid(data=data_frame, vars=parameter_names, diag_sharey=False)
        pair_grid.figure.set_size_inches(10, 10)

        def plot_diag(x: pd.Series, **_kwargs: Any) -> None:
            self._plot_1d_posterior(
                x,
                weights,
                parameters,
                posterior_summary,
                map_estimate,
                most_likely_evaluated_sample,
            )
            if self.plot_map_estimate:
                axes = plt.gca()
                axes.set_ylim(bottom=0)

        pair_grid.map_diag(plot_diag)

        if len(parameter_names) > 1:

            def plot_offdiag(x: pd.Series, y: pd.Series, **_kwargs: Any) -> None:
                contours.append(self._plot_2d_posterior(x, y, weights))

            def plot_upper(x: pd.Series, y: pd.Series, **_kwargs: Any) -> None:
                self._plot_training_samples(x, y, results, iteration, parameters)

            pair_grid.map_offdiag(plot_offdiag)
            pair_grid.map_upper(plot_upper)

        pair_grid.figure.suptitle(
            f"Marginal posterior distributions - iteration {iteration}",
            y=0.995,
            fontsize=14,
            fontweight="bold",
        )
        self._format_pair_grid_figure(
            pair_grid.figure,
            contours[0] if contours else None,
            parameters,
            most_likely_evaluated_sample,
            pair_grid.diag_axes,
        )
        return pair_grid

    def _particles_to_dataframe(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
    ) -> pd.DataFrame:
        """Convert posterior particles into a weighted DataFrame."""
        particles = np.asarray(results["particles"][iteration], dtype=float)
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)
        if particles.shape[1] != parameters.num_parameters:
            raise ValueError("Particle dimension does not match the parameters.")

        data_frame = pd.DataFrame(particles, columns=parameters.parameters_keys)
        weights = np.asarray(results["weights"][iteration], dtype=float).reshape(-1)
        if weights.size != particles.shape[0]:
            raise ValueError("Number of weights does not match the number of particles.")
        data_frame["weights"] = weights
        return data_frame

    def _plot_1d_posterior(
        self,
        x: pd.Series,
        weights: ArrayLike,
        parameters: Parameters,
        posterior_summary: PosteriorSummary,
        map_estimate: MAPEstimate | None = None,
        most_likely_evaluated_sample: EvaluatedSample | None = None,
        **_kwargs: Any,
    ) -> None:
        """Plot one univariate marginal posterior."""
        axes = plt.gca()
        name = str(x.name)
        values = np.asarray(x, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        bounds = self.plot_bounds[name]
        grid = np.linspace(bounds[0], bounds[1], self.kde_num_points)
        density = GaussianKDE(values, weights=weights).pdf(grid)

        axes.plot(
            grid,
            density,
            label=(
                "Posterior Density"
                if len(parameters.parameters_keys) == 1
                else "Marginal Posterior Density"
            ),
        )
        axes.fill_between(grid, density, alpha=0.3)

        parameter = parameters.dict.get(name)
        if parameter is not None and parameter.dimension == 1:
            prior_density = np.asarray(parameter.pdf(grid)).reshape(-1)
            axes.plot(grid, prior_density, linestyle="-", label="Prior Density")

        parameter_index = parameters.parameters_keys.index(name)
        axes.axvspan(
            posterior_summary.credible_interval_lower[parameter_index],
            posterior_summary.credible_interval_upper[parameter_index],
            color="tab:blue",
            alpha=0.12,
            label="95% credible interval",
        )

        if self.plot_map_estimate and map_estimate is not None:
            axes.axvline(
                map_estimate.sample[parameter_index],
                color="tab:green",
                linestyle="--",
                linewidth=1,
                label="MAP estimate",
            )

        if most_likely_evaluated_sample is not None:
            axes.axvline(
                most_likely_evaluated_sample.sample[parameter_index],
                color="tab:purple",
                linestyle="-.",
                linewidth=1,
                label="Most likely evaluated sample",
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

        kde = GaussianKDE(np.column_stack([x_values, y_values]), weights=weights)
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_bounds[0], x_bounds[1], self.kde_grid_size),
            np.linspace(y_bounds[0], y_bounds[1], self.kde_grid_size),
        )
        density = kde.pdf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        contour = axes.contourf(grid_x, grid_y, density, levels=self.contour_levels, cmap="plasma")
        axes.set_xlim(x_bounds)
        axes.set_ylim(y_bounds)
        return contour

    def _plot_training_samples(
        self,
        x: pd.Series,
        y: pd.Series,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
    ) -> None:
        """Overlay adaptive training samples in bivariate panels."""
        x_index = parameters.parameters_keys.index(str(x.name))
        y_index = parameters.parameters_keys.index(str(y.name))

        plt.scatter(
            results["x_train"][iteration][:, x_index],
            results["x_train"][iteration][:, y_index],
            marker=".",
            color="gray",
            label="Training Samples",
            s=15,
        )
        plt.scatter(
            results["x_train_new"][iteration][:, x_index],
            results["x_train_new"][iteration][:, y_index],
            marker="+",
            color="green",
            linewidth=2,
            label="Next Training Samples",
            s=35,
        )
        plt.scatter(
            results["x_train_failed"][iteration][:, x_index],
            results["x_train_failed"][iteration][:, y_index],
            marker="x",
            color="red",
            linewidth=2,
            label="Failed Training Samples",
            s=20,
        )

    @staticmethod
    def _evaluated_sample_text(
        most_likely_evaluated_sample: EvaluatedSample,
        parameters: Parameters,
    ) -> str:
        """Describe the highest-likelihood evaluated training sample."""
        evaluated_parameters = "\n".join(
            f"  {name}={most_likely_evaluated_sample.sample[i]:.3f}"
            for i, name in enumerate(parameters.parameters_keys)
        )
        model_output = np.array2string(
            most_likely_evaluated_sample.model_output,
            precision=3,
            separator=", ",
            suppress_small=False,
        )
        return (
            rf"$\bf{{Most\ likely\ evaluated\ sample}}$"
            f"\n  Training sample ID: {most_likely_evaluated_sample.training_sample_index}\n"
            f"  Model output = {model_output}\n"
            f"{evaluated_parameters}"
        )

    @classmethod
    def _posterior_summary(
        cls,
        particles: ArrayLike,
        weights: ArrayLike,
        credible_interval: tuple[float, float] = (0.025, 0.975),
    ) -> PosteriorSummary:
        """Calculate weighted posterior summary statistics."""
        particles = np.asarray(particles, dtype=float)
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)
        normalized_weights = cls._normalized_weights(weights, particles.shape[0])
        quantiles = cls._weighted_quantiles(
            particles,
            normalized_weights,
            np.array([credible_interval[0], 0.5, credible_interval[1]]),
        )
        return PosteriorSummary(
            mean=np.average(particles, axis=0, weights=normalized_weights),
            median=quantiles[1],
            credible_interval_lower=quantiles[0],
            credible_interval_upper=quantiles[2],
        )

    @staticmethod
    def _normalized_weights(weights: ArrayLike, num_particles: int) -> np.ndarray:
        """Validate and normalize particle weights."""
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != num_particles:
            raise ValueError("Number of weights does not match the number of particles.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Particle weights must be finite and non-negative.")
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise ValueError("Particle weights must have a positive sum.")
        return weights / weight_sum

    @staticmethod
    def _weighted_quantiles(
        values: np.ndarray,
        normalized_weights: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Calculate weighted quantiles for each column of a sample matrix."""
        quantiles = np.empty((probabilities.size, values.shape[1]))
        for dimension in range(values.shape[1]):
            order = np.argsort(values[:, dimension])
            sorted_values = values[order, dimension]
            sorted_weights = normalized_weights[order]
            cumulative_weights = np.cumsum(sorted_weights)
            quantiles[:, dimension] = np.interp(
                probabilities,
                cumulative_weights,
                sorted_values,
                left=sorted_values[0],
                right=sorted_values[-1],
            )
        return quantiles

    def _plot_parameter_evolution(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
    ) -> Figure:
        """Plot posterior statistics by iteration."""
        summaries = [
            self._posterior_summary(results["particles"][i], results["weights"][i])
            for i in range(iteration + 1)
        ]
        means = np.vstack([summary.mean for summary in summaries])
        medians = np.vstack([summary.median for summary in summaries])
        lower = np.vstack([summary.credible_interval_lower for summary in summaries])
        upper = np.vstack([summary.credible_interval_upper for summary in summaries])
        iterations = np.arange(iteration + 1)
        map_estimates = None
        most_likely_evaluated_samples = None
        if self.plot_map_estimate:
            map_estimates = np.vstack(
                [self._get_map_estimate(results, i).sample for i in iterations]
            )
            most_likely_evaluated_samples = np.vstack(
                [self._get_most_likely_evaluated_sample(results, i).sample for i in iterations]
            )

        figure, axes = plt.subplots(
            parameters.num_parameters,
            1,
            figsize=(9, max(3.2, 2.4 * parameters.num_parameters)),
            sharex=True,
            squeeze=False,
        )
        for parameter_index, (axes_row, parameter_name) in enumerate(
            zip(axes, parameters.parameters_keys)
        ):
            parameter_axes = axes_row[0]
            parameter_axes.fill_between(
                iterations,
                lower[:, parameter_index],
                upper[:, parameter_index],
                color="tab:blue",
                alpha=0.18,
                label="95% credible interval",
            )
            parameter_axes.plot(
                iterations,
                means[:, parameter_index],
                color="tab:green",
                marker="o",
                label="Posterior mean",
            )
            parameter_axes.plot(
                iterations,
                medians[:, parameter_index],
                color="tab:orange",
                linestyle="-.",
                marker=".",
                label="Posterior median",
            )
            if map_estimates is not None and most_likely_evaluated_samples is not None:
                parameter_axes.plot(
                    iterations,
                    map_estimates[:, parameter_index],
                    color="grey",
                    linestyle="--",
                    marker="x",
                    label="MAP estimate",
                )
                parameter_axes.plot(
                    iterations,
                    most_likely_evaluated_samples[:, parameter_index],
                    color="tab:purple",
                    linestyle=":",
                    marker="s",
                    label="Most likely evaluated sample",
                )
            parameter_axes.set_ylabel(parameter_name)
            parameter_axes.grid(alpha=0.25)

        axes[-1, 0].set_xlabel("Adaptive iteration")
        axes[-1, 0].set_xticks(iterations)
        figure.suptitle("Posterior parameter evolution", fontsize=14, fontweight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        return figure

    def _format_pair_grid_figure(
        self,
        figure: Figure,
        contour: QuadContourSet | None,
        parameters: Parameters,
        most_likely_evaluated_sample: EvaluatedSample | None,
        diagonal_axes: Sequence[Any],
    ) -> None:
        """Add figure-level legend and colorbar."""
        if most_likely_evaluated_sample is not None:
            figure.text(
                1.02,
                0.98,
                self._evaluated_sample_text(
                    most_likely_evaluated_sample,
                    parameters,
                ),
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
            )
        legend = self._draw_legend_without_duplicates(
            figure,
            additional_axes=diagonal_axes,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        if contour is not None and legend is not None:
            self._add_colorbar_below_legend(figure, legend, contour, parameters)

    def _plot_bounds(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
    ) -> dict[str, Bounds]:
        """Determine plot bounds from particles and training samples."""
        all_samples = self._all_plot_samples(results, iteration, parameters)

        plot_bounds = {}
        for index, name in enumerate(parameters.parameters_keys):
            values = all_samples[:, index]
            finite = values[np.isfinite(values)]

            if finite.size == 0:
                lower, upper = 0.0, 1.0
            else:
                lower = float(np.min(finite))
                upper = float(np.max(finite))

            parameter = parameters.dict.get(name)
            if isinstance(parameter, Uniform):
                lower_bound = np.asarray(parameter.lower_bound, dtype=float).reshape(-1)[0]
                upper_bound = np.asarray(parameter.upper_bound, dtype=float).reshape(-1)[0]
                lower = min(lower, float(lower_bound))
                upper = max(upper, float(upper_bound))

            margin = (
                max(1.0, abs(lower) * 0.1) if np.isclose(lower, upper) else 0.05 * (upper - lower)
            )
            plot_bounds[name] = (lower - margin, upper + margin)

        return plot_bounds

    def _all_plot_samples(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
    ) -> np.ndarray:
        """Return all samples that should be visible in the plots."""
        return np.vstack(
            [
                np.asarray(results[key][iteration], dtype=float).reshape(
                    -1, parameters.num_parameters
                )
                for key in ("particles", "x_train", "x_train_new", "x_train_failed")
            ]
        )

    @staticmethod
    def _draw_legend_without_duplicates(
        figure: Figure,
        additional_axes: Sequence[Any] = (),
        **legend_kwargs: Any,
    ) -> Any:
        """Draw a single figure-level legend without duplicate labels."""
        handles, labels = [], []
        for axes in [*figure.axes, *additional_axes]:
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
        parameters: Parameters,
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
        colorbar.set_label(
            "Posterior Density"
            if len(parameters.parameters_keys) == 2
            else "Marginal Posterior Density"
        )
        if len(contour.levels) >= 2:
            colorbar.set_ticks([contour.levels[0], contour.levels[-1]])
            colorbar.set_ticklabels(["Low", "High"])

    @staticmethod
    def _get_map_estimate(
        results: AdaptiveSamplingResults,
        iteration: int,
    ) -> MAPEstimate:
        """Return the MAP estimate from posterior particles."""
        log_posterior = np.asarray(results["log_posterior"][iteration], dtype=float)
        map_index = int(np.argmax(log_posterior))
        sample = np.asarray(results["particles"][iteration][map_index], dtype=float)
        return MAPEstimate(sample=sample)

    @staticmethod
    def _get_most_likely_evaluated_sample(
        results: AdaptiveSamplingResults,
        iteration: int,
    ) -> EvaluatedSample:
        """Return the evaluated sample with the highest log likelihood."""
        x_train = np.asarray(results["x_train"][iteration], dtype=float)
        log_likelihood = np.asarray(results["y_train"][iteration], dtype=float).reshape(-1)
        if x_train.shape[0] != log_likelihood.size:
            raise ValueError("Training samples and log-likelihood values do not match.")
        sample_index = int(np.argmax(log_likelihood))
        model_output = np.asarray(results["model_outputs"][iteration][sample_index], dtype=float)
        return EvaluatedSample(
            sample=x_train[sample_index],
            model_output=model_output,
            training_sample_index=sample_index,
        )
