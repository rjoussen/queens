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
"""Log-uniform distribution."""

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class LogUniform(Continuous):
    """Log-uniform distribution class.

    Support is strictly positive and bounded by ``lower_bound`` and ``upper_bound``.

    Attributes:
        lower_bound: Lower bound(s) of the distribution.
        upper_bound: Upper bound(s) of the distribution.
        log_width: Width(s) in log-space.
        pdf_const: Constant for the evaluation of the PDF.
        logpdf_const: Constant for the evaluation of the log-PDF.
    """

    @log_init_args
    def __init__(self, lower_bound: ArrayLike, upper_bound: ArrayLike) -> None:
        """Initialize log-uniform distribution.

        Args:
            lower_bound: Lower bound(s) of the distribution
            upper_bound: Upper bound(s) of the distribution
        """
        lower_bound = np.array(lower_bound, dtype=float).reshape(-1)
        upper_bound = np.array(upper_bound, dtype=float).reshape(-1)
        super().check_positivity(lower_bound=lower_bound, upper_bound=upper_bound)
        super().check_bounds(lower_bound, upper_bound)

        log_width = np.log(upper_bound / lower_bound)
        scaled_mean = np.expm1(log_width) / log_width
        mean = lower_bound * scaled_mean

        scaled_second_moment = np.expm1(2.0 * log_width) / (2.0 * log_width)
        scaled_variance = scaled_second_moment - scaled_mean**2
        narrow_interval = np.abs(log_width) < 1e-4
        if np.any(narrow_interval):
            width = log_width[narrow_interval]
            scaled_variance[narrow_interval] = width**2 * (
                1.0 / 12.0
                + width
                * (
                    1.0 / 12.0
                    + width * (17.0 / 360.0 + width * (7.0 / 360.0 + width * (43.0 / 6720.0)))
                )
            )
        variance = lower_bound**2 * scaled_variance

        super().__init__(mean=mean, covariance=np.diag(variance), dimension=mean.size)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_width = log_width
        self.pdf_const = 1.0 / np.prod(log_width)
        self.logpdf_const = np.sum(np.log(log_width))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated

        Returns:
            CDF at positions
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        cdf = np.zeros(x.shape[0])
        positive_support = np.all(x > 0, axis=1)

        if np.any(positive_support):
            x_positive = x[positive_support]
            cdf[positive_support] = np.prod(
                np.clip(
                    (np.log(x_positive) - np.log(self.lower_bound)) / self.log_width,
                    a_min=np.zeros(self.dimension),
                    a_max=np.ones(self.dimension),
                ),
                axis=1,
            )

        return cdf

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Drawn samples from the distribution
        """
        samples = np.exp(
            np.random.uniform(
                low=np.log(self.lower_bound),
                high=np.log(self.upper_bound),
                size=(num_draws, self.dimension),
            )
        )
        return samples

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated

        Returns:
            Log-PDF at positions
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        within_bounds = (
            (x >= self.lower_bound).all(axis=1)
            * (x <= self.upper_bound).all(axis=1)
            * (x > 0).all(axis=1)
        )
        logpdf = np.full(x.shape[0], -np.inf)

        if np.any(within_bounds):
            logpdf[within_bounds] = -np.sum(np.log(x[within_bounds]), axis=1) - self.logpdf_const

        return logpdf

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to *x*.

        Args:
            x: Positions at which the gradient of log-PDF is evaluated

        Returns:
            Gradient of the log-PDF evaluated at positions
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        within_bounds = (x >= self.lower_bound).all(axis=1) & (x <= self.upper_bound).all(axis=1)
        grad_logpdf = np.full(x.shape, np.nan)
        grad_logpdf[within_bounds] = -1.0 / x[within_bounds]
        return grad_logpdf

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated

        Returns:
            PDF at positions
        """
        return np.exp(self.logpdf(x))

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Percent point function (inverse of CDF — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated

        Returns:
            Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = np.exp(
            scipy.stats.uniform.ppf(q=quantiles, loc=np.log(self.lower_bound), scale=self.log_width)
        ).reshape(-1)
        return ppf
