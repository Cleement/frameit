# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np


def setup_geodesic_polar_ax(ax, *, deg_ticks: bool = True, step_deg: int = 30) -> None:
    """
    Configure Matplotlib polar axes for geodesic (North-up, clockwise) convention.

    Sets theta zero to North and direction to clockwise, matching the meteorological
    convention where 0° is North and 90° is East.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or numpy.ndarray
        One polar Axes object or an array of Axes (e.g. from ``plt.subplots``).
    deg_ticks : bool, optional
        If True, replace the default radian ticks with degree labels. Default True.
    step_deg : int, optional
        Angular spacing between tick marks, in degrees. Default 30.
    """
    axs = ax.ravel().tolist() if isinstance(ax, np.ndarray) else [ax]

    for a in axs:
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)

        if deg_ticks:
            deg = np.arange(0, 360, int(step_deg))
            a.set_xticks(np.deg2rad(deg))
            a.set_xticklabels([f"{d:d}°" for d in deg])
