Rotate polar axes
=================

Function setup\_geodesic\_polar\_ax
-----------------------------------

Apply a compass-like layout to a polar plot: north faces up and angles
increase clockwise, matching standard map and geodesic conventions.

Accepts a single polar axis or an array of axes — useful when working
with multi-panel figures.

.. py:function:: setup_geodesic_polar_ax(ax, *, deg_ticks=True, step_deg=30)

   :param ax: A polar axis or an array of polar axes to configure.
   :param deg_ticks: Show degree labels on the angular axis. Defaults to ``True``.
   :param step_deg: Spacing between degree labels in degrees. Defaults to ``30``.
   :returns: ``None`` — axes are modified in place.
