"""Per-robot plan marker colors and the combined, latched plan visualization."""

import re

from visualization_msgs.msg import Marker, MarkerArray

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def hex_to_rgba(value, alpha=1.0):
    """``#rrggbb`` -> (r, g, b, a) floats, or None if ``value`` is not a hex color."""
    m = _HEX_RE.match(str(value or "").strip())
    if m is None:
        return None
    h = m.group(1)
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)) + (float(alpha),)


def parse_robot_colors(value):
    """``robot_colors`` param -> {robot_name: (r, g, b, a)}.

    Accepts ``"name:#hex,name:#hex"`` or a string array of ``name:#hex`` items
    (the format dcist_launch_system.robot_colors emits). Malformed entries and
    placeholders like ``"none"`` are skipped so a bad param never stops planning.
    """
    items = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    colors = {}
    for item in items:
        name, sep, color = str(item).strip().partition(":")
        rgba = hex_to_rgba(color)
        if sep and name.strip() and rgba is not None:
            colors[name.strip()] = rgba
    return colors


def combine_plan_markers(markers_by_robot):
    """One MarkerArray holding every robot's latest plan markers.

    RViz2 applies DELETEALL to the whole MarkerArray display (all namespaces),
    so the array leads with DELETEALL to drop stale plans and then always carries
    every robot's markers; a late (transient_local) subscriber gets the full
    multi-robot picture from this single message.
    """
    clear = Marker()
    clear.action = Marker.DELETEALL
    ma = MarkerArray()
    ma.markers.append(clear)
    for robot_name in sorted(markers_by_robot):
        ma.markers.extend(markers_by_robot[robot_name])
    return ma
