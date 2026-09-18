"""Convert SVG A commands into cubic Bezier curves.

The arc is converted according to the SVG specification, then split
into pieces of at most 90 degrees and approximated with cubic Bézier
segments.
"""

import math

__all__ = ["arc_to_cubics"]

# A cubic Bezier works well for an arc up to 90 degrees.
# Larger arcs are split into smaller pieces for better accuracy.
MAX_SWEEP = math.pi / 2.0


def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
    """
    Calculate the signed angle from vector (ux, uy) to vector (vx, vy).

    Parameters:
        ux, uy: x and y components of the first vector.
        vx, vy: x and y components of the second vector.

    Returns:
        The angle in radians.
        Positive and negative signs indicate the turn direction.
    """
    denominator = math.hypot(ux, uy) * math.hypot(vx, vy)
    if denominator == 0.0:
        return 0.0

    # clamps cosine to -1 to 1, acos rejects anything outside
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / denominator))
    theta = math.acos(cosine)

    # Cross product sign decides whether the angle is positive or negative.
    return -theta if (ux * vy - uy * vx) < 0.0 else theta


def arc_to_cubics(x1, y1, rx, ry, rotation, large_arc, sweep, x2, y2):
    """Convert one SVG arc command into cubic Bezier segments.

    Parameters:
        x1, y1: Starting point of the arc.
        rx, ry: Horizontal and vertical radii of the ellipse.
        rotation: Rotation of the ellipse's x-axis in degrees.
        large_arc: 0 for the smaller arc, 1 for the larger arc.
        sweep: 0 for counterclockwise, 1 for clockwise.
        x2, y2: Ending point of the arc.

    Returns:
        A list of cubic Bezier segments as
        (c1x, c1y, c2x, c2y, x, y).
    """
    # Reject non-finite or invalid values
    for v in (x1, y1, rx, ry, rotation, x2, y2):
        if not math.isfinite(v):
            return []

    # If the arc's start and end points are identical, don't draw the arc.
    if x1 == x2 and y1 == y2:
        return []

    rx, ry = abs(rx), abs(ry)

    # If rx or ry is zero, draw a straight line instead of an arc.
    if rx == 0.0 or ry == 0.0:
        return [(x1, y1, x2, y2, x2, y2)]

    phi = math.radians(rotation % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    # Step 1: Find half the distance between the start and end points,
    #  then consider the ellipse's rotation.
    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    # Step 2: Check if the radii are large enough to reach both endpoints
    # If they are too small, increase both radii until they are large enough.
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    # Step 3: Calculate the values needed to find the ellipse's center.
    numerator = (rx * rx * ry * ry
                 - rx * rx * y1p * y1p
                 - ry * ry * x1p * x1p)
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p

    if denominator == 0.0:
        return [(x1, y1, x2, y2, x2, y2)]

    # Prevent a tiny negative value from floating-point rounding before taking the square root.
    coefficient = math.sqrt(max(0.0, numerator / denominator))

    # Use the two flags to choose the correct ellipse center.
    if large_arc == sweep:
        coefficient = -coefficient

    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx

    # Step 4: Rotate the centre back into the original frame
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    # Step 5: Find the start and end directions of the arc.
    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry

    theta1 = _angle(1.0, 0.0, ux, uy)
    delta = _angle(ux, uy, vx, vy)

    # the sweep flag decides which way round
    if sweep == 0.0 and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep == 1.0 and delta < 0.0:
        delta += 2.0 * math.pi

    # Step 6: Split the arc into pieces of at most 90 degrees.
    count = max(1, int(math.ceil(abs(delta) / MAX_SWEEP - 1e-9)))
    step = delta / count

    # Calculate how far the Bezier control points should extend along the tangent.
    alpha = (4.0 / 3.0) * math.tan(step / 4.0)

    # Store the resulting cubic Bezier segments.
    segments = []

    # Start at the beginning of the arc.
    theta = theta1
    px, py = x1, y1

    for _ in range(count):
        theta_next = theta + step

        # tangent direction at the start of this piece
        d1x = -rx * cos_phi * math.sin(theta) - ry * sin_phi * math.cos(theta)
        d1y = -rx * sin_phi * math.sin(theta) + ry * cos_phi * math.cos(theta)

        # where this piece ends
        ex = cx + rx * cos_phi * math.cos(theta_next) - ry * sin_phi * math.sin(theta_next)
        ey = cy + rx * sin_phi * math.cos(theta_next) + ry * cos_phi * math.sin(theta_next)

        # tangent direction at the end
        d2x = -rx * cos_phi * math.sin(theta_next) - ry * sin_phi * math.cos(theta_next)
        d2y = -rx * sin_phi * math.sin(theta_next) + ry * cos_phi * math.cos(theta_next)

        segments.append((
            px + alpha * d1x, py + alpha * d1y,   # first control point
            ex - alpha * d2x, ey - alpha * d2y,   # second control point
            ex, ey,                               # endpoint
        ))

        px, py = ex, ey
        theta = theta_next

    # The last piece must end exactly at the requested endpoint.
    last = segments[-1]
    segments[-1] = (last[0], last[1], last[2], last[3], x2, y2)

    return segments
