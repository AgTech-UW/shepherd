"""Path-following guidance laws for a Dubins-car leader.

Extracted from the exp5 sandbox so several experiments can share one
implementation. Every tunable is an explicit argument rather than a module
global: that is what lets a single run call the same law twice with different
gains, which is exactly what the before/after comparison figures need.

  vector field -- Nelson, Barber, McLain & Beard, "Vector Field Path Following
                  for Miniature Air Vehicles", IEEE T-RO 23(3), 2007.
"""
import numpy as np


def guidanceWaypoint(x, y, wpX, wpY, gain):
    """Aim straight at the active waypoint. The legacy law, kept for contrast.

    This oscillates, and the reason is structural rather than a tuning fault:
    it nulls POSITION error but says nothing about heading. Arriving at a
    waypoint from off-line means arriving with close to 90 degrees of heading
    error every single time, so the vehicle overshoots, turns back, and limit
    cycles. Useful precisely because it reproduces that failure on demand.
    """
    return (wpX - x) * gain, (wpY - y) * gain


def guidanceVectorField(x, y, segA, segB, gain, kCross, chiInf, magnitude):
    """Vector field path following (Nelson et al. 2007).

    Defines, for every point in the plane, the heading you SHOULD hold. Far
    off the line you approach at chiInf; as cross-track error shrinks the
    desired heading rotates smoothly onto the path direction. Cross-track and
    heading error therefore null at the same instant, which is what removes
    the limit cycle the waypoint law suffers from.

    segA -> segB  : the path segment being tracked
    kCross        : how sharply the field bends back to the line. Too small is
                    a lazy recovery; too large demands more curvature than the
                    car has and you get chatter. 0.4-0.6 is the useful band at
                    R = 5.
    chiInf        : approach angle when far off the line, in radians. Must be
                    strictly under pi/2 -- 90 degrees means "fly straight at
                    the line", which is not trackable.
    magnitude     : length of the returned vector. MUST be constant. Using the
                    distance-to-waypoint here (as the old code did) makes the
                    vector shrink from 10 to 0 across each leg, silently
                    changing the balance between guidance and avoidance
                    depending on where in the leg the encounter happens.
    """
    seg = np.asarray(segB, dtype=float) - np.asarray(segA, dtype=float)
    L = np.linalg.norm(seg)
    if L < 1e-9:                      # degenerate segment: caller should fall
        return None                   # back to the waypoint law

    u = seg / L                                   # along-track unit vector
    n = np.array([-u[1], u[0]])                   # left normal
    e = (x - segA[0]) * n[0] + (y - segA[1]) * n[1]   # signed cross-track error

    chiPath = np.arctan2(u[1], u[0])
    # |e| large  -> atan saturates -> chiD = chiPath -+ chiInf
    # e == 0     -> atan is 0      -> chiD = chiPath exactly
    chiD = chiPath - chiInf * (2.0 / np.pi) * np.arctan(kCross * e)

    return magnitude * gain * np.cos(chiD), magnitude * gain * np.sin(chiD)


def crossTrackError(xs, ys, refPath):
    """Signed perpendicular distance from a reference line, over a whole run.

    This is the number a guidance law is supposed to improve, so it lives here
    next to the laws rather than in the experiment's reporting code.
    """
    a = refPath[0]
    u = (refPath[-1] - a) / np.linalg.norm(refPath[-1] - a)
    n = np.array([-u[1], u[0]])
    return (xs - a[0]) * n[0] + (ys - a[1]) * n[1]