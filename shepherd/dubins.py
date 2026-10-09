"""Shortest Dubins paths between two poses, as executable arcs.

Six candidate words (LSL, RSR, LSR, RSL, RLR, LRL), shortest wins. Formulas
follow Shkel & Lumelsky (2001) in the normalised form used by A. Walker's
dubins.c. The output is a list of (turn, length) pieces: turn = +1 left,
-1 right, 0 straight, length in metres of arc. The environment executes
those at whatever speed the leader is holding.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def _mod(a):
    return a - TWO_PI * np.floor(a / TWO_PI)


def _words(a, b, d):
    sa, sb, ca, cb = np.sin(a), np.sin(b), np.cos(a), np.cos(b)
    cab = np.cos(a - b)
    out = {}

    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp = np.arctan2(cb - ca, d + sa - sb)
        out['LSL'] = (_mod(-a + tmp), np.sqrt(p2), _mod(b - tmp))

    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp = np.arctan2(ca - cb, d - sa + sb)
        out['RSR'] = (_mod(a - tmp), np.sqrt(p2), _mod(-b + tmp))

    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = np.sqrt(p2)
        tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        out['LSR'] = (_mod(-a + tmp), p, _mod(-_mod(b) + tmp))

    p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = np.sqrt(p2)
        tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        out['RSL'] = (_mod(a - tmp), p, _mod(b - tmp))

    tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1:
        p = _mod(TWO_PI - np.arccos(tmp))
        t = _mod(a - np.arctan2(ca - cb, d - sa + sb) + p / 2.0)
        out['RLR'] = (t, p, _mod(a - b - t + p))

    tmp = (6 - d * d + 2 * cab + 2 * d * (-sa + sb)) / 8.0
    if abs(tmp) <= 1:
        p = _mod(TWO_PI - np.arccos(tmp))
        t = _mod(-a - np.arctan2(ca - cb, d + sa - sb) + p / 2.0)
        out['LRL'] = (t, p, _mod(_mod(b) - a - t + p))
    return out


_TURN = {'L': 1, 'R': -1, 'S': 0}


def dubinsShortest(q0, q1, R):
    """Shortest Dubins path from pose q0 to pose q1 at turn radius R.

    Returns (word, pieces, length) with pieces = [(turn, metres), ...].
    """
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    d = np.hypot(dx, dy) / R
    th = _mod(np.arctan2(dy, dx))
    a, b = _mod(q0[2] - th), _mod(q1[2] - th)
    best = None
    for word, (t, p, q) in _words(a, b, d).items():
        L = t + p + q
        if best is None or L < best[2]:
            best = (word, (t, p, q), L)
    word, params, L = best
    pieces = [(_TURN[c], float(s * R)) for c, s in zip(word, params) if s * R > 1e-9]
    return word, pieces, float(L * R)


def integrateArcs(q0, pieces, R, step=0.05):
    """Exact endpoint and a sampled polyline of a list of (turn, metres) pieces."""
    x, y, th = map(float, q0)
    pts = [(x, y)]
    for turn, L in pieces:
        n = max(1, int(np.ceil(L / step)))
        ds = L / n
        for _ in range(n):
            if turn == 0:
                x += ds * np.cos(th)
                y += ds * np.sin(th)
            else:
                k = turn / R
                thN = th + k * ds
                x += (np.sin(thN) - np.sin(th)) / k
                y += -(np.cos(thN) - np.cos(th)) / k
                th = thN
            pts.append((x, y))
    return np.array([x, y, th]), np.array(pts)
