"""A single total ordering over every agent on the field.

WHY A TOTAL ORDER
-----------------
The leaders needed a priority rule because two vehicles that both stop are
deadlocked forever. Exactly the same argument applies to the wingmen, and to
wingman-versus-leader encounters. A speed floor (let everyone crawl rather
than freeze) makes deadlock slow instead of resolving it, and leaves the
outcome dependent on whatever asymmetry happens to be in the initial
conditions.

Duhaut et al. (2007), as reviewed in Grover, Liu & Sycara (IJRR 42(6), 2023):
without information about the global state of the system, deadlock cannot be
avoided; priority is the standard resolution. Grover et al. further show the
number of admissible deadlock configurations grows roughly exponentially in
agent count -- so for a field of two captains and a dozen wingmen, reasoning
about the equilibrium is hopeless and a cheap static rank is the right tool.

THE ORDERING
------------
Each agent gets a key. LOWER sorts first and means STAND-ON: hold course, do
not yield. HIGHER means GIVE-WAY: you are the one who moves.

    key = (tier, squadRank, indexWithinSquad)

  tier 0 = captains, tier 1 = wingmen.
      A captain always outranks every wingman, of any squadron. This is the
      "captains trump fans, always" rule, and it is what makes a captain's
      path-following exact: nothing a wingman does can perturb it.

  squadRank ranks the squadrons against each other, so a whole squadron can
      be given way to as a unit.

  indexWithinSquad breaks the remaining ties, so no two agents anywhere on
      the field compare equal. Totality is the point: a rule that is usually
      decisive still deadlocks on the cases where it is not.

WHAT YIELDING MEANS DEPENDS ON WHO YOU ARE YIELDING TO
------------------------------------------------------
This is the part that is easy to get wrong, and it is not symmetric.

  wingman yielding to WINGMAN -> brake. Both are manoeuvrable, both can be
      given way to, and the ranking guarantees only one of any pair brakes.
      Nobody ends up parked in anybody's path.

  wingman yielding to CAPTAIN -> get out of the lane, at speed. NEVER brake.
      A captain is a constant-speed vehicle that does not yield to wingmen by
      construction, so a stopped wingman in front of one is not avoiding a
      collision, it is guaranteeing one -- you become a stationary target.
      The correct response is lateral: leave the corridor at full speed.
"""
import numpy as np


CAPTAIN_TIER = 0
WINGMAN_TIER = 1


def buildRanks(squad, nLeaders, squadRank=None):
    """Rank keys for every agent, captains first then wingmen.

    squad     : squadron index per wingman
    nLeaders  : number of captains (one per squadron)
    squadRank : ordering over squadrons, indexed by squadron id. None gives
                the identity, i.e. squadron 0 outranks squadron 1.

    Returns (capKeys, fanKeys) as (n, 3) integer arrays. Compare rows
    lexicographically: smaller is stand-on.
    """
    if squadRank is None:
        squadRank = np.arange(nLeaders)
    squadRank = np.asarray(squadRank)

    capKeys = np.stack([np.full(nLeaders, CAPTAIN_TIER),
                        squadRank,
                        np.zeros(nLeaders, dtype=int)], axis=1)

    # index within squadron, assigned in order of appearance
    idxWithin = np.zeros(len(squad), dtype=int)
    for k in range(nLeaders):
        m = (squad == k)
        idxWithin[m] = np.arange(np.count_nonzero(m))

    fanKeys = np.stack([np.full(len(squad), WINGMAN_TIER),
                        squadRank[squad],
                        idxWithin], axis=1)
    return capKeys, fanKeys


def _lexLess(A, B):
    """Row-wise lexicographic A < B for (n, 3) integer key arrays broadcast
    into (n, m, 3). Returns an (n, m) boolean.

    Written out rather than using np.lexsort because we need the pairwise
    comparison matrix, not a sort order.
    """
    a0, a1, a2 = A[..., 0], A[..., 1], A[..., 2]
    b0, b1, b2 = B[..., 0], B[..., 1], B[..., 2]
    return ((a0 < b0) |
            ((a0 == b0) & (a1 < b1)) |
            ((a0 == b0) & (a1 == b1) & (a2 < b2)))


def yieldMatrix(fanKeys, otherKeys):
    """(nFan, nOther) boolean: does fan i give way to agent j?

    True where j outranks i, i.e. j's key sorts strictly first.
    """
    A = fanKeys[:, None, :]
    B = otherKeys[None, :, :]
    return _lexLess(B, A)


def fanBrakeRanked(fanX, fanY, fanVx, fanVy, fanKeys,
                   brakeRange, brakeStop, floor=0.0):
    """Speed multiplier per wingman, braking only for HIGHER-RANKED WINGMEN.

    Captains are deliberately excluded -- see the module docstring. Braking in
    front of a vehicle that never yields is worse than not braking.

    Returns a multiplier applied to DISPLACEMENT only, leaving the velocity
    state alone so heading stays well defined and the Ackermann clamp keeps
    working on the next step.

    floor: minimum multiplier. With a total order this can safely be 0.0 --
    only one of any pair ever brakes, so a full stop cannot deadlock. A
    non-zero floor is available for comparison against the old symmetric
    scheme, where it was load-bearing.
    """
    n = fanX.size
    if n == 0:
        return np.ones(0)

    dx = fanX[None, :] - fanX[:, None]      # from i to j
    dy = fanY[None, :] - fanY[:, None]
    d = np.hypot(dx, dy)
    dSafe = np.where(d > 1e-9, d, 1.0)

    # closing rate along the line of sight; negative = closing
    rdot = ((fanVx[None, :] - fanVx[:, None]) * dx +
            (fanVy[None, :] - fanVy[:, None]) * dy) / dSafe

    givesWay = yieldMatrix(fanKeys, fanKeys)        # (n, n)
    near = (d > 1e-9) & (d < brakeRange) & (rdot < 0.0) & givesWay

    dMin = np.where(near, d, np.inf).min(axis=1)
    s = (dMin - brakeStop) / (brakeRange - brakeStop)
    return np.where(np.isfinite(dMin), np.clip(s, floor, 1.0), 1.0)


def corridorClear(fanX, fanY, capX, capY, capTh, range_, halfWidth, gain):
    """Lateral push out of a captain's lane. The wingman-versus-captain rule.

    Decomposes each wingman's offset from each captain into along-track (a)
    and cross-track (c) in that captain's body frame. A wingman that is AHEAD
    of the captain (a > 0), within range_, and inside the lane (|c| <
    halfWidth) is pushed along the captain's normal, away from the centreline,
    hardest when dead centre and when closest.

    This is the "get out of the way at speed" half of the asymmetry. It moves
    the wingman sideways rather than slowing it, which is the only response
    that actually clears a vehicle that will not yield.

    Vectorised over (wingmen, captains).
    """
    ux, uy = np.cos(capTh), np.sin(capTh)           # (nCap,)
    nx, ny = -uy, ux

    dx = fanX[:, None] - capX[None, :]              # (nFan, nCap)
    dy = fanY[:, None] - capY[None, :]
    a = dx * ux[None, :] + dy * uy[None, :]         # along-track
    c = dx * nx[None, :] + dy * ny[None, :]         # cross-track

    inLane = (a > 0.0) & (a <= range_) & (np.abs(c) < halfWidth)
    side = np.where(c >= 0, 1.0, -1.0)              # exit the near side
    side = np.where(np.abs(c) < 1e-6, 1.0, side)    # dead centre: consistent

    w = gain * (1.0 - np.abs(c) / halfWidth) * (1.0 - a / range_)
    w = np.where(inLane, w, 0.0)

    return ((w * side * nx[None, :]).sum(axis=1),
            (w * side * ny[None, :]).sum(axis=1))


def spawnFormation(capX, capY, capTh, nPerSquad, spacing, rows,
                   minRadius, speed, jitter=0.0, rng=None):
    """Deterministic squadron spawn: every wingman starts in its slot, outside
    a clear radius of its captain, heading the same way the captain is.

    This replaces random gaussian scatter around the captain. On a real field
    the initial and final states are known quantities -- vehicles are placed,
    not sprinkled -- so a random start is modelling a condition that does not
    occur, and it does so in the worst possible way: gaussian scatter puts
    some fraction of wingmen INSIDE the protected range of the captain and of
    each other at t = 0. Those agents begin in violation and spend the opening
    seconds resolving a conflict that should never have existed, which
    contaminates the collision statistics with artefacts of the initial
    condition rather than the behaviour under study.

    Three guarantees, all enforced by construction rather than by rejection
    sampling:

      1. Every wingman is at least `minRadius` from its captain. Slots are
         laid out at along-track offset -(minRadius + row * spacing), so the
         along-track component alone already exceeds minRadius.
      2. Adjacent slots are `spacing` apart. Set spacing > opts.PR or the
         formation is self-repelling: separation will fight the slot pull and
         the formation cannot hold.
      3. Every wingman starts on its captain's heading, at a common speed. No
         initial heading transient, so the opening seconds are not spent
         turning the squadron around.

    jitter: optional gaussian noise on slot positions, default 0 for a fully
    deterministic start. Use it only to test robustness to placement error.

    Returns (fanX, fanY, fanVx, fanVy, squad, slotAlong, slotLateral), all
    flat across squadrons.
    """
    capX = np.atleast_1d(capX)
    capY = np.atleast_1d(capY)
    capTh = np.atleast_1d(capTh)
    nCap = len(capX)

    slotA, slotL = formationSlots(nPerSquad, spacing, rows, minRadius)

    fanX, fanY, fanVx, fanVy, squad = [], [], [], [], []
    for k in range(nCap):
        ux, uy = np.cos(capTh[k]), np.sin(capTh[k])
        x = capX[k] + slotA * ux - slotL * uy
        y = capY[k] + slotA * uy + slotL * ux
        if jitter > 0.0:
            r = rng if rng is not None else np.random
            x = x + r.normal(0, jitter, nPerSquad)
            y = y + r.normal(0, jitter, nPerSquad)
        fanX.append(x)
        fanY.append(y)
        # everyone points the way their captain points
        fanVx.append(np.full(nPerSquad, speed * ux))
        fanVy.append(np.full(nPerSquad, speed * uy))
        squad.append(np.full(nPerSquad, k))

    return (np.concatenate(fanX), np.concatenate(fanY),
            np.concatenate(fanVx), np.concatenate(fanVy),
            np.concatenate(squad),
            np.tile(slotA, nCap), np.tile(slotL, nCap))


def sectorCapacity(minRadius, maxRadius, halfAngle, minSep, slack=3.0):
    """How many wingmen fit in the rear sector, and the radius needed for n.

    The annular sector has area halfAngle * (Rmax^2 - Rmin^2). Hexagonal
    packing of points at separation s consumes about (sqrt(3)/2) s^2 each, so
    the geometric capacity is that ratio.

    But geometric capacity is not ACHIEVABLE capacity for rejection sampling.
    Placing points uniformly at random and rejecting clashes stalls long
    before the packing limit -- at 90% density essentially every candidate is
    rejected. `slack` is the safety factor on area; 3.0 keeps the acceptance
    rate high enough that placement terminates quickly.

    Returns (capacity, requiredMaxRadius) where capacity is how many fit in
    the given band, and requiredMaxRadius is the outer radius that would be
    needed for a requested count (see radiusFor below).
    """
    area = halfAngle * (maxRadius ** 2 - minRadius ** 2)
    per = slack * (np.sqrt(3.0) / 2.0) * minSep ** 2
    return int(area / per)


def radiusForCount(n, minRadius, halfAngle, minSep, slack=3.0):
    """Outer radius needed to place n wingmen comfortably. Inverse of above."""
    per = slack * (np.sqrt(3.0) / 2.0) * minSep ** 2
    return float(np.sqrt(minRadius ** 2 + n * per / halfAngle))


def spawnRandomBehind(capX, capY, capTh, nPerSquad, minRadius, maxRadius,
                      halfAngle, minSep, speed, rng=None, maxTries=4000,
                      autoExpand=True, verbose=False):
    """Random squadron spawn in the sector BEHIND each captain, headings aligned.

    Placement is random, which is what you want when the point is that the
    formation assembles itself rather than being handed to the controller. But
    it is random subject to constraints, which is the difference between this
    and a gaussian blob:

      minRadius, maxRadius : radial band behind the captain. minRadius keeps
          every wingman outside the captain's protected range at t = 0, so
          nobody opens the run already in violation.
      halfAngle : half-width of the rear sector, in radians. pi/2 is the whole
          rear hemisphere; smaller values keep the squadron in a tighter cone
          astern.
      minSep : minimum distance between any two wingmen of the same squadron.
          Set this above opts.PR or wingmen spawn inside each other's
          protected range and the opening seconds are spent unpiling.

    autoExpand : if the requested band cannot comfortably hold nPerSquad
        wingmen, push maxRadius out to a radius that can, rather than failing.
        This is on by default because the alternative is worse in both
        directions: raising kills a long parameter sweep partway through, and
        silently returning an overlapping layout defeats the entire purpose of
        the function. Expanding the band changes only how far back the
        squadron starts, which the formation controller then closes up anyway.

        The capacity check is a real packing bound, not a guess -- see
        sectorCapacity(). Rejection sampling stalls well below the geometric
        packing limit, so the bound carries a slack factor.

    Headings are NOT random: every wingman starts on its captain's heading at
    a common speed, so there is no initial turning transient. That is the part
    that is known on a real field even when the exact positions are not.

    Returns (fanX, fanY, fanVx, fanVy, squad), flat across squadrons.
    """
    capX = np.atleast_1d(capX)
    capY = np.atleast_1d(capY)
    capTh = np.atleast_1d(capTh)
    r = rng if rng is not None else np.random

    if autoExpand:
        need = radiusForCount(nPerSquad, minRadius, halfAngle, minSep)
        if need > maxRadius:
            if verbose:
                print('spawnRandomBehind: band [%.1f, %.1f] too small for %d '
                      'wingmen at minSep=%.1f; expanding outer radius to %.1f'
                      % (minRadius, maxRadius, nPerSquad, minSep, need))
            maxRadius = need

    fanX, fanY, fanVx, fanVy, squad = [], [], [], [], []
    for k in range(len(capX)):
        ux, uy = np.cos(capTh[k]), np.sin(capTh[k])
        nx, ny = -uy, ux
        xs, ys = [], []
        tries = 0
        while len(xs) < nPerSquad:
            tries += 1
            if tries > maxTries:
                raise RuntimeError(
                    'spawnRandomBehind: could not place %d wingmen with '
                    'minSep=%.2f in band [%.1f, %.1f], halfAngle=%.2f rad '
                    '(capacity about %d). Widen the band, widen halfAngle, '
                    'reduce minSep, or leave autoExpand=True.'
                    % (nPerSquad, minSep, minRadius, maxRadius, halfAngle,
                       sectorCapacity(minRadius, maxRadius, halfAngle,
                                      minSep)))
            # uniform in the annular sector, area-weighted so the band does
            # not bunch toward the inner radius
            rad = np.sqrt(r.uniform(minRadius ** 2, maxRadius ** 2))
            ang = r.uniform(-halfAngle, halfAngle)
            # measured from DIRECTLY ASTERN, hence the leading minus
            along = -rad * np.cos(ang)
            lat = rad * np.sin(ang)
            x = capX[k] + along * ux + lat * nx
            y = capY[k] + along * uy + lat * ny
            if xs:
                d = np.hypot(np.array(xs) - x, np.array(ys) - y)
                if d.min() < minSep:
                    continue
            xs.append(x)
            ys.append(y)
        fanX.append(np.array(xs))
        fanY.append(np.array(ys))
        fanVx.append(np.full(nPerSquad, speed * ux))
        fanVy.append(np.full(nPerSquad, speed * uy))
        squad.append(np.full(nPerSquad, k))

    return (np.concatenate(fanX), np.concatenate(fanY),
            np.concatenate(fanVx), np.concatenate(fanVy),
            np.concatenate(squad))


def formationSlots(n, spacing, rows, minRadius=0.0, shape='column',
                   sweep=None):
    """Slot offsets (along, lateral) in the captain's body frame.

    A whole squadron sharing ONE attractor is a guaranteed pile-up: every
    wingman's target is a point every other wingman is also trying to occupy
    and simultaneously required to avoid, so the equilibrium is a jammed shell
    at radius PR around the captain. Distinct attractors are separated by
    construction -- nobody is trying to stand where somebody else stands.

    Slots trail the captain in rows, alternating left and right of the
    centreline so the formation is balanced. `minRadius` pushes the whole
    formation back so that even the nearest slot clears the captain by that
    much -- since the along-track offset alone is at least minRadius, the
    Euclidean distance is too, with no trigonometry needed.

    shape:
      'column'  -- trailing grid, `rows` wingmen abreast per rank. The
                   original layout. Compact, but wingmen directly astern sit
                   in each other's wake and in the captain's corridor.
      'vee'     -- flying V. Wingmen alternate left and right, each one
                   further out AND further back than the last, so no wingman
                   is directly behind another and none is in the captain's
                   lane. This is the formation birds and aircraft actually
                   use, for exactly that reason.
      'echelon' -- all wingmen stacked to one side in a diagonal line. Useful
                   when the corridor on one side must stay clear.

    sweep: for 'vee' and 'echelon', the angle back from abeam, in radians.
           None gives 45 degrees. Smaller is a wider, flatter V; larger is a
           tighter, deeper one.
    """
    if sweep is None:
        sweep = np.pi / 4.0

    along, lat = [], []

    if shape == 'column':
        for i in range(n):
            row = i // rows
            col = i % rows
            offset = ((col + 1) // 2) * spacing * (1 if col % 2 else -1)
            along.append(-(minRadius + row * spacing))
            lat.append(offset)

    elif shape == 'vee':
        # rank 1 is the pair closest to the captain, alternating sides.
        # Distance along the V arm grows with rank, so lateral and along-track
        # offsets both grow -- that is what keeps every wingman clear of the
        # one ahead of it and out of the captain's corridor.
        for i in range(n):
            rank = i // 2 + 1
            side = 1.0 if i % 2 == 0 else -1.0
            arm = minRadius + (rank - 1) * spacing
            along.append(-arm * np.cos(sweep))
            lat.append(side * arm * np.sin(sweep))

    elif shape == 'echelon':
        for i in range(n):
            arm = minRadius + i * spacing
            along.append(-arm * np.cos(sweep))
            lat.append(arm * np.sin(sweep))

    else:
        raise ValueError('unknown formation shape %r' % (shape,))

    return np.array(along, dtype=float), np.array(lat, dtype=float)