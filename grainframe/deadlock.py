"""Infeasibility detection and priority-based deadlock resolution.

WHY THIS MODULE EXISTS
----------------------
A Dubins car has a minimum turn radius. Inside some distance there is simply
no steering input that clears an oncoming vehicle -- the manoeuvre is not
merely expensive, it is kinematically impossible. No amount of gain tuning on
the repulsion term fixes that; the gain saturates against the heading clamp
and the vehicle orbits uselessly. Braking is the one control authority that
does not depend on turn radius, so a leader that cannot steer clear must stop.

That immediately raises the question this module answers: if BOTH vehicles
stop, nobody moves again. Deadlock.

THE LITERATURE
--------------
Deadlock in reactive multi-robot collision avoidance is a well-characterised
phenomenon rather than an implementation defect:

  Grover, Liu & Sycara, "The Before, During, and After of Multi-robot
  Deadlock", IJRR 42(6):317-336, 2023.
      Deadlock is a force-equilibrium on the robots, and it occurs precisely
      to preserve safety when safety is on the brink of being violated. The
      deadlock set is non-empty and lies on the BOUNDARY of the safe set --
      i.e. a deadlocked configuration is a safe one, which is why our frozen
      leaders never collide, they just never arrive. The authors also show the
      number of admissible deadlock configurations grows roughly
      exponentially in the number of robots, which is the practical argument
      against trying to detect and reason about the equilibrium directly and
      in favour of a cheap static rule like the one below.

  Duhaut et al. (2007), as summarised in the above.
      Without sharing information about the global situation of the system,
      deadlock cannot be avoided. Their proposal is a random-move plus
      priority-based resolution rule. The negative result is the important
      part for us: a purely local symmetric rule CANNOT get out of this, so
      some asymmetry has to be injected from outside. Priority is that
      asymmetry.

  Zhou et al. (2017) and others use a geometric convention instead -- the
      rightmost agent moves first -- which needs no identity or ranking at
      all. See avoidance.swirl() for the handedness version of the same idea.
      Worth contrasting: geometric conventions scale without coordination,
      static ranks require agreed identities but are trivially auditable.

COLREGS MAPPING
---------------
The maritime rules encode the same structure and are the natural framing for
a paper about rules of the road:

  Rule 16  Give-way vessel takes early and substantial action to keep clear.
  Rule 17  Stand-on vessel maintains course and speed.
  Rule 17(b)  If collision cannot be avoided by the give-way vessel's action
           alone, the stand-on vessel shall take such action as will best aid
           to avoid collision.
  Rule 18  A static hierarchy BY VESSEL TYPE (not under command, restricted in
           ability to manoeuvre, constrained by draft, fishing, sailing,
           power-driven). This is the closest analogue to leaderPriority.

One caution, because it cuts against the obvious design. Rule 15 assigns
give-way by GEOMETRY (the vessel with the other on her starboard side gives
way), and Rule 14 says that in a HEAD-ON both vessels alter to starboard --
there is no give-way/stand-on distinction at all. So a static rank is the
right model for mixed vehicle classes, but for a symmetric head-on the
literature's answer is a mutual handed turn, not a hierarchy.
"""
import numpy as np


def canAvoid(x, y, th, xOther, yOther, thOther, opts,
             lookahead, margin, vOther=None):
    """Can this vehicle clear the other by turning as hard as it legally can?

    Rolls both forward `lookahead` steps: the other holds heading at vOther,
    I turn at my maximum rate (v/R) in whichever direction opens the gap. If
    even that best case stays closer than `margin`, no feasible avoidance
    exists and the caller should stop.

    Returns True if a feasible avoidance manoeuvre exists.

    lookahead is a DISTANCE in disguise: steps * v * dt. At v=1, dt=0.01, a
    value of 3200 is 32 units, about one full 2*pi*R circle at R=5. Values
    that do not cover at least a full turning circle silently collapse this
    into "am I inside margin right now", which is not a feasibility test.

    vOther: pass 0.0 for an already-stopped vehicle. This matters more than it
    looks. Assuming a frozen obstacle is still closing at full speed makes the
    situation appear permanently infeasible, so the second vehicle freezes in
    sympathy and the pair locks up forever -- a deadlock manufactured entirely
    by the feasibility test's own pessimism.

    Vectorised: under a constant turn rate the rollout is closed-form, so this
    is one cumsum rather than a Python loop over the horizon.
    """
    dt, v, R = opts.dt, opts.v, opts.R
    if vOther is None:
        vOther = v
    dpsiMax = (v / R) * dt                  # most I can rotate per step

    # Which way opens the gap? Sign of cross(heading, toward-other).
    toX, toY = xOther - x, yOther - y
    cross = np.cos(th) * toY - np.sin(th) * toX
    turn = -np.sign(cross) * dpsiMax        # turn AWAY

    n = np.arange(1, lookahead + 1)
    thi = th + turn * n                     # my heading each step
    xi = x + np.cumsum(v * np.cos(thi) * dt)
    yi = y + np.cumsum(v * np.sin(thi) * dt)
    xo = xOther + vOther * np.cos(thOther) * dt * n
    yo = yOther + vOther * np.sin(thOther) * dt * n

    return float(np.min(np.hypot(xi - xo, yi - yo))) >= margin


def yieldDecision(k, other, trapped, priority, prevStopped, useRule17b=False):
    """Should vehicle k stop? Priority rule, COLREGs Rules 16/17.

    k, other    : indices of the two vehicles in question
    trapped     : bool, canAvoid() already came back False for k
    priority    : list of ranks indexed by vehicle. LOWER = stand-on (holds
                  course), HIGHER = give-way (yields first).
    prevStopped : freeze flags from a SNAPSHOT taken before any vehicle in
                  this timestep decided. Reading a partially-updated list here
                  makes loop index act as a hidden priority rule, which is a
                  genuinely nasty bug because the simulation still looks
                  plausible -- somebody yields, it just isn't the one the
                  ranking says.
    useRule17b  : see the warning below.

    Ties in rank fall back to index order, so a tie can never leave both
    vehicles standing on. That fallback is the whole point: the resolution
    must be total, not merely usually-decisive.
    """
    if not trapped:
        return False

    # Rule 16: the lower-ranked vehicle gives way. Tuple comparison appends
    # the index as a tiebreak so the ordering is total.
    if (priority[k], k) > (priority[other], other):
        return True

    # Rule 17(b): the give-way vehicle has already yielded and I am STILL
    # trapped, so its action alone was not enough and I must act too.
    #
    # WARNING, measured: implementing "act" as "stop" produces permanent
    # gridlock in a head-on. Once the give-way vehicle halts it becomes a
    # stationary obstacle dead ahead, which never stops being infeasible, so
    # the stand-on vehicle freezes too and neither recovers. Observed at
    # PR_RED = PR_GOLD = 3.0: both leaders frozen ~11000 steps, zero
    # cross-track, neither reaching its goal. Note this is exactly Grover et
    # al.'s characterisation -- a safe configuration on the boundary of the
    # safe set, holding forever.
    #
    # The correct reading of 17(b) is "such action as will best aid to avoid
    # collision", which in a head-on means TURN (Rule 14), not stop. That is a
    # change to the steering path, not to this predicate, so it is left off by
    # default here rather than shipped broken.
    if useRule17b and prevStopped[other]:
        return True

    return False