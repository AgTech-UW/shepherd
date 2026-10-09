"""Vectorised multi-squadron boids update.

One call resolves every fanboid in every squadron for one timestep. The rules
are stock Reynolds (1987): separation against everyone, cohesion and alignment
within your own squadron only.

WHY VECTORISED. The obvious implementation loops over boids and calls a
per-boid rule function. That is what this replaced, and it was the dominant
cost of the whole simulation -- not the arithmetic, the dispatch. NumPy costs
roughly a microsecond per call regardless of array size, so a per-boid loop
over 4-to-8-element arrays spends nearly all its time in overhead. Measured on
an 8391-step run: 569 us/step at 3 fans per squad and 1758 us/step at 12,
against 232 and 257 us/step for this version. Note the scaling, which is the
real point -- the loop version grows linearly in fanboid count, this one is
nearly flat.
"""
import numpy as np


def flockStep(fanX, fanY, fanVx, fanVy, squad, leadX, leadY, leadTh, opts,
              sepOnlyOnEncounter=True, encounterRange=None,
              slotAlong=None, slotLateral=None,
              swirl=0.0, swirlHeadOnOnly=True, slotPullCap=None):
    """Desired velocities for every fanboid, one squadron-aware pass.

    fanX/fanY/fanVx/fanVy : flat arrays over ALL fanboids, every squadron
    squad                 : squadron index per fanboid, same length
    leadX/leadY/leadTh    : arrays over leaders, one per squadron
    returns               : (vxDes, vyDes), same shape as fanX

    Nothing here is hard-coded to two squadrons -- squadron membership is a
    mask comparison, so N squadrons is the same code.

    sepOnlyOnEncounter: suspend cohesion and alignment for a fanboid while any
    FOREIGN agent is within encounterRange.

    That switch is the fix for a specific failure. Cross-squadron coupling is
    already separation-only, because cohesion and alignment are fed same-squad
    neighbours only. The counter-productive term is a fanboid's OWN squadron:
    while two squadrons interpenetrate, cohesion pulls each boid back toward
    its squadmates' centre of mass and alignment pulls its heading back to the
    squad average, and both point straight back into the lane it is trying to
    separate out of. The rules cancel and the swerve never executes. Reynolds
    gives separation strict priority for exactly this reason.

    encounterRange: None -> opts.VR, the range cohesion and alignment act over
    anyway. Raising it above VR makes squadrons stop flocking EARLIER, which
    sounds backwards but gives each boid more runway to clear the lane before
    contact.
    """
    v = opts.v
    lvx, lvy = v * np.cos(leadTh), v * np.sin(leadTh)

    # Every agent a fanboid can perceive: all fanboids, then all leaders.
    # esquad tags each column with its squadron so "same squad" is a mask.
    eX = np.concatenate([fanX, leadX])
    eY = np.concatenate([fanY, leadY])
    eVx = np.concatenate([fanVx, lvx])
    eVy = np.concatenate([fanVy, lvy])
    esquad = np.concatenate([squad, np.arange(len(leadX))])

    # (nFan, nFan+nLeaders): row i is fanboid i's view of every agent j
    dx = fanX[:, None] - eX[None, :]        # vector FROM j TO i
    dy = fanY[:, None] - eY[None, :]
    D = np.hypot(dx, dy)

    notSelf = D > 1e-9
    same = esquad[None, :] == squad[:, None]

    # --- separation: EVERYONE inside the protected range, all squadrons ---
    prot = notSelf & (D <= opts.PR)
    # Force profile. 'flat' is the published form, F = SF * d, largest just
    # inside PR and vanishing at contact. 'norm' is the Reynolds 1999 form,
    # offset normalised and weighted 1/r, so F ~ SF * PR / D. The PR factor
    # scales it so the two profiles span the same magnitudes over (0, PR],
    # mirror-imaged: flat runs from ~0 at contact up to SF*PR at the
    # boundary, norm runs from SF*PR near contact down to SF at the
    # boundary. The weight applies to the swirl below as well, so the
    # rotated component keeps the same profile as the radial one.
    if getattr(opts, 'SEP_PROFILE', 'flat') == 'norm':
        Dsafe = np.where(D > 1e-9, D, 1e-9)
        wgt = opts.PR / (Dsafe * Dsafe)
    else:
        wgt = 1.0
    sepX = dx * wgt * prot
    sepY = dy * wgt * prot

    # --- SWIRL: a shared handedness, applied per neighbour pair ---
    #
    # Pure repulsion CANNOT resolve a symmetric head-on. Both agents sit on the
    # mirror line of the configuration, the repulsive force points ALONG that
    # line, and no gain produces lateral separation -- raising it just pushes
    # them back down their own tracks. Braking does not help either: slowing
    # for an agent that is closing head-on leaves you in its path with less
    # authority to steer out, and if only one of the pair yields the other runs
    # it down.
    #
    # Rotating the repulsion toward one side gives both agents a consistent
    # reason to go the same way relative to each other, so they pass
    # starboard-to-starboard. This is COLREGs Rule 14, which for a head-on
    # prescribes a mutual turn to starboard rather than a give-way/stand-on
    # hierarchy -- and unlike a priority rule it needs no agreed identities or
    # ranking between the agents, which is what lets it work between squadrons.
    #
    # swirlHeadOnOnly restricts it to genuinely head-on pairs (closing, and
    # roughly reciprocal courses). Applying a rotation to an overtaking pair
    # would push the slower agent sideways for no reason; that case is what the
    # brake is for. Set False to swirl every repulsion, which reproduces the
    # captain-level capSwirl behaviour in multifleet.
    if swirl != 0.0:
        if swirlHeadOnOnly:
            # closing rate along the line of sight; negative = closing
            rdot = ((eVx[None, :] - fanVx[:, None]) * (-dx) +
                    (eVy[None, :] - fanVy[:, None]) * (-dy))
            # reciprocal courses: my heading vs theirs, negative dot = opposed
            hdot = (fanVx[:, None] * eVx[None, :] +
                    fanVy[:, None] * eVy[None, :])
            headOn = (rdot < 0.0) & (hdot < 0.0)
        else:
            headOn = np.ones_like(prot, dtype=bool)

        use = prot & headOn
        # rotate 90 degrees in the world frame: (x, y) -> (-y, x). The sign
        # convention is checked on the canonical case in avoidance.swirl().
        sepX = np.where(use, sepX + swirl * (-dy * wgt * prot), sepX)
        sepY = np.where(use, sepY + swirl * (dx * wgt * prot), sepY)

    sX = np.sum(sepX, axis=1) * opts.SF
    sY = np.sum(sepY, axis=1) * opts.SF

    # --- cohesion + alignment: own squadron only, inside visual range ---
    vis = notSelf & (D <= opts.VR) & same
    cnt = vis.sum(axis=1)
    safeCnt = np.where(cnt > 0, cnt, 1)            # avoid 0/0 on empty rows
    xAvg = (eX[None, :] * vis).sum(axis=1) / safeCnt
    yAvg = (eY[None, :] * vis).sum(axis=1) / safeCnt
    vxAvg = (eVx[None, :] * vis).sum(axis=1) / safeCnt
    vyAvg = (eVy[None, :] * vis).sum(axis=1) / safeCnt

    cX = (xAvg - fanX) * opts.CF
    cY = (yAvg - fanY) * opts.CF
    aX = (vxAvg - fanVx) * opts.AF
    aY = (vyAvg - fanVy) * opts.AF

    seen = cnt > 0                                 # nobody in sight -> zero
    cX, cY, aX, aY = cX * seen, cY * seen, aX * seen, aY * seen

    # --- separation-only while a foreign agent is close ---
    if sepOnlyOnEncounter:
        encR = opts.VR if encounterRange is None else encounterRange
        engaged = (notSelf & ~same & (D <= encR)).any(axis=1)
        flock = ~engaged
        cX, cY, aX, aY = cX * flock, cY * flock, aX * flock, aY * flock

    # --- pull toward MY SLOT in my leader's formation, or the leader itself ---
    # One shared attractor for a whole squadron is a guaranteed pile-up: every
    # wingman's target is a point every other wingman is also trying to reach
    # and simultaneously required to avoid, so the equilibrium is a jammed
    # shell at radius PR around the leader. Measured: the dominant collision
    # type is a wingman against its OWN leader, gap 0.00 at 6 per squad.
    # Distinct slots are separated by construction, so the formation becomes
    # the controller rather than just the initial condition.
    if slotAlong is None:
        tx, ty = leadX[squad], leadY[squad]
    else:
        ux, uy = np.cos(leadTh[squad]), np.sin(leadTh[squad])
        tx = leadX[squad] + slotAlong * ux - slotLateral * uy
        ty = leadY[squad] + slotAlong * uy + slotLateral * ux
    lX = (tx - fanX) * opts.fanLeaderFactor
    lY = (ty - fanY) * opts.fanLeaderFactor

    # --- cap the slot pull so it cannot outgun separation at close range ---
    #
    # The flat Reynolds separation is F = SF * d: it SHRINKS as agents close,
    # reaching zero at contact. The slot pull is fanLeaderFactor * (distance
    # to slot) and does NOT shrink -- a wingman 20 from its slot is pulled at
    # 20 regardless of what it is about to hit.
    #
    # Setting the two equal gives the balance distance: SF*d = k*L, so
    # d = k*L/SF. Outside it separation wins; INSIDE it separation is weaker
    # and keeps weakening, so anything that crosses is drawn all the way in.
    # It behaves as a trapdoor rather than a collision. Measured: a wingman
    # reached 0.15 of its own captain, where separation was 1.5 against a slot
    # pull of about 20.
    #
    # Note this got WORSE when slot spacing was scaled up, because larger
    # spacing means larger typical slot distances and therefore a larger pull.
    # The capacity fix created this.
    #
    # A cap is a blunt instrument -- the principled fix is a separation profile
    # that grows at contact (Reynolds' 1/r weighting) -- but it is a direct
    # test of whether the ratio is really what is trapping them.
    if slotPullCap is not None:
        mag = np.hypot(lX, lY)
        scale = np.where(mag > slotPullCap,
                         slotPullCap / np.where(mag > 1e-9, mag, 1.0), 1.0)
        lX, lY = lX * scale, lY * scale

    dt = opts.dt
    return (fanVx + (sX + aX + cX + lX) * dt,
            fanVy + (sY + aY + cY + lY) * dt)