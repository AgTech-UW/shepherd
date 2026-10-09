"""A generic, tunable farm: yard lane, field gate, a field worked in passes,
and hazards placed at random per seed.

Everything is set by a FarmSpec, which loads from and saves to JSON, so a
field can be edited by hand now and drawn in an editor later:

    spec = FarmSpec.load('farms/default.json')   # or FarmSpec() for defaults
    course = buildFarm(spec, seed=3)
    env = FarmEnv(course, instruction='wet')

The mission, in order:
  lane      out of the yard down a lane between two shelterbelts, with a bend
  gate      through the field gate (a gap in the field-edge fence)
  approach  line up on the first pass
  pass1     work the first pass, east
  headland  turn round at the east end and line up on the second pass
  pass2     work the second pass, west
  return    run down the west headland back to the gate
  gate/park out through the gate and park on a pose outside it

Coordinates: +x is east, +y is north. The field occupies [0, L] x [0, W].
Machines: one leader and n followers, each pulling an implement `swath`
metres wide. Pass spacing is the width of the full working line (n+1 lanes
of `swath`), so two passes with the line fully spread cover the field.

Hazards (all randomised within the spec's ranges, per seed):
  slough  a wet low spot. Passable, but anything in it is slowed to
          `wetSpeed`, and every machine-second spent in it is counted as
          soil damage (rutting, compaction). Not an obstacle.
  rock    a rock pile. Solid: touching it is a collision.
  pole    a power pole. Solid, small.
Fences and shelterbelts are solid too. Followers sense none of it; they
are boids holding slots.
"""
import json
from dataclasses import dataclass, field, asdict
import numpy as np

from .courses import Course


@dataclass
class FarmSpec:
    # field
    fieldLength: float = 240.0        # east-west, m
    swath: float = 12.0               # implement width per machine, m (a 40 ft drill)
    nFollowers: int = 8
    passes: int = 2
    edgeMargin: float = 6.0           # between the working line and the fence, m
    passStartX: float = 30.0
    passEndInset: float = 32.0        # pass ends this far short of the east fence
    headlandInset: float = 20.0       # headland track this far from the east fence
    returnX: float = 18.0             # west headland track
    fencePostR: float = 1.0
    # gate and lane
    gateY: float = 30.0
    gateWidth: float = 16.0
    lane: list = field(default_factory=lambda: [[-110.0, -30.0], [-60.0, -30.0], [-32.0, 30.0], [-12.0, 30.0]])
    laneHalfWidth: float = 10.0
    treeR: float = 1.5
    park: list = field(default_factory=lambda: [-30.0, 30.0, float(np.pi)])
    # hazards: counts and ranges; lateral is measured from a pass centre line
    sloughs: int = 1
    sloughRadius: list = field(default_factory=lambda: [10.0, 16.0])
    sloughLateral: float = 36.0
    rocks: int = 1
    rockRadius: list = field(default_factory=lambda: [2.0, 3.5])
    rockLateral: float = 40.0
    poles: int = 1
    poleRadius: float = 0.8
    hazardX: list = field(default_factory=lambda: [70.0, 190.0])
    hazardMinSep: float = 30.0
    wetSpeed: float = 0.6             # anything in a slough is slowed to this
    northStrip: float = 25.0          # depth of the strip along the north fence (held-out instruction)
    sensorRange: float = 80.0
    coverageCell: float = 2.0

    @property
    def lanes(self):
        return self.nFollowers + 1

    @property
    def band(self):
        return self.lanes * self.swath

    @property
    def fieldWidth(self):
        return self.passes * self.band + 2 * self.edgeMargin

    def passY(self, k):
        return self.edgeMargin + self.band * (k + 0.5)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1)


@dataclass
class FarmCourse(Course):
    spec: FarmSpec = None
    seed: int = 0
    phases: list = None               # phase of the leg that ends at route[i]
    passLegs: dict = None             # route index of a pass end -> (startXY, endXY)
    sloughs: np.ndarray = None        # (k, 3) x, y, r (soft)
    hazards: list = None              # dicts for observations: type, x, y, r
    field: tuple = None               # (0, L, 0, W)


def _row(a, b, r, spacing=1.5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.hypot(*(b - a))
    n = max(2, int(np.ceil(L / (spacing * r))) + 1)
    t = np.linspace(0, 1, n)[:, None]
    p = a + t * (b - a)
    return np.column_stack([p, np.full(n, r)])


def _offsetPolyline(pts, d):
    """Polyline offset by d to the left (mitred joints)."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    seg /= np.hypot(seg[:, 0], seg[:, 1])[:, None]
    nrm = np.column_stack([-seg[:, 1], seg[:, 0]])
    out = [pts[0] + d * nrm[0]]
    for i in range(1, len(pts) - 1):
        m = nrm[i - 1] + nrm[i]
        m /= np.hypot(*m)
        out.append(pts[i] + d * m / max(0.3, float(m @ nrm[i])))
    out.append(pts[-1] + d * nrm[-1])
    return np.array(out)


def buildFarm(spec=None, seed=0):
    spec = spec or FarmSpec()
    rng = np.random.default_rng(seed)
    L, W = spec.fieldLength, spec.fieldWidth
    obs = []

    # field-edge fence with a gate gap on the west side
    g0, g1 = spec.gateY - spec.gateWidth / 2, spec.gateY + spec.gateWidth / 2
    r = spec.fencePostR
    obs += [_row((0, 0), (L, 0), r), _row((L, 0), (L, W), r), _row((L, W), (0, W), r),
            _row((0, W), (0, g1), r), _row((0, g0), (0, 0), r)]

    # lane: two shelterbelts either side of the lane centre line, stopping at the fence
    lane = np.asarray(spec.lane, float)
    d = spec.laneHalfWidth + spec.treeR
    for side in (+d, -d):
        poly = _offsetPolyline(lane, side)
        poly[-1, 0] = -spec.treeR - 1.0          # end the tree rows at the fence line
        for a, b in zip(poly[:-1], poly[1:]):
            obs.append(_row(a, b, spec.treeR))

    # route with phases
    y = [spec.passY(k) for k in range(spec.passes)]
    xs, xe = spec.passStartX, L - spec.passEndInset
    xh, xr = L - spec.headlandInset, spec.returnX
    route, phases = [], []

    def add(p, ph):
        route.append((float(p[0]), float(p[1])))
        phases.append(ph)

    for p in lane[1:]:
        add(p, 'lane')
    add((xr, spec.gateY), 'gate')
    add((xs, y[0]), 'approach')
    passLegs = {}
    for k in range(spec.passes):
        east = (k % 2 == 0)
        a, b = ((xs, y[k]), (xe, y[k])) if east else ((xe, y[k]), (xs, y[k]))
        add(b, 'pass%d' % (k + 1))
        passLegs[len(route) - 1] = (a, b)
        if k < spec.passes - 1:
            hx = xh if east else xr
            add((hx, y[k] + 12), 'headland')
            add((hx, y[k + 1] - 12), 'headland')
            add(((xe if east else xs), y[k + 1]), 'headland')
    lastEast = (spec.passes - 1) % 2 == 0
    if lastEast:                                   # finish at the east end: come back along the north headland
        add((xh, y[-1] + 12), 'return')
        add((xh, W - 10), 'return')
        add((xr, W - 10), 'return')
    add((xr, y[-1] - 14) if not lastEast else (xr, W - 24), 'return')
    add((xr, spec.gateY + 14), 'return')
    add((-10.0, spec.gateY), 'gate')

    # hazards, placed per seed
    hazards, sloughs = [], []
    placed = []

    def place(kind, rad, latMax):
        for _ in range(200):
            k = int(rng.integers(spec.passes))
            px = rng.uniform(*spec.hazardX)
            py = y[k] + rng.uniform(-latMax, latMax)
            if all(np.hypot(px - qx, py - qy) > spec.hazardMinSep for qx, qy in placed):
                placed.append((px, py))
                return px, py
        raise RuntimeError('could not place %s' % kind)

    for _ in range(spec.sloughs):
        rad = rng.uniform(*spec.sloughRadius)
        px, py = place('slough', rad, spec.sloughLateral)
        sloughs.append((px, py, rad))
        hazards.append(dict(type='slough', x=px, y=py, r=rad))
    for _ in range(spec.rocks):
        rad = rng.uniform(*spec.rockRadius)
        px, py = place('rock', rad, spec.rockLateral)
        obs.append(np.array([[px, py, rad]]))
        hazards.append(dict(type='rock', x=px, y=py, r=rad))
    for _ in range(spec.poles):
        px, py = place('pole', spec.poleRadius, spec.band / 2 - 4)
        obs.append(np.array([[px, py, spec.poleRadius]]))
        hazards.append(dict(type='pole', x=px, y=py, r=spec.poleRadius))

    start = (float(lane[0, 0]), float(lane[0, 1]),
             float(np.arctan2(lane[1, 1] - lane[0, 1], lane[1, 0] - lane[0, 0])))
    return FarmCourse(
        name='farm', start=start, goal=tuple(map(float, spec.park)),
        obstacles=np.vstack(obs), route=route,
        bounds=(float(lane[:, 0].min()) - 15, L + 15, min(-15.0, float(lane[:, 1].min()) - 15), W + 15),
        description='Yard lane, field gate, %d passes of a %.0f x %.0f m field, back out the gate.'
                    % (spec.passes, L, W),
        spec=spec, seed=seed, phases=phases, passLegs=passLegs,
        sloughs=np.array(sloughs).reshape(-1, 3), hazards=hazards, field=(0.0, L, 0.0, W))
