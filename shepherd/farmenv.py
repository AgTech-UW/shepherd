"""FarmEnv: ShepherdEnv on a FarmCourse, plus terrain and farm metrics.

Terrain: anything inside a slough is slowed to spec.wetSpeed.
Metrics, per episode:
  coverage        fraction of workable field (field minus sloughs and rock /
                  pole footprints) passed over by at least one implement.
                  Each machine works a disc of radius swath/2, which traced
                  along its path is a band one swath wide.
  sloughExposure  machine-seconds spent inside sloughs (soil damage)
  northExposure   machine-seconds inside the strip along the north fence
                  (used only to score the held-out instruction; never shown
                  to any decider)
  collisions      as in ShepherdEnv: fences, trees, rocks, poles, each other

Instructions are what the operator says. Each has a cost used to SCORE
runs and, for the oracle search only, to choose actions. The language model
never sees the weights, only the words.
"""
import numpy as np

from .env import ShepherdEnv
from .farm import FarmCourse

INSTRUCTIONS = {
    'wet': dict(
        text="It's wet out there. Stay out of the low spots and don't rut up the field, "
             "even if it means leaving some ground undone.",
        w=dict(col=10.0, slough=2.0, north=0.0, coverage=100.0, time=0.01), heldOut=False),
    'weather': dict(
        text="Rain's coming in tonight. Get as much of this field done as you can before it hits.",
        w=dict(col=10.0, slough=0.1, north=0.0, coverage=300.0, time=0.05), heldOut=False),
    'north': dict(
        text="Neighbour's got cattle along the north fence. Keep the machines well back from that side.",
        w=dict(col=10.0, slough=0.5, north=1.0, coverage=150.0, time=0.01), heldOut=True),
}


def instructionCost(m, key):
    """Lower is better. m holds collisions, sloughExposure, northExposure,
    coverage (0-1), time."""
    w = INSTRUCTIONS[key]['w']
    return (w['col'] * m['collisions'] + w['slough'] * m['sloughExposure'] +
            w['north'] * m['northExposure'] - w['coverage'] * m['coverage'] + w['time'] * m['time'])


class FarmEnv(ShepherdEnv):
    def __init__(self, course: FarmCourse, instruction='wet', **kw):
        self.instruction = instruction
        sp = course.spec
        L, W = course.field[1], course.field[3]
        c = sp.coverageCell
        gx, gy = np.meshgrid(np.arange(c / 2, L, c), np.arange(c / 2, W, c))
        self._cellX, self._cellY = gx.ravel(), gy.ravel()
        blocked = np.zeros(self._cellX.size, bool)
        for (x, y, r) in course.sloughs:
            blocked |= np.hypot(self._cellX - x, self._cellY - y) < r
        for h in course.hazards:
            if h['type'] != 'slough':
                blocked |= np.hypot(self._cellX - h['x'], self._cellY - h['y']) < h['r']
        self._workable = ~blocked
        self._static = ('_cellX', '_cellY', '_workable')
        kw.setdefault('nFan', sp.nFollowers)
        kw.setdefault('formation', dict(shape='column', spacing=12, frame='track'))
        kw.setdefault('captureRadius', 8.0)
        kw.setdefault('lostRadius', 60.0)
        super().__init__(course, **kw)

    # ---- lifecycle ----
    def reset(self):
        self.covered = np.zeros(self._cellX.size, bool)
        self.sloughExposure = 0.0
        self.northExposure = 0.0
        self.passOffset = 0.0
        self.seen = set()
        super().reset()
        self.routeRadius = 45.0          # the base reset sets 12; the farm needs more
        # machines start lined up in the yard, single file behind the leader
        if self.nFan:
            rng = np.random.default_rng(self.seed + 7919)
            c, s = np.cos(self.th), np.sin(self.th)
            k = np.arange(1, self.nFan + 1)
            self.fanX = self.x - c * 12.0 * k + rng.normal(0, 0.7, self.nFan)
            self.fanY = self.y - s * 12.0 * k + rng.normal(0, 0.7, self.nFan)
            self.fanVx = np.full(self.nFan, self.speed * c)
            self.fanVy = np.full(self.nFan, self.speed * s)
            spec, self.formation = self.formation, {}
            self.setFormation(spec, blend=0.0, count=False)
        self.log = {k: [] for k in self.log}
        self._log()
        return self.observe()

    # ---- route: phases and passing ----
    def phase(self):
        ph = self.course.phases
        return ph[self.routeIdx] if self.routeIdx < len(ph) else 'park'

    def _passed(self, i):
        """Passed once the leader crosses the perpendicular to the INCOMING
        leg at waypoint i, within routeRadius. Using the incoming leg keeps
        this right when the pass is being driven with a lateral offset."""
        rt = self.course.route
        w = np.asarray(rt[i], float)
        leg = self.course.passLegs.get(i)
        if leg is not None:
            # end of a pass: reached once the leader crosses the end line,
            # however far the pass has been shifted sideways
            a, b = np.asarray(leg[0], float), np.asarray(leg[1], float)
            return float((self.x - b[0]) * (b[0] - a[0]) + (self.y - b[1]) * (b[1] - a[1])) >= 0.0
        d = np.hypot(w[0] - self.x, w[1] - self.y)
        if d < 3.0:
            return True
        if d > self.routeRadius:
            return False
        prev = np.asarray(rt[i - 1] if i > 0 else self.course.start[:2], float)
        u = w - prev
        return float((self.x - w[0]) * u[0] + (self.y - w[1]) * u[1]) >= 0.0

    def passFrame(self):
        """(start, end, unit along, unit left) of the pass being driven, or None."""
        leg = self.course.passLegs.get(self.routeIdx)
        if leg is None:
            return None
        a, b = np.asarray(leg[0], float), np.asarray(leg[1], float)
        u = (b - a) / np.hypot(*(b - a))
        return a, b, u, np.array([-u[1], u[0]])

    def offsetLimits(self, lanes):
        """Allowed pass offsets for a formation whose machines sit at lateral
        positions `lanes` (leader at 0): the working band must stay inside
        the field with the spec's edge margin."""
        fr = self.passFrame()
        if fr is None:
            return 0.0, 0.0
        a, b, u, n = fr
        sp = self.course.spec
        L, W = self.course.field[1], self.course.field[3]
        half = sp.swath / 2
        # distance from the pass line to each fence along the left normal
        if abs(n[1]) > 0.5:              # pass runs east-west
            left = (W - a[1]) if n[1] > 0 else a[1]
            right = a[1] if n[1] > 0 else (W - a[1])
        else:
            left = (L - a[0]) if n[0] > 0 else a[0]
            right = a[0] if n[0] > 0 else (L - a[0])
        lo = -(right - sp.edgeMargin) - (min(lanes) - half)
        hi = (left - sp.edgeMargin) - (max(lanes) + half)
        cap = sp.band / 2 - sp.edgeMargin          # a pass stays on its own strip
        return float(max(lo, -cap)), float(min(hi, cap))

    # ---- terrain ----
    def _inSlough(self, x, y):
        s = self.course.sloughs
        if not len(s):
            return np.zeros(np.shape(x), bool)
        return (np.hypot(np.asarray(x)[..., None] - s[:, 0], np.asarray(y)[..., None] - s[:, 1]) < s[:, 2]).any(-1)

    def leaderSpeedCap(self):
        return self.course.spec.wetSpeed if self._inSlough(self.x, self.y) else np.inf

    def capFanSpeeds(self):
        if not self.nFan:
            return
        wet = self._inSlough(self.fanX, self.fanY)
        if wet.any():
            sp = np.hypot(self.fanVx, self.fanVy)
            k = np.where(wet & (sp > self.course.spec.wetSpeed),
                         self.course.spec.wetSpeed / np.maximum(sp, 1e-9), 1.0)
            self.fanVx, self.fanVy = self.fanVx * k, self.fanVy * k

    def postStep(self):
        dt = self.opts.dt
        ax = np.concatenate([[self.x], self.fanX])
        ay = np.concatenate([[self.y], self.fanY])
        self.sloughExposure += float(self._inSlough(ax, ay).sum()) * dt
        L, W = self.course.field[1], self.course.field[3]
        inField = (ax > 0) & (ax < L) & (ay > 0) & (ay < W)
        self.northExposure += float(np.sum(inField & (ay > W - self.course.spec.northStrip))) * dt
        if self.steps % 50 == 0:                       # every 0.5 s: mark worked cells
            r = self.course.spec.swath / 2
            ax, ay = ax[inField], ay[inField]
            if ax.size:
                near = (np.abs(self._cellX[:, None] - ax[None, :]) < r) & \
                       (np.abs(self._cellY[:, None] - ay[None, :]) < r)
                cand = near.any(1) & ~self.covered
                if cand.any():
                    idx = np.nonzero(cand)[0]
                    d2 = (self._cellX[idx, None] - ax[None, :]) ** 2 + (self._cellY[idx, None] - ay[None, :]) ** 2
                    self.covered[idx[(d2 < r * r).any(1)]] = True

    def coverage(self):
        return float(np.sum(self.covered & self._workable) / max(1, np.sum(self._workable)))

    # ---- hazards as an agent would be told about them ----
    def hazardsInRange(self, rng=None):
        rng = rng or self.course.spec.sensorRange
        out = []
        c, s = np.cos(self.th), np.sin(self.th)
        for i, h in enumerate(self.course.hazards):
            dx, dy = h['x'] - self.x, h['y'] - self.y
            d = np.hypot(dx, dy) - h['r']
            if d > rng:
                continue
            ahead, lateral = dx * c + dy * s, -dx * s + dy * c
            out.append(dict(id=i, type=h['type'], aheadM=round(float(ahead), 1),
                            leftM=round(float(lateral), 1), radiusM=round(float(h['r']), 1),
                            edgeDistanceM=round(float(max(0.0, d)), 1)))
        return out

    def metrics(self):
        r = super().result()
        r.update(coverage=self.coverage(), sloughExposure=self.sloughExposure,
                 northExposure=self.northExposure, instruction=self.instruction)
        return r

    def result(self):
        r = self.metrics()
        for k in INSTRUCTIONS:
            r['cost_' + k] = instructionCost(r, k)
        r['cost'] = r['cost_' + self.instruction]
        return r
