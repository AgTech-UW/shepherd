"""All tunables in one place. Edit here, nowhere else."""
from dataclasses import dataclass, field


@dataclass
class SimOptions:
    # ---- Boids rule gains ----
    TF: float = 2.0          # Turning factor (wall push strength)
    VR: float = 8.0          # Visual range (how far they see neighbors)
    PR: float = 10.0         # Protected range (crash avoidance zone)
    CF: float = 0.8          # Cohesion factor (pull to group center)
    SF: float = 10.0          # Separation factor (push away from close neighbors)
    AF: float = 2.0          # Alignment factor (match group speed/heading)
    # Separation force profile. 'flat' is the published form, magnitude
    # growing with distance up to PR and vanishing at contact. 'norm' is the
    # Reynolds 1999 form, offset normalised and weighted 1/r, scaled so its
    # magnitude at contact matches the flat profile's magnitude at the PR
    # boundary. Same dynamic range, mirror-image profile.
    SEP_PROFILE: str = 'flat'
    maxSpeed: float = 2.0
    minSpeed: float = 0.8
    safetyF: float = 5.0     # Margin from the walls

    # ---- Dubins vehicle (filled from saved_params CSV by load_dataset) ----
    R: float = 5.0           # Falcon turning radius
    v: float = 1.0           # Falcon speed
    dt: float = 0.01         # timestep

    # ---- Captain / navigation ----
    N: int = 10              # number of distractor boids
    leaderFactor: float = 1.0
    falconBoidsWeight: float = 0.4
    dAng: float = 0.005
    collectRadius: float = 0.75   # bullseye size for goal
    passWindow: float = 5.0
    hysteresisSteps: float = 2.5  # hysteresis = hysteresisSteps * v * dt
    seed: int = 8
    tMaxPathFactor: float = 1.75  # tMax = factor * refLen / v + tMaxPad
    tMaxPad: float = 10.0
    maxClumps: int = 3            # random starting groups for fanboids

    # Ackermann constraint: boid turn radius as fraction of Falcon's
    RboidFactor: float = 1

    # ---- Fanboid parameters ----
    fanLeaderFactor: float = 1.0     # pull toward the captain
    fanLeaderVelocityFactor: float = 0.0  # heading/velocity match to the captain.
                                     # Used in dynamics.fanLeaderRule but was never
                                     # defined here, so fanboid runs crashed. 0.0 =
                                     # position pull only (pre-velocity-term code).
    falconSeesFanboids: bool = False # keep captain unperturbed by default
    nFanShowcase: int = 10
    fanSweepN: tuple = (1, 2, 3, 5, 7, 12, 20)
    fanTrials: int = 5
    sigmaPos: float = 1.0            # pos error capture tolerance
    sigmaTh: float = 1.0             # heading error capture tolerance

    # ---- Collision counting ----
    bodyRadius: float = 0.5          # physical radius of each agent (a disk)
    boidsSeeCaptain: bool = False    # if True, DISTRACTORS also get Cap in their
                                     # neighbor list so they can avoid him
                                     # (fanboids always see him regardless)

    # ---- Multi-fleet (two captains crossing) ----
    capAvoidWeight: float = 2.0      # how hard a captain reacts when someone
                                     # enters his protected range (0 = never dodge)
    capPR: float = 12.0              # a captain's OWN protected range. Bigger than
                                     # the boids' PR on purpose: a Dubins car with
                                     # turn radius R needs ~sqrt(2*R*d) of runway to
                                     # sidestep by d, so he must react EARLY.
    capAvoidsOwnFans: bool = False   # should a captain dodge his own fanboids?
                                     # False = his own fleet keeps clear of him,
                                     # so he only reacts to OTHER fleets.
    capSwirl: float = 1.2            # "rules of the road" tiebreaker: everyone
                                     # also veers RIGHT when avoiding. Pure
                                     # push-apart CANNOT solve a symmetric
                                     # crossing (both agents get shoved along the
                                     # same mirror line), so a consistent
                                     # handedness is what actually breaks the tie.

    @property
    def collisionRadius(self):
        # Two disks touch when their centers are within the sum of radii = 2*r
        return 2.0 * self.bodyRadius

    @property
    def Rboid(self):
        return self.RboidFactor * self.R # this is the boid turn radius, which is smaller than the Falcon's

    @property
    def hysteresis(self):
        return self.hysteresisSteps * self.v * self.dt

    def tMax(self, refLen):
        return self.tMaxPathFactor * refLen / self.v + self.tMaxPad