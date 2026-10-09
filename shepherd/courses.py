"""Obstacle courses. Each one is a start pose, a goal pose, circular
obstacles, and a hand-designed route the physics baseline follows.

Obstacles are circles (x, y, r). Walls are rows of circles, which keeps the
geometry to one primitive and makes 'clearance' one number per obstacle.
All lengths in metres, turn radius R = 5 as in the paper.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Course:
    name: str
    start: tuple                 # (x, y, heading)
    goal: tuple                  # (x, y, heading) the leader must hit
    obstacles: np.ndarray        # (K, 3) x, y, r
    route: list                  # waypoints for the baseline, excluding goal
    bounds: tuple                # (xmin, xmax, ymin, ymax) for plotting/obs
    description: str = ''


def _wall(x0, y0, x1, y1, r=1.5, gap=None):
    """A row of circles from (x0,y0) to (x1,y1), spaced so they overlap."""
    L = np.hypot(x1 - x0, y1 - y0)
    n = max(2, int(np.ceil(L / (1.5 * r))) + 1)
    ts = np.linspace(0, 1, n)
    pts = np.column_stack([x0 + ts * (x1 - x0), y0 + ts * (y1 - y0), np.full(n, r)])
    return pts


def open_field():
    return Course(
        name='open',
        start=(0.0, 0.0, 0.0),
        goal=(80.0, 30.0, np.pi / 2),
        obstacles=np.zeros((0, 3)),
        route=[],
        bounds=(-20, 110, -30, 60),
        description='No obstacles. Pure question: can you hit the pose '
                    'without your own flock running into you?')


def gate():
    # a wall across x = 50 with a 16 m gap centred at y = 0
    top = _wall(50, 8, 50, 40)
    bot = _wall(50, -40, 50, -8)
    return Course(
        name='gate',
        start=(0.0, 10.0, 0.0),
        goal=(100.0, -10.0, 0.0),
        obstacles=np.vstack([top, bot]),
        route=[(30.0, 0.0), (70.0, 0.0)],
        bounds=(-20, 120, -45, 45),
        description='Wall at x=50 with a 16 m gap at y in [-8, 8]. The gap is '
                    'narrower than two protected ranges, so the flock has '
                    'to squeeze.')


def slalom():
    posts = np.array([[30, 0, 3.0], [60, 0, 3.0], [90, 0, 3.0], [120, 0, 3.0]])
    return Course(
        name='slalom',
        start=(0.0, 0.0, 0.0),
        goal=(150.0, 0.0, 0.0),
        obstacles=posts,
        route=[(30, 15), (60, -15), (90, 15), (120, -15)],
        bounds=(-20, 170, -40, 40),
        description='Four posts on the centreline. Weave between them.')


def uturn():
    # a long divider the leader has to go around, ending pointing back home
    wall = _wall(10, 0, 90, 0)
    return Course(
        name='uturn',
        start=(0.0, -15.0, 0.0),
        goal=(0.0, 15.0, np.pi),
        obstacles=wall,
        route=[(80, -15), (110, 0), (80, 15)],
        bounds=(-20, 130, -45, 45),
        description='Divider along y=0 from x=10 to x=90. Go round the end '
                    'and come back on the other side, pointing home. A '
                    'headland turn, the case where the paper\'s formations '
                    'scatter.')


def _polyWall(pts, r=1.5):
    return np.vstack([_wall(*a, *b, r=r) for a, b in zip(pts[:-1], pts[1:])])


def narrows():
    # wide (44 m) -> narrow (14 m) -> wide (44 m). Inner wall faces at
    # |y| = 22 and |y| = 7. The formation that is right in the wide parts is
    # wrong in the narrow one, and the switch has to happen before the step.
    W, w = 22.0, 7.0
    top = _polyWall([(30, W), (70, W), (70, w), (130, w), (130, W), (170, W)])
    bot = _polyWall([(30, -W), (70, -W), (70, -w), (130, -w), (130, -W), (170, -W)])
    return Course(
        name='narrows',
        start=(0.0, 0.0, 0.0),
        goal=(195.0, 0.0, 0.0),
        obstacles=np.vstack([top, bot]),
        route=[(30, 0), (70, 0), (130, 0), (170, 0)],
        bounds=(-25, 210, -45, 45),
        description='Channel: 44 m wide for x in [30,70], 14 m wide for x in '
                    '[70,130], 44 m wide for x in [130,170]. Walls block '
                    'the boids, who cannot see them.')


def dogleg():
    # 20 m corridor east along y=0, then a 90 degree left turn north
    top = _polyWall([(30, 10), (90, 10), (90, 85)])
    bot = _polyWall([(30, -10), (110, -10), (110, 85)])
    return Course(
        name='dogleg',
        start=(0.0, 0.0, 0.0),
        goal=(100.0, 110.0, np.pi / 2),
        obstacles=np.vstack([top, bot]),
        route=[(30, 0), (90, 0), (100, 10), (100, 85)],
        bounds=(-25, 135, -30, 125),
        description='20 m wide corridor east from x=30, turning 90 degrees '
                    'left at x=100 and running north to y=85. The corner is '
                    'where followers cut through walls.')


COURSES = {'open': open_field, 'gate': gate, 'slalom': slalom, 'uturn': uturn,
           'narrows': narrows, 'dogleg': dogleg}


def get_course(name):
    return COURSES[name]()
