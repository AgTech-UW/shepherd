"""Reusable playback animation for any grainframe experiment.

The animator is deliberately dumb about WHERE the data comes from. You hand it
a list of "tracks" -- each track is one moving thing (a leader, a fanboid) with
its x,y history and a color -- and it plays them back with trails and a clock.

Every experiment logs data a bit differently, so each one does a few lines of
glue to package its logs as tracks, then calls animate_tracks(). Helpers for
the two common cases (flat arrays, and Fleet objects) are provided below.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Track:
    """One moving agent's trajectory for the animation.

    x, y  : equal-length arrays of positions over time
    color : matplotlib color
    size  : marker size (big for leaders, small for fanboids)
    label : legend label, or None to skip the legend for this track
    """
    def __init__(self, x, y, color, size=6, label=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.color = color
        self.size = size
        self.label = label


def animate_tracks(tracks, dt=0.01, frameSkip=25, tail=400, title='',
                   subtitleFn=None, refPaths=None, save=None, fps=20,
                   blit=True):
    """Play back a list of Track objects.

    tracks     : list of Track
    dt         : sim timestep (for the clock display)
    frameSkip  : plot every Nth sim step (smaller = slower, smoother)
    tail       : trail length, in sim steps
    title      : figure title
    subtitleFn : optional function(t) -> str, shown each frame (e.g. a live
                 distance readout). t is the sim-step index.
    refPaths   : optional list of (x, y, color) dotted reference paths to draw
    save       : optional filename ('foo.gif' or 'foo.mp4') to write instead of
                 (or as well as) showing
    fps        : frames per second when saving
    blit       : redraw only the artists that changed instead of the whole
                 canvas. This is the difference between ~15 fps and smooth
                 playback with a dozen-odd tracks. Pass blit=False if you are
                 debugging and want a plain full redraw.

    Returns the FuncAnimation (keep a reference or playback stops).
    """
    T = min(len(tr.x) for tr in tracks)
    frames = range(0, T, frameSkip)

    fig, ax = plt.subplots(figsize=(9, 8))

    if refPaths:
        for rx, ry, rc in refPaths:
            ax.plot(rx, ry, ':', color=rc, alpha=0.5, zorder=1)

    # one dot + one trail per track
    dots, trails = [], []
    for tr in tracks:
        dot, = ax.plot([], [], 'o', color=tr.color, markersize=tr.size,
                       zorder=6, label=tr.label)
        trail, = ax.plot([], [], '-', color=tr.color,
                         linewidth=1.6 if tr.size >= 8 else 0.5,
                         alpha=0.9 if tr.size >= 8 else 0.3, zorder=5)
        dots.append(dot)
        trails.append(trail)
    ax.set_title(title)
    # The live readout must be an artist INSIDE the axes: blitting only
    # repaints the axes bbox, and a figure/axes TITLE sits outside it, so a
    # set_title() clock would either never update or force a full redraw.
    clock = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top',
                    ha='left', fontsize=9, zorder=10)

    # fixed axes so the view doesn't jump
    allX = np.concatenate([tr.x[:T] for tr in tracks])
    allY = np.concatenate([tr.y[:T] for tr in tracks])
    pad = 8
    ax.set_xlim(allX.min() - pad, allX.max() + pad)
    ax.set_ylim(allY.min() - pad, allY.max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if any(tr.label for tr in tracks):
        ax.legend(fontsize=8, loc='upper right')

    def update(t):
        t0 = max(0, t - tail)
        for tr, dot, trail in zip(tracks, dots, trails):
            dot.set_data([tr.x[t]], [tr.y[t]])
            trail.set_data(tr.x[t0:t + 1], tr.y[t0:t + 1])
        clock.set_text(subtitleFn(t) if subtitleFn is not None else '')
        return dots + trails + [clock]

    # cache_frame_data=False: FuncAnimation otherwise retains every frame's
    # returned artists for the whole run, which on a few hundred frames is a
    # steady memory climb and a slow drift in playback rate.
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps,
                         blit=blit, cache_frame_data=False)

    if save:
        if save.endswith('.gif'):
            anim.save(save, writer=PillowWriter(fps=fps))
        else:
            anim.save(save, fps=fps)
        print('saved animation to %s' % save)

    plt.show()
    return anim


# ---------------- glue helpers for the two common data shapes ----------------

def tracks_from_arrays(leaders, fans, leaderColors, fanColors,
                       leaderLabels=None):
    """Build tracks from flat arrays (the sandbox style).

    leaders      : list of (x_array, y_array) per leader
    fans         : list of (x_array, y_array) per fanboid
    leaderColors : list of colors, one per leader
    fanColors    : list of colors, one per fanboid
    leaderLabels : optional list of legend labels per leader
    """
    tracks = []
    for k, (lx, ly) in enumerate(leaders):
        lbl = leaderLabels[k] if leaderLabels else None
        tracks.append(Track(lx, ly, leaderColors[k], size=10, label=lbl))
    for i, (fxi, fyi) in enumerate(fans):
        tracks.append(Track(fxi, fyi, fanColors[i], size=6))
    return tracks


def tracks_from_fleets(fleets):
    """Build tracks from Fleet objects (the multifleet style)."""
    tracks = []
    for fl in fleets:
        tracks.append(Track(fl.log['x'], fl.log['y'], fl.color, size=10,
                            label=fl.name))
        if fl.log['fanX'].size > 0:
            fx = fl.log['fanX']
            fy = fl.log['fanY']
            for b in range(fx.shape[1]):
                tracks.append(Track(fx[:, b], fy[:, b], fl.color, size=6))
    return tracks