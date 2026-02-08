#!/usr/bin/env python
"""
SOFAI Demo: Thinking Fast and Slow in AI 
================================================================
Three agents navigate a 9x9 gridworld from Start to Goal:
  - System 1 Only  : Fast, heuristic 
  - SOFAI (MCA)    : Dynamically switches between S1 and S2
  - System 2 Only  : Slow, deliberative 

Animation speed is proportional to actual execution time:
  S1 moves fastest, S2 moves slowest, SOFAI in between.

CONTROLS:
  SPACE       Pause / Resume
  RIGHT       Step forward
  LEFT        Step backward
  R           Restart
  +/-         Speed up / slow down
  Q / ESC     Quit

Run:
    python demo.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import random
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import max_ent.examples.grid_9_by_9 as G
from max_ent.utility.support import create_world, count_states, total_reward
from max_ent.gridworld import Directions
from mc.self import ModelSelf
from mc.system1 import System1Solver
from mc.system2 import System2Solver
from mc.mca import MCA

# ─── Configuration ────────────────────────────────────────────────────────────
SEED = 42
N_TRAJECTORIES = 50
N_DISPLAY_TRAJ = 8
BASE_INTERVAL = 80        # ms per "tick" of the fastest agent (S1)

TH1 = 200
TH2 = 0.5
TH3 = 0.5
TH5 = 0
TH7 = 0.9

BLUE   = [21, 9, 59, 1, 0, 20]
GREEN  = [42, 18, 76, 41, 23, 30]
CS     = [63, 74, 13, 39, 48, 38]
CA     = [Directions.DOWN_LEFT, Directions.UP_LEFT]
CC     = [1, 2]
START  = 7
GOAL   = 65

# ─── Clean Color Scheme (White Background) ────────────────────────────────────
BG_COLOR         = "#ffffff"
GRID_COLOR       = "#d0d0d0"
S1_COLOR         = "#e74c3c"      # Red
S2_COLOR         = "#3498db"      # Blue
MCA_COLOR        = "#f39c12"      # Orange
GOAL_COLOR       = "#27ae60"      # Green
START_COLOR      = "#34495e"      # Dark gray
CONSTRAINT_COLOR = "#c0392b"      # Dark red
BLUE_CELL        = "#5dade2"      # Light blue
GREEN_CELL       = "#58d68d"      # Light green
VIOLATION_COLOR  = "#e74c3c"      # Red
TEXT_COLOR       = "#2c3e50"      # Dark blue-gray
LEGEND_BG        = "#f8f9fa"      # Very light gray


# ─── World & agents ──────────────────────────────────────────────────────────

def setup_worlds():
    random.seed(SEED); np.random.seed(SEED)
    print("Setting up worlds …")
    n, n_cfg, demo_n, _ = create_world('Nominal', BLUE, GREEN,
                                        start=START, goal=GOAL, draw=False)
    c, c_cfg, demo_c, _ = create_world('Constrained', BLUE, GREEN, CS, CA, CC,
                                        start=START, goal=GOAL, check=True, draw=False)
    return n, n_cfg, c, c_cfg, demo_c


def run_agents(n, c, demo_c):
    import time
    random.seed(SEED); np.random.seed(SEED)
    results = {}

    # SOFAI
    print("Running SOFAI …")
    ms = ModelSelf(n, c, demo_c)
    mca = MCA(s1=System1Solver(), s2=System2Solver(), modelSelf=ms,
              threshold1=TH1, threshold2=TH2, threshold3=TH3,
              threshold5=TH5, threshold7=TH7)
    t0 = time.time()
    demo_mca = mca.generate_trajectories(N_TRAJECTORIES)
    dt = time.time() - t0
    pct_s1, pct_s2 = mca.getStatistics()
    results['SOFAI'] = dict(demo=demo_mca, color=MCA_COLOR, mca=mca,
                            traj_stats=mca.trajectory_stat,
                            label='SOFAI (Hybrid)', s1_pct=pct_s1, s2_pct=pct_s2,
                            exec_time=dt, avg_time=dt / N_TRAJECTORIES)

    # System 1
    print("Running System 1 …")
    ms1 = ModelSelf(n, c, demo_c)
    mca1 = MCA(s1=System1Solver(), s2=System2Solver(), modelSelf=ms1,
               threshold1=TH1, threshold2=TH2, threshold3=TH3, only_s1=True)
    t0 = time.time()
    demo1 = mca1.generate_trajectories(N_TRAJECTORIES)
    dt = time.time() - t0
    results['System 1'] = dict(demo=demo1, color=S1_COLOR, mca=mca1,
                               traj_stats=mca1.trajectory_stat,
                               label='System 1 (Fast)', exec_time=dt,
                               avg_time=dt / N_TRAJECTORIES)

    # System 2
    print("Running System 2 …")
    ms2 = ModelSelf(n, c, None)
    mca2 = MCA(s1=System1Solver(), s2=System2Solver(), modelSelf=ms2,
               threshold1=TH1, threshold2=TH2, threshold3=TH3,
               threshold5=TH5, only_s2=True)
    t0 = time.time()
    demo2 = mca2.generate_trajectories(N_TRAJECTORIES)
    dt = time.time() - t0
    results['System 2'] = dict(demo=demo2, color=S2_COLOR, mca=mca2,
                               traj_stats=mca2.trajectory_stat,
                               label='System 2 (Slow)', exec_time=dt,
                               avg_time=dt / N_TRAJECTORIES)
    return results


def get_stats(trajectories, c_mdp, n, constraints):
    info = count_states(trajectories, c_mdp, n, constraints)
    return dict(avg_length=info[1], avg_reward=info[2], avg_violated=info[4],
                counters=info[5])


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_grid(ax, world, sp, title="", title_color="black"):
    """Draw the 9×9 grid background, colored cells, constraints, S and G."""
    size = world.size
    
    # Lighter background with subtle gradient
    bg = np.zeros((size, size))
    for s in range(size * size):
        x, y = world.state_index_to_point(s)
        bg[y, x] = sp[s]
    ax.imshow(bg, origin='lower', cmap='Greys', vmin=-60, vmax=15, alpha=0.15)

    # Grid lines
    for i in range(size + 1):
        ax.plot([i-.5, i-.5], [-.5, size-.5], color=GRID_COLOR, lw=1.0, alpha=0.6)
        ax.plot([-.5, size-.5], [i-.5, i-.5], color=GRID_COLOR, lw=1.0, alpha=0.6)

    # Blue cells (reward zones)
    for s in BLUE:
        x, y = world.state_index_to_point(s)
        ax.add_patch(Rectangle((x-.42, y-.42), .84, .84, 
                                facecolor=BLUE_CELL, edgecolor=BLUE_CELL,
                                alpha=0.3, lw=2.0))
    
    # Green cells (reward zones)
    for s in GREEN:
        x, y = world.state_index_to_point(s)
        ax.add_patch(Rectangle((x-.42, y-.42), .84, .84, 
                                facecolor=GREEN_CELL, edgecolor=GREEN_CELL,
                                alpha=0.3, lw=2.0))
    
    # Constraint states (forbidden)
    for s in CS:
        x, y = world.state_index_to_point(s)
        ax.plot(x, y, 'X', color=CONSTRAINT_COLOR, ms=18, mew=3.5, alpha=0.8)

    # Start position
    sx, sy = world.state_index_to_point(START)
    ax.add_patch(Circle((sx, sy), 0.35, facecolor=START_COLOR, 
                        edgecolor='black', linewidth=2, zorder=10, alpha=0.9))
    ax.text(sx, sy, 'S', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', zorder=11)
    
    # Goal position
    gx, gy = world.state_index_to_point(GOAL)
    ax.plot(gx, gy, '*', color=GOAL_COLOR, ms=28, mew=2, 
            markeredgecolor='#1e8449', zorder=10, alpha=0.95)
    ax.text(gx, gy-0.75, 'GOAL', ha='center', va='top', fontsize=7,
            fontweight='bold', color=GOAL_COLOR, zorder=11,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                     edgecolor=GOAL_COLOR, linewidth=1.5, alpha=0.95))

    ax.set_xlim(-.5, size-.5); ax.set_ylim(-.5, size-.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', color=title_color, 
                 pad=15, bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='white', edgecolor=title_color, 
                                  linewidth=2, alpha=0.95))


def draw_trajectory_segment(ax, world, traj, color, n_steps,
                            system_usage=None, alpha=.85, lw=4.0, zorder=8, 
                            force_single_color=False):
    """Draw trajectory up to *n_steps*; return artist list."""
    trans = traj.transitions()
    if not trans or n_steps <= 0:
        return []
    steps = min(n_steps, len(trans))
    arts = []

    for i in range(steps):
        s0, _, s1 = trans[i]
        x0, y0 = world.state_index_to_point(s0)
        x1, y1 = world.state_index_to_point(s1)

        seg_c = color
        # Only use system colors if not forcing single color
        if not force_single_color and system_usage is not None and i < len(system_usage):
            seg_c = S1_COLOR if system_usage[i] == 1 else S2_COLOR

        # Highlight violations
        if s0 in CS or s1 in CS:
            a, = ax.plot([x0, x1], [y0, y1], color=VIOLATION_COLOR,
                         alpha=.4, lw=lw+4, zorder=zorder-1, 
                         solid_capstyle='round', linestyle='--')
            arts.append(a)

        # Main path
        a, = ax.plot([x0, x1], [y0, y1], color=seg_c, alpha=alpha,
                     lw=lw, zorder=zorder, solid_capstyle='round')
        arts.append(a)

    # Current position marker
    last = trans[steps-1][2]
    lx, ly = world.state_index_to_point(last)
    g, = ax.plot(lx, ly, 'o', color=color, ms=24, alpha=.25, zorder=zorder, mew=0)
    m, = ax.plot(lx, ly, 'o', color=color, ms=13, mec='white', mew=2.5, zorder=zorder+1)
    arts += [g, m]
    return arts


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

def create_legend(legend_ax, results):
    """Compact, non-overlapping legend: 2 columns x 3 rows with a reserved title band."""

    legend_ax.set_facecolor(LEGEND_BG)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    # Border with padding
    pad = 0.01
    legend_ax.add_patch(Rectangle(
        (pad, pad), 1 - 2*pad, 1 - 2*pad,
        fill=False, edgecolor="#cccccc", linewidth=1.2
    ))

    # Title (single!)
    legend_ax.text(
        0.5, 0.85, "Legend",
        ha="center", va="center",
        fontsize=12, fontweight="semibold",
        color=TEXT_COLOR
    )

    # Items (label, kind)
    items = [
        ("Start (S)", "start"),
        ("Goal", "goal"),
        ("Blue Reward Zone", "blue"),
        ("Green Reward Zone", "green"),
        ("Forbidden State", "forbid"),
        ("Constraint Violation", "viol"),
    ]

    # Layout: 2 columns x 3 rows under the title
    # Reserve a bigger gap between title and first row by lowering y_top.
    y_top, y_bot = 0.72, 0.22
    ys = np.linspace(y_top, y_bot, 3)  # 3 rows

    x_icon = [0.12, 0.57]
    x_text = [0.18, 0.63]

    # Icon geometry
    rect_w, rect_h = 0.08, 0.10
    line_half = 0.04

    def draw(kind, x, y):
        if kind == "start":
            legend_ax.plot(
                x, y, marker="o", linestyle="None",
                markersize=11,
                markerfacecolor=START_COLOR,
                markeredgecolor="black",
                markeredgewidth=1.4
            )
        elif kind == "goal":
            legend_ax.plot(
                x, y, marker="*", linestyle="None",
                markersize=15,
                markerfacecolor=GOAL_COLOR,
                markeredgecolor="#1e8449",
                markeredgewidth=1.3
            )
        elif kind == "blue":
            legend_ax.add_patch(Rectangle(
                (x - rect_w/2, y - rect_h/2),
                rect_w, rect_h,
                facecolor=BLUE_CELL, edgecolor=BLUE_CELL,
                alpha=0.30, linewidth=2
            ))
        elif kind == "green":
            legend_ax.add_patch(Rectangle(
                (x - rect_w/2, y - rect_h/2),
                rect_w, rect_h,
                facecolor=GREEN_CELL, edgecolor=GREEN_CELL,
                alpha=0.30, linewidth=2
            ))
        elif kind == "forbid":
            # Use lowercase 'x' (cleaner) and explicit color on edge
            legend_ax.plot(
                x, y, marker="x", linestyle="None",
                markersize=9,
                color=CONSTRAINT_COLOR,
                markeredgewidth=2.2
            )
        elif kind == "viol":
            legend_ax.plot(
                [x - line_half, x + line_half], [y, y],
                color=VIOLATION_COLOR,
                linewidth=4,
                linestyle="--",
                alpha=0.75
            )

    # Fill column-wise (3 rows left, 3 rows right)
    for i, (label, kind) in enumerate(items):
        col = 0 if i < 3 else 1
        row = i if i < 3 else i - 3

        xi = x_icon[col]
        xt = x_text[col]
        y = ys[row]

        draw(kind, xi, y)
        legend_ax.text(
            xt, y, label,
            ha="left", va="center",
            fontsize=11,
            color=TEXT_COLOR
        )

# ─── Proportional-speed animation state ──────────────────────────────────────

class ProportionalAnimator:
    """
    Each agent accumulates 'time' every global tick.
    The fastest agent (lowest avg_time) advances 1 step per tick.
    Slower agents advance proportionally fewer steps per tick.

    This means S1 visually moves the fastest, S2 the slowest.
    """

    def __init__(self, agent_names, avg_times, traj_lengths):
        self.names = agent_names
        self.traj_lengths = {n: traj_lengths[n] for n in agent_names}

        # Compute speed relative to fastest agent
        fastest = min(avg_times[n] for n in agent_names)
        # speed_ratio > 1 means slower; we invert so faster agent has higher rate
        self.step_rate = {}
        for n in agent_names:
            ratio = fastest / avg_times[n]   # 1.0 for fastest, <1 for slower
            self.step_rate[n] = ratio

        # Accumulated fractional steps
        self.accum = {n: 0.0 for n in agent_names}
        self.current_step = {n: 0 for n in agent_names}

        self.max_ticks = self._estimate_max_ticks()
        self.tick = 0
        self.playing = True
        self.speed_mult = 1.0    # global speed multiplier

    def _estimate_max_ticks(self):
        """How many ticks until the slowest agent finishes."""
        worst = 0
        for n in self.names:
            ticks_needed = self.traj_lengths[n] / self.step_rate[n]
            worst = max(worst, ticks_needed)
        return int(worst) + 15   # small buffer

    def advance(self):
        """Advance one global tick. Returns dict of current step per agent."""
        if self.playing:
            self.tick += 1
            for n in self.names:
                if self.current_step[n] < self.traj_lengths[n]:
                    self.accum[n] += self.step_rate[n] * self.speed_mult
                    while self.accum[n] >= 1.0 and self.current_step[n] < self.traj_lengths[n]:
                        self.current_step[n] += 1
                        self.accum[n] -= 1.0
        return dict(self.current_step)

    def get_steps(self):
        return dict(self.current_step)

    def all_done(self):
        return all(self.current_step[n] >= self.traj_lengths[n] for n in self.names)

    # Controls
    def toggle(self):
        self.playing = not self.playing

    def step_forward(self):
        was = self.playing
        self.playing = True
        self.advance()
        self.playing = was

    def step_backward(self):
        # Simple: reset and replay to tick-1
        target = max(0, self.tick - 2)
        self.restart()
        for _ in range(target):
            self.playing = True
            self.advance()
        self.playing = True  # restore

    def restart(self):
        self.tick = 0
        self.accum = {n: 0.0 for n in self.names}
        self.current_step = {n: 0 for n in self.names}

    def speed_up(self):
        self.speed_mult = min(5.0, self.speed_mult + 0.25)

    def slow_down(self):
        self.speed_mult = max(0.25, self.speed_mult - 0.25)


# ─── Helper function for parsing metrics ─────────────────────────────────────

def _parse_float(x):
    """Parse float from string with optional units (e.g., '17.258 s' -> 17.258)"""
    return float(str(x).strip().split()[0])


# ─── Main demo ────────────────────────────────────────────────────────────────

def create_demo():
    n, n_cfg, c, c_cfg, demo_c = setup_worlds()
    results = run_agents(n, c, demo_c)

    constraints = dict(blue=BLUE, green=GREEN, cs=CS, ca=CA)
    world = c_cfg.mdp.world
    sp = c_cfg.state_penalties

    agent_order = ['System 1', 'SOFAI', 'System 2']

    # ──────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────
    stats = {}
    for name in agent_order:
        stats[name] = get_stats(
            results[name]['demo'].trajectories,
            c_cfg.mdp, n, constraints
        )

    # Representative trajectories
    anim_trajs, anim_sys = {}, {}
    for name in agent_order:
        trajs = results[name]['demo'].trajectories
        lengths = [len(t.transitions()) for t in trajs]
        idx = int(np.argmin([abs(l - np.median(lengths)) for l in lengths]))
        anim_trajs[name] = trajs[idx]

        ts = results[name].get('traj_stats', None)
        anim_sys[name] = ts[idx] if ts and idx < len(ts) else None

    traj_lens = {name: len(anim_trajs[name].transitions())
                 for name in agent_order}

    avg_times = {name: results[name]['avg_time']
                 for name in agent_order}

    # Print speed ratios
    fastest_name = min(avg_times, key=avg_times.get)
    print("\nAnimation speed ratios:")
    for name in agent_order:
        ratio = avg_times[name] / avg_times[fastest_name]
        print(f"{name}: {ratio:.2f}x slower")

    animator = ProportionalAnimator(agent_order, avg_times, traj_lens)

    # ──────────────────────────────────────────────────────────────────────
    # Figure
    # ──────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor="white")

    fig.suptitle(
        "SOFAI Gridworld Demo: Thinking Fast and Slow in AI",
        fontsize=20,
        fontweight="semibold",
        y=0.985,
        color=TEXT_COLOR
    )

    gs = gridspec.GridSpec(
        5, 3,
        figure=fig,
        height_ratios = [3.0, 0.5, 1.25, 0.25, 0.8],
        hspace=0.0,
        wspace=0.22,
        left=0.06,
        right=0.94,
        top=0.88,
        bottom=0.07
    )

    # ──────────────────────────────────────────────────────────────────────
    # Panels
    # ──────────────────────────────────────────────────────────────────────
    axes = {}

    for i, name in enumerate(agent_order):

        ax = fig.add_subplot(gs[0, i], facecolor="white")

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#dddddd")

        draw_grid(
            ax,
            world,
            sp,
            title=results[name]['label'],
            title_color=results[name]['color']
        )

        axes[name] = ax

    # Spacer
    fig.add_subplot(gs[1, :]).axis("off")

    # ──────────────────────────────────────────────────────────────────────
    # Table
    # ──────────────────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    col_labels = [
        'Agent',
        'Avg Length',
        'Avg Reward',
        'Avg Violations',
        'Total Time',
        'Time / Traj'
    ]

    rows = []
    row_colors = []

    for name in agent_order:

        s = stats[name]
        r = results[name]

        rows.append([
            name,
            f"{s['avg_length']:.1f}",
            f"{s['avg_reward']:.1f}",
            f"{s['avg_violated']:.2f}",
            f"{r['exec_time']:.3f} s",
            f"{r['avg_time']*1000:.1f} ms",
        ])

        row_colors.append(r['color'])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.9]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():

        cell.set_edgecolor("#e0e0e0")
        cell.set_linewidth(0.8)

        if r == 0:

            cell.set_facecolor("#f3f3f3")
            cell.set_text_props(
                fontweight="semibold",
                color=TEXT_COLOR
            )

        else:

            cell.set_facecolor("white")
            cell.set_text_props(color=TEXT_COLOR)

            if c == 0:

                cell.set_text_props(
                    color=row_colors[r-1],
                    fontweight="semibold"
                )

    # ──────────────────────────────────────────────────────────────────────
    # Highlight best metrics
    # ──────────────────────────────────────────────────────────────────────

    metric_rules = {
        1: True,   # minimize length
        2: False,  # maximize reward
        3: True,   # minimize violations
        4: True,   # minimize total time
        5: True    # minimize time/traj
    }

    best_vals = {}

    for col, minimize in metric_rules.items():

        vals = [_parse_float(rows[i][col])
                for i in range(len(rows))]

        best_vals[col] = min(vals) if minimize else max(vals)

    BEST_BG = "#f6fff6"

    scores = [0] * len(rows)

    for r in range(1, len(rows)+1):

        for col, minimize in metric_rules.items():

            v = _parse_float(rows[r-1][col])

            if v == best_vals[col]:

                cell = tbl[(r, col)]

                cell.set_facecolor(BEST_BG)

                cell.set_text_props(
                    fontweight="semibold",
                    color=GOAL_COLOR
                )

                scores[r-1] += 1

    best_agent_idx = int(np.argmax(scores))

    tbl[(best_agent_idx+1, 0)].set_text_props(
        fontweight="bold"
    )

    # Spacer
    fig.add_subplot(gs[3, :]).axis("off")

    # ──────────────────────────────────────────────────────────────────────
    # Step counters
    # ──────────────────────────────────────────────────────────────────────
    step_texts = {}

    for name in agent_order:

        ax = axes[name]

        t = ax.text(
            0.5,
            -0.12,  # Changed from -0.16 to bring closer to grid
            "",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="semibold",
            color=results[name]['color'],
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#e0e0e0"
            )
        )

        step_texts[name] = t

    # ──────────────────────────────────────────────────────────────────────
    # Footer
    # ──────────────────────────────────────────────────────────────────────
    ax_footer = fig.add_subplot(gs[4, :])
    ax_footer.axis("off")

    # Legenda: 80%, controlli: 20%
    ax_leg = ax_footer.inset_axes([0.00, 0.20, 1.00, 0.80])
    ax_ctl = ax_footer.inset_axes([0.00, 0.00, 1.00, 0.20])

    ax_leg.axis("off")
    ax_ctl.axis("off")

    create_legend(ax_leg, results)

    ax_ctl.text(
        0.5, 0.50,
        "Controls: SPACE play/pause • ←/→ step • R restart • +/- speed • Q quit",
        ha="center", va="center",
        fontsize=10, color="#666666"
    )

    # ──────────────────────────────────────────────────────────────────────
    # Animation container
    # ──────────────────────────────────────────────────────────────────────
    live_arts = {name: [] for name in agent_order}


    def animate(_frame):
        steps = animator.advance()

        for name in agent_order:
            for a in live_arts[name]:
                a.remove()
            live_arts[name] = []

            # Force single color for SOFAI to keep it yellow/orange
            force_color = (name == 'SOFAI')
            
            arts = draw_trajectory_segment(
                axes[name], world, anim_trajs[name],
                results[name]['color'], steps[name],
                system_usage=anim_sys[name],
                force_single_color=force_color)
            live_arts[name] = arts

            # Update step counter
            total = traj_lens[name]
            cur = min(steps[name], total)
            step_texts[name].set_text(f"Step {cur} / {total}")

        # Auto-restart when all done
        if animator.all_done():
            animator.restart()

    def on_key(event):
        if event.key == ' ':
            animator.toggle()
        elif event.key == 'right':
            animator.step_forward()
        elif event.key == 'left':
            animator.step_backward()
        elif event.key == 'r':
            animator.restart()
        elif event.key in ('+', '='):
            animator.speed_up()
        elif event.key == '-':
            animator.slow_down()
        elif event.key in ('q', 'escape'):
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)

    _anim = animation.FuncAnimation(fig, animate,
                                    interval=BASE_INTERVAL,
                                    blit=False, repeat=True)
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("  SOFAI Demo")
    print("=" * 60)
    create_demo()