"""
Generate the site's hero artwork: a phase portrait of a genetic toggle switch.

The system is bistable. Trajectories started anywhere in the plane flow toward
one of two stable states, separated by a saddle point. That two-basin geometry is
the visual signature of cell-fate decisions and viability boundaries.

Run:  python tools/generate_phase_portrait.py
Out:  assets/phase-portrait.svg   (transparent background, teal strokes)

Everything here is intentionally plain so you can swap in your own model:
edit `du_dt`, `dv_dt`, and the domain, then re-run.
"""

import numpy as np

# ----------------------------------------------------------------------
# 1. The model: a symmetric genetic toggle switch
#      du/dt = A / (1 + v^N) - u
#      dv/dt = A / (1 + u^N) - v
# ----------------------------------------------------------------------
A = 3.0      # max production rate
N = 3.0      # Hill coefficient (cooperativity); higher -> sharper switch

def du_dt(u, v):
    return A / (1.0 + v**N) - u

def dv_dt(u, v):
    return A / (1.0 + u**N) - v

# State space window we draw
U_MAX = 3.2
V_MAX = 3.2

# SVG canvas (square)
SIZE = 1000

def to_svg(u, v):
    """Map a point in state space to SVG pixel coordinates (y is flipped)."""
    x = u / U_MAX * SIZE
    y = SIZE - (v / V_MAX * SIZE)
    return x, y

# ----------------------------------------------------------------------
# 2. Integrate one trajectory with classic Runge-Kutta (RK4)
# ----------------------------------------------------------------------
def trajectory(u0, v0, steps=140, dt=0.06):
    pts = []
    u, v = u0, v0
    for _ in range(steps):
        pts.append((u, v))
        k1u, k1v = du_dt(u, v),             dv_dt(u, v)
        k2u, k2v = du_dt(u + 0.5*dt*k1u, v + 0.5*dt*k1v), dv_dt(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
        k3u, k3v = du_dt(u + 0.5*dt*k2u, v + 0.5*dt*k2v), dv_dt(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
        k4u, k4v = du_dt(u + dt*k3u, v + dt*k3v),         dv_dt(u + dt*k3u, v + dt*k3v)
        u += dt/6.0 * (k1u + 2*k2u + 2*k3u + k4u)
        v += dt/6.0 * (k1v + 2*k2v + 2*k3v + k4v)
        u = min(max(u, 0.0), U_MAX)
        v = min(max(v, 0.0), V_MAX)
    return pts

# ----------------------------------------------------------------------
# 3. Find the fixed points (where both rates vanish) on the line scan,
#    then classify each as stable or a saddle using the Jacobian.
# ----------------------------------------------------------------------
def fixed_points():
    us = np.linspace(0.001, U_MAX, 4000)
    # On the v-nullcline, v = A/(1+u^N). Substitute and look for sign changes.
    g = A / (1.0 + (A / (1.0 + us**N))**N) - us
    roots = []
    for i in range(len(us) - 1):
        if g[i] == 0 or g[i] * g[i+1] < 0:
            u = us[i] - g[i] * (us[i+1] - us[i]) / (g[i+1] - g[i])
            v = A / (1.0 + u**N)
            roots.append((u, v))
    return roots

def is_stable(u, v, h=1e-4):
    # Numerical Jacobian; stable if both eigenvalues have negative real part.
    j11 = (du_dt(u+h, v) - du_dt(u-h, v)) / (2*h)
    j12 = (du_dt(u, v+h) - du_dt(u, v-h)) / (2*h)
    j21 = (dv_dt(u+h, v) - dv_dt(u-h, v)) / (2*h)
    j22 = (dv_dt(u, v+h) - dv_dt(u, v-h)) / (2*h)
    eig = np.linalg.eigvals([[j11, j12], [j21, j22]])
    return np.all(eig.real < 0)

# ----------------------------------------------------------------------
# 4. Build the SVG
# ----------------------------------------------------------------------
def build_svg():
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" '
        f'width="{SIZE}" height="{SIZE}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    )

    TEAL = "#4fe0cf"

    # --- streamlines: seed a grid of starting points -------------------
    seeds = np.linspace(0.12, U_MAX - 0.12, 11)
    for u0 in seeds:
        for v0 in seeds:
            pts = trajectory(u0, v0)
            d = "M " + " L ".join(f"{to_svg(u,v)[0]:.1f} {to_svg(u,v)[1]:.1f}" for u, v in pts[::2])
            parts.append(f'<path d="{d}" stroke="{TEAL}" stroke-width="1.2" opacity="0.24"/>')

    # --- nullclines (brighter) -----------------------------------------
    vs = np.linspace(0, V_MAX, 200)
    u_null = A / (1.0 + vs**N)                      # du/dt = 0
    d = "M " + " L ".join(f"{to_svg(u,v)[0]:.1f} {to_svg(u,v)[1]:.1f}" for u, v in zip(u_null, vs))
    parts.append(f'<path d="{d}" stroke="{TEAL}" stroke-width="2.4" opacity="0.85"/>')

    us = np.linspace(0, U_MAX, 200)
    v_null = A / (1.0 + us**N)                      # dv/dt = 0
    d = "M " + " L ".join(f"{to_svg(u,v)[0]:.1f} {to_svg(u,v)[1]:.1f}" for u, v in zip(us, v_null))
    parts.append(f'<path d="{d}" stroke="{TEAL}" stroke-width="2.4" opacity="0.55"/>')

    # --- fixed points --------------------------------------------------
    for u, v in fixed_points():
        x, y = to_svg(u, v)
        if is_stable(u, v):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{TEAL}" opacity="0.95"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="{TEAL}" opacity="0.18"/>')
        else:  # saddle: open ring
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" stroke="{TEAL}" stroke-width="2.4" opacity="0.9"/>')

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    svg = build_svg()
    with open("assets/phase-portrait.svg", "w") as f:
        f.write(svg)
    print("Wrote assets/phase-portrait.svg")
