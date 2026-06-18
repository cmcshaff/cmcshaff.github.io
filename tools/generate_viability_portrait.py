"""
Viability portrait (single-cell physiology model, McShaffrey & Beer 2026).

A geometric construct, not a scientific plot: muted survival-outcome regions
as a quiet backdrop, with the viability boundary, the organizing manifolds,
and the limit sets as the vibrant, glowing details. No axes, no labels.

Out: assets/viability-portrait.svg
"""

import base64, io
import numpy as np
from PIL import Image, ImageFilter

# ---- model -------------------------------------------------------------
rchem, K1, h1, kd, F1, F2 = 0.04, 4.0, 3.0, 0.05, 28.0, 6.0
LO, HI = 0.1, 20.0

def hill(m):
    m = np.maximum(m, 0.0)
    return m**h1 / (K1**h1 + m**h1)

def deriv(M1, M2):
    return (rchem*hill(M2)*F1 - kd*M1, rchem*hill(M1)*F2 - kd*M2)

# ---- 1. classify initial conditions by survival outcome (vectorized) ---
RES = 320
ax = np.linspace(0, 20, RES)
G1, G2 = np.meshgrid(ax, ax)
M1 = G1.flatten().astype(float); M2 = G2.flatten().astype(float)
DEATHZONE, A, T1, T2, T3 = 4, 0, 1, 2, 3
outcome = np.full(M1.size, -1)
outcome[(M1 < LO) | (M2 < LO) | (M1 + M2 > HI)] = DEATHZONE
active = outcome == -1
dt = 0.4
for _ in range(1600):
    a = active
    k1 = deriv(M1, M2)
    k2 = deriv(M1 + 0.5*dt*k1[0], M2 + 0.5*dt*k1[1])
    k3 = deriv(M1 + 0.5*dt*k2[0], M2 + 0.5*dt*k2[1])
    k4 = deriv(M1 + dt*k3[0], M2 + dt*k3[1])
    M1 = M1 + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * a
    M2 = M2 + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * a
    c3 = a & (M1 + M2 > HI); c1 = a & (M1 < LO); c2 = a & (M2 < LO)
    outcome[c3] = T3; active &= ~c3
    outcome[c1] = T1; active &= ~c1
    outcome[c2] = T2; active &= ~c2
    if not active.any():
        break
outcome[outcome == -1] = A
outcome = outcome.reshape(RES, RES)

# muted, low-saturation fields (quiet backdrop)
fill = {DEATHZONE: (7, 10, 16),
        A:  (22, 48, 46),    # survive  - muted teal
        T3: (60, 48, 38),    # burst    - muted warm brown
        T2: (54, 44, 64),    # starve M2- muted plum
        T1: (40, 46, 70)}    # starve M1- muted indigo
rgb = np.zeros((RES, RES, 3), np.uint8)
for k, c in fill.items():
    rgb[outcome == k] = c
img = Image.fromarray(rgb[::-1]).filter(ImageFilter.GaussianBlur(1.4))
buf = io.BytesIO(); img.save(buf, format="PNG")
field_b64 = base64.b64encode(buf.getvalue()).decode()

# ---- 2. organizing structures (scalar integration) ---------------------
def f(s):
    m1, m2 = s
    return np.array([rchem*(max(m2,0)**h1/(K1**h1+max(m2,0)**h1))*F1 - kd*m1,
                     rchem*(max(m1,0)**h1/(K1**h1+max(m1,0)**h1))*F2 - kd*m2])

def alive(s):
    return s[0] >= LO and s[1] >= LO and s[0] + s[1] <= HI

def integ(s0, h, n=6000, stop_attr=False):
    s = np.array(s0, float); pts = [s.copy()]
    for _ in range(n):
        k1 = f(s); k2 = f(s+0.5*h*k1); k3 = f(s+0.5*h*k2); k4 = f(s+h*k3)
        s = s + h/6*(k1+2*k2+2*k3+k4)
        if not alive(s):
            pts.append(s.copy()); break
        pts.append(s.copy())
        if stop_attr and np.linalg.norm(f(s)) < 5e-5:
            break
    return np.array(pts)

def jac(s, e=1e-6):
    J = np.zeros((2, 2))
    for i in range(2):
        sp = s.copy(); sm = s.copy(); sp[i] += e; sm[i] -= e
        J[:, i] = (f(sp) - f(sm)) / (2*e)
    return J

def newton(s):
    s = np.array(s, float)
    for _ in range(60):
        s = s - np.linalg.solve(jac(s), f(s))
    return s

attractor = newton([13.811, 4.686])
saddle = newton([4.008, 2.407])
w, V = np.linalg.eig(jac(saddle))
unstable = V[:, np.argmax(w.real)].real
stable = V[:, np.argmin(w.real)].real
e0 = 0.05
Wu = [integ(saddle + sgn*e0*unstable, +0.3, stop_attr=True) for sgn in (+1, -1)]
Ws = [integ(saddle + sgn*e0*stable, -0.3) for sgn in (+1, -1)]
Mman = integ(np.array([14.836, 5.164]), -0.3)
Oman = integ(np.array([0.1, 0.1]), -0.3)
mortality_pt = np.array([14.836, 5.164]); ordering_pt = np.array([0.1, 0.1])

# ---- 3. assemble SVG (geometric construct, no axes / labels) -----------
PAD, PLOT = 16, 640
W = H = PLOT + 2*PAD
def X(m): return PAD + m/20.0*PLOT
def Y(m): return PAD + PLOT - m/20.0*PLOT
def dpath(p): return "M " + " L ".join(f"{X(a):.1f} {Y(b):.1f}" for a, b in p)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" fill="none" '
     f'stroke-linecap="round" stroke-linejoin="round">']
s.append(f'<defs><clipPath id="pc"><rect x="{PAD}" y="{PAD}" width="{PLOT}" height="{PLOT}" rx="14"/></clipPath></defs>')

# muted region backdrop
s.append(f'<image x="{PAD}" y="{PAD}" width="{PLOT}" height="{PLOT}" clip-path="url(#pc)" '
         f'preserveAspectRatio="none" href="data:image/png;base64,{field_b64}"/>')

def beam(p, color, core, glow, glow_op, core_op=0.97):
    d = dpath(p)
    return (f'<path d="{d}" stroke="{color}" stroke-width="{glow}" opacity="{glow_op}"/>'
            f'<path d="{d}" stroke="{color}" stroke-width="{core}" opacity="{core_op}"/>')

g = ['<g clip-path="url(#pc)">']
# unstable + stable manifolds of the saddle
for seg in Wu:
    g.append(beam(seg, "#ff8a6e", 1.8, 8, 0.16, 0.85))
for seg in Ws:
    g.append(beam(seg, "#8fb6ff", 1.8, 8, 0.16, 0.9))
# mortality + ordering manifolds (the signature structures)
g.append(beam(Mman, "#ff5ea8", 2.8, 12, 0.24))
g.append(beam(Oman, "#b06bff", 2.8, 12, 0.24))
g.append('</g>')
s.extend(g)

# viability boundary (the triangle) - vibrant cyan frame
tri = "M " + " L ".join(f"{X(a):.1f} {Y(b):.1f}" for a, b in [(0,0),(20,0),(0,20)]) + " Z"
s.append(f'<path d="{tri}" stroke="#46c6ff" stroke-width="11" opacity="0.16"/>')
s.append(f'<path d="{tri}" stroke="#74d6ff" stroke-width="2.4" opacity="0.95"/>')

# limit sets + organizing points
def glow_dot(p, color, r):
    return (f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{r*3:.0f}" fill="{color}" opacity="0.15"/>'
            f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{r*1.7:.0f}" fill="{color}" opacity="0.22"/>'
            f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{r}" fill="{color}"/>')
s.append(glow_dot(attractor, "#74f0ff", 8))                              # viable attractor
s.append(f'<circle cx="{X(saddle[0]):.1f}" cy="{Y(saddle[1]):.1f}" r="6.5" '
         f'fill="#0c1118" stroke="#8dffc0" stroke-width="2.6"/>')         # saddle (open)
s.append(glow_dot(mortality_pt, "#ff5ea8", 4.5))
s.append(glow_dot(ordering_pt, "#b06bff", 4.5))

s.append("</svg>")
open("assets/viability-portrait.svg", "w").write("".join(s))
print("done; regions:", {k: int((outcome == k).sum()) for k in fill})
