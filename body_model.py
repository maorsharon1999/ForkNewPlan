"""Anatomical patient body traces for Fork Simulator V5.

Replaces the torso-ellipsoid + head-sphere + shoulder-markers + rest-arm-line
primitives in dynamic_simulation._static_traces with more human-like geometry:
torso, hip pad, neck cylinder, head sphere, shoulder-cap spheres, and a
tubular rest arm — all assembled from pure numpy, no extra runtime deps.

Public API:
    body_traces(mouth, dominant_hand, table_z) -> list[go.BaseTraceType]
"""

import numpy as np
import plotly.graph_objects as go

# ── Anatomical constants (mirror dynamic_simulation.py) ──────────────────────
TORSO_CENTER = (0.0,  0.22,  0.06)
TORSO_RADII  = (0.09, 0.06,  0.17)
HIP_CENTER   = (0.0,  0.21, -0.09)
HIP_RADII    = (0.11, 0.07,  0.085)
SHOULDER_X   = 0.10
SHOULDER_Y   = 0.22
SHOULDER_Z   = 0.22
SHOULDER_R   = 0.046
NECK_RADIUS  = 0.024
HEAD_RADIUS  = 0.065
REST_HAND_DX = 0.22
REST_HAND_Y  = 0.10
SKIN_COLOR   = "#e6b89c"
SHIRT_COLOR  = "#c06018"


# ── Primitive mesh builders ──────────────────────────────────────────────────

def _sphere_vf(cx, cy, cz, r, n=14):
    phi   = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)
    P, T  = np.meshgrid(phi, theta)
    x = cx + r * np.sin(P) * np.cos(T)
    y = cy + r * np.sin(P) * np.sin(T)
    z = cz + r * np.cos(P)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    idx   = np.arange(n * n).reshape(n, n)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            faces += [[a, b, c], [b, d, c]]
    return verts, np.array(faces, dtype=np.int32)


def _ellipsoid_vf(cx, cy, cz, rx, ry, rz, n=14):
    v, f = _sphere_vf(0.0, 0.0, 0.0, 1.0, n)
    return v * np.array([rx, ry, rz]) + np.array([cx, cy, cz]), f


def _cylinder_vf(p0, p1, r, n=16):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    axis   = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    u   = axis / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1  = np.cross(u, ref);  v1 /= np.linalg.norm(v1)
    v2  = np.cross(u, v1)
    th  = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.outer(np.cos(th), v1) + np.outer(np.sin(th), v2)
    ring0 = p0 + r * circ
    ring1 = p1 + r * circ
    verts = np.vstack([ring0, ring1])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + i], [j, n + j, n + i]]
    return verts, np.array(faces, dtype=np.int32)


def _combine(parts):
    all_v, all_f = [], []
    offset = 0
    for v, f in parts:
        if len(v) == 0:
            continue
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.vstack(all_v), np.vstack(all_f)


def _mesh3d(v, f, color, opacity, name):
    showlegend = bool(name and not name.startswith("_"))
    return go.Mesh3d(
        x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
        i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
        color=color, opacity=opacity, flatshading=False,
        name=name, showlegend=showlegend,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def body_traces(mouth, dominant_hand, table_z):
    """Return Plotly traces for the seated patient body.

    Emits:
      - Combined shirt-colored Mesh3d: torso, hip, shoulder caps, rest-arm tube,
        rest hand (replaces the ellipsoid + shoulder-dots + arm-line + rest-hand)
      - Combined skin-colored Mesh3d: neck cylinder + head sphere
        (replaces the head sphere)

    The dominant-arm traces (from shoulder to fork hand) remain in build_figure
    and are NOT produced here.
    """
    mouth = np.asarray(mouth, float)
    sign  = -1.0 if dominant_hand != "Left" else 1.0

    # ── Anatomical positions ─────────────────────────────────────────────────
    head_center   = mouth + np.array([0.0, HEAD_RADIUS, HEAD_RADIUS * 0.30])
    torso_top     = np.array([TORSO_CENTER[0], TORSO_CENTER[1],
                               TORSO_CENTER[2] + TORSO_RADII[2]])
    neck_bottom   = torso_top + np.array([0.0, 0.0, 0.008])
    neck_top      = head_center - np.array([0.0, 0.0, HEAD_RADIUS * 0.85])
    dom_shoulder  = np.array([ sign * SHOULDER_X, SHOULDER_Y, SHOULDER_Z])
    rest_shoulder = np.array([-sign * SHOULDER_X, SHOULDER_Y, SHOULDER_Z])
    rest_hand_pos = np.array([-sign * REST_HAND_DX, REST_HAND_Y, table_z + 0.025])

    # ── Shirt-colored body ───────────────────────────────────────────────────
    shirt_parts = [
        _ellipsoid_vf(*TORSO_CENTER, *TORSO_RADII),
        _ellipsoid_vf(*HIP_CENTER,   *HIP_RADII),
        _sphere_vf(*dom_shoulder,  SHOULDER_R),
        _sphere_vf(*rest_shoulder, SHOULDER_R),
        _cylinder_vf(rest_shoulder, rest_hand_pos, 0.022),
        _sphere_vf(*rest_hand_pos, 0.028),
    ]
    sv, sf = _combine(shirt_parts)

    # ── Skin-colored head + neck ─────────────────────────────────────────────
    skin_parts = [
        _cylinder_vf(neck_bottom, neck_top, NECK_RADIUS),
        _sphere_vf(*head_center, HEAD_RADIUS),
    ]
    kv, kf = _combine(skin_parts)

    traces = []
    if len(sv):
        traces.append(_mesh3d(sv, sf, SHIRT_COLOR, opacity=0.70, name="Body"))
    if len(kv):
        traces.append(_mesh3d(kv, kf, SKIN_COLOR,  opacity=0.78, name="Head"))
    return traces
