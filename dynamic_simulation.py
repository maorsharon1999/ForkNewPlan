"""
Fork Simulator V5 — Plotly Dash research viewer for ForkEVG eating-kinematics.
Features: gyro-complementary orientation, peak-cluster mouth, serial action IDs,
XAI reasoning panel, face-camera default, table collision clamp.
Run: python dynamic_simulation.py  →  http://127.0.0.1:8050

Ported verbatim from Forkevg/fork_simulator.py.
Single functional change: CSV_ROOT now reads from cfg.ANNOTATED_CSV_DIR so the
viewer automatically picks up annotated files produced by ForkNewPlan/main.py.
"""

import functools
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, Patch, State, callback_context, dcc, html, dash_table

import config as cfg
import body_model

# ── CONFIG ─────────────────────────────────────────────────────────────────────

FORK_LENGTH      = 0.20      # sensor → tine tip (m)
HANDLE_BACK_LEN  = 0.02      # stub behind sensor (m)
HANDLE_FWD_LEN   = 0.06      # dark-wood grip section forward of sensor (m)
HANDLE_COLOR     = "#4a3b2c" # dark wood
TINE_LENGTH    = 0.030
TINE_OFFSETS   = [-0.018, -0.006, 0.006, 0.018]

FPS            = 30
TRAIL_MAX      = 150
CSV_ROOT       = Path(cfg.ANNOTATED_CSV_DIR)   # ← ForkNewPlan output dir

PLATE_RADIUS   = 0.08
HEAD_RADIUS    = 0.065
MOUTH_HALF     = 0.022
NOSE_EXTEND    = 0.040
LO_CLUSTER_PCT = 0.03
SPEED_CHOICES  = [1, 2, 4, 8, 16]

TORSO_CENTER   = (0.0,  0.22,  0.06)
TORSO_RADII    = (0.09, 0.06, 0.17)
SHOULDER_X     = 0.10
SHOULDER_Y     = 0.22
SHOULDER_Z     = 0.22
REST_HAND_DX   = 0.22
REST_HAND_Y    = 0.10
SKIN_COLOR     = "#e6b89c"
SHIRT_COLOR    = "#c06018"

TABLE_X        = (-0.30,  0.30)
TABLE_Y        = (-0.30,  0.30)
AXIS_X         = (-0.40,  0.40)
AXIS_Y         = (-0.40,  0.40)
AXIS_Z         = (-0.26,  0.40)

ACTION_COLORS  = {"Noise": "#888888", "Stab": "#e63946", "Scoop": "#2dc653"}
ACTION_DEFAULT = "#aaaaaa"
HAND_COLORS    = {"Left": "#3a86ff", "Right": "#8338ec", "Unknown": "#6c757d"}


# ── ORIENTATION ────────────────────────────────────────────────────────────────

def _integrate_orientation(acc, gyro_deg, ts_ms):
    """Complementary filter: accel gives tilt, gyro_z adds yaw.
    Returns fork_dir (N,3) and acc_hat (N,3).
    Fork body convention: -Z_body points handle→prongs."""
    n = len(ts_ms)
    deg2rad = np.pi / 180.0

    dt = np.empty(n)
    dt[0] = 1.0 / 30.0
    if n > 1:
        dt[1:] = np.clip(np.diff(ts_ms.astype(float)) / 1000.0, 1e-4, 0.1)

    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    acc_hat  = acc / norms
    base_dir = -acc_hat

    increments = gyro_deg[:, 2] * deg2rad * dt
    yaw = np.cumsum(increments)
    if n > 10:
        t = np.arange(n, dtype=float)
        slope = np.polyfit(t, yaw, 1)[0]
        yaw -= slope * t

    cy, sy = np.cos(yaw), np.sin(yaw)
    fx = cy * base_dir[:, 0] - sy * base_dir[:, 1]
    fy = sy * base_dir[:, 0] + cy * base_dir[:, 1]
    fz = base_dir[:, 2]
    fork_dir = np.stack([fx, fy, fz], axis=1)

    fd_norm = np.linalg.norm(fork_dir, axis=1, keepdims=True)
    fork_dir = fork_dir / np.where(fd_norm < 1e-9, 1.0, fd_norm)
    return fork_dir.astype(float), acc_hat.astype(float)


# ── DATA ───────────────────────────────────────────────────────────────────────

def discover_csvs(root: Path):
    files = sorted(root.rglob("*.csv"))
    return [{"label": f"{f.parent.name} / {f.name}", "value": str(f)} for f in files]


def _derive_landmarks_v5(legacy_tip):
    """Plate = bottom-3% tip_z cluster.
    Mouth = highest repeatable peak cluster (prominence-filtered local maxima)."""
    zs = legacy_tip[:, 2]
    n  = len(zs)

    k_lo = max(5, int(n * LO_CLUSTER_PCT))
    lo_idx   = np.argsort(zs)[:k_lo]
    plate_xy = legacy_tip[lo_idx, :2].mean(axis=0)

    med, std = float(np.median(zs)), float(np.std(zs))
    threshold = med + 0.5 * std
    peaks = []
    for i in range(3, n - 3):
        if (zs[i] > threshold
                and zs[i] > zs[i - 3:i].max()
                and zs[i] > zs[i + 1:i + 4].max()):
            peaks.append(i)

    if len(peaks) >= 4:
        peaks    = np.array(peaks)
        peak_zs  = zs[peaks]
        z_top    = float(peak_zs.max())
        top_mask = peak_zs >= z_top * 0.85
        mouth    = legacy_tip[peaks[top_mask]].mean(axis=0)
    else:
        k_hi  = max(5, int(n * 0.03))
        hi_idx = np.argsort(zs)[-k_hi:]
        mouth  = legacy_tip[hi_idx].mean(axis=0)

    return plate_xy.astype(float), mouth.astype(float)


def _enumerate_actions(motions, acc_hat_z, fork_dir_z, gyro_mag):
    """Group consecutive same-Motion rows; attach XAI reasoning per group."""
    groups, counts, cur = [], {}, None
    for i, m in enumerate(motions):
        if cur is None or m != cur["motion"]:
            if cur is not None:
                groups.append(cur)
            counts[m] = counts.get(m, 0) + 1
            cur = {"start": i, "end": i, "motion": m, "seq": counts[m]}
        else:
            cur["end"] = i
    if cur is not None:
        groups.append(cur)

    for g in groups:
        g["id"] = f"{g['motion']} #{g['seq']}"
        s, e, dur = g["start"], g["end"] + 1, g["end"] - g["start"] + 1
        m = g["motion"]
        if m == "Stab":
            peak_az = float(np.abs(acc_hat_z[s:e]).max())
            g["reason"] = (
                f"Stab detected: sharp downward impact\n"
                f"  peak |acc_z / |acc|| = {peak_az:.3f}\n"
                f"  duration = {dur} frames"
            )
        elif m == "Scoop":
            dz = float(np.diff(fork_dir_z[s:e]).mean()) if dur > 1 else 0.0
            g["reason"] = (
                f"Scoop detected: upward lifting arc\n"
                f"  mean Δfork_dir_z / frame = {dz:+.5f}\n"
                f"  duration = {dur} frames"
            )
        else:
            mean_gyro = float(gyro_mag[s:e].mean()) if dur > 0 else 0.0
            g["reason"] = (
                f"Noise: low-amplitude / transition\n"
                f"  mean |ω| = {mean_gyro:.2f} °/s\n"
                f"  duration = {dur} frames"
            )
    return groups


def _find_group(groups, idx):
    lo, hi = 0, len(groups) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if groups[mid]["end"] < idx:
            lo = mid + 1
        elif groups[mid]["start"] > idx:
            hi = mid - 1
        else:
            return groups[mid]
    return groups[0] if groups else {"id": "—", "motion": "Noise", "reason": ""}


def _fmt_time(sec):
    sec = max(0.0, float(sec))
    m   = int(sec // 60)
    s   = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


@functools.lru_cache(maxsize=16)
def load_file(path: str):
    df     = pd.read_csv(path)
    acc    = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyro   = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    ts_ms  = df["timestamp"].to_numpy(dtype=float)

    norms_a   = np.linalg.norm(acc, axis=1, keepdims=True)
    norms_a   = np.where(norms_a < 1e-9, 1.0, norms_a)
    legacy_tip = (-FORK_LENGTH * acc / norms_a).astype(float)
    table_z    = float(legacy_tip[:, 2].min())
    plate_xy, mouth = _derive_landmarks_v5(legacy_tip)

    fork_dir, acc_hat = _integrate_orientation(acc, gyro, ts_ms)

    plate_anchor = np.array([plate_xy[0], plate_xy[1], table_z + 0.05])
    mouth_anchor = np.array([mouth[0], mouth[1] - 0.05, mouth[2] - 0.04])
    z_lo  = float(legacy_tip[:, 2].min())
    z_hi  = float(legacy_tip[:, 2].max())
    span  = max(z_hi - z_lo, 1e-6)
    prog  = np.clip((legacy_tip[:, 2] - z_lo) / span, 0.0, 1.0)
    hand_xyz = (plate_anchor + prog[:, None] * (mouth_anchor - plate_anchor)).astype(float)

    tip_xyz = (hand_xyz + FORK_LENGTH * fork_dir).astype(float)

    below = tip_xyz[:, 2] < table_z + 0.002
    if below.any():
        tip_xyz[below, 2] = table_z + 0.002

    hand_counts = df["Hand"].value_counts()
    non_unk     = hand_counts[hand_counts.index != "Unknown"]
    dominant    = str(non_unk.index[0]) if len(non_unk) else "Right"
    dom_count   = int(hand_counts.get(dominant, 0))
    hand_reason = (f"{dominant} — most frequent non-Unknown label\n"
                   f"  ({dom_count} of {len(df)} rows)")

    gyro_mag = np.linalg.norm(gyro, axis=1)
    groups   = _enumerate_actions(
        list(df["Motion"].astype(str)),
        acc_hat[:, 2],
        fork_dir[:, 2],
        gyro_mag,
    )

    t_sec = (ts_ms - float(ts_ms[0])) / 1000.0
    telem_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.5, 0.5],
                               vertical_spacing=0.06)
    gyro_cols = [("gyro_x", "#e63946"), ("gyro_y", "#2dc653"), ("gyro_z", "#3a86ff")]
    for col, clr in gyro_cols:
        telem_fig.add_trace(go.Scatter(
            x=t_sec, y=df[col].to_numpy(dtype=float),
            mode="lines", line=dict(color=clr, width=1),
            name=col), row=1, col=1)
    pos_cols = [("X", hand_xyz[:, 0], "#e63946"),
                ("Y", hand_xyz[:, 1], "#2dc653"),
                ("Z", hand_xyz[:, 2], "#3a86ff")]
    for lbl, arr, clr in pos_cols:
        telem_fig.add_trace(go.Scatter(
            x=t_sec, y=arr,
            mode="lines", line=dict(color=clr, width=1),
            name=f"pos_{lbl}"), row=2, col=1)
    telem_fig.update_yaxes(title_text="ω (°/s)", row=1, col=1,
                            title_font=dict(size=11))
    telem_fig.update_yaxes(title_text="pos (m)", row=2, col=1,
                            title_font=dict(size=11))
    telem_fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                            title_font=dict(size=11))
    telem_fig.update_layout(
        margin=dict(l=55, r=10, t=20, b=30),
        paper_bgcolor="#ffffff", plot_bgcolor="#f9f9f9",
        legend=dict(x=1.0, y=1.0, xanchor="right",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        uirevision=path,
        shapes=[],
    )

    return {
        "df": df, "tip_xyz": tip_xyz, "hand_xyz": hand_xyz,
        "fork_dir": fork_dir, "table_z": table_z,
        "plate_xy": plate_xy, "mouth": mouth,
        "dominant_hand": dominant, "hand_reason": hand_reason,
        "groups": groups, "ts_ms": ts_ms,
        "t0_ms": float(ts_ms[0]), "duration_ms": float(ts_ms[-1] - ts_ms[0]),
        "telem_fig": telem_fig,
    }


# ── GEOMETRY HELPERS ───────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _sphere_mesh(cx, cy, cz, r, n=14):
    lat = np.linspace(0, np.pi, n)
    lon = np.linspace(0, 2 * np.pi, 2 * n)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    x = (cx + r * np.sin(la) * np.cos(lo)).ravel()
    y = (cy + r * np.sin(la) * np.sin(lo)).ravel()
    z = (cz + r * np.cos(la)).ravel()
    nr, nc, ii, jj, kk = n, 2 * n, [], [], []
    for row in range(nr - 1):
        for col in range(nc - 1):
            v0 = row * nc + col
            v1 = row * nc + (col + 1) % nc
            v2 = (row + 1) * nc + col
            v3 = (row + 1) * nc + (col + 1) % nc
            ii += [v0, v0]; jj += [v1, v3]; kk += [v3, v2]
    return dict(x=x, y=y, z=z, i=ii, j=jj, k=kk)


def _tine_segments(tip_world, fork_dir_unit):
    """4 fork tines from tip. fork_dir_unit: handle→prongs unit vector."""
    d = fork_dir_unit / (np.linalg.norm(fork_dir_unit) + 1e-9)
    z_hat = np.array([0.0, 0.0, 1.0])
    cross = np.cross(d, z_hat)
    if np.linalg.norm(cross) < 1e-9:
        cross = np.cross(d, np.array([1.0, 0.0, 0.0]))
    u = cross / np.linalg.norm(cross)
    xs, ys, zs = [], [], []
    for s in TINE_OFFSETS:
        base = tip_world + s * u
        end  = base + TINE_LENGTH * d
        xs += [base[0], end[0], None]
        ys += [base[1], end[1], None]
        zs += [base[2], end[2], None]
    return xs, ys, zs


@functools.lru_cache(maxsize=8)
def _disc_mesh(cx, cy, cz, r, n=24):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.concatenate([[cx], cx + r * np.cos(theta)])
    ys = np.concatenate([[cy], cy + r * np.sin(theta)])
    zs = np.full(n + 1, cz)
    ii = [0] * n
    jj = list(range(1, n + 1))
    kk = [(i % n) + 1 for i in range(1, n + 1)]
    return dict(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk)


@functools.lru_cache(maxsize=8)
def _ellipsoid_mesh(cx, cy, cz, rx, ry, rz, n=14):
    lat = np.linspace(0, np.pi, n)
    lon = np.linspace(0, 2 * np.pi, 2 * n)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    x = (cx + rx * np.sin(la) * np.cos(lo)).ravel()
    y = (cy + ry * np.sin(la) * np.sin(lo)).ravel()
    z = (cz + rz * np.cos(la)).ravel()
    nr, nc, ii, jj, kk = n, 2 * n, [], [], []
    for row in range(nr - 1):
        for col in range(nc - 1):
            v0 = row * nc + col
            v1 = row * nc + (col + 1) % nc
            v2 = (row + 1) * nc + col
            v3 = (row + 1) * nc + (col + 1) % nc
            ii += [v0, v0]; jj += [v1, v3]; kk += [v3, v2]
    return dict(x=x, y=y, z=z, i=ii, j=jj, k=kk)


# ── STATIC SCENE ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _static_traces(table_z, plate_xy, mouth, dominant_hand="Right"):
    plate_xy = np.array(plate_xy, dtype=float)
    mouth    = np.array(mouth,    dtype=float)

    head_center = mouth + np.array([0.0, HEAD_RADIUS,        HEAD_RADIUS * 0.30])
    nose_base   = mouth + np.array([0.0, 0.0,                0.015])
    nose_tip    = mouth + np.array([0.0, -NOSE_EXTEND,       0.020])
    mouth_L     = mouth + np.array([-MOUTH_HALF, 0.0, 0.0])
    mouth_R     = mouth + np.array([ MOUTH_HALF, 0.0, 0.0])

    sign          = -1.0 if dominant_hand != "Left" else 1.0
    dom_shoulder  = np.array([sign  * SHOULDER_X, SHOULDER_Y, SHOULDER_Z])
    rest_shoulder = np.array([-sign * SHOULDER_X, SHOULDER_Y, SHOULDER_Z])
    rest_hand_pos = np.array([-sign * REST_HAND_DX, REST_HAND_Y, table_z + 0.025])

    traces = []
    traces.extend(body_model.body_traces(mouth, dominant_hand, table_z))

    traces.append(go.Mesh3d(
        x=[TABLE_X[0], TABLE_X[1], TABLE_X[1], TABLE_X[0]],
        y=[TABLE_Y[0], TABLE_Y[0], TABLE_Y[1], TABLE_Y[1]],
        z=[table_z] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
        color="burlywood", opacity=0.40, flatshading=True,
        name="Table", showlegend=True))

    traces.append(go.Mesh3d(
        **_disc_mesh(float(plate_xy[0]), float(plate_xy[1]), table_z + 0.002, PLATE_RADIUS),
        color="#ffffff", opacity=0.90, flatshading=True,
        name="Plate", showlegend=True))
    theta = np.linspace(0, 2 * np.pi, 50)
    traces.append(go.Scatter3d(
        x=plate_xy[0] + PLATE_RADIUS * np.cos(theta),
        y=plate_xy[1] + PLATE_RADIUS * np.sin(theta),
        z=np.full_like(theta, table_z + 0.003),
        mode="lines", line=dict(color="#b0b0b0", width=3),
        showlegend=False, name="_plate_rim"))

    traces.append(go.Scatter3d(
        x=[nose_base[0], nose_tip[0]], y=[nose_base[1], nose_tip[1]],
        z=[nose_base[2], nose_tip[2]],
        mode="lines+markers",
        line=dict(color="#c77b4a", width=5),
        marker=dict(size=[0, 5], color="#c77b4a"),
        name="Nose"))

    traces.append(go.Scatter3d(
        x=[mouth_L[0], mouth_R[0]], y=[mouth_L[1], mouth_R[1]],
        z=[mouth_L[2], mouth_R[2]],
        mode="lines", line=dict(color="#c1121f", width=7),
        name="Mouth"))

    lx = [float(head_center[0]), float(nose_tip[0]) - 0.04,
          float(mouth_R[0]) + 0.04, float(plate_xy[0]),
          TABLE_X[0] + 0.02, float(TORSO_CENTER[0]),
          float(rest_hand_pos[0])]
    ly = [float(head_center[1]), float(nose_tip[1]),
          float(mouth_R[1]), float(plate_xy[1]),
          TABLE_Y[0] + 0.02, float(TORSO_CENTER[1]) - 0.12,
          float(rest_hand_pos[1])]
    lz = [float(head_center[2]) + HEAD_RADIUS + 0.03,
          float(nose_tip[2]) + 0.02, float(mouth_R[2]) + 0.02,
          table_z + 0.02, table_z + 0.02,
          float(TORSO_CENTER[2]) + TORSO_RADII[2] + 0.03,
          float(rest_hand_pos[2]) + 0.04]
    traces.append(go.Scatter3d(
        x=lx, y=ly, z=lz, mode="text",
        text=["HEAD", "NOSE", "MOUTH", "PLATE", "TABLE", "BODY", "REST"],
        textfont=dict(size=13, color="#333333"),
        showlegend=False, name="_labels"))

    return traces


# ── PER-FRAME FIGURE ───────────────────────────────────────────────────────────

def _axis_cfg(title, rng):
    return dict(title=dict(text=title, font=dict(size=11)),
                range=list(rng), autorange=False,
                showbackground=True, backgroundcolor="#f4f4f4",
                gridcolor="#dddddd", zerolinecolor="#aaaaaa")


def build_figure(tip_xyz, hand_xyz, fork_dir, table_z, plate_xy, mouth,
                 frame_idx, motion, uirevision, dominant_hand="Right"):
    sensor   = hand_xyz[frame_idx]
    tip      = tip_xyz[frame_idx]
    fdir     = fork_dir[frame_idx]
    color    = ACTION_COLORS.get(motion, ACTION_DEFAULT)

    butt_pt  = sensor - HANDLE_BACK_LEN * fdir
    grip_end = sensor + HANDLE_FWD_LEN * fdir

    sign         = -1.0 if dominant_hand != "Left" else 1.0
    dom_shoulder = np.array([sign * SHOULDER_X, SHOULDER_Y, SHOULDER_Z])

    trail_start = max(0, frame_idx - TRAIL_MAX)
    trail = hand_xyz[trail_start: frame_idx + 1]

    traces = list(_static_traces(table_z, tuple(plate_xy), tuple(mouth), dominant_hand))

    if len(trail) > 1:
        traces.append(go.Scatter3d(
            x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
            mode="lines", line=dict(color="lightgray", width=2), name="Trail"))

    traces.append(go.Scatter3d(
        x=[dom_shoulder[0], sensor[0]], y=[dom_shoulder[1], sensor[1]],
        z=[dom_shoulder[2], sensor[2]],
        mode="lines", line=dict(color=SKIN_COLOR, width=8),
        showlegend=False, name="_active_arm"))

    traces.append(go.Scatter3d(
        x=[sensor[0]], y=[sensor[1]], z=[sensor[2]],
        mode="markers", marker=dict(size=12, color="#ffb347"),
        name="Hand"))

    traces.append(go.Scatter3d(
        x=[butt_pt[0], sensor[0]], y=[butt_pt[1], sensor[1]],
        z=[butt_pt[2], sensor[2]],
        mode="lines", line=dict(color=HANDLE_COLOR, width=14),
        showlegend=False, name="_handle_rear"))

    traces.append(go.Scatter3d(
        x=[sensor[0], grip_end[0]], y=[sensor[1], grip_end[1]],
        z=[sensor[2], grip_end[2]],
        mode="lines", line=dict(color=HANDLE_COLOR, width=14),
        name="Handle"))

    traces.append(go.Scatter3d(
        x=[grip_end[0], tip[0]], y=[grip_end[1], tip[1]],
        z=[grip_end[2], tip[2]],
        mode="lines", line=dict(color=color, width=8),
        name="Fork"))

    traces.append(go.Scatter3d(
        x=[sensor[0]], y=[sensor[1]], z=[sensor[2]],
        mode="markers",
        marker=dict(size=10, color="#ffffff", symbol="circle",
                    line=dict(color=color, width=3)),
        name="Sensor"))

    xs, ys, zs = _tine_segments(tip, fdir)
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=5),
        showlegend=False, name="_tines"))

    traces.append(go.Scatter3d(
        x=[sensor[0] - 0.04, tip[0] + 0.03],
        y=[sensor[1],         tip[1]],
        z=[sensor[2] + 0.04,  tip[2] + 0.03],
        mode="text", text=["SENSOR", "TIP"],
        textfont=dict(size=12, color="#333333"),
        showlegend=False, name="_dyn_labels"))

    xspan = AXIS_X[1] - AXIS_X[0]
    yspan = AXIS_Y[1] - AXIS_Y[0]
    zspan = AXIS_Z[1] - AXIS_Z[0]
    mv    = np.array(mouth)

    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision=uirevision,
        margin=dict(l=0, r=0, t=28, b=0),
        scene=dict(
            xaxis=_axis_cfg("X (right)", AXIS_X),
            yaxis=_axis_cfg("Y (patient)", AXIS_Y),
            zaxis=_axis_cfg("Z (up)",   AXIS_Z),
            aspectmode="manual",
            aspectratio=dict(x=xspan / zspan, y=yspan / zspan, z=1.0),
            camera=dict(
                eye=dict(x=0.0, y=float(mv[1]) - 1.60, z=float(mv[2]) + 0.30),
                center=dict(x=0.0, y=float(mv[1]),      z=float(mv[2])),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)",
                    font=dict(size=11)),
        paper_bgcolor="#ffffff",
    )
    return fig


# ── APP LAYOUT ─────────────────────────────────────────────────────────────────

app    = dash.Dash(__name__, title="Fork Simulator V5")
server = app.server

csv_options  = discover_csvs(CSV_ROOT)
default_file = csv_options[0]["value"] if csv_options else None

_badge_base = dict(
    textAlign="center", padding="10px 0", borderRadius="6px",
    fontSize="18px", fontWeight="bold", color="white",
    background=ACTION_DEFAULT, letterSpacing="2px",
)
_panel = dict(
    display="flex", flexDirection="column", gap="10px",
    padding="14px", width="400px", flexShrink="0",
    overflowY="auto", background="#ffffff",
    boxShadow="-2px 0 8px rgba(0,0,0,0.12)",
    fontFamily="'Courier New', monospace", fontSize="13px",
)

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row",
           "height": "100vh", "background": "#eeeeee"},
    children=[

        html.Div(style={"flex": "1 1 auto", "display": "flex",
                        "flexDirection": "column", "padding": "8px",
                        "gap": "4px"},
                 children=[
                     dcc.Graph(id="scene", style={"flex": "65 1 0"},
                               config={"scrollZoom": True}),
                     dcc.Graph(id="telemetry", style={"flex": "35 1 0"},
                               config={"displayModeBar": False}),
                 ]),

        html.Div(style=_panel, children=[

            html.H3("Fork Simulator V5",
                    style={"margin": "0", "fontFamily": "sans-serif"}),

            html.Label("File", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="file-picker", options=csv_options,
                         value=default_file, clearable=False,
                         style={"fontSize": "12px"}),

            html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"},
                     children=[
                         html.Label("Speed:", style={"whiteSpace": "nowrap"}),
                         dcc.Dropdown(id="speed",
                                      options=[{"label": f"{s}x", "value": s}
                                               for s in SPEED_CHOICES],
                                      value=1, clearable=False,
                                      style={"width": "80px", "fontSize": "13px"}),
                         html.Button("Play",  id="play-btn",  n_clicks=0,
                                     style={"flex": "1", "padding": "7px",
                                            "cursor": "pointer", "borderRadius": "4px"}),
                         html.Button("Pause", id="pause-btn", n_clicks=0,
                                     style={"flex": "1", "padding": "7px",
                                            "cursor": "pointer", "borderRadius": "4px"}),
                     ]),

            html.Label("Frame", style={"fontWeight": "bold"}),
            dcc.Slider(id="frame-slider", min=0, max=1000, step=1, value=0,
                       marks=None,
                       tooltip={"always_visible": True, "placement": "bottom"}),

            html.Div(id="frame-id",
                     style={"background": "#f0f0f0", "padding": "6px",
                            "borderRadius": "4px", "whiteSpace": "pre"}),

            html.Label("Action", style={"fontWeight": "bold"}),
            html.Div(id="action-badge", style=_badge_base),

            html.Label("Hand", style={"fontWeight": "bold"}),
            html.Div(id="hand-badge", style={**_badge_base, "background": "#6c757d"}),

            html.Label("Reasoning (XAI)", style={"fontWeight": "bold"}),
            html.Pre(id="reasoning",
                     style={"background": "#f8f8f0", "padding": "8px",
                            "borderRadius": "4px", "fontSize": "11px",
                            "whiteSpace": "pre-wrap", "overflowY": "auto",
                            "border": "1px solid #ddd", "maxHeight": "140px"}),

            html.Div(id="tip-coords",
                     style={"background": "#f0f0f0", "padding": "6px",
                            "borderRadius": "4px", "whiteSpace": "pre"}),

            html.Label("Raw CSV row", style={"fontWeight": "bold"}),
            html.Div(style={"overflowX": "auto", "fontSize": "11px"},
                     children=[dash_table.DataTable(
                         id="raw-row",
                         style_table={"overflowX": "auto"},
                         style_cell={"padding": "4px 6px", "fontFamily": "monospace"},
                         style_header={"fontWeight": "bold", "background": "#e8e8e8"},
                     )]),

            dcc.Interval(id="tick", interval=1000 // FPS, disabled=True),
            dcc.Store(id="play-state", data={"playing": False}),
            dcc.Store(id="file-state", data={}),
        ]),
    ],
)


# ── CALLBACKS ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("file-state",   "data"),
    Output("frame-slider", "max"),
    Output("frame-slider", "value"),
    Output("telemetry",    "figure"),
    Input("file-picker", "value"),
)
def cb_load_file(path):
    empty_telem = go.Figure()
    if not path:
        return {}, 0, 0, empty_telem
    data = load_file(path)
    n    = len(data["df"])
    return ({"path": path, "n_frames": n,
             "table_z":  data["table_z"],
             "plate_xy": data["plate_xy"].tolist(),
             "mouth":    data["mouth"].tolist(),
             "dominant_hand": data["dominant_hand"]},
            n - 1, 0, data["telem_fig"])


@app.callback(
    Output("play-state", "data"),
    Output("tick",       "disabled"),
    Input("play-btn",  "n_clicks"),
    Input("pause-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_play_pause(play_n, pause_n):
    playing = callback_context.triggered_id == "play-btn"
    return {"playing": playing}, not playing


@app.callback(
    Output("frame-slider", "value", allow_duplicate=True),
    Input("tick", "n_intervals"),
    State("frame-slider", "value"),
    State("file-state",   "data"),
    State("play-state",   "data"),
    State("speed",        "value"),
    prevent_initial_call=True,
)
def cb_tick(_, frame_idx, file_state, play_state, speed):
    if not play_state.get("playing") or not file_state:
        return dash.no_update
    n = file_state.get("n_frames", 1)
    return (int(frame_idx) + int(speed or 1)) % n


@app.callback(
    Output("scene",        "figure"),
    Output("frame-id",     "children"),
    Output("action-badge", "children"),
    Output("action-badge", "style"),
    Output("hand-badge",   "children"),
    Output("hand-badge",   "style"),
    Output("reasoning",    "children"),
    Output("tip-coords",   "children"),
    Output("raw-row",      "data"),
    Output("raw-row",      "columns"),
    Output("telemetry",    "figure", allow_duplicate=True),
    Input("frame-slider", "value"),
    Input("file-state",   "data"),
    prevent_initial_call="initial_duplicate",
)
def cb_render(frame_idx, file_state):
    _act_style  = {**_badge_base, "background": ACTION_DEFAULT}
    _hand_style = {**_badge_base, "background": "#6c757d"}
    empty_fig   = go.Figure()
    blank = (empty_fig, "—", "—", _act_style, "—", _hand_style, "—", "—", [], [],
             dash.no_update)
    if not file_state or "path" not in file_state:
        return blank

    path    = file_state["path"]
    cached  = load_file(path)
    df      = cached["df"]
    tip_xyz = cached["tip_xyz"]
    hand_xyz = cached["hand_xyz"]
    fork_dir = cached["fork_dir"]
    plate_xy = cached["plate_xy"]
    mouth    = cached["mouth"]
    table_z  = cached["table_z"]
    groups   = cached["groups"]
    ts_ms    = cached["ts_ms"]
    t0_ms    = cached["t0_ms"]
    dur_ms   = cached["duration_ms"]
    n        = len(df)

    idx = max(0, min(int(frame_idx or 0), n - 1))
    row = df.iloc[idx]
    ts  = float(ts_ms[idx])

    motion        = str(row.get("Motion", "Noise"))
    hand_label    = str(row.get("Hand",   "Unknown"))
    dominant_hand = cached["dominant_hand"]
    tip           = tip_xyz[idx]
    sensor        = hand_xyz[idx]

    fig = build_figure(tip_xyz, hand_xyz, fork_dir, table_z, plate_xy, mouth,
                       idx, motion, uirevision=path, dominant_hand=dominant_hand)

    grp         = _find_group(groups, idx)
    action_id   = grp["id"]
    mot_reason  = grp.get("reason", "")
    hand_reason = cached["hand_reason"]
    reasoning   = (f"Hand: {dominant_hand}\n  {hand_reason}\n\n"
                   f"Motion: {action_id}\n  {mot_reason}")

    elapsed_s = (ts - t0_ms) / 1000.0
    total_s   = dur_ms / 1000.0
    frame_text = (f"Row {idx} / {n}   {_fmt_time(elapsed_s)} / {_fmt_time(total_s)}\n"
                  f"ts = {int(ts)} ms")

    coords_text = (f"sensor x={sensor[0]:+.4f}  y={sensor[1]:+.4f}  z={sensor[2]:+.4f}\n"
                   f"tip    x={tip[0]:+.4f}  y={tip[1]:+.4f}  z={tip[2]:+.4f}")

    act_style  = {**_badge_base, "background": ACTION_COLORS.get(motion, ACTION_DEFAULT)}
    hand_style = {**_badge_base, "background": HAND_COLORS.get(hand_label, "#6c757d")}

    playhead_x = (float(ts_ms[idx]) - t0_ms) / 1000.0
    telem_patch = Patch()
    telem_patch["layout"]["shapes"] = [
        dict(type="line", xref="x",  yref="paper",
             x0=playhead_x, x1=playhead_x, y0=0.5, y1=1.0,
             line=dict(color="#ff0000", width=2, dash="dot")),
        dict(type="line", xref="x2", yref="paper",
             x0=playhead_x, x1=playhead_x, y0=0.0, y1=0.5,
             line=dict(color="#ff0000", width=2, dash="dot")),
    ]

    return (fig, frame_text,
            action_id, act_style,
            hand_label, hand_style,
            reasoning,
            coords_text,
            [row.to_dict()], [{"name": c, "id": c} for c in df.columns],
            telem_patch)


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nFork Simulator V5 — open http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
