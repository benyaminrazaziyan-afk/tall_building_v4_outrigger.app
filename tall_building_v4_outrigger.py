
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

G = 9.81

st.set_page_config(page_title="Outrigger Framework", layout="wide")


# =========================
# DATA MODELS
# =========================
@dataclass
class Material:
    E: float
    fy: float
    density: float = 7850.0


@dataclass
class CoreWallSection:
    thickness: float
    length_x: float
    length_y: float
    n_web_x: int = 2
    n_web_y: int = 2
    cracked_factor: float = 0.45

    @property
    def area(self) -> float:
        return self.n_web_x * self.length_x * self.thickness + self.n_web_y * self.length_y * self.thickness

    def inertia_y(self) -> float:
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        I_from_ywalls = self.n_web_y * (t * ly**3 / 12.0)
        arm = max(0.25 * lx, t / 2.0)
        A_x = lx * t
        I_from_xwalls = self.n_web_x * (lx * t**3 / 12.0 + A_x * arm**2)
        return self.cracked_factor * (I_from_ywalls + I_from_xwalls)

    def inertia_x(self) -> float:
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        I_from_xwalls = self.n_web_x * (t * lx**3 / 12.0)
        arm = max(0.25 * ly, t / 2.0)
        A_y = ly * t
        I_from_ywalls = self.n_web_y * (ly * t**3 / 12.0 + A_y * arm**2)
        return self.cracked_factor * (I_from_xwalls + I_from_ywalls)


@dataclass
class ColumnSection:
    b: float
    h: float
    cracked_factor: float = 0.70

    @property
    def area(self) -> float:
        return self.b * self.h

    def inertia_x(self) -> float:
        return self.cracked_factor * self.b * self.h**3 / 12.0

    def inertia_y(self) -> float:
        return self.cracked_factor * self.h * self.b**3 / 12.0


@dataclass
class OutriggerLevel:
    story: int
    axis: str = "x"
    truss_depth: float = 4.0
    chord_area: float = 0.12
    diagonal_area: float = 0.06
    truss_E: float = 200e9


@dataclass
class TowerInput:
    n_story: int
    story_height: float
    plan_x: float
    plan_y: float
    floor_mass: float
    core: CoreWallSection
    perimeter_column: ColumnSection
    corner_column: ColumnSection
    n_perimeter_columns_x_face: int
    n_perimeter_columns_y_face: int
    outriggers: List[OutriggerLevel] = field(default_factory=list)
    column_material: Material = field(default_factory=lambda: Material(E=32e9, fy=420e6))
    wall_material: Material = field(default_factory=lambda: Material(E=32e9, fy=40e6))
    design_base_shear_ratio: float = 0.02
    core_stiffness_modifier: float = 1.0
    column_stiffness_modifier: float = 1.0

    def height(self) -> float:
        return self.n_story * self.story_height

    def bay_arm(self, axis: str) -> float:
        if axis.lower() == "x":
            return max((self.plan_x - self.core.length_x) / 2.0, 0.5)
        return max((self.plan_y - self.core.length_y) / 2.0, 0.5)

    def n_engaged_columns_per_side(self, axis: str) -> int:
        return self.n_perimeter_columns_y_face if axis.lower() == "x" else self.n_perimeter_columns_x_face


# =========================
# ANALYSIS CORE
# =========================
def generalized_eigh(K: np.ndarray, M: np.ndarray):
    m = np.diag(M).astype(float)
    A = K / m[:, None]
    vals, vecs = np.linalg.eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def _ensure_vector_mass(inp: TowerInput) -> np.ndarray:
    return np.full(inp.n_story, inp.floor_mass, dtype=float)


def _assemble_shear_building_K(story_k: np.ndarray) -> np.ndarray:
    n = len(story_k)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += story_k[i]
        else:
            K[i, i] += story_k[i]
            K[i, i - 1] -= story_k[i]
            K[i - 1, i] -= story_k[i]
            K[i - 1, i - 1] += story_k[i]
    return K


def _assemble_diagonal_mass(masses: np.ndarray) -> np.ndarray:
    return np.diag(masses)


def _story_stiffness_from_core(inp: TowerInput, axis: str) -> np.ndarray:
    I = inp.core.inertia_y() if axis.lower() == "x" else inp.core.inertia_x()
    E = inp.wall_material.E
    H = inp.height()
    k_global = inp.core_stiffness_modifier * 3.0 * E * I / max(H**3, 1e-12)
    return np.full(inp.n_story, inp.n_story * k_global, dtype=float)


def _story_stiffness_from_columns(inp: TowerInput, axis: str) -> np.ndarray:
    E = inp.column_material.E
    if axis.lower() == "x":
        I_per = inp.perimeter_column.inertia_y()
        I_cor = inp.corner_column.inertia_y()
    else:
        I_per = inp.perimeter_column.inertia_x()
        I_cor = inp.corner_column.inertia_x()
    n_per_cols = 2 * inp.n_engaged_columns_per_side(axis)
    n_corner = 4
    I_total = n_per_cols * I_per + n_corner * I_cor
    H = inp.height()
    k_global = inp.column_stiffness_modifier * 3.0 * E * I_total / max(H**3, 1e-12)
    return np.full(inp.n_story, inp.n_story * k_global, dtype=float)


def _engaged_column_axial_stiffness(inp: TowerInput, story: int, axis: str) -> float:
    E = inp.column_material.E
    L = max(story * inp.story_height, inp.story_height)
    n_side = inp.n_engaged_columns_per_side(axis)
    A_per = inp.perimeter_column.area
    A_cor = inp.corner_column.area
    A_side_total = n_side * A_per + 1.0 * A_cor
    return E * A_side_total / L


def _outrigger_truss_tip_stiffness(arm: float, truss_depth: float, chord_area: float, diagonal_area: float, truss_E: float) -> float:
    d = max(truss_depth, 0.25)
    b = max(arm, 0.25)
    Ld = math.sqrt(b**2 + d**2)
    Ieq = 0.5 * chord_area * d**2
    k_bend = 3.0 * truss_E * Ieq / max(b**3, 1e-12)
    k_diag = 2.0 * truss_E * diagonal_area * d**2 / max(Ld**3, 1e-12)
    return k_bend + k_diag


def analyze_tower(inp: TowerInput, axis: str = "x"):
    masses = _ensure_vector_mass(inp)
    k_total = _story_stiffness_from_core(inp, axis) + _story_stiffness_from_columns(inp, axis)

    outrigger_rows = []
    for ou in inp.outriggers:
        if ou.axis.lower() != axis.lower():
            continue
        arm = inp.bay_arm(axis)
        k_truss_tip = _outrigger_truss_tip_stiffness(arm, ou.truss_depth, ou.chord_area, ou.diagonal_area, ou.truss_E)
        k_col_tip = _engaged_column_axial_stiffness(inp, ou.story, axis)
        k_tip = 1.0 / (1.0 / max(k_truss_tip, 1e-12) + 1.0 / max(k_col_tip, 1e-12))
        k_rot = 2.0 * k_tip * arm**2
        k_add = k_rot / max(ou.story * inp.story_height**2, 1e-12)
        k_total[:ou.story] += k_add

        outrigger_rows.append({
            "story": ou.story,
            "axis": axis,
            "arm_m": arm,
            "k_truss_tip": k_truss_tip,
            "k_column_tip": k_col_tip,
            "k_tip_combined": k_tip,
            "k_rot": k_rot,
            "distributed_story_stiffness": k_add,
        })

    K = _assemble_shear_building_K(k_total)
    M = _assemble_diagonal_mass(masses)
    w2, phi = generalized_eigh(K, M)
    mask = w2 > 1e-9
    w = np.sqrt(w2[mask])
    phi = phi[:, mask]

    periods = 2.0 * np.pi / w
    frequencies = w / (2.0 * np.pi)

    for i in range(phi.shape[1]):
        scale = phi[-1, i] if abs(phi[-1, i]) > 1e-12 else np.max(np.abs(phi[:, i]))
        phi[:, i] = phi[:, i] / scale
        if phi[-1, i] < 0:
            phi[:, i] *= -1

    weights = masses * G
    Vb = inp.design_base_shear_ratio * np.sum(weights)
    z = np.arange(1, inp.n_story + 1) * inp.story_height
    lateral_profile = z / np.sum(z)
    F = Vb * lateral_profile

    u = np.linalg.solve(K, F)
    drifts = np.diff(np.hstack(([0.0], u))) / inp.story_height
    max_drift = float(np.max(np.abs(drifts)))

    outrigger_axial = {}
    for row in outrigger_rows:
        j = int(row["story"]) - 1
        theta = u[j] / max((j + 1) * inp.story_height, 1e-12)
        delta_tip = theta * row["arm_m"]
        axial = row["k_tip_combined"] * delta_tip
        outrigger_axial[int(row["story"])] = axial

    story_df = pd.DataFrame({
        "Story": np.arange(1, inp.n_story + 1),
        "Story Stiffness (N/m)": k_total,
        "Floor Disp. (mm)": u * 1000.0,
        "Drift Ratio": drifts,
    })

    ou_df = pd.DataFrame(outrigger_rows)
    if not ou_df.empty:
        ou_df["Axial Force per Side (kN)"] = ou_df["story"].map(lambda s: outrigger_axial.get(int(s), 0.0) / 1000.0)

    results = {
        "periods": periods,
        "frequencies": frequencies,
        "mode_shapes": phi,
        "story_df": story_df,
        "outrigger_df": ou_df,
        "roof_disp_mm": float(u[-1] * 1000.0),
        "max_drift_ratio": max_drift,
        "base_shear_kN": float(Vb / 1000.0),
        "max_outrigger_axial_kN": float(max(outrigger_axial.values(), default=0.0) / 1000.0),
        "input": inp,
    }
    return results


# =========================
# PLOTS
# =========================
def plot_plan(inp: TowerInput, axis: str = "x"):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], lw=2)

    cx0 = 0.5 * (inp.plan_x - inp.core.length_x)
    cy0 = 0.5 * (inp.plan_y - inp.core.length_y)
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core.length_x, inp.core.length_y, fill=False, lw=2))

    nxf = inp.n_perimeter_columns_x_face
    nyf = inp.n_perimeter_columns_y_face

    xs = np.linspace(0, inp.plan_x, nxf + 2)[1:-1]
    ys = np.linspace(0, inp.plan_y, nyf + 2)[1:-1]

    for x in xs:
        ax.plot(x, 0, "ks")
        ax.plot(x, inp.plan_y, "ks")
    for y in ys:
        ax.plot(0, y, "ks")
        ax.plot(inp.plan_x, y, "ks")

    ax.plot([0, 0, inp.plan_x, inp.plan_x], [0, inp.plan_y, 0, inp.plan_y], "ko", markersize=7)

    if axis.lower() == "x":
        ymid = inp.plan_y / 2.0
        ax.plot([cx0, 0], [ymid, ymid], lw=3)
        ax.plot([cx0 + inp.core.length_x, inp.plan_x], [ymid, ymid], lw=3)
        ax.text(inp.plan_x / 2.0, ymid + 1.0, "Outrigger axis = X", ha="center")
    else:
        xmid = inp.plan_x / 2.0
        ax.plot([xmid, xmid], [cy0, 0], lw=3)
        ax.plot([xmid, xmid], [cy0 + inp.core.length_y, inp.plan_y], lw=3)
        ax.text(xmid + 1.0, inp.plan_y / 2.0, "Outrigger axis = Y", rotation=90, va="center")

    ax.set_title("Plan View")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    return fig


def plot_elevation(inp: TowerInput):
    fig, ax = plt.subplots(figsize=(6, 9))
    H = inp.height()
    x0, x1 = 0.0, 8.0

    ax.plot([x0, x0], [0, H], lw=2)
    ax.plot([x1, x1], [0, H], lw=2)
    ax.plot([x0, x1], [0, 0], lw=2)
    ax.plot([x0, x1], [H, H], lw=2)

    core_x0 = 2.5
    core_x1 = 5.5
    ax.plot([core_x0, core_x0], [0, H], lw=2)
    ax.plot([core_x1, core_x1], [0, H], lw=2)

    for i in range(inp.n_story + 1):
        z = i * inp.story_height
        ax.plot([x0, x1], [z, z], lw=0.4, alpha=0.4)

    for ou in inp.outriggers:
        z = ou.story * inp.story_height
        ax.plot([x0 - 0.8, x1 + 0.8], [z, z], lw=2.5)
        ax.text(x1 + 1.0, z, f"O @ {ou.story}", va="center")

    ax.set_title("Elevation View")
    ax.set_xlabel("Schematic width")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.25)
    return fig


def plot_modes(results, n_modes: int = 3):
    phi = results["mode_shapes"]
    periods = results["periods"]
    n_story = results["input"].n_story
    story_height = results["input"].story_height
    z = np.arange(1, n_story + 1) * story_height

    n_modes = min(n_modes, phi.shape[1])
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in range(n_modes):
        ax.plot(phi[:, i], z, label=f"Mode {i+1} (T={periods[i]:.3f} s)")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_title("Mode Shapes")
    ax.set_xlabel("Normalized modal displacement")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


# =========================
# STREAMLIT UI
# =========================
st.title("Core–Outrigger Preliminary Framework")
st.write("این نسخه برای اجرا در Streamlit بازنویسی شده است و مستقیم با `streamlit run` بالا می‌آید.")

with st.sidebar:
    st.header("Input")
    n_story = st.number_input("Number of stories", min_value=2, value=60, step=1)
    story_height = st.number_input("Story height (m)", min_value=2.0, value=3.2, step=0.1)
    plan_x = st.number_input("Plan X (m)", min_value=10.0, value=80.0, step=1.0)
    plan_y = st.number_input("Plan Y (m)", min_value=10.0, value=80.0, step=1.0)
    floor_mass = st.number_input("Floor mass (kg)", min_value=1000.0, value=900000.0, step=10000.0)

    st.subheader("Core")
    core_t = st.number_input("Core thickness (m)", min_value=0.1, value=0.60, step=0.05)
    core_lx = st.number_input("Core length X (m)", min_value=1.0, value=18.0, step=1.0)
    core_ly = st.number_input("Core length Y (m)", min_value=1.0, value=18.0, step=1.0)

    st.subheader("Columns")
    per_b = st.number_input("Perimeter column b (m)", min_value=0.2, value=0.95, step=0.05)
    per_h = st.number_input("Perimeter column h (m)", min_value=0.2, value=0.95, step=0.05)
    cor_b = st.number_input("Corner column b (m)", min_value=0.2, value=1.10, step=0.05)
    cor_h = st.number_input("Corner column h (m)", min_value=0.2, value=1.10, step=0.05)

    npx = st.number_input("Columns on X face", min_value=1, value=7, step=1)
    npy = st.number_input("Columns on Y face", min_value=1, value=7, step=1)

    st.subheader("Materials")
    E_col = st.number_input("Column E (Pa)", min_value=1e9, value=32e9, step=1e9, format="%.2e")
    fy_col = st.number_input("Column fy (Pa)", min_value=1e6, value=420e6, step=1e6, format="%.2e")
    E_wall = st.number_input("Wall E (Pa)", min_value=1e9, value=32e9, step=1e9, format="%.2e")
    fy_wall = st.number_input("Wall fy/strength proxy (Pa)", min_value=1e6, value=40e6, step=1e6, format="%.2e")

    st.subheader("Lateral load")
    base_shear_ratio = st.number_input("Base shear ratio", min_value=0.001, value=0.02, step=0.001, format="%.3f")

    st.subheader("Outriggers")
    axis = st.selectbox("Analysis axis", ["x", "y"])
    n_ou = st.number_input("Number of outriggers", min_value=0, max_value=5, value=3, step=1)

    ou_list = []
    default_stories = [15, 30, 45, 20, 40]
    for i in range(int(n_ou)):
        st.markdown(f"**Outrigger {i+1}**")
        story = st.number_input(f"Story O{i+1}", min_value=1, max_value=int(n_story), value=min(default_stories[i], int(n_story)), step=1, key=f"story_{i}")
        truss_depth = st.number_input(f"Truss depth O{i+1} (m)", min_value=0.5, value=4.0, step=0.5, key=f"d_{i}")
        chord_area = st.number_input(f"Chord area O{i+1} (m²)", min_value=0.001, value=0.12, step=0.01, key=f"ca_{i}")
        diagonal_area = st.number_input(f"Diagonal area O{i+1} (m²)", min_value=0.001, value=0.06, step=0.01, key=f"da_{i}")
        ou_list.append(OutriggerLevel(story=int(story), axis=axis, truss_depth=truss_depth, chord_area=chord_area, diagonal_area=diagonal_area))

    run_button = st.button("Run Analysis", type="primary")

if run_button:
    try:
        inp = TowerInput(
            n_story=int(n_story),
            story_height=float(story_height),
            plan_x=float(plan_x),
            plan_y=float(plan_y),
            floor_mass=float(floor_mass),
            core=CoreWallSection(
                thickness=float(core_t),
                length_x=float(core_lx),
                length_y=float(core_ly),
                n_web_x=2,
                n_web_y=2,
                cracked_factor=0.45,
            ),
            perimeter_column=ColumnSection(b=float(per_b), h=float(per_h), cracked_factor=0.70),
            corner_column=ColumnSection(b=float(cor_b), h=float(cor_h), cracked_factor=0.70),
            n_perimeter_columns_x_face=int(npx),
            n_perimeter_columns_y_face=int(npy),
            outriggers=ou_list,
            column_material=Material(E=float(E_col), fy=float(fy_col)),
            wall_material=Material(E=float(E_wall), fy=float(fy_wall)),
            design_base_shear_ratio=float(base_shear_ratio),
        )

        results = analyze_tower(inp, axis=axis)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T1 (s)", f"{results['periods'][0]:.3f}")
        c2.metric("Roof disp. (mm)", f"{results['roof_disp_mm']:.2f}")
        c3.metric("Max drift ratio", f"{results['max_drift_ratio']:.5f}")
        c4.metric("Base shear (kN)", f"{results['base_shear_kN']:.1f}")

        c5, c6 = st.columns(2)
        c5.metric("Max outrigger axial (kN)", f"{results['max_outrigger_axial_kN']:.1f}")
        c6.metric("No. of outriggers", f"{len(inp.outriggers)}")

        tab1, tab2, tab3, tab4 = st.tabs(["Story results", "Outriggers", "Plan/Elevation", "Modes"])

        with tab1:
            st.dataframe(results["story_df"], use_container_width=True)

        with tab2:
            if results["outrigger_df"].empty:
                st.info("No outriggers defined.")
            else:
                st.dataframe(results["outrigger_df"], use_container_width=True)

        with tab3:
            colA, colB = st.columns(2)
            with colA:
                fig1 = plot_plan(inp, axis=axis)
                st.pyplot(fig1, clear_figure=True)
            with colB:
                fig2 = plot_elevation(inp)
                st.pyplot(fig2, clear_figure=True)

        with tab4:
            fig3 = plot_modes(results, n_modes=3)
            st.pyplot(fig3, clear_figure=True)

    except Exception as e:
        st.exception(e)
else:
    st.info("ورودی‌ها را تنظیم کن و روی Run Analysis بزن.")
