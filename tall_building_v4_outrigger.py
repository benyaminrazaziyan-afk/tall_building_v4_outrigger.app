
"""
tower_flexural_mdof_v9.py

Realistic preliminary tower-period framework
============================================

This version replaces the previous shear-building stiffness model with a
flexural cantilever MDOF model.

Why this version exists
-----------------------
For tall towers, the first period is usually governed by global flexural
cantilever behavior, not by independent storey shear springs. A shear-building
approximation using k = 12EI/h^3 at every storey can produce unrealistically
short periods. This version uses Euler-Bernoulli beam elements with two DOFs
per level:

    DOF 1: lateral displacement u
    DOF 2: rotation theta

The eigenvalue problem is:

    [K]{phi} = omega^2 [M]{phi}

The first modal period is:

    T1 = 2*pi/omega1

Engineering use
---------------
This is still a preliminary framework, but it is much closer to real tower
behavior than the previous shear-spring model.

Recommended workflow
--------------------
1. Build an ETABS model.
2. Extract/estimate effective EI along height.
3. Use this app to calibrate EI reduction factor until T1 matches ETABS.
4. Use the calibrated factor for conceptual comparisons:
   - no outrigger
   - belt truss
   - pipe bracing
   - different outrigger levels
   - section stiffness sensitivity

Author: Benyamin
Version: v9.0-flexural-mdof
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import pi, sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# CONSTANTS
# ============================================================

G = 9.81
GAMMA_RC = 25.0
RHO_STEEL = 7850.0
APP_VERSION = "v9.0-flexural-mdof"


# ============================================================
# DATA DEFINITIONS
# ============================================================

class Direction(str, Enum):
    X = "X"
    Y = "Y"


class OutriggerType(str, Enum):
    NONE = "None"
    BELT_TRUSS = "Belt Truss"
    PIPE_BRACE = "Pipe Bracing"


@dataclass
class TowerInput:
    # Geometry
    n_story: int = 60
    story_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0

    # Core
    core_x_m: float = 12.0
    core_y_m: float = 10.0
    wall_t_base_m: float = 0.55
    wall_t_top_m: float = 0.30

    # Perimeter columns
    n_perimeter_columns: int = 28
    column_dim_base_m: float = 1.20
    column_dim_top_m: float = 0.70
    column_cracked_factor: float = 0.70

    # Materials
    Ec_MPa: float = 34000.0

    # Effective stiffness calibration
    # This is important. Cracked concrete, coupling beams, foundation flexibility,
    # diaphragm effects, and modeling assumptions can reduce global stiffness.
    global_EI_factor: float = 0.20

    # Mass
    floor_mass_ton: float = 2500.0

    # Outrigger
    use_outrigger: bool = True
    outrigger_type: OutriggerType = OutriggerType.BELT_TRUSS
    outrigger_story_1: int = 30
    outrigger_story_2: int = 42
    outrigger_chord_area_m2: float = 0.08
    outrigger_diagonal_area_m2: float = 0.04
    outrigger_depth_m: float = 3.0
    outrigger_efficiency: float = 0.35

    # Optional ETABS calibration target
    etabs_target_period_s: float = 25.0

    @property
    def height_m(self) -> float:
        return self.n_story * self.story_height_m

    @property
    def E_N_m2(self) -> float:
        return self.Ec_MPa * 1e6

    @property
    def floor_mass_kg(self) -> float:
        return self.floor_mass_ton * 1000.0


# ============================================================
# SECTION PROPERTIES
# ============================================================

def interpolate_by_height(base_value: float, top_value: float, story: int, n_story: int) -> float:
    r = (story - 1) / max(n_story - 1, 1)
    return base_value + r * (top_value - base_value)


def wall_tube_inertia(core_x: float, core_y: float, t: float) -> Tuple[float, float]:
    """
    Approximate closed rectangular core inertia.

    Ix: bending about X axis, resists Y translation.
    Iy: bending about Y axis, resists X translation.

    This is a thin-wall closed-core approximation using outer minus inner rectangle.
    """
    bx = core_x
    by = core_y
    ix_outer = bx * by**3 / 12.0
    iy_outer = by * bx**3 / 12.0

    bx_i = max(core_x - 2.0 * t, 0.10)
    by_i = max(core_y - 2.0 * t, 0.10)

    ix_inner = bx_i * by_i**3 / 12.0
    iy_inner = by_i * bx_i**3 / 12.0

    Ix = max(ix_outer - ix_inner, 1e-9)
    Iy = max(iy_outer - iy_inner, 1e-9)

    return Ix, Iy


def perimeter_column_inertia(inp: TowerInput, col_dim: float) -> Tuple[float, float]:
    """
    Approximate perimeter column contribution to global bending stiffness.

    For tower overturning, perimeter column axial stiffness contributes strongly
    through plan lever arms. This is a simplified transformed bending stiffness:

        I_global = sum(A_i * r_i^2) + sum(I_local)

    This is not final design, but it is much better than ignoring column lever arms.
    """
    A = col_dim * col_dim
    I_local = col_dim**4 / 12.0

    # Distribute columns approximately along rectangular perimeter.
    n = max(inp.n_perimeter_columns, 4)
    xs = []
    ys = []

    # Four sides, approximate equal distribution.
    n_side = max(n // 4, 1)

    for i in range(n_side):
        x = -inp.plan_x_m / 2 + i * inp.plan_x_m / max(n_side - 1, 1)
        xs.append(x)
        ys.append(-inp.plan_y_m / 2)
        xs.append(x)
        ys.append(inp.plan_y_m / 2)

    for i in range(n_side):
        y = -inp.plan_y_m / 2 + i * inp.plan_y_m / max(n_side - 1, 1)
        xs.append(-inp.plan_x_m / 2)
        ys.append(y)
        xs.append(inp.plan_x_m / 2)
        ys.append(y)

    xs = np.array(xs[:n], dtype=float)
    ys = np.array(ys[:n], dtype=float)

    # Ix resists Y translation, lever arm in y.
    Ix = np.sum(I_local + A * ys**2)
    # Iy resists X translation, lever arm in x.
    Iy = np.sum(I_local + A * xs**2)

    return inp.column_cracked_factor * Ix, inp.column_cracked_factor * Iy


def effective_EI_profile(inp: TowerInput, direction: Direction) -> np.ndarray:
    """
    Return EI per storey element.

    For X translation, use bending about Y axis: EI_y.
    For Y translation, use bending about X axis: EI_x.
    """
    EI = []

    for story in range(1, inp.n_story + 1):
        t = interpolate_by_height(inp.wall_t_base_m, inp.wall_t_top_m, story, inp.n_story)
        col_dim = interpolate_by_height(inp.column_dim_base_m, inp.column_dim_top_m, story, inp.n_story)

        Ix_core, Iy_core = wall_tube_inertia(inp.core_x_m, inp.core_y_m, t)
        Ix_col, Iy_col = perimeter_column_inertia(inp, col_dim)

        if direction == Direction.X:
            I = Iy_core + Iy_col
        else:
            I = Ix_core + Ix_col

        EI.append(inp.E_N_m2 * I * inp.global_EI_factor)

    return np.array(EI, dtype=float)


# ============================================================
# OUTRIGGER MODEL
# ============================================================

def outrigger_eta(system: OutriggerType) -> float:
    if system == OutriggerType.BELT_TRUSS:
        return 1.00
    if system == OutriggerType.PIPE_BRACE:
        return 0.65
    return 0.0


def outrigger_rotational_spring(inp: TowerInput, story: int, direction: Direction) -> float:
    """
    Equivalent rotational spring at an outrigger level.

    Outrigger mechanism:
    - core rotation creates vertical displacement demand at perimeter columns
    - outrigger transfers this action through axial members
    - this restrains core rotation

    Simplified:
        Ktheta ≈ eta * EA/L * arm^2

    Units:
        EA/L = N/m
        arm^2 = m²
        Ktheta = N*m/rad
    """
    if not inp.use_outrigger:
        return 0.0

    if story not in [inp.outrigger_story_1, inp.outrigger_story_2]:
        return 0.0

    if inp.outrigger_type == OutriggerType.NONE:
        return 0.0

    E = inp.E_N_m2
    eta = outrigger_eta(inp.outrigger_type) * inp.outrigger_efficiency

    if direction == Direction.X:
        arm = max((inp.plan_x_m - inp.core_x_m) / 2.0, 1.0)
    else:
        arm = max((inp.plan_y_m - inp.core_y_m) / 2.0, 1.0)

    L_diag = sqrt(arm**2 + inp.outrigger_depth_m**2)

    k_ax_chord = E * inp.outrigger_chord_area_m2 / arm
    k_ax_diag = E * inp.outrigger_diagonal_area_m2 / L_diag

    k_ax = 2.0 * (k_ax_chord + k_ax_diag)

    Ktheta = eta * k_ax * arm**2

    return Ktheta


# ============================================================
# FINITE ELEMENT MDOF ASSEMBLY
# ============================================================

def beam_element_stiffness(EI: float, L: float) -> np.ndarray:
    """
    Euler-Bernoulli beam element stiffness matrix for lateral bending.

    DOFs:
        [u_i, theta_i, u_j, theta_j]
    """
    return EI / L**3 * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2],
        ],
        dtype=float,
    )


def assemble_flexural_mk(inp: TowerInput, direction: Direction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble global K and M for flexural cantilever tower.

    Nodes:
        0 = base fixed
        1..n = floors

    DOFs at each node:
        u, theta

    Base DOFs are removed.
    """
    n = inp.n_story
    L = inp.story_height_m
    ndof_total = 2 * (n + 1)

    K = np.zeros((ndof_total, ndof_total), dtype=float)
    M = np.zeros((ndof_total, ndof_total), dtype=float)

    EI_profile = effective_EI_profile(inp, direction)

    for e in range(n):
        EI = EI_profile[e]
        ke = beam_element_stiffness(EI, L)

        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]

        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]

    # Lump floor masses at translational DOFs.
    for node in range(1, n + 1):
        u_dof = 2 * node
        theta_dof = 2 * node + 1

        M[u_dof, u_dof] += inp.floor_mass_kg

        # Small rotational inertia regularization.
        # This avoids singular M for rotational DOFs without dominating periods.
        M[theta_dof, theta_dof] += inp.floor_mass_kg * (inp.story_height_m**2) * 1e-4

    # Add outrigger rotational springs at actual levels.
    for story in range(1, n + 1):
        Ktheta = outrigger_rotational_spring(inp, story, direction)
        if Ktheta > 0:
            theta_dof = 2 * story + 1
            K[theta_dof, theta_dof] += Ktheta

    # Remove fixed base DOFs: node 0 u and theta.
    free = list(range(2, ndof_total))
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]

    return Mff, Kff


def solve_modal(inp: TowerInput, direction: Direction, n_modes: int = 5):
    M, K = assemble_flexural_mk(inp, direction)

    # Solve generalized eigenproblem by M^-1 K.
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    keep = eigvals > 1e-8
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    n_modes = min(n_modes, len(eigvals))
    omega = np.sqrt(eigvals[:n_modes])

    periods = 2 * pi / omega
    freqs = omega / (2 * pi)

    # Extract floor displacement DOFs from free DOF vector.
    # Free vector order: node1 u, node1 theta, node2 u, node2 theta, ...
    shapes = []
    mass_ratios = []
    cumulative = []

    total_mass = inp.floor_mass_kg * inp.n_story
    ones = np.zeros((2 * inp.n_story, 1))
    ones[0::2, 0] = 1.0

    cum = 0.0

    for i in range(n_modes):
        phi = eigvecs[:, i].reshape(-1, 1)

        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ ones) / denom).item()
        meff = gamma**2 * denom
        ratio = meff / total_mass
        cum += ratio

        floor_disp = phi.flatten()[0::2]
        if abs(floor_disp[-1]) > 1e-12:
            floor_disp = floor_disp / floor_disp[-1]
        if floor_disp[-1] < 0:
            floor_disp *= -1

        shapes.append(floor_disp)
        mass_ratios.append(ratio)
        cumulative.append(cum)

    return {
        "direction": direction.value,
        "periods": periods,
        "frequencies": freqs,
        "shapes": shapes,
        "mass_ratios": mass_ratios,
        "cumulative": cumulative,
    }


# ============================================================
# CALIBRATION
# ============================================================

def required_EI_factor_for_target(inp: TowerInput, target_T: float, direction: Direction) -> float:
    """
    Since T roughly scales as 1/sqrt(EI), the required EI factor is:

        factor_new = factor_old * (T_current / T_target)^2
    """
    modal = solve_modal(inp, direction, n_modes=1)
    T_current = float(modal["periods"][0])
    return inp.global_EI_factor * (T_current / target_T) ** 2


def story_property_table(inp: TowerInput) -> pd.DataFrame:
    rows = []
    EI_x = effective_EI_profile(inp, Direction.X)
    EI_y = effective_EI_profile(inp, Direction.Y)

    for story in range(1, inp.n_story + 1):
        t = interpolate_by_height(inp.wall_t_base_m, inp.wall_t_top_m, story, inp.n_story)
        col = interpolate_by_height(inp.column_dim_base_m, inp.column_dim_top_m, story, inp.n_story)
        ktheta_x = outrigger_rotational_spring(inp, story, Direction.X)
        ktheta_y = outrigger_rotational_spring(inp, story, Direction.Y)

        rows.append(
            {
                "Story": story,
                "Elevation (m)": story * inp.story_height_m,
                "Wall t (m)": t,
                "Column dim (m)": col,
                "EI for X translation (GN.m²)": EI_x[story - 1] / 1e9,
                "EI for Y translation (GN.m²)": EI_y[story - 1] / 1e9,
                "Outrigger Ktheta X (GN.m/rad)": ktheta_x / 1e9,
                "Outrigger Ktheta Y (GN.m/rad)": ktheta_y / 1e9,
            }
        )

    return pd.DataFrame(rows)


def modal_table(modal: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mode": list(range(1, len(modal["periods"]) + 1)),
            "Direction": modal["direction"],
            "Period (s)": modal["periods"],
            "Frequency (Hz)": modal["frequencies"],
            "Effective mass (%)": [100 * x for x in modal["mass_ratios"]],
            "Cumulative mass (%)": [100 * x for x in modal["cumulative"]],
        }
    )


# ============================================================
# PLOTS
# ============================================================

def plot_modes(inp: TowerInput, modal: dict):
    y = np.arange(1, inp.n_story + 1) * inp.story_height_m
    n_modes = len(modal["shapes"])

    fig, axes = plt.subplots(1, n_modes, figsize=(16, 6))
    if n_modes == 1:
        axes = [axes]

    for i in range(n_modes):
        ax = axes[i]
        phi = modal["shapes"][i]
        ax.plot(phi, y, linewidth=2)
        ax.scatter(phi, y, s=12)
        ax.axvline(0, linestyle="--", linewidth=0.8)

        if inp.use_outrigger:
            for st in [inp.outrigger_story_1, inp.outrigger_story_2]:
                if 1 <= st <= inp.n_story:
                    ax.axhline(st * inp.story_height_m, linestyle=":", alpha=0.6)

        ax.set_title(f"Mode {i+1}\nT={modal['periods'][i]:.2f}s")
        ax.set_xlabel("Normalized u")
        if i == 0:
            ax.set_ylabel("Height (m)")
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Flexural MDOF mode shapes - Direction {modal['direction']}")
    fig.tight_layout()
    return fig


def plot_EI(inp: TowerInput):
    df = story_property_table(inp)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(df["EI for X translation (GN.m²)"], df["Story"], label="EI for X translation")
    ax.plot(df["EI for Y translation (GN.m²)"], df["Story"], label="EI for Y translation")

    ax.set_xlabel("Effective EI (GN.m²)")
    ax.set_ylabel("Story")
    ax.set_title("Effective flexural stiffness profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_plan(inp: TowerInput):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [-inp.plan_x_m/2, inp.plan_x_m/2, inp.plan_x_m/2, -inp.plan_x_m/2, -inp.plan_x_m/2],
        [-inp.plan_y_m/2, -inp.plan_y_m/2, inp.plan_y_m/2, inp.plan_y_m/2, -inp.plan_y_m/2],
        color="black",
    )

    # Core
    ax.add_patch(
        plt.Rectangle(
            (-inp.core_x_m/2, -inp.core_y_m/2),
            inp.core_x_m,
            inp.core_y_m,
            fill=False,
            edgecolor="green",
            linewidth=2.5,
        )
    )

    # Approx perimeter columns
    n = max(inp.n_perimeter_columns, 4)
    n_side = max(n // 4, 1)
    pts = []

    for i in range(n_side):
        x = -inp.plan_x_m/2 + i * inp.plan_x_m / max(n_side - 1, 1)
        pts.append((x, -inp.plan_y_m/2))
        pts.append((x, inp.plan_y_m/2))

    for i in range(n_side):
        y = -inp.plan_y_m/2 + i * inp.plan_y_m / max(n_side - 1, 1)
        pts.append((-inp.plan_x_m/2, y))
        pts.append((inp.plan_x_m/2, y))

    pts = pts[:n]

    col_dim = inp.column_dim_base_m
    for x, y in pts:
        ax.add_patch(
            plt.Rectangle(
                (x-col_dim/2, y-col_dim/2),
                col_dim,
                col_dim,
                facecolor="darkred",
                alpha=0.85,
            )
        )

    if inp.use_outrigger and inp.outrigger_type != OutriggerType.NONE:
        ax.plot([-inp.plan_x_m/2, -inp.core_x_m/2], [0, 0], color="orange", linewidth=4)
        ax.plot([inp.core_x_m/2, inp.plan_x_m/2], [0, 0], color="orange", linewidth=4)
        ax.plot([0, 0], [-inp.plan_y_m/2, -inp.core_y_m/2], color="orange", linewidth=4)
        ax.plot([0, 0], [inp.core_y_m/2, inp.plan_y_m/2], color="orange", linewidth=4)

    ax.set_title("Simplified structural plan")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    return fig


# ============================================================
# REPORT
# ============================================================

def build_report(inp: TowerInput, modal_x: dict, modal_y: dict) -> str:
    T_x = float(modal_x["periods"][0])
    T_y = float(modal_y["periods"][0])
    gov = "X" if T_x >= T_y else "Y"
    T_gov = max(T_x, T_y)

    req_factor_x = required_EI_factor_for_target(inp, inp.etabs_target_period_s, Direction.X)
    req_factor_y = required_EI_factor_for_target(inp, inp.etabs_target_period_s, Direction.Y)

    lines = []
    lines.append("=" * 90)
    lines.append("FLEXURAL MDOF TOWER PRE-DESIGN REPORT")
    lines.append("=" * 90)
    lines.append(f"Version: {APP_VERSION}")
    lines.append("")
    lines.append("1. MODEL TYPE")
    lines.append("-" * 90)
    lines.append("The tower is modeled as a flexural cantilever using Euler-Bernoulli beam elements.")
    lines.append("Each floor node has two DOFs: lateral displacement and rotation.")
    lines.append("The eigenvalue equation is [K]{phi} = omega²[M]{phi}.")
    lines.append("")
    lines.append("2. PERIOD RESULTS")
    lines.append("-" * 90)
    lines.append(f"T1 X direction = {T_x:.3f} s")
    lines.append(f"T1 Y direction = {T_y:.3f} s")
    lines.append(f"Governing direction = {gov}")
    lines.append(f"Governing T1 = {T_gov:.3f} s")
    lines.append(f"ETABS target entered = {inp.etabs_target_period_s:.3f} s")
    lines.append("")
    lines.append("3. CALIBRATION")
    lines.append("-" * 90)
    lines.append(f"Current global EI factor = {inp.global_EI_factor:.5f}")
    lines.append(f"Required EI factor to match ETABS in X = {req_factor_x:.5f}")
    lines.append(f"Required EI factor to match ETABS in Y = {req_factor_y:.5f}")
    lines.append("")
    lines.append("4. ENGINEERING JUDGMENT")
    lines.append("-" * 90)
    lines.append(
        "If ETABS gives a period close to 25 s, the model must be dominated by global "
        "cantilever flexibility, cracked stiffness reduction, coupling-beam flexibility, "
        "foundation flexibility, or lower effective lateral stiffness than a simple gross-section model."
    )
    lines.append(
        "The previous shear-building spring model was not appropriate because it transformed "
        "each storey EI into an unrealistically large storey spring. This flexural model is "
        "the correct direction for a defensible preliminary tower framework."
    )
    lines.append("")
    lines.append("5. LIMITATION")
    lines.append("-" * 90)
    lines.append(
        "This model is still preliminary. It must be calibrated against ETABS or another "
        "3D model before being used for thesis-level parametric conclusions."
    )

    return "\n".join(lines)


# ============================================================
# STREAMLIT APP
# ============================================================

def read_sidebar() -> TowerInput:
    st.sidebar.header("1. Tower geometry")

    n_story = st.sidebar.number_input("Stories", 10, 120, 60)
    story_height = st.sidebar.number_input("Story height (m)", 2.8, 5.0, 3.2)
    plan_x = st.sidebar.number_input("Plan X (m)", 20.0, 200.0, 48.0)
    plan_y = st.sidebar.number_input("Plan Y (m)", 20.0, 200.0, 42.0)

    st.sidebar.header("2. Core")
    core_x = st.sidebar.number_input("Core X (m)", 4.0, 80.0, 12.0)
    core_y = st.sidebar.number_input("Core Y (m)", 4.0, 80.0, 10.0)
    wall_base = st.sidebar.number_input("Wall thickness base (m)", 0.20, 2.00, 0.55)
    wall_top = st.sidebar.number_input("Wall thickness top (m)", 0.15, 1.50, 0.30)

    st.sidebar.header("3. Columns")
    n_cols = st.sidebar.number_input("Perimeter column count", 4, 120, 28)
    col_base = st.sidebar.number_input("Column dim base (m)", 0.30, 3.00, 1.20)
    col_top = st.sidebar.number_input("Column dim top (m)", 0.20, 2.00, 0.70)
    col_crack = st.sidebar.number_input("Column cracked factor", 0.05, 1.00, 0.70)

    st.sidebar.header("4. Material and calibration")
    Ec = st.sidebar.number_input("Ec (MPa)", 20000.0, 60000.0, 34000.0)
    EI_factor = st.sidebar.number_input("Global EI calibration factor", 0.001, 1.000, 0.020, format="%.4f")
    floor_mass = st.sidebar.number_input("Floor mass (ton)", 100.0, 20000.0, 2500.0)

    st.sidebar.header("5. Outrigger")
    use_out = st.sidebar.checkbox("Use outrigger", value=True)
    out_type_name = st.sidebar.selectbox(
        "Outrigger type",
        [OutriggerType.BELT_TRUSS.value, OutriggerType.PIPE_BRACE.value, OutriggerType.NONE.value],
        index=0,
    )
    out1 = st.sidebar.number_input("Outrigger story 1", 1, int(n_story), int(round(0.50 * n_story)))
    out2 = st.sidebar.number_input("Outrigger story 2", 1, int(n_story), int(round(0.70 * n_story)))
    out_eff = st.sidebar.number_input("Outrigger efficiency", 0.01, 1.00, 0.35)
    A_chord = st.sidebar.number_input("Chord area (m²)", 0.001, 1.000, 0.08, format="%.4f")
    A_diag = st.sidebar.number_input("Diagonal area (m²)", 0.001, 1.000, 0.04, format="%.4f")
    out_depth = st.sidebar.number_input("Outrigger depth (m)", 1.0, 10.0, 3.0)

    st.sidebar.header("6. ETABS calibration")
    etabs_T = st.sidebar.number_input("ETABS target T1 (s)", 1.0, 80.0, 25.0)

    return TowerInput(
        n_story=int(n_story),
        story_height_m=float(story_height),
        plan_x_m=float(plan_x),
        plan_y_m=float(plan_y),
        core_x_m=float(core_x),
        core_y_m=float(core_y),
        wall_t_base_m=float(wall_base),
        wall_t_top_m=float(wall_top),
        n_perimeter_columns=int(n_cols),
        column_dim_base_m=float(col_base),
        column_dim_top_m=float(col_top),
        column_cracked_factor=float(col_crack),
        Ec_MPa=float(Ec),
        global_EI_factor=float(EI_factor),
        floor_mass_ton=float(floor_mass),
        use_outrigger=bool(use_out),
        outrigger_type=OutriggerType(out_type_name),
        outrigger_story_1=int(out1),
        outrigger_story_2=int(out2),
        outrigger_chord_area_m2=float(A_chord),
        outrigger_diagonal_area_m2=float(A_diag),
        outrigger_depth_m=float(out_depth),
        outrigger_efficiency=float(out_eff),
        etabs_target_period_s=float(etabs_T),
    )


def main():
    st.set_page_config(page_title="Flexural MDOF Tower v9", layout="wide")

    st.title("Flexural MDOF Tower Pre-Design Framework")
    st.caption(APP_VERSION)

    st.warning(
        "This version replaces the unrealistic shear-building spring model with a flexural cantilever MDOF model. "
        "It is intended to calibrate against ETABS periods such as 25 s."
    )

    inp = read_sidebar()

    modal_x = solve_modal(inp, Direction.X, n_modes=5)
    modal_y = solve_modal(inp, Direction.Y, n_modes=5)

    T_x = float(modal_x["periods"][0])
    T_y = float(modal_y["periods"][0])
    T_gov = max(T_x, T_y)
    gov_dir = "X" if T_x >= T_y else "Y"

    req_factor_x = required_EI_factor_for_target(inp, inp.etabs_target_period_s, Direction.X)
    req_factor_y = required_EI_factor_for_target(inp, inp.etabs_target_period_s, Direction.Y)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T1 X (s)", f"{T_x:.2f}")
    c2.metric("T1 Y (s)", f"{T_y:.2f}")
    c3.metric("Governing T1 (s)", f"{T_gov:.2f}")
    c4.metric("Gov. direction", gov_dir)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Current EI factor", f"{inp.global_EI_factor:.4f}")
    c6.metric("Required factor X", f"{req_factor_x:.4f}")
    c7.metric("Required factor Y", f"{req_factor_y:.4f}")
    c8.metric("ETABS target (s)", f"{inp.etabs_target_period_s:.2f}")

    st.info(
        "Calibration rule: if the app period is shorter than ETABS, reduce the Global EI calibration factor. "
        "The required factors above estimate what factor would match the ETABS target."
    )

    tabs = st.tabs(
        [
            "Mode shapes X",
            "Mode shapes Y",
            "EI profile",
            "Plan",
            "Modal tables",
            "Story properties",
            "Report",
        ]
    )

    with tabs[0]:
        st.pyplot(plot_modes(inp, modal_x), use_container_width=True)

    with tabs[1]:
        st.pyplot(plot_modes(inp, modal_y), use_container_width=True)

    with tabs[2]:
        st.pyplot(plot_EI(inp), use_container_width=True)

    with tabs[3]:
        st.pyplot(plot_plan(inp), use_container_width=True)

    with tabs[4]:
        st.markdown("### X Direction")
        st.dataframe(modal_table(modal_x), use_container_width=True, hide_index=True)
        st.markdown("### Y Direction")
        st.dataframe(modal_table(modal_y), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.dataframe(story_property_table(inp), use_container_width=True, hide_index=True)

    with tabs[6]:
        report = build_report(inp, modal_x, modal_y)
        st.text_area("Report", report, height=520)
        st.download_button(
            "Download report",
            data=report.encode("utf-8"),
            file_name="flexural_mdof_tower_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
