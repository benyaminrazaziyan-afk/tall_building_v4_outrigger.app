from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import pi, sqrt
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:
    class _StreamlitStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop
    st = _StreamlitStub()

try:
    from scipy.linalg import eigh
except Exception:
    eigh = None


# ===== Embedded MDOF Framework =====

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.linalg import eigh
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for generalized eigenvalue solution") from exc


# ---------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------

@dataclass
class Material:
    E: float                 # Pa
    fy: float                # Pa
    density: float = 7850.0  # kg/m3


@dataclass
class CoreWallSection:
    thickness: float         # m
    length_x: float          # m
    length_y: float          # m
    n_web_x: int = 2
    n_web_y: int = 2
    cracked_factor: float = 0.45

    @property
    def area(self) -> float:
        return self.n_web_x * self.length_x * self.thickness + self.n_web_y * self.length_y * self.thickness

    def inertia_y(self) -> float:
        """
        Equivalent in-plane bending inertia for X-direction sway
        using walls parallel to Y plus parallel-axis contribution of X walls.
        This is a simplified closed-form estimate.
        """
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        # walls normal to X contribute mainly by own strong axis
        I_from_ywalls = self.n_web_y * (t * ly**3 / 12.0)
        # walls parallel to X contribute with some lever arm; use half-core spacing
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
    b: float                 # m
    h: float                 # m
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
    axis: str = "x"                 # "x" or "y"
    truss_depth: float = 3.0        # m
    chord_area: float = 0.08        # m2
    diagonal_area: float = 0.04     # m2
    truss_E: float = 200e9          # Pa

    def validate(self) -> None:
        if self.axis.lower() not in {"x", "y"}:
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")
        if self.story < 1:
            raise ValueError("story must be >= 1")


@dataclass
class TowerInput:
    n_story: int
    story_height: float
    plan_x: float
    plan_y: float
    floor_mass: float                # kg per typical story
    core: CoreWallSection
    perimeter_column: ColumnSection
    corner_column: ColumnSection
    n_perimeter_columns_x_face: int
    n_perimeter_columns_y_face: int
    outriggers: List[OutriggerLevel] = field(default_factory=list)
    concrete_E: float = 32e9
    drift_limit_ratio: float = 1 / 500.0
    column_material: Material = field(default_factory=lambda: Material(E=32e9, fy=420e6))
    wall_material: Material = field(default_factory=lambda: Material(E=32e9, fy=40e6))
    use_same_floor_mass_all_levels: bool = True

    def height(self) -> float:
        return self.n_story * self.story_height

    def bay_arm(self, axis: str) -> float:
        """
        Lever arm from core face to perimeter column line.
        """
        axis = axis.lower()
        if axis == "x":
            return max((self.plan_x - self.core.length_x) / 2.0, 0.5)
        return max((self.plan_y - self.core.length_y) / 2.0, 0.5)

    def n_engaged_columns_per_side(self, axis: str) -> int:
        axis = axis.lower()
        return self.n_perimeter_columns_y_face if axis == "x" else self.n_perimeter_columns_x_face

    def validate(self) -> None:
        if self.n_story < 2:
            raise ValueError("n_story must be >= 2")
        if self.story_height <= 0:
            raise ValueError("story_height must be > 0")
        if self.floor_mass <= 0:
            raise ValueError("floor_mass must be > 0")
        for ou in self.outriggers:
            ou.validate()
            if ou.story > self.n_story:
                raise ValueError(f"outrigger story {ou.story} > n_story {self.n_story}")


@dataclass
class OutriggerMechanics:
    story: int
    axis: str
    arm: float
    k_truss_tip: float            # N/m
    k_column_tip: float           # N/m
    k_tip_combined: float         # N/m
    k_story_equiv: float          # N/m
    k_rot: float                  # N*m/rad
    axial_force_per_side: float   # N for unit interstory drift proxy


@dataclass
class AnalysisResult:
    periods: np.ndarray
    frequencies: np.ndarray
    mode_shapes: np.ndarray
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    story_stiffness_base: np.ndarray
    story_stiffness_total: np.ndarray
    outriggers: List[OutriggerMechanics]
    top_displacement_under_unit_profile: np.ndarray
    outrigger_axial_by_level: dict[int, float]
    max_drift_ratio_unit_profile: float
    input_model: TowerInput


@dataclass
class DesignIterationResult:
    iteration: int
    period_1: float
    perimeter_col_b: float
    perimeter_col_h: float
    corner_col_b: float
    corner_col_h: float
    core_thickness: float
    max_outrigger_axial: float
    max_drift_unit_profile: float


# ---------------------------------------------------------------------
# BASIC STRUCTURAL FUNCTIONS
# ---------------------------------------------------------------------

def _ensure_vector_mass(inp: TowerInput) -> np.ndarray:
    return np.full(inp.n_story, inp.floor_mass, dtype=float)


def _story_stiffness_from_core(inp: TowerInput, axis: str) -> np.ndarray:
    """
    Approximate each story lateral stiffness contribution from the core.
    k_story ≈ 12 E I / h^3
    """
    axis = axis.lower()
    I = inp.core.inertia_y() if axis == "x" else inp.core.inertia_x()
    E = inp.wall_material.E
    k_story = 12.0 * E * I / inp.story_height**3
    return np.full(inp.n_story, k_story, dtype=float)


def _story_stiffness_from_columns(inp: TowerInput, axis: str) -> np.ndarray:
    """
    Approximate each story lateral stiffness contribution from all perimeter/corner columns.
    """
    axis = axis.lower()
    E = inp.column_material.E

    if axis == "x":
        I_per = inp.perimeter_column.inertia_y()
        I_cor = inp.corner_column.inertia_y()
    else:
        I_per = inp.perimeter_column.inertia_x()
        I_cor = inp.corner_column.inertia_x()

    n_per_cols = 2 * inp.n_engaged_columns_per_side(axis)
    n_corner = 4
    k_per = n_per_cols * 12.0 * E * I_per / inp.story_height**3
    k_cor = n_corner * 12.0 * E * I_cor / inp.story_height**3
    return np.full(inp.n_story, k_per + k_cor, dtype=float)


def _assemble_shear_building_K(story_k: np.ndarray) -> np.ndarray:
    """
    Standard n-story shear-building stiffness matrix from story stiffnesses.
    """
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


def _mode_participation_factor(masses: np.ndarray, phi: np.ndarray) -> float:
    ones = np.ones_like(phi)
    num = phi.T @ np.diag(masses) @ ones
    den = phi.T @ np.diag(masses) @ phi
    return float(num / den)


def _equivalent_static_profile_from_mode(masses: np.ndarray, phi: np.ndarray) -> np.ndarray:
    gamma = _mode_participation_factor(masses, phi)
    return gamma * masses * phi


# ---------------------------------------------------------------------
# OUTRIGGER MECHANICS
# ---------------------------------------------------------------------

def _engaged_column_axial_stiffness(inp: TowerInput, story: int, axis: str) -> float:
    """
    Axial stiffness of engaged perimeter columns from foundation to the outrigger level.
    Uses k = sum(EA/L) for one side. Two sides participate through the outrigger arm.
    """
    axis = axis.lower()
    E = inp.column_material.E
    L = max(story * inp.story_height, inp.story_height)
    n_side = inp.n_engaged_columns_per_side(axis)

    # Take face columns as perimeter columns; corners are counted separately at a reduced share.
    A_per = inp.perimeter_column.area
    A_cor = inp.corner_column.area

    # One side columns plus two corners at 50% participation each.
    A_side_total = n_side * A_per + 1.0 * A_cor
    return E * A_side_total / L


def _outrigger_truss_tip_stiffness(arm: float, truss_depth: float, chord_area: float, diagonal_area: float, truss_E: float) -> float:
    """
    Vertical tip stiffness of one outrigger arm from:
      - chord couple bending surrogate
      - diagonal racking surrogate
    This is simplified but physically meaningful and dimensionally consistent.
    """
    d = max(truss_depth, 0.25)
    b = max(arm, 0.25)
    Ld = sqrt(b**2 + d**2)

    # Chord-couple equivalent flexural inertia
    Ieq = 0.5 * chord_area * d**2
    k_bend = 3.0 * truss_E * Ieq / b**3

    # Diagonal contribution
    k_diag = 2.0 * truss_E * diagonal_area * d**2 / max(Ld**3, 1e-12)

    return k_bend + k_diag


def outrigger_mechanics(inp: TowerInput, ou: OutriggerLevel) -> OutriggerMechanics:
    axis = ou.axis.lower()
    arm = inp.bay_arm(axis)
    k_truss_tip = _outrigger_truss_tip_stiffness(
        arm=arm,
        truss_depth=ou.truss_depth,
        chord_area=ou.chord_area,
        diagonal_area=ou.diagonal_area,
        truss_E=ou.truss_E,
    )
    k_col_tip = _engaged_column_axial_stiffness(inp, ou.story, axis)

    # series combination: truss tip flexibility + column axial flexibility
    k_tip = 1.0 / (1.0 / max(k_truss_tip, 1e-12) + 1.0 / max(k_col_tip, 1e-12))

    # convert to rotational restraint at core:
    # K_theta = 2 * k_tip * arm^2  (two symmetric sides)
    k_rot = 2.0 * k_tip * arm**2

    # convert rotational restraint into equivalent translational floor stiffness contribution
    # u_j ≈ theta * z_j  => F/u ≈ K_theta / z_j^2
    z = ou.story * inp.story_height
    k_story_equiv = k_rot / max(z**2, 1e-12)

    # For reporting only: representative axial force per side for unit story displacement proxy
    # theta_proxy = 1 / z, delta_tip = arm / z
    axial_force_per_side = k_tip * arm / max(z, 1e-12)

    return OutriggerMechanics(
        story=ou.story,
        axis=axis,
        arm=arm,
        k_truss_tip=k_truss_tip,
        k_column_tip=k_col_tip,
        k_tip_combined=k_tip,
        k_story_equiv=k_story_equiv,
        k_rot=k_rot,
        axial_force_per_side=axial_force_per_side,
    )


# ---------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------

def analyze_tower(inp: TowerInput, axis: str = "x") -> AnalysisResult:
    inp.validate()
    axis = axis.lower()

    m = _ensure_vector_mass(inp)
    k_core = _story_stiffness_from_core(inp, axis)
    k_cols = _story_stiffness_from_columns(inp, axis)
    k_total = k_core + k_cols

    ou_mech: List[OutriggerMechanics] = []
    for ou in inp.outriggers:
        if ou.axis.lower() == axis:
            mech = outrigger_mechanics(inp, ou)
            ou_mech.append(mech)
            k_total[ou.story - 1] += mech.k_story_equiv

    K = _assemble_shear_building_K(k_total)
    M = _assemble_diagonal_mass(m)

    w2, phi = eigh(K, M)
    positive = w2 > 1e-9
    w2 = w2[positive]
    phi = phi[:, positive]
    w = np.sqrt(w2)
    periods = 2.0 * pi / w
    freqs = w / (2.0 * pi)

    # normalize each mode to roof displacement = +1
    phi_n = phi.copy()
    for i in range(phi_n.shape[1]):
        scale = phi_n[-1, i] if abs(phi_n[-1, i]) > 1e-12 else np.max(np.abs(phi_n[:, i]))
        phi_n[:, i] = phi_n[:, i] / scale
        if phi_n[-1, i] < 0:
            phi_n[:, i] *= -1.0

    # unit lateral load profile increasing with height
    heights = np.arange(1, inp.n_story + 1) * inp.story_height
    f_unit = heights / heights.sum()
    u = np.linalg.solve(K, f_unit)

    drift = np.diff(np.hstack([[0.0], u])) / inp.story_height
    max_drift = float(np.max(np.abs(drift)))

    ou_axial = {}
    for mech in ou_mech:
        j = mech.story - 1
        theta_proxy = u[j] / max(mech.story * inp.story_height, 1e-12)
        delta_tip = theta_proxy * mech.arm
        ou_axial[j + 1] = mech.k_tip_combined * delta_tip

    return AnalysisResult(
        periods=periods,
        frequencies=freqs,
        mode_shapes=phi_n,
        mass_matrix=M,
        stiffness_matrix=K,
        story_stiffness_base=k_core + k_cols,
        story_stiffness_total=k_total,
        outriggers=ou_mech,
        top_displacement_under_unit_profile=u,
        outrigger_axial_by_level=ou_axial,
        max_drift_ratio_unit_profile=max_drift,
        input_model=inp,
    )


# ---------------------------------------------------------------------
# SECTION UPDATE / RE-DESIGN
# ---------------------------------------------------------------------

def _update_perimeter_columns(inp: TowerInput, result: AnalysisResult, min_dim: float = 0.70, max_dim: float = 2.00) -> ColumnSection:
    """
    Increase perimeter column section according to max outrigger axial force.
    """
    Nmax = max(result.outrigger_axial_by_level.values(), default=0.0)
    if Nmax <= 0:
        return inp.perimeter_column

    fy = inp.column_material.fy
    demand_area = 1.20 * Nmax / max(0.35 * fy, 1e-12)
    old_a = inp.perimeter_column.area
    target_a = max(old_a, demand_area)

    side = min(max_dim, max(min_dim, sqrt(target_a)))
    return ColumnSection(b=side, h=side, cracked_factor=inp.perimeter_column.cracked_factor)


def _update_corner_columns(inp: TowerInput, perimeter_col: ColumnSection, min_dim: float = 0.80, max_dim: float = 2.20) -> ColumnSection:
    target_a = max(inp.corner_column.area, 1.25 * perimeter_col.area)
    side = min(max_dim, max(min_dim, sqrt(target_a)))
    return ColumnSection(b=side, h=side, cracked_factor=inp.corner_column.cracked_factor)


def _update_core(inp: TowerInput, result: AnalysisResult, target_period_reduction_ratio: float = 0.96) -> CoreWallSection:
    """
    If the outrigger is active and the first-mode period is still too large,
    thicken the core moderately. This is a design-feedback approximation.
    """
    if len(result.periods) == 0:
        return inp.core

    T1 = float(result.periods[0])

    # reference without outrigger using same sections
    bare = replace(inp, outriggers=[])
    bare_result = analyze_tower(bare, axis="x")
    Tbare = float(bare_result.periods[0])

    if T1 <= target_period_reduction_ratio * Tbare:
        return inp.core

    ratio = min(max(T1 / max(target_period_reduction_ratio * Tbare, 1e-12), 1.0), 1.15)
    new_t = min(inp.core.thickness * ratio, 1.50)
    return replace(inp.core, thickness=new_t)


def redesign_with_outriggers(inp: TowerInput, axis: str = "x", max_iter: int = 6, verbose: bool = False) -> tuple[TowerInput, AnalysisResult, pd.DataFrame]:
    """
    Iterative loop:
    1) Analyze with current sections.
    2) Read outrigger axial demand.
    3) Update perimeter/corner columns.
    4) Update core if needed.
    5) Re-analyze until converged.
    """
    model = replace(inp)
    hist: list[DesignIterationResult] = []

    for it in range(1, max_iter + 1):
        res = analyze_tower(model, axis=axis)

        new_per = _update_perimeter_columns(model, res)
        new_cor = _update_corner_columns(model, new_per)
        tmp_model = replace(model, perimeter_column=new_per, corner_column=new_cor)
        res_tmp = analyze_tower(tmp_model, axis=axis)
        new_core = _update_core(tmp_model, res_tmp)
        new_model = replace(tmp_model, core=new_core)

        max_ax = max(res.outrigger_axial_by_level.values(), default=0.0)
        hist.append(
            DesignIterationResult(
                iteration=it,
                period_1=float(res.periods[0]),
                perimeter_col_b=model.perimeter_column.b,
                perimeter_col_h=model.perimeter_column.h,
                corner_col_b=model.corner_column.b,
                corner_col_h=model.corner_column.h,
                core_thickness=model.core.thickness,
                max_outrigger_axial=max_ax,
                max_drift_unit_profile=res.max_drift_ratio_unit_profile,
            )
        )

        if verbose:
            print(
                f"it={it:02d}, T1={res.periods[0]:.4f} s, "
                f"PerCol={model.perimeter_column.b:.3f}x{model.perimeter_column.h:.3f} m, "
                f"Core t={model.core.thickness:.3f} m, "
                f"N_ou,max={max_ax/1e3:.1f} kN"
            )

        # convergence
        diffs = [
            abs(new_model.perimeter_column.area - model.perimeter_column.area) / max(model.perimeter_column.area, 1e-12),
            abs(new_model.corner_column.area - model.corner_column.area) / max(model.corner_column.area, 1e-12),
            abs(new_model.core.thickness - model.core.thickness) / max(model.core.thickness, 1e-12),
        ]
        model = new_model
        if max(diffs) < 0.01:
            break

    final = analyze_tower(model, axis=axis)
    hist_df = pd.DataFrame([x.__dict__ for x in hist])
    return model, final, hist_df


# ---------------------------------------------------------------------
# STUDIES / COMPARISON
# ---------------------------------------------------------------------

def root_outrigger_study(
    base_inp: TowerInput,
    candidate_stories: Sequence[int],
    counts: Sequence[int] = (0, 1, 2, 3),
    axis: str = "x",
    redesign: bool = True,
) -> pd.DataFrame:
    """
    Compare 0..n outriggers using first candidate stories.
    """
    rows = []
    template_ou = OutriggerLevel(story=1, axis=axis)

    for c in counts:
        levels = []
        for s in list(candidate_stories)[:c]:
            levels.append(replace(template_ou, story=int(s), axis=axis))
        inp = replace(base_inp, outriggers=levels)

        if redesign:
            model, res, _ = redesign_with_outriggers(inp, axis=axis, max_iter=6, verbose=False)
        else:
            model = inp
            res = analyze_tower(inp, axis=axis)

        rows.append(
            {
                "n_outriggers": c,
                "stories": [ou.story for ou in model.outriggers],
                "T1_s": float(res.periods[0]),
                "T2_s": float(res.periods[1]) if len(res.periods) > 1 else np.nan,
                "perimeter_col_m": f"{model.perimeter_column.b:.3f} x {model.perimeter_column.h:.3f}",
                "corner_col_m": f"{model.corner_column.b:.3f} x {model.corner_column.h:.3f}",
                "core_t_m": model.core.thickness,
                "max_outrigger_axial_kN": max(res.outrigger_axial_by_level.values(), default=0.0) / 1e3,
                "max_drift_ratio_unit": res.max_drift_ratio_unit_profile,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_plan(inp: TowerInput, axis: str = "x", figsize: tuple[float, float] = (7, 7)):
    fig, ax = plt.subplots(figsize=figsize)

    # building perimeter
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], lw=2)

    # core centered
    cx0 = 0.5 * (inp.plan_x - inp.core.length_x)
    cy0 = 0.5 * (inp.plan_y - inp.core.length_y)
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core.length_x, inp.core.length_y, fill=False, lw=2))

    axis = axis.lower()
    if axis == "x":
        xL = 0.0
        xR = inp.plan_x
        ymid = inp.plan_y / 2.0
        ax.plot([cx0, xL], [ymid, ymid], lw=3)
        ax.plot([cx0 + inp.core.length_x, xR], [ymid, ymid], lw=3)
        ax.text(inp.plan_x / 2.0, ymid + 0.8, "Outrigger axis = X", ha="center")
    else:
        yB = 0.0
        yT = inp.plan_y
        xmid = inp.plan_x / 2.0
        ax.plot([xmid, xmid], [cy0, yB], lw=3)
        ax.plot([xmid, xmid], [cy0 + inp.core.length_y, yT], lw=3)
        ax.text(xmid + 0.8, inp.plan_y / 2.0, "Outrigger axis = Y", rotation=90, va="center")

    ax.set_aspect("equal")
    ax.set_title("Plan sketch: core and outrigger arm direction")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_elevation(inp: TowerInput, figsize: tuple[float, float] = (6, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    H = inp.height()
    x0, x1 = 0.0, inp.plan_x * 0.12

    # tower outline
    ax.plot([x0, x0], [0, H], lw=2)
    ax.plot([x1, x1], [0, H], lw=2)
    ax.plot([x0, x1], [0, 0], lw=2)
    ax.plot([x0, x1], [H, H], lw=2)

    # story lines
    for i in range(inp.n_story + 1):
        z = i * inp.story_height
        ax.plot([x0, x1], [z, z], lw=0.4, alpha=0.4)

    # outriggers
    for ou in inp.outriggers:
        z = ou.story * inp.story_height
        ax.plot([x0 - 0.15 * x1, x1 + 0.15 * x1], [z, z], lw=2.5)
        ax.text(x1 + 0.18 * x1, z, f"O @ {ou.story}", va="center")

    ax.set_title("Elevation sketch: outrigger levels")
    ax.set_xlabel("schematic width")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_modes(result: AnalysisResult, n_modes: int = 3, scale: float = 1.0):
    n_modes = min(n_modes, result.mode_shapes.shape[1])
    z = np.arange(1, result.input_model.n_story + 1) * result.input_model.story_height
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in range(n_modes):
        ax.plot(scale * result.mode_shapes[:, i], z, label=f"Mode {i+1} (T={result.periods[i]:.3f}s)")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Normalized modal displacement")
    ax.set_ylabel("Height (m)")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_design_history(hist_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_df["iteration"], hist_df["period_1"], marker="o", label="T1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Period T1 (s)")
    ax.set_title("Redesign history")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


# ---------------------------------------------------------------------
# EXAMPLE / DEFAULT MODEL
# ---------------------------------------------------------------------

def example_input() -> TowerInput:
    return TowerInput(
        n_story=60,
        story_height=3.2,
        plan_x=80.0,
        plan_y=80.0,
        floor_mass=9.0e5,  # 900 t/story
        core=CoreWallSection(
            thickness=0.60,
            length_x=18.0,
            length_y=18.0,
            n_web_x=2,
            n_web_y=2,
            cracked_factor=0.45,
        ),
        perimeter_column=ColumnSection(b=0.95, h=0.95, cracked_factor=0.70),
        corner_column=ColumnSection(b=1.10, h=1.10, cracked_factor=0.70),
        n_perimeter_columns_x_face=7,
        n_perimeter_columns_y_face=7,
        outriggers=[],
        concrete_E=32e9,
        column_material=Material(E=32e9, fy=420e6),
        wall_material=Material(E=32e9, fy=40e6),
    )


def example_usage():
    inp = example_input()

    print("=== Bare tower ===")
    bare = analyze_tower(inp, axis="x")
    print(f"T1 = {bare.periods[0]:.4f} s")

    print("\n=== Tower with outriggers at 15, 30, 45 ===")
    ou_levels = [
        OutriggerLevel(story=15, axis="x"),
        OutriggerLevel(story=30, axis="x"),
        OutriggerLevel(story=45, axis="x"),
    ]
    inp2 = replace(inp, outriggers=ou_levels)
    model2, res2, hist = redesign_with_outriggers(inp2, axis="x", max_iter=6, verbose=True)
    print(f"Final T1 = {res2.periods[0]:.4f} s")
    print(f"Final perimeter column = {model2.perimeter_column.b:.3f} x {model2.perimeter_column.h:.3f} m")
    print(f"Final core thickness = {model2.core.thickness:.3f} m")

    print("\n=== Comparison study ===")
    study = root_outrigger_study(inp, candidate_stories=[15, 30, 45], counts=(0, 1, 2, 3), axis="x", redesign=True)
    print(study.to_string(index=False))

    return inp, bare, model2, res2, hist, study



# ===== Streamlit App =====


from dataclasses import asdict
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:
    class _StreamlitStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop
    st = _StreamlitStub()


AUTHOR_NAME = "Benyamin"
APP_VERSION = "v5.0-Integrated-MDOF"

st.set_page_config(
    page_title="Tall Building Integrated MDOF + Outrigger",
    layout="wide",
    initial_sidebar_state="expanded",
)


def streamlit_input_panel() -> tuple[TowerInput, str]:
    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", min_value=2, max_value=120, value=60, step=1)
        story_height = st.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=3.2)
        plan_x = st.number_input("Plan X (m)", min_value=20.0, max_value=300.0, value=80.0)
        n_perim_x_face = st.number_input("Perimeter columns on X face", min_value=2, max_value=30, value=7, step=1)
        floor_mass = st.number_input("Typical floor mass (kg)", min_value=1e5, max_value=5e6, value=9.0e5, step=1e4, format="%.0f")
    with c2:
        plan_y = st.number_input("Plan Y (m)", min_value=20.0, max_value=300.0, value=80.0)
        n_perim_y_face = st.number_input("Perimeter columns on Y face", min_value=2, max_value=30, value=7, step=1)
        axis = st.selectbox("Analysis axis", ["x", "y"], index=0)
        drift_den = st.number_input("Drift denominator", min_value=200.0, max_value=2000.0, value=500.0)

    st.markdown("### Core walls")
    c3, c4, c5 = st.columns(3)
    with c3:
        core_t = st.number_input("Core wall thickness (m)", min_value=0.20, max_value=1.50, value=0.60, step=0.05)
        core_lx = st.number_input("Core dimension X (m)", min_value=6.0, max_value=40.0, value=18.0, step=0.5)
    with c4:
        core_ly = st.number_input("Core dimension Y (m)", min_value=6.0, max_value=40.0, value=18.0, step=0.5)
        core_nx = st.number_input("Number of X core webs", min_value=1, max_value=6, value=2, step=1)
    with c5:
        core_ny = st.number_input("Number of Y core webs", min_value=1, max_value=6, value=2, step=1)
        core_cr = st.number_input("Core cracked factor", min_value=0.10, max_value=1.00, value=0.45, step=0.05)

    st.markdown("### Columns")
    c6, c7 = st.columns(2)
    with c6:
        perim_b = st.number_input("Perimeter column b (m)", min_value=0.40, max_value=2.50, value=0.95, step=0.05)
        perim_h = st.number_input("Perimeter column h (m)", min_value=0.40, max_value=2.50, value=0.95, step=0.05)
        perim_cr = st.number_input("Perimeter cracked factor", min_value=0.10, max_value=1.00, value=0.70, step=0.05)
    with c7:
        corner_b = st.number_input("Corner column b (m)", min_value=0.40, max_value=2.50, value=1.10, step=0.05)
        corner_h = st.number_input("Corner column h (m)", min_value=0.40, max_value=2.50, value=1.10, step=0.05)
        corner_cr = st.number_input("Corner cracked factor", min_value=0.10, max_value=1.00, value=0.70, step=0.05)

    st.markdown("### Materials")
    c8, c9 = st.columns(2)
    with c8:
        Ec = st.number_input("Concrete E (Pa)", min_value=20e9, max_value=60e9, value=32e9, step=1e9, format="%.0f")
        fy_col = st.number_input("Column fy (Pa)", min_value=200e6, max_value=700e6, value=420e6, step=10e6, format="%.0f")
    with c9:
        fy_wall = st.number_input("Wall strength proxy fy (Pa)", min_value=20e6, max_value=100e6, value=40e6, step=1e6, format="%.0f")

    st.markdown("### Outriggers")
    c10, c11, c12 = st.columns(3)
    with c10:
        n_or = st.number_input("Number of outriggers", min_value=0, max_value=5, value=0, step=1)
    with c11:
        truss_depth = st.number_input("Truss depth (m)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        chord_area = st.number_input("Chord area (m²)", min_value=0.01, max_value=0.50, value=0.08, step=0.01, format="%.3f")
    with c12:
        diag_area = st.number_input("Diagonal area (m²)", min_value=0.01, max_value=0.50, value=0.04, step=0.01, format="%.3f")
        Es_truss = st.number_input("Truss steel E (Pa)", min_value=100e9, max_value=250e9, value=200e9, step=10e9, format="%.0f")

    ou_story_levels: List[int] = []
    if int(n_or) > 0:
        st.markdown("**Outrigger story levels**")
        cols = st.columns(min(3, int(n_or)))
        defaults = [15, 30, 45, 50, 55]
        for i in range(int(n_or)):
            with cols[i % len(cols)]:
                lev = st.number_input(
                    f"Outrigger {i+1} story",
                    min_value=1,
                    max_value=int(n_story),
                    value=min(defaults[i], int(n_story)),
                    step=1,
                    key=f"or_story_{i}"
                )
                ou_story_levels.append(int(lev))

    core = CoreWallSection(
        thickness=float(core_t),
        length_x=float(core_lx),
        length_y=float(core_ly),
        n_web_x=int(core_nx),
        n_web_y=int(core_ny),
        cracked_factor=float(core_cr),
    )
    perim = ColumnSection(b=float(perim_b), h=float(perim_h), cracked_factor=float(perim_cr))
    corner = ColumnSection(b=float(corner_b), h=float(corner_h), cracked_factor=float(corner_cr))

    outriggers = [
        OutriggerLevel(
            story=s,
            axis=str(axis),
            truss_depth=float(truss_depth),
            chord_area=float(chord_area),
            diagonal_area=float(diag_area),
            truss_E=float(Es_truss),
        )
        for s in ou_story_levels
    ]

    inp = TowerInput(
        n_story=int(n_story),
        story_height=float(story_height),
        plan_x=float(plan_x),
        plan_y=float(plan_y),
        floor_mass=float(floor_mass),
        core=core,
        perimeter_column=perim,
        corner_column=corner,
        n_perimeter_columns_x_face=int(n_perim_x_face),
        n_perimeter_columns_y_face=int(n_perim_y_face),
        outriggers=outriggers,
        concrete_E=float(Ec),
        drift_limit_ratio=1.0 / float(drift_den),
        column_material=Material(E=float(Ec), fy=float(fy_col)),
        wall_material=Material(E=float(Ec), fy=float(fy_wall)),
    )
    return inp, str(axis)


def build_report(inp: TowerInput, axis: str, designed_model: TowerInput, final_result, hist_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("TALL BUILDING INTEGRATED MDOF + OUTRIGGER REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("INPUT MODEL")
    lines.append("-" * 78)
    lines.append(f"Stories                         = {inp.n_story}")
    lines.append(f"Story height                    = {inp.story_height:.3f} m")
    lines.append(f"Total height                    = {inp.height():.3f} m")
    lines.append(f"Plan                            = {inp.plan_x:.3f} x {inp.plan_y:.3f} m")
    lines.append(f"Analysis axis                   = {axis.upper()}")
    lines.append(f"Typical floor mass              = {inp.floor_mass:,.0f} kg")
    lines.append("")
    lines.append("FINAL DESIGNED SECTIONS")
    lines.append("-" * 78)
    lines.append(f"Core thickness                  = {designed_model.core.thickness:.3f} m")
    lines.append(f"Core dimensions                 = {designed_model.core.length_x:.3f} x {designed_model.core.length_y:.3f} m")
    lines.append(f"Perimeter column                = {designed_model.perimeter_column.b:.3f} x {designed_model.perimeter_column.h:.3f} m")
    lines.append(f"Corner column                   = {designed_model.corner_column.b:.3f} x {designed_model.corner_column.h:.3f} m")
    lines.append("")
    lines.append("DYNAMIC RESPONSE")
    lines.append("-" * 78)
    for i, T in enumerate(final_result.periods[:5], start=1):
        lines.append(f"Mode {i} period                 = {T:.4f} s")
    lines.append(f"Max drift ratio (unit profile)  = {final_result.max_drift_ratio_unit_profile:.6f}")
    lines.append("")
    if final_result.outriggers:
        lines.append("OUTRIGGER MECHANICS")
        lines.append("-" * 78)
        for mech in final_result.outriggers:
            lines.append(f"Story {mech.story:>3} | axis={mech.axis.upper()} | arm={mech.arm:.3f} m | "
                         f"k_truss={mech.k_truss_tip:,.3e} N/m | k_col={mech.k_column_tip:,.3e} N/m | "
                         f"k_story={mech.k_story_equiv:,.3e} N/m")
        lines.append("")
    if not hist_df.empty:
        lines.append("ITERATION HISTORY")
        lines.append("-" * 78)
        lines.append(hist_df.to_string(index=False))
        lines.append("")
    return "\n".join(lines)



def _candidate_stories(inp: TowerInput) -> list[int]:
    defaults = sorted(set([
        max(2, round(0.25 * inp.n_story)),
        max(3, round(0.50 * inp.n_story)),
        max(4, round(0.75 * inp.n_story)),
    ]))
    return defaults


def replace_for_study(inp: TowerInput) -> TowerInput:
    # same geometry/materials, no outriggers; study will add them
    return TowerInput(
        n_story=inp.n_story,
        story_height=inp.story_height,
        plan_x=inp.plan_x,
        plan_y=inp.plan_y,
        floor_mass=inp.floor_mass,
        core=inp.core,
        perimeter_column=inp.perimeter_column,
        corner_column=inp.corner_column,
        n_perimeter_columns_x_face=inp.n_perimeter_columns_x_face,
        n_perimeter_columns_y_face=inp.n_perimeter_columns_y_face,
        outriggers=[],
        concrete_E=inp.concrete_E,
        drift_limit_ratio=inp.drift_limit_ratio,
        column_material=inp.column_material,
        wall_material=inp.wall_material,
    )


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 0.7rem; padding-bottom: 0.7rem; max-width: 100%;}
    .stButton button {width: 100%; font-weight: 700; height: 3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tall Building Integrated MDOF + Outrigger")
st.caption(f"Prepared by {AUTHOR_NAME} | {APP_VERSION}")
st.info(
    "This version keeps the Streamlit UI, but the solver is driven by the new "
    "MDOF outrigger framework. The outrigger is assembled at its real level and "
    "feeds back into section re-sizing."
)

if "designed_model" not in st.session_state:
    st.session_state.designed_model = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "study_df" not in st.session_state:
    st.session_state.study_df = None
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

left_col, right_col = st.columns([1.0, 2.2], gap="medium")

with left_col:
    inp, axis = streamlit_input_panel()
    b1, b2 = st.columns(2)

    with b1:
        if st.button("ANALYZE / REDESIGN"):
            try:
                with st.spinner("Running integrated MDOF redesign..."):
                    model, result, hist_df = redesign_with_outriggers(inp, axis=axis, max_iter=6, verbose=False)
                    study_df = root_outrigger_study(
                        replace_for_study(inp),
                        candidate_stories=_candidate_stories(inp),
                        counts=(0, 1, 2, 3),
                        axis=axis,
                        redesign=True,
                    )
                    st.session_state.designed_model = model
                    st.session_state.analysis_result = result
                    st.session_state.history_df = hist_df
                    st.session_state.study_df = study_df
                    st.session_state.report_text = build_report(inp, axis, model, result, hist_df)
                st.success("Integrated analysis completed.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    with b2:
        if st.session_state.report_text:
            st.download_button(
                "SAVE REPORT",
                data=st.session_state.report_text.encode("utf-8"),
                file_name="integrated_mdof_outrigger_report.txt",
                mime="text/plain",
            )
        else:
            st.button("SAVE REPORT", disabled=True)

with right_col:
    if st.session_state.analysis_result is None:
        st.info("Run ANALYZE / REDESIGN to generate the new integrated results.")
    else:
        model = st.session_state.designed_model
        result = st.session_state.analysis_result
        hist_df = st.session_state.history_df
        study_df = st.session_state.study_df

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T1 (s)", f"{result.periods[0]:.3f}")
        c2.metric("T2 (s)", f"{result.periods[1]:.3f}" if len(result.periods) > 1 else "-")
        c3.metric("Max drift ratio", f"{result.max_drift_ratio_unit_profile:.5f}")
        c4.metric("Outriggers", len(result.outriggers))

        d1, d2, d3 = st.columns(3)
        d1.metric("Perim. column", f"{model.perimeter_column.b:.2f} x {model.perimeter_column.h:.2f} m")
        d2.metric("Corner column", f"{model.corner_column.b:.2f} x {model.corner_column.h:.2f} m")
        d3.metric("Core thickness", f"{model.core.thickness:.2f} m")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Plan / Elevation", "Modes", "Design History", "0-3 Outrigger Study", "Report"]
        )

        with tab1:
            a, b = st.columns(2)
            with a:
                st.pyplot(plot_plan(model, axis=axis), use_container_width=True)
            with b:
                st.pyplot(plot_elevation(model), use_container_width=True)

        with tab2:
            st.pyplot(plot_modes(result, n_modes=5), use_container_width=True)

        with tab3:
            if hist_df is not None and not hist_df.empty:
                st.pyplot(plot_design_history(hist_df), use_container_width=True)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.info("No iteration history available.")

        with tab4:
            if study_df is not None and not study_df.empty:
                st.dataframe(study_df, use_container_width=True, hide_index=True)
            else:
                st.info("Study table not available.")

        with tab5:
            st.text_area("", st.session_state.report_text, height=520, label_visibility="collapsed")
