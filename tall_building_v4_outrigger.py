"""
tower_mdof_professional_v7_stable.py

Stable professional Streamlit app for preliminary tower building pre-design.

Main engineering features
-------------------------
1. Story-by-story lateral stiffness.
2. X and Y direction MDOF modal analysis.
3. Empirical target-period control.
4. Iterative section stiffness scaling.
5. Outrigger placement at real story levels.
6. Belt-truss vs pipe-bracing efficiency comparison.
7. Tables, plots, plan view, and downloadable report.

Important limitation
--------------------
This is a preliminary design and research tool. It is not a replacement for
ETABS/SAP2000/Abaqus, final code design, nonlinear analysis, P-Delta analysis,
or wind tunnel/serviceability design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
GAMMA_RC = 25.0          # kN/m3
RHO_STEEL = 7850.0      # kg/m3
APP_VERSION = "v7.0-stable-professional-mdof"


# ============================================================
# ENUMS
# ============================================================

class Direction(str, Enum):
    X = "X"
    Y = "Y"


class OutriggerType(str, Enum):
    BELT_TRUSS = "Belt Truss"
    PIPE_BRACE = "Pipe Bracing"


# ============================================================
# INPUT DATA CLASSES
# ============================================================

@dataclass
class OutriggerInput:
    story: int
    system: OutriggerType
    depth_m: float
    chord_area_m2: float
    diagonal_area_m2: float
    active: bool = True


@dataclass
class ModelInput:
    # Geometry
    n_story: int = 60
    story_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6

    # Core geometry ratios
    core_ratio_x: float = 0.24
    core_ratio_y: float = 0.22
    opening_ratio_x: float = 0.16
    opening_ratio_y: float = 0.14

    # Materials
    fck_mpa: float = 60.0
    Ec_mpa: float = 34000.0
    fy_mpa: float = 420.0

    # Loads
    DL_kN_m2: float = 3.5
    LL_kN_m2: float = 2.5
    finishes_kN_m2: float = 1.5
    live_load_mass_factor: float = 0.25
    facade_kN_m: float = 1.2
    seismic_mass_factor: float = 1.0
    base_shear_coeff: float = 0.015

    # Period target
    Ct: float = 0.0488
    x_exp: float = 0.75
    upper_period_factor: float = 1.20
    target_position_factor: float = 0.80

    # Limits
    drift_limit: float = 1.0 / 500.0
    min_wall_t_m: float = 0.30
    max_wall_t_m: float = 1.00
    min_col_m: float = 0.70
    max_col_m: float = 1.60
    min_slab_t_m: float = 0.22
    max_slab_t_m: float = 0.35
    min_beam_b_m: float = 0.40
    max_beam_b_m: float = 0.90
    min_beam_h_m: float = 0.80
    max_beam_h_m: float = 1.80

    # Cracked stiffness factors
    wall_cracked: float = 0.35
    column_cracked: float = 0.70
    outrigger_connection_eff: float = 0.80

    # Layout
    lower_wall_count: int = 8
    middle_wall_count: int = 6
    upper_wall_count: int = 4
    perimeter_col_factor: float = 1.10
    corner_col_factor: float = 1.30

    # Reinforcement ratios for quantity estimate
    wall_rebar_ratio: float = 0.004
    column_rebar_ratio: float = 0.012
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.004

    outriggers: List[OutriggerInput] = field(default_factory=list)

    @property
    def height_m(self) -> float:
        return self.n_story * self.story_height_m

    @property
    def floor_area_m2(self) -> float:
        return self.plan_x_m * self.plan_y_m

    @property
    def bay_x_m(self) -> float:
        return self.plan_x_m / max(self.n_bays_x, 1)

    @property
    def bay_y_m(self) -> float:
        return self.plan_y_m / max(self.n_bays_y, 1)

    @property
    def Ec_pa(self) -> float:
        return self.Ec_mpa * 1e6


@dataclass
class StoryResult:
    story: int
    elevation_m: float
    zone: str

    wall_t_m: float
    wall_count: int
    core_x_m: float
    core_y_m: float
    open_x_m: float
    open_y_m: float

    interior_col_m: float
    perimeter_col_m: float
    corner_col_m: float
    slab_t_m: float
    beam_b_m: float
    beam_h_m: float

    mass_kg: float
    weight_kN: float

    kx_core: float
    ky_core: float
    kx_col: float
    ky_col: float
    kx_out: float
    ky_out: float

    concrete_m3: float
    steel_kg: float

    @property
    def kx_total(self) -> float:
        return self.kx_core + self.kx_col + self.kx_out

    @property
    def ky_total(self) -> float:
        return self.ky_core + self.ky_col + self.ky_out


@dataclass
class ModalResult:
    direction: Direction
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[np.ndarray]
    effective_mass_ratios: List[float]
    cumulative_mass_ratios: List[float]


@dataclass
class DesignResult:
    inp: ModelInput
    stories: List[StoryResult]
    modal_x: ModalResult
    modal_y: ModalResult
    t_ref: float
    t_target: float
    t_upper: float
    governing_direction: Direction
    governing_period: float
    period_error: float
    drift_x_m: float
    drift_y_m: float
    drift_x_ratio: float
    drift_y_ratio: float
    total_weight_kN: float
    total_mass_kg: float
    total_concrete_m3: float
    total_steel_kg: float
    core_scale: float
    column_scale: float
    outrigger_scale: float
    iteration_table: pd.DataFrame
    messages: List[str]


# ============================================================
# BASIC ENGINEERING FUNCTIONS
# ============================================================

def validate_input(inp: ModelInput) -> None:
    errors = []
    if inp.n_story < 1:
        errors.append("n_story must be at least 1.")
    if inp.story_height_m <= 0:
        errors.append("story_height_m must be positive.")
    if inp.plan_x_m <= 0 or inp.plan_y_m <= 0:
        errors.append("plan dimensions must be positive.")
    if inp.Ec_mpa <= 0:
        errors.append("Ec must be positive.")
    if not (0.05 <= inp.wall_cracked <= 1.0):
        errors.append("wall_cracked must be between 0.05 and 1.0.")
    if not (0.05 <= inp.column_cracked <= 1.0):
        errors.append("column_cracked must be between 0.05 and 1.0.")
    for o in inp.outriggers:
        if o.story < 1 or o.story > inp.n_story:
            errors.append(f"Outrigger story {o.story} is outside the building height.")
        if o.depth_m <= 0 or o.chord_area_m2 <= 0 or o.diagonal_area_m2 <= 0:
            errors.append("Outrigger dimensions and areas must be positive.")
    if errors:
        raise ValueError(" | ".join(errors))


def empirical_periods(inp: ModelInput) -> Tuple[float, float, float]:
    t_ref = inp.Ct * inp.height_m ** inp.x_exp
    t_upper = inp.upper_period_factor * t_ref
    t_target = t_ref + inp.target_position_factor * (t_upper - t_ref)
    return t_ref, t_target, t_upper


def zone_name(inp: ModelInput, story: int) -> str:
    z1 = max(1, round(0.30 * inp.n_story))
    z2 = max(z1 + 1, round(0.70 * inp.n_story))
    if story <= z1:
        return "Lower"
    if story <= z2:
        return "Middle"
    return "Upper"


def wall_count(inp: ModelInput, story: int) -> int:
    z = zone_name(inp, story)
    if z == "Lower":
        return inp.lower_wall_count
    if z == "Middle":
        return inp.middle_wall_count
    return inp.upper_wall_count


def core_dimensions(inp: ModelInput) -> Tuple[float, float, float, float]:
    core_x = max(8.0, inp.core_ratio_x * inp.plan_x_m)
    core_y = max(8.0, inp.core_ratio_y * inp.plan_y_m)
    open_x = min(inp.opening_ratio_x * inp.plan_x_m, core_x - 2.0)
    open_y = min(inp.opening_ratio_y * inp.plan_y_m, core_y - 2.0)
    open_x = max(open_x, 2.0)
    open_y = max(open_y, 2.0)
    return core_x, core_y, open_x, open_y


def wall_thickness(inp: ModelInput, story: int, core_scale: float) -> float:
    z = zone_name(inp, story)
    zone_factor = {"Lower": 1.00, "Middle": 0.82, "Upper": 0.65}[z]
    t = (inp.height_m / 180.0) * zone_factor * core_scale
    return float(np.clip(t, inp.min_wall_t_m, inp.max_wall_t_m))


def column_dimensions(inp: ModelInput, story: int, column_scale: float) -> Tuple[float, float, float]:
    z = zone_name(inp, story)
    zone_factor = {"Lower": 1.00, "Middle": 0.86, "Upper": 0.72}[z]
    base = 0.85 * zone_factor * column_scale
    inter = float(np.clip(base, inp.min_col_m, inp.max_col_m))
    perim = float(np.clip(base * inp.perimeter_col_factor, inp.min_col_m, inp.max_col_m))
    corner = float(np.clip(base * inp.corner_col_factor, inp.min_col_m, inp.max_col_m))
    return inter, perim, corner


def slab_thickness(inp: ModelInput, column_scale: float) -> float:
    span = max(inp.bay_x_m, inp.bay_y_m)
    t = span / 30.0 * (0.95 + 0.10 * column_scale)
    return float(np.clip(t, inp.min_slab_t_m, inp.max_slab_t_m))


def beam_size(inp: ModelInput, column_scale: float) -> Tuple[float, float]:
    span = max(inp.bay_x_m, inp.bay_y_m)
    h = span / 12.0 * (0.95 + 0.10 * column_scale)
    h = float(np.clip(h, inp.min_beam_h_m, inp.max_beam_h_m))
    b = float(np.clip(0.45 * h, inp.min_beam_b_m, inp.max_beam_b_m))
    return b, h


def column_counts(inp: ModelInput) -> Tuple[int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return interior, perimeter, corner


def rect_I_square(dim: float) -> float:
    return dim ** 4 / 12.0


def story_k_from_EI(EI: float, h: float) -> float:
    return 12.0 * EI / max(h ** 3, 1e-9)


def core_inertia(inp: ModelInput, story: int, t: float) -> Tuple[float, float]:
    """Return approximate cracked Ix and Iy of the core."""
    core_x, core_y, _, _ = core_dimensions(inp)
    wc = wall_count(inp, story)

    # Ix resists Y-direction; Iy resists X-direction.
    Ix = 2.0 * (core_x * t**3 / 12.0 + core_x * t * (core_y / 2.0) ** 2)
    Ix += 2.0 * (t * core_y**3 / 12.0)

    Iy = 2.0 * (core_y * t**3 / 12.0 + core_y * t * (core_x / 2.0) ** 2)
    Iy += 2.0 * (t * core_x**3 / 12.0)

    if wc >= 6:
        Ix += 2.0 * (t * (0.45 * core_y) ** 3 / 12.0)
        Iy += 2.0 * ((0.45 * core_x) * t**3 / 12.0 + (0.45 * core_x) * t * (0.22 * core_x) ** 2)

    if wc >= 8:
        Ix += 2.0 * ((0.45 * core_y) * t**3 / 12.0 + (0.45 * core_y) * t * (0.22 * core_y) ** 2)
        Iy += 2.0 * (t * (0.45 * core_x) ** 3 / 12.0)

    return inp.wall_cracked * Ix, inp.wall_cracked * Iy


def column_group_inertia(inp: ModelInput, inter: float, perim: float, corner: float) -> Tuple[float, float]:
    ni, np_, nc = column_counts(inp)
    I = ni * rect_I_square(inter) + np_ * rect_I_square(perim) + nc * rect_I_square(corner)
    I *= inp.column_cracked
    return I, I


def outrigger_efficiency(system: OutriggerType) -> float:
    if system == OutriggerType.BELT_TRUSS:
        return 1.00
    if system == OutriggerType.PIPE_BRACE:
        return 0.70
    return 0.0


def outrigger_material_factor(system: OutriggerType) -> float:
    if system == OutriggerType.BELT_TRUSS:
        return 1.00
    if system == OutriggerType.PIPE_BRACE:
        return 1.15
    return 1.00


def outrigger_stiffness(inp: ModelInput, story: int, outrigger_scale: float) -> Tuple[float, float, float]:
    core_x, core_y, _, _ = core_dimensions(inp)
    arm_x = max(0.5 * (inp.plan_x_m - core_x), 1.0)
    arm_y = max(0.5 * (inp.plan_y_m - core_y), 1.0)

    kx = 0.0
    ky = 0.0
    steel_kg = 0.0

    for o in inp.outriggers:
        if not o.active or o.story != story:
            continue

        eta = outrigger_efficiency(o.system) * inp.outrigger_connection_eff
        mat_factor = outrigger_material_factor(o.system)

        A_chord = o.chord_area_m2 * outrigger_scale
        A_diag = o.diagonal_area_m2 * outrigger_scale
        Lx = sqrt(arm_x ** 2 + o.depth_m ** 2)
        Ly = sqrt(arm_y ** 2 + o.depth_m ** 2)

        kx += eta * (2.0 * inp.Ec_pa * A_chord / arm_x + 2.0 * inp.Ec_pa * A_diag / Lx)
        ky += eta * (2.0 * inp.Ec_pa * A_chord / arm_y + 2.0 * inp.Ec_pa * A_diag / Ly)

        volume = 2.0 * A_chord * (arm_x + arm_y) + 2.0 * A_diag * (Lx + Ly)
        steel_kg += volume * RHO_STEEL * mat_factor

    return kx, ky, steel_kg


def floor_mass_quantities(
    inp: ModelInput,
    wall_t: float,
    inter_col: float,
    perim_col: float,
    corner_col: float,
    slab_t: float,
    beam_b: float,
    beam_h: float,
    out_steel_kg: float,
) -> Tuple[float, float, float, float]:
    A = inp.floor_area_m2

    slab_vol = A * slab_t
    beam_lines = inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1)
    beam_len = beam_lines * 0.5 * (inp.bay_x_m + inp.bay_y_m)
    beam_vol = beam_b * beam_h * beam_len

    ni, np_, nc = column_counts(inp)
    col_vol = (
        ni * inter_col**2 * inp.story_height_m
        + np_ * perim_col**2 * inp.story_height_m
        + nc * corner_col**2 * inp.story_height_m
    )

    core_x, core_y, _, _ = core_dimensions(inp)
    wall_len = 2.0 * (core_x + core_y)
    wall_vol = wall_len * wall_t * inp.story_height_m

    concrete_m3 = slab_vol + beam_vol + col_vol + wall_vol

    steel_kg = (
        inp.slab_rebar_ratio * slab_vol * RHO_STEEL
        + inp.beam_rebar_ratio * beam_vol * RHO_STEEL
        + inp.column_rebar_ratio * col_vol * RHO_STEEL
        + inp.wall_rebar_ratio * wall_vol * RHO_STEEL
        + out_steel_kg
    )

    gravity_area_kN = (inp.DL_kN_m2 + inp.finishes_kN_m2 + inp.live_load_mass_factor * inp.LL_kN_m2) * A
    concrete_weight_kN = GAMMA_RC * concrete_m3
    steel_weight_kN = steel_kg * G / 1000.0
    facade_weight_kN = inp.facade_kN_m * 2.0 * (inp.plan_x_m + inp.plan_y_m)

    weight_kN = (gravity_area_kN + concrete_weight_kN + steel_weight_kN + facade_weight_kN) * inp.seismic_mass_factor
    mass_kg = weight_kN * 1000.0 / G

    return mass_kg, weight_kN, concrete_m3, steel_kg


# ============================================================
# STORY BUILDING AND MODAL ANALYSIS
# ============================================================

def build_stories(inp: ModelInput, core_scale: float, column_scale: float, outrigger_scale: float) -> List[StoryResult]:
    stories: List[StoryResult] = []
    core_x, core_y, open_x, open_y = core_dimensions(inp)

    for story in range(1, inp.n_story + 1):
        zone = zone_name(inp, story)
        wc = wall_count(inp, story)
        wt = wall_thickness(inp, story, core_scale)
        inter, perim, corner = column_dimensions(inp, story, column_scale)
        slab_t = slab_thickness(inp, column_scale)
        beam_b, beam_h = beam_size(inp, column_scale)

        Ix_core, Iy_core = core_inertia(inp, story, wt)
        Ix_col, Iy_col = column_group_inertia(inp, inter, perim, corner)

        # X displacement is resisted by Iy; Y displacement is resisted by Ix.
        kx_core = story_k_from_EI(inp.Ec_pa * Iy_core, inp.story_height_m)
        ky_core = story_k_from_EI(inp.Ec_pa * Ix_core, inp.story_height_m)
        kx_col = story_k_from_EI(inp.Ec_pa * Iy_col, inp.story_height_m)
        ky_col = story_k_from_EI(inp.Ec_pa * Ix_col, inp.story_height_m)

        kx_out, ky_out, out_steel = outrigger_stiffness(inp, story, outrigger_scale)

        # Local coupling effect at outrigger story only.
        if kx_out > 0.0:
            kx_out += 0.15 * min(kx_core, kx_col)
        if ky_out > 0.0:
            ky_out += 0.15 * min(ky_core, ky_col)

        mass_kg, weight_kN, concrete_m3, steel_kg = floor_mass_quantities(
            inp, wt, inter, perim, corner, slab_t, beam_b, beam_h, out_steel
        )

        stories.append(
            StoryResult(
                story=story,
                elevation_m=story * inp.story_height_m,
                zone=zone,
                wall_t_m=wt,
                wall_count=wc,
                core_x_m=core_x,
                core_y_m=core_y,
                open_x_m=open_x,
                open_y_m=open_y,
                interior_col_m=inter,
                perimeter_col_m=perim,
                corner_col_m=corner,
                slab_t_m=slab_t,
                beam_b_m=beam_b,
                beam_h_m=beam_h,
                mass_kg=mass_kg,
                weight_kN=weight_kN,
                kx_core=kx_core,
                ky_core=ky_core,
                kx_col=kx_col,
                ky_col=ky_col,
                kx_out=kx_out,
                ky_out=ky_out,
                concrete_m3=concrete_m3,
                steel_kg=steel_kg,
            )
        )

    return stories


def assemble_mk(stories: List[StoryResult], direction: Direction) -> Tuple[np.ndarray, np.ndarray]:
    n = len(stories)
    masses = np.array([s.mass_kg for s in stories], dtype=float)
    M = np.diag(masses)

    if direction == Direction.X:
        k_story = np.array([s.kx_total for s in stories], dtype=float)
    else:
        k_story = np.array([s.ky_total for s in stories], dtype=float)

    K = np.zeros((n, n), dtype=float)

    for i in range(n):
        k = max(k_story[i], 1e-9)
        if i == 0:
            K[i, i] += k
        else:
            K[i, i] += k
            K[i, i - 1] -= k
            K[i - 1, i] -= k
            K[i - 1, i - 1] += k

    return M, K


def solve_modal(stories: List[StoryResult], direction: Direction, n_modes: int = 5) -> ModalResult:
    M, K = assemble_mk(stories, direction)

    # More stable than inv(M) @ K
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    valid = eigvals > 1e-8
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    m = min(n_modes, len(eigvals))
    omegas = np.sqrt(eigvals[:m])
    periods = [float(2.0 * pi / w) for w in omegas]
    freqs = [float(w / (2.0 * pi)) for w in omegas]

    total_mass = float(np.trace(M))
    ones = np.ones((len(stories), 1))

    shapes: List[np.ndarray] = []
    ratios: List[float] = []
    cumulative: List[float] = []
    cum = 0.0

    for i in range(m):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = float(phi.T @ M @ phi)
        gamma = float((phi.T @ M @ ones) / denom)
        meff = gamma**2 * denom
        ratio = float(meff / total_mass)
        cum += ratio

        ph = phi.flatten()
        if abs(ph[-1]) > 1e-12:
            ph = ph / ph[-1]
        if ph[-1] < 0:
            ph = -ph

        shapes.append(ph)
        ratios.append(ratio)
        cumulative.append(float(cum))

    return ModalResult(direction, periods, freqs, shapes, ratios, cumulative)


def base_shear_N(inp: ModelInput, total_weight_kN: float) -> float:
    return inp.base_shear_coeff * total_weight_kN * 1000.0


def top_drift(stories: List[StoryResult], direction: Direction, Vbase_N: float) -> float:
    n = len(stories)
    elevations = np.array([s.elevation_m for s in stories], dtype=float)
    distribution = elevations / max(np.sum(elevations), 1e-9)
    F = Vbase_N * distribution

    V = np.zeros(n)
    for i in range(n - 1, -1, -1):
        V[i] = F[i] + (V[i + 1] if i < n - 1 else 0.0)

    if direction == Direction.X:
        k = np.array([s.kx_total for s in stories])
    else:
        k = np.array([s.ky_total for s in stories])

    drift = np.sum(V / np.maximum(k, 1e-9))
    return float(drift)


# ============================================================
# DESIGN ITERATION
# ============================================================

def evaluate(inp: ModelInput, core_scale: float, column_scale: float, outrigger_scale: float) -> dict:
    stories = build_stories(inp, core_scale, column_scale, outrigger_scale)
    modal_x = solve_modal(stories, Direction.X)
    modal_y = solve_modal(stories, Direction.Y)
    t_ref, t_target, t_upper = empirical_periods(inp)

    tx = modal_x.periods_s[0]
    ty = modal_y.periods_s[0]
    if tx >= ty:
        gov_dir = Direction.X
        gov_T = tx
    else:
        gov_dir = Direction.Y
        gov_T = ty

    total_weight = sum(s.weight_kN for s in stories)
    total_mass = sum(s.mass_kg for s in stories)
    total_concrete = sum(s.concrete_m3 for s in stories)
    total_steel = sum(s.steel_kg for s in stories)

    V = base_shear_N(inp, total_weight)
    dx = top_drift(stories, Direction.X, V)
    dy = top_drift(stories, Direction.Y, V)

    return {
        "stories": stories,
        "modal_x": modal_x,
        "modal_y": modal_y,
        "t_ref": t_ref,
        "t_target": t_target,
        "t_upper": t_upper,
        "gov_dir": gov_dir,
        "gov_T": gov_T,
        "period_error": abs(gov_T - t_target) / max(t_target, 1e-9),
        "total_weight": total_weight,
        "total_mass": total_mass,
        "total_concrete": total_concrete,
        "total_steel": total_steel,
        "drift_x": dx,
        "drift_y": dy,
        "drift_x_ratio": dx / inp.height_m,
        "drift_y_ratio": dy / inp.height_m,
    }


def run_design(inp: ModelInput, max_iter: int = 30, tolerance: float = 0.025) -> DesignResult:
    validate_input(inp)

    core_scale = 1.0
    column_scale = 1.0
    outrigger_scale = 1.0

    best_score = 1e99
    best = None
    logs = []

    for it in range(1, max_iter + 1):
        ev = evaluate(inp, core_scale, column_scale, outrigger_scale)

        drift_ratio_max = max(ev["drift_x_ratio"], ev["drift_y_ratio"])
        drift_over = max(drift_ratio_max / inp.drift_limit - 1.0, 0.0)
        period_over = max(ev["gov_T"] / ev["t_upper"] - 1.0, 0.0)

        score = (
            1200.0 * ev["period_error"] ** 2
            + 1600.0 * drift_over ** 2
            + 2000.0 * period_over ** 2
            + 0.000002 * ev["total_concrete"]
            + 0.0000005 * ev["total_steel"]
        )

        logs.append(
            {
                "Iteration": it,
                "Core scale": core_scale,
                "Column scale": column_scale,
                "Outrigger scale": outrigger_scale,
                "T governing (s)": ev["gov_T"],
                "T target (s)": ev["t_target"],
                "T upper (s)": ev["t_upper"],
                "Error (%)": 100.0 * ev["period_error"],
                "Drift X": ev["drift_x_ratio"],
                "Drift Y": ev["drift_y_ratio"],
                "Weight (MN)": ev["total_weight"] / 1000.0,
                "Concrete (m3)": ev["total_concrete"],
                "Steel (t)": ev["total_steel"] / 1000.0,
                "Direction": ev["gov_dir"].value,
            }
        )

        if score < best_score:
            best_score = score
            best = (core_scale, column_scale, outrigger_scale, ev)

        ok_period = ev["period_error"] <= tolerance and ev["gov_T"] <= ev["t_upper"]
        ok_drift = drift_ratio_max <= inp.drift_limit
        if ok_period and ok_drift:
            best = (core_scale, column_scale, outrigger_scale, ev)
            break

        # T ~ sqrt(M/K), so K_required / K_current = (T_current / T_target)^2
        stiffness_ratio = (ev["gov_T"] / max(ev["t_target"], 1e-9)) ** 2

        core_factor = stiffness_ratio ** (0.55 / 3.0)
        col_factor = stiffness_ratio ** (0.30 / 4.0)
        out_factor = stiffness_ratio ** 0.15

        if drift_ratio_max > inp.drift_limit:
            df = min(1.20, (drift_ratio_max / inp.drift_limit) ** 0.20)
            core_factor *= df
            col_factor *= df
            out_factor *= df

        damping = 0.65
        core_scale *= 1.0 + damping * (core_factor - 1.0)
        column_scale *= 1.0 + damping * (col_factor - 1.0)
        outrigger_scale *= 1.0 + damping * (out_factor - 1.0)

        core_scale = float(np.clip(core_scale, 0.40, 2.50))
        column_scale = float(np.clip(column_scale, 0.40, 2.50))
        outrigger_scale = float(np.clip(outrigger_scale, 0.40, 3.00))

    if best is None:
        raise RuntimeError("No valid design result was produced.")

    core_scale, column_scale, outrigger_scale, ev = best
    messages = []
    messages.append("Upper period check: OK" if ev["gov_T"] <= ev["t_upper"] else "Upper period check: NOT OK")
    messages.append("Drift check: OK" if max(ev["drift_x_ratio"], ev["drift_y_ratio"]) <= inp.drift_limit else "Drift check: NOT OK")
    messages.append("This is a preliminary MDOF shear-building model. Final design requires calibrated 3D analysis.")

    return DesignResult(
        inp=inp,
        stories=ev["stories"],
        modal_x=ev["modal_x"],
        modal_y=ev["modal_y"],
        t_ref=ev["t_ref"],
        t_target=ev["t_target"],
        t_upper=ev["t_upper"],
        governing_direction=ev["gov_dir"],
        governing_period=ev["gov_T"],
        period_error=ev["period_error"],
        drift_x_m=ev["drift_x"],
        drift_y_m=ev["drift_y"],
        drift_x_ratio=ev["drift_x_ratio"],
        drift_y_ratio=ev["drift_y_ratio"],
        total_weight_kN=ev["total_weight"],
        total_mass_kg=ev["total_mass"],
        total_concrete_m3=ev["total_concrete"],
        total_steel_kg=ev["total_steel"],
        core_scale=core_scale,
        column_scale=column_scale,
        outrigger_scale=outrigger_scale,
        iteration_table=pd.DataFrame(logs),
        messages=messages,
    )


# ============================================================
# TABLES AND PLOTS
# ============================================================

def story_table(res: DesignResult) -> pd.DataFrame:
    rows = []
    for s in res.stories:
        rows.append(
            {
                "Story": s.story,
                "Elevation (m)": s.elevation_m,
                "Zone": s.zone,
                "Wall t (m)": s.wall_t_m,
                "Wall count": s.wall_count,
                "Interior col (m)": s.interior_col_m,
                "Perimeter col (m)": s.perimeter_col_m,
                "Corner col (m)": s.corner_col_m,
                "Kx core (MN/m)": s.kx_core / 1e6,
                "Kx col (MN/m)": s.kx_col / 1e6,
                "Kx out (MN/m)": s.kx_out / 1e6,
                "Kx total (MN/m)": s.kx_total / 1e6,
                "Ky core (MN/m)": s.ky_core / 1e6,
                "Ky col (MN/m)": s.ky_col / 1e6,
                "Ky out (MN/m)": s.ky_out / 1e6,
                "Ky total (MN/m)": s.ky_total / 1e6,
                "Mass (t)": s.mass_kg / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def modal_table(modal: ModalResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mode": list(range(1, len(modal.periods_s) + 1)),
            "Direction": modal.direction.value,
            "Period (s)": modal.periods_s,
            "Frequency (Hz)": modal.frequencies_hz,
            "Eff. Mass (%)": [100.0 * x for x in modal.effective_mass_ratios],
            "Cumulative Mass (%)": [100.0 * x for x in modal.cumulative_mass_ratios],
        }
    )


def sustainability_table(res: DesignResult) -> pd.DataFrame:
    total_area = res.inp.floor_area_m2 * res.inp.n_story
    return pd.DataFrame(
        {
            "Indicator": [
                "Total concrete",
                "Total steel",
                "Concrete intensity",
                "Steel intensity",
                "Estimated seismic weight",
            ],
            "Value": [
                res.total_concrete_m3,
                res.total_steel_kg,
                res.total_concrete_m3 / max(total_area, 1e-9),
                res.total_steel_kg / max(total_area, 1e-9),
                res.total_weight_kN,
            ],
            "Unit": ["m3", "kg", "m3/m2", "kg/m2", "kN"],
        }
    )


def plot_stiffness(res: DesignResult, direction: Direction):
    df = story_table(res)
    fig, ax = plt.subplots(figsize=(8, 8))
    if direction == Direction.X:
        ax.plot(df["Kx core (MN/m)"], df["Story"], label="Core")
        ax.plot(df["Kx col (MN/m)"], df["Story"], label="Columns")
        ax.plot(df["Kx out (MN/m)"], df["Story"], label="Outrigger")
        ax.plot(df["Kx total (MN/m)"], df["Story"], linewidth=2.5, label="Total")
    else:
        ax.plot(df["Ky core (MN/m)"], df["Story"], label="Core")
        ax.plot(df["Ky col (MN/m)"], df["Story"], label="Columns")
        ax.plot(df["Ky out (MN/m)"], df["Story"], label="Outrigger")
        ax.plot(df["Ky total (MN/m)"], df["Story"], linewidth=2.5, label="Total")
    ax.set_title(f"Story stiffness profile - {direction.value}")
    ax.set_xlabel("Stiffness (MN/m)")
    ax.set_ylabel("Story")
    ax.grid(True, alpha=0.30)
    ax.legend()
    return fig


def plot_modes(res: DesignResult, modal: ModalResult):
    y = np.array([s.elevation_m for s in res.stories])
    n = min(5, len(modal.mode_shapes))
    fig, axes = plt.subplots(1, n, figsize=(16, 6))
    if n == 1:
        axes = [axes]
    out_stories = [o.story for o in res.inp.outriggers if o.active]
    for i in range(n):
        ax = axes[i]
        phi = modal.mode_shapes[i]
        phi = phi / max(np.max(np.abs(phi)), 1e-9)
        ax.plot(phi, y, linewidth=2.0)
        ax.scatter(phi, y, s=12)
        ax.axvline(0.0, linestyle="--", linewidth=0.8)
        for st_ in out_stories:
            ax.axhline(st_ * res.inp.story_height_m, linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_title(f"Mode {i+1}\nT={modal.periods_s[i]:.3f}s")
        ax.set_xlabel("Normalized shape")
        if i == 0:
            ax.set_ylabel("Elevation (m)")
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Mode shapes - {modal.direction.value}")
    fig.tight_layout()
    return fig


def plot_iteration(res: DesignResult):
    df = res.iteration_table
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(df["Iteration"], df["T governing (s)"], marker="o", label="Governing")
    axes[0, 0].plot(df["Iteration"], df["T target (s)"], linestyle="--", label="Target")
    axes[0, 0].plot(df["Iteration"], df["T upper (s)"], linestyle=":", label="Upper")
    axes[0, 0].set_title("Period convergence")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Period (s)")
    axes[0, 0].grid(True, alpha=0.30)
    axes[0, 0].legend()

    axes[0, 1].plot(df["Iteration"], df["Error (%)"], marker="s")
    axes[0, 1].axhline(2.5, linestyle="--")
    axes[0, 1].set_title("Period error")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Error (%)")
    axes[0, 1].grid(True, alpha=0.30)

    axes[1, 0].plot(df["Iteration"], df["Core scale"], marker="o", label="Core")
    axes[1, 0].plot(df["Iteration"], df["Column scale"], marker="s", label="Columns")
    axes[1, 0].plot(df["Iteration"], df["Outrigger scale"], marker="^", label="Outrigger")
    axes[1, 0].set_title("Scale factors")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].grid(True, alpha=0.30)
    axes[1, 0].legend()

    axes[1, 1].plot(df["Iteration"], df["Steel (t)"], marker="d")
    axes[1, 1].set_title("Steel trend")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Steel (t)")
    axes[1, 1].grid(True, alpha=0.30)

    fig.tight_layout()
    return fig


def plot_plan(res: DesignResult, story: int):
    inp = res.inp
    s = next(x for x in res.stories if x.story == story)
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black")

    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#d0d0d0", linewidth=0.7)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#d0d0d0", linewidth=0.7)

    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x_m
            y = j * inp.bay_y_m
            is_corner = (i in [0, inp.n_bays_x]) and (j in [0, inp.n_bays_y])
            is_perim = (i in [0, inp.n_bays_x] or j in [0, inp.n_bays_y]) and not is_corner
            if is_corner:
                dim = s.corner_col_m
            elif is_perim:
                dim = s.perimeter_col_m
            else:
                dim = s.interior_col_m
            ax.add_patch(plt.Rectangle((x - dim/2, y - dim/2), dim, dim, facecolor="#8B0000", edgecolor="black", alpha=0.85))

    cx0 = 0.5 * (inp.plan_x_m - s.core_x_m)
    cy0 = 0.5 * (inp.plan_y_m - s.core_y_m)
    ox0 = 0.5 * (inp.plan_x_m - s.open_x_m)
    oy0 = 0.5 * (inp.plan_y_m - s.open_y_m)

    ax.add_patch(plt.Rectangle((cx0, cy0), s.core_x_m, s.core_y_m, fill=False, edgecolor="#2E8B57", linewidth=2.4))
    ax.add_patch(plt.Rectangle((ox0, oy0), s.open_x_m, s.open_y_m, fill=False, edgecolor="#2E8B57", linestyle="--", linewidth=1.2))

    active_out = [o for o in inp.outriggers if o.active and o.story == story]
    if active_out:
        mx = inp.plan_x_m / 2.0
        my = inp.plan_y_m / 2.0
        ax.plot([0, cx0], [my, my], linewidth=4.5, color="#FF8C00")
        ax.plot([cx0 + s.core_x_m, inp.plan_x_m], [my, my], linewidth=4.5, color="#FF8C00")
        ax.plot([mx, mx], [0, cy0], linewidth=4.5, color="#FF8C00")
        ax.plot([mx, mx], [cy0 + s.core_y_m, inp.plan_y_m], linewidth=4.5, color="#FF8C00")
        label = ", ".join([o.system.value for o in active_out])
        ax.text(mx, my, f"Outrigger\n{label}", ha="center", va="center", fontsize=9)

    ax.set_title(f"Plan view - Story {story} | Zone: {s.zone}\nWall t={s.wall_t_m:.2f} m | Interior col={s.interior_col_m:.2f} m | Corner col={s.corner_col_m:.2f} m")
    ax.set_aspect("equal")
    ax.set_xlim(-2, inp.plan_x_m + 2)
    ax.set_ylim(-2, inp.plan_y_m + 2)
    ax.grid(False)
    return fig


def build_report(res: DesignResult) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("PROFESSIONAL TOWER PRE-DESIGN REPORT")
    lines.append("=" * 80)
    lines.append(f"Version: {APP_VERSION}")
    lines.append("")
    lines.append("GLOBAL PERIOD TARGETING")
    lines.append("-" * 80)
    lines.append(f"Reference empirical period : {res.t_ref:.3f} s")
    lines.append(f"Target period              : {res.t_target:.3f} s")
    lines.append(f"Upper period limit         : {res.t_upper:.3f} s")
    lines.append(f"Governing direction        : {res.governing_direction.value}")
    lines.append(f"Governing MDOF period      : {res.governing_period:.3f} s")
    lines.append(f"Period error               : {100.0 * res.period_error:.2f} %")
    lines.append("")
    lines.append("DRIFT SCREENING")
    lines.append("-" * 80)
    lines.append(f"Top drift X                : {res.drift_x_m:.4f} m")
    lines.append(f"Top drift Y                : {res.drift_y_m:.4f} m")
    lines.append(f"Drift ratio X              : {res.drift_x_ratio:.6f}")
    lines.append(f"Drift ratio Y              : {res.drift_y_ratio:.6f}")
    lines.append(f"Allowable drift ratio      : {res.inp.drift_limit:.6f}")
    lines.append("")
    lines.append("MATERIAL ESTIMATE")
    lines.append("-" * 80)
    lines.append(f"Total seismic weight       : {res.total_weight_kN:,.0f} kN")
    lines.append(f"Total mass                 : {res.total_mass_kg:,.0f} kg")
    lines.append(f"Total concrete             : {res.total_concrete_m3:,.0f} m3")
    lines.append(f"Total steel                : {res.total_steel_kg:,.0f} kg")
    lines.append("")
    lines.append("FINAL SCALE FACTORS")
    lines.append("-" * 80)
    lines.append(f"Core scale                 : {res.core_scale:.3f}")
    lines.append(f"Column scale               : {res.column_scale:.3f}")
    lines.append(f"Outrigger scale            : {res.outrigger_scale:.3f}")
    lines.append("")
    lines.append("OUTRIGGER SYSTEMS")
    lines.append("-" * 80)
    if res.inp.outriggers:
        for o in res.inp.outriggers:
            lines.append(f"Story {o.story}: {o.system.value}, depth={o.depth_m:.2f} m, A_chord={o.chord_area_m2:.4f} m2, A_diag={o.diagonal_area_m2:.4f} m2")
    else:
        lines.append("No outrigger system used.")
    lines.append("")
    lines.append("STATUS")
    lines.append("-" * 80)
    for m in res.messages:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("ENGINEERING CONCLUSION")
    lines.append("-" * 80)
    lines.append("For preliminary tower design, belt-truss outriggers are generally more efficient than pipe bracing for controlling global period and drift because they couple the core and perimeter columns more directly. Pipe bracing can still be useful where architecture or constructability demands it, but its lower global coupling efficiency should be checked using the modal and stiffness profiles.")
    lines.append("")
    lines.append("This tool is suitable for preliminary sizing and research comparison only. Final design requires calibrated 3D finite-element analysis, torsional checks, P-Delta effects, wind/seismic load combinations, and code-based member design.")
    return "\n".join(lines)


# ============================================================
# STREAMLIT UI
# ============================================================

def sidebar_inputs() -> ModelInput:
    st.sidebar.header("1. Geometry")
    n_story = st.sidebar.number_input("Stories", 10, 120, 60, step=1)
    story_height = st.sidebar.number_input("Story height (m)", 2.8, 5.0, 3.2)
    plan_x = st.sidebar.number_input("Plan X (m)", 20.0, 200.0, 48.0)
    plan_y = st.sidebar.number_input("Plan Y (m)", 20.0, 200.0, 42.0)
    n_bays_x = st.sidebar.number_input("Bays X", 2, 20, 6, step=1)
    n_bays_y = st.sidebar.number_input("Bays Y", 2, 20, 6, step=1)

    core_ratio_x = st.sidebar.number_input("Core ratio X", 0.10, 0.45, 0.24)
    core_ratio_y = st.sidebar.number_input("Core ratio Y", 0.10, 0.45, 0.22)
    opening_ratio_x = st.sidebar.number_input("Opening ratio X", 0.05, 0.35, 0.16)
    opening_ratio_y = st.sidebar.number_input("Opening ratio Y", 0.05, 0.35, 0.14)

    st.sidebar.header("2. Materials")
    fck = st.sidebar.number_input("fck (MPa)", 25.0, 100.0, 60.0)
    Ec = st.sidebar.number_input("Ec (MPa)", 22000.0, 60000.0, 34000.0)
    fy = st.sidebar.number_input("fy (MPa)", 300.0, 700.0, 420.0)

    st.sidebar.header("3. Loads")
    DL = st.sidebar.number_input("DL (kN/m2)", 0.0, 15.0, 3.5)
    LL = st.sidebar.number_input("LL (kN/m2)", 0.0, 10.0, 2.5)
    finishes = st.sidebar.number_input("Finishes (kN/m2)", 0.0, 8.0, 1.5)
    facade = st.sidebar.number_input("Facade load (kN/m)", 0.0, 20.0, 1.2)
    live_mass = st.sidebar.number_input("Live load mass factor", 0.0, 1.0, 0.25)
    seismic_mass = st.sidebar.number_input("Seismic mass factor", 0.5, 1.5, 1.0)
    base_shear_coeff = st.sidebar.number_input("Preliminary base shear coeff.", 0.001, 0.100, 0.015)

    st.sidebar.header("4. Period target")
    Ct = st.sidebar.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f")
    x_exp = st.sidebar.number_input("x exponent", 0.10, 1.50, 0.75)
    upper_factor = st.sidebar.number_input("Upper period factor", 1.00, 2.00, 1.20)
    target_factor = st.sidebar.number_input("Target position factor", 0.10, 0.95, 0.80)

    st.sidebar.header("5. Limits and cracked factors")
    drift_den = st.sidebar.number_input("Drift denominator", 250.0, 2000.0, 500.0)
    wall_cracked = st.sidebar.number_input("Wall cracked factor", 0.05, 1.00, 0.35)
    col_cracked = st.sidebar.number_input("Column cracked factor", 0.05, 1.00, 0.70)
    out_eff = st.sidebar.number_input("Outrigger connection efficiency", 0.30, 1.00, 0.80)

    st.sidebar.header("6. Section limits")
    min_wall = st.sidebar.number_input("Min wall t (m)", 0.20, 1.00, 0.30)
    max_wall = st.sidebar.number_input("Max wall t (m)", 0.40, 2.00, 1.00)
    min_col = st.sidebar.number_input("Min column dim (m)", 0.40, 2.00, 0.70)
    max_col = st.sidebar.number_input("Max column dim (m)", 0.80, 3.00, 1.60)
    min_slab = st.sidebar.number_input("Min slab t (m)", 0.15, 0.50, 0.22)
    max_slab = st.sidebar.number_input("Max slab t (m)", 0.20, 0.70, 0.35)

    st.sidebar.header("7. Layout")
    lower_wc = st.sidebar.number_input("Lower wall count", 4, 8, 8, step=1)
    middle_wc = st.sidebar.number_input("Middle wall count", 4, 8, 6, step=1)
    upper_wc = st.sidebar.number_input("Upper wall count", 4, 8, 4, step=1)
    perim_factor = st.sidebar.number_input("Perimeter column factor", 1.00, 2.00, 1.10)
    corner_factor = st.sidebar.number_input("Corner column factor", 1.00, 2.00, 1.30)

    st.sidebar.header("8. Outriggers")
    n_out = st.sidebar.number_input("Number of outriggers", 0, 5, 2, step=1)
    outriggers: List[OutriggerInput] = []
    for i in range(int(n_out)):
        st.sidebar.markdown(f"Outrigger {i+1}")
        default_story = int(round((i + 1) * int(n_story) / (int(n_out) + 1)))
        story = st.sidebar.number_input(f"Outrigger story {i+1}", 1, int(n_story), default_story, step=1, key=f"out_story_{i}")
        system_text = st.sidebar.selectbox(f"System {i+1}", [OutriggerType.BELT_TRUSS.value, OutriggerType.PIPE_BRACE.value], key=f"out_type_{i}")
        depth = st.sidebar.number_input(f"Depth {i+1} (m)", 1.0, 8.0, 3.0, key=f"out_depth_{i}")
        chord = st.sidebar.number_input(f"Chord/main area {i+1} (m2)", 0.005, 0.500, 0.08, key=f"out_chord_{i}")
        diag = st.sidebar.number_input(f"Diagonal area {i+1} (m2)", 0.005, 0.500, 0.04, key=f"out_diag_{i}")
        outriggers.append(OutriggerInput(int(story), OutriggerType(system_text), float(depth), float(chord), float(diag), True))

    return ModelInput(
        n_story=int(n_story),
        story_height_m=float(story_height),
        plan_x_m=float(plan_x),
        plan_y_m=float(plan_y),
        n_bays_x=int(n_bays_x),
        n_bays_y=int(n_bays_y),
        core_ratio_x=float(core_ratio_x),
        core_ratio_y=float(core_ratio_y),
        opening_ratio_x=float(opening_ratio_x),
        opening_ratio_y=float(opening_ratio_y),
        fck_mpa=float(fck),
        Ec_mpa=float(Ec),
        fy_mpa=float(fy),
        DL_kN_m2=float(DL),
        LL_kN_m2=float(LL),
        finishes_kN_m2=float(finishes),
        live_load_mass_factor=float(live_mass),
        facade_kN_m=float(facade),
        seismic_mass_factor=float(seismic_mass),
        base_shear_coeff=float(base_shear_coeff),
        Ct=float(Ct),
        x_exp=float(x_exp),
        upper_period_factor=float(upper_factor),
        target_position_factor=float(target_factor),
        drift_limit=1.0 / float(drift_den),
        wall_cracked=float(wall_cracked),
        column_cracked=float(col_cracked),
        outrigger_connection_eff=float(out_eff),
        min_wall_t_m=float(min_wall),
        max_wall_t_m=float(max_wall),
        min_col_m=float(min_col),
        max_col_m=float(max_col),
        min_slab_t_m=float(min_slab),
        max_slab_t_m=float(max_slab),
        lower_wall_count=int(lower_wc),
        middle_wall_count=int(middle_wc),
        upper_wall_count=int(upper_wc),
        perimeter_col_factor=float(perim_factor),
        corner_col_factor=float(corner_factor),
        outriggers=outriggers,
    )


def main() -> None:
    st.set_page_config(page_title="Tower MDOF Professional v7", layout="wide")

    st.title("Tower Preliminary Design - Professional MDOF v7")
    st.caption(APP_VERSION)
    st.info("Stable version: X/Y MDOF modal analysis, story-by-story stiffness, period targeting, and outrigger comparison.")

    inp = sidebar_inputs()

    if "res_v7" not in st.session_state:
        st.session_state.res_v7 = None
    if "report_v7" not in st.session_state:
        st.session_state.report_v7 = ""

    if st.button("Run professional MDOF analysis", type="primary"):
        try:
            with st.spinner("Running analysis..."):
                res = run_design(inp)
                st.session_state.res_v7 = res
                st.session_state.report_v7 = build_report(res)
            st.success("Analysis completed.")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    res = st.session_state.res_v7
    if res is None:
        st.warning("Run the analysis to see results.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T_ref (s)", f"{res.t_ref:.3f}")
    c2.metric("T_target (s)", f"{res.t_target:.3f}")
    c3.metric("T_gov (s)", f"{res.governing_period:.3f}")
    c4.metric("Error (%)", f"{100.0 * res.period_error:.2f}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Governing direction", res.governing_direction.value)
    d2.metric("Drift X", f"{res.drift_x_ratio:.5f}")
    d3.metric("Drift Y", f"{res.drift_y_ratio:.5f}")
    d4.metric("Weight (MN)", f"{res.total_weight_kN / 1000.0:.1f}")

    tabs = st.tabs([
        "Stiffness X",
        "Stiffness Y",
        "Modes X",
        "Modes Y",
        "Convergence",
        "Plan",
        "Story Table",
        "Modal Tables",
        "Sustainability",
        "Report",
    ])

    with tabs[0]:
        st.pyplot(plot_stiffness(res, Direction.X), use_container_width=True)
    with tabs[1]:
        st.pyplot(plot_stiffness(res, Direction.Y), use_container_width=True)
    with tabs[2]:
        st.pyplot(plot_modes(res, res.modal_x), use_container_width=True)
    with tabs[3]:
        st.pyplot(plot_modes(res, res.modal_y), use_container_width=True)
    with tabs[4]:
        st.pyplot(plot_iteration(res), use_container_width=True)
        st.dataframe(res.iteration_table, use_container_width=True, hide_index=True)
    with tabs[5]:
        story = st.slider("Story", 1, res.inp.n_story, max(1, res.inp.n_story // 2))
        st.pyplot(plot_plan(res, story), use_container_width=True)
    with tabs[6]:
        st.dataframe(story_table(res), use_container_width=True, hide_index=True)
    with tabs[7]:
        st.markdown("### X Direction")
        st.dataframe(modal_table(res.modal_x), use_container_width=True, hide_index=True)
        st.markdown("### Y Direction")
        st.dataframe(modal_table(res.modal_y), use_container_width=True, hide_index=True)
    with tabs[8]:
        st.dataframe(sustainability_table(res), use_container_width=True, hide_index=True)
        st.markdown("The sustainable option is the system that satisfies target period and drift with the lowest reliable material demand.")
    with tabs[9]:
        st.text_area("Report", st.session_state.report_v7, height=520)
        st.download_button(
            "Download report",
            data=st.session_state.report_v7.encode("utf-8"),
            file_name="tower_mdof_professional_report_v7.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
