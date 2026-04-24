
"""
phd_tower_pre_design_app.py

PhD-defensible preliminary tower pre-design framework
====================================================

Purpose
-------
This application is a PRELIMINARY structural engineering framework for tall building
pre-design. It is intended for academic comparison, early-stage sizing, and structural
system evaluation, not final member design.

Main capabilities
-----------------
1. Story-by-story mass estimation.
2. Story-by-story lateral stiffness estimation in X and Y directions.
3. MDOF shear-building matrix assembly.
4. Modal period extraction from eigenvalue analysis.
5. Modal effective mass participation calculation.
6. Empirical target-period comparison.
7. Iterative section-stiffness scaling toward target period.
8. Outrigger comparison:
   - Belt truss outrigger
   - Pipe-braced outrigger
9. Sustainability indicators:
   - concrete intensity
   - steel intensity
   - stiffness gained per tonne of outrigger steel

Engineering status
------------------
This is a rational preliminary model. It is not a substitute for a full 3D FEM design
model. Final design must include torsion, diaphragm modeling, P-Delta effects,
foundation flexibility, load combinations, member design, wind/seismic code checks,
and calibrated cracked-section stiffness.

Author: Benyamin
Version: v8.0-phd-defensible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi, sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# 0. CONSTANTS
# ============================================================

G = 9.81                         # m/s²
GAMMA_RC = 25.0                  # kN/m³
RHO_STEEL = 7850.0               # kg/m³
APP_VERSION = "v8.0-phd-defensible"


# ============================================================
# 1. DATA DEFINITIONS
# ============================================================

class OutriggerType(str, Enum):
    NONE = "None"
    BELT_TRUSS = "Belt Truss"
    PIPE_BRACE = "Pipe Bracing"


class Direction(str, Enum):
    X = "X"
    Y = "Y"


@dataclass
class Material:
    """Material properties."""
    Ec_MPa: float = 34000.0
    fck_MPa: float = 60.0
    fy_MPa: float = 420.0

    @property
    def Ec_N_m2(self) -> float:
        return self.Ec_MPa * 1e6


@dataclass
class Geometry:
    """Building geometry."""
    n_story: int = 60
    story_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6
    core_ratio_x: float = 0.24
    core_ratio_y: float = 0.22
    core_opening_ratio_x: float = 0.16
    core_opening_ratio_y: float = 0.14

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


@dataclass
class Loads:
    """Preliminary gravity and seismic mass assumptions."""
    DL_kN_m2: float = 3.5
    LL_kN_m2: float = 2.5
    finish_kN_m2: float = 1.5
    live_load_mass_factor: float = 0.25
    facade_line_kN_m: float = 1.2
    preliminary_base_shear_coeff: float = 0.015


@dataclass
class SectionLimits:
    """Practical preliminary section bounds."""
    min_wall_t_m: float = 0.30
    max_wall_t_m: float = 1.20
    min_col_dim_m: float = 0.70
    max_col_dim_m: float = 1.80
    min_slab_t_m: float = 0.22
    max_slab_t_m: float = 0.40
    min_beam_b_m: float = 0.40
    max_beam_b_m: float = 1.00
    min_beam_h_m: float = 0.75
    max_beam_h_m: float = 1.80


@dataclass
class StiffnessFactors:
    """Effective stiffness factors."""
    wall_cracked_factor: float = 0.35
    column_cracked_factor: float = 0.70
    outrigger_connection_factor: float = 0.80


@dataclass
class PeriodTarget:
    """
    Empirical target period definition.

    T_ref = Ct * H^x
    T_upper = upper_factor * T_ref
    T_target = T_ref + beta * (T_upper - T_ref)
    """
    Ct: float = 0.0488
    x_exp: float = 0.75
    upper_factor: float = 1.20
    beta: float = 0.80

    def compute(self, H_m: float) -> Tuple[float, float, float]:
        T_ref = self.Ct * H_m ** self.x_exp
        T_upper = self.upper_factor * T_ref
        T_target = T_ref + self.beta * (T_upper - T_ref)
        return T_ref, T_target, T_upper


@dataclass
class OutriggerInput:
    story: int = 30
    system: OutriggerType = OutriggerType.BELT_TRUSS
    depth_m: float = 3.0
    chord_area_m2: float = 0.08
    diagonal_area_m2: float = 0.04
    active: bool = True


@dataclass
class BuildingInput:
    material: Material = field(default_factory=Material)
    geometry: Geometry = field(default_factory=Geometry)
    loads: Loads = field(default_factory=Loads)
    limits: SectionLimits = field(default_factory=SectionLimits)
    factors: StiffnessFactors = field(default_factory=StiffnessFactors)
    period: PeriodTarget = field(default_factory=PeriodTarget)

    drift_limit_ratio: float = 1 / 500

    lower_wall_count: int = 8
    middle_wall_count: int = 6
    upper_wall_count: int = 4

    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30

    wall_rebar_ratio: float = 0.004
    column_rebar_ratio: float = 0.012
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.004

    outriggers: List[OutriggerInput] = field(default_factory=list)


@dataclass
class StoryResult:
    story: int
    elevation_m: float
    zone: str

    wall_t_m: float
    wall_count: int
    interior_col_m: float
    perimeter_col_m: float
    corner_col_m: float
    slab_t_m: float
    beam_b_m: float
    beam_h_m: float

    mass_kg: float
    weight_kN: float
    concrete_m3: float
    steel_kg: float

    kx_core_N_m: float
    ky_core_N_m: float
    kx_columns_N_m: float
    ky_columns_N_m: float
    kx_outrigger_N_m: float
    ky_outrigger_N_m: float
    outrigger_steel_kg: float

    @property
    def kx_total_N_m(self) -> float:
        return self.kx_core_N_m + self.kx_columns_N_m + self.kx_outrigger_N_m

    @property
    def ky_total_N_m(self) -> float:
        return self.ky_core_N_m + self.ky_columns_N_m + self.ky_outrigger_N_m


@dataclass
class ModalResult:
    direction: Direction
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[np.ndarray]
    effective_mass_ratio: List[float]
    cumulative_mass_ratio: List[float]


@dataclass
class DesignResult:
    input: BuildingInput
    stories: List[StoryResult]
    modal_x: ModalResult
    modal_y: ModalResult
    T_ref_s: float
    T_target_s: float
    T_upper_s: float
    governing_direction: Direction
    governing_period_s: float
    period_error_percent: float
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
    iteration_df: pd.DataFrame
    messages: List[str]


# ============================================================
# 2. VALIDATION
# ============================================================

def validate_input(inp: BuildingInput) -> List[str]:
    """Return validation errors. Empty list means input is acceptable."""
    g = inp.geometry
    m = inp.material
    errors = []

    if g.n_story < 1:
        errors.append("Number of stories must be positive.")
    if g.story_height_m <= 0:
        errors.append("Story height must be positive.")
    if g.plan_x_m <= 0 or g.plan_y_m <= 0:
        errors.append("Plan dimensions must be positive.")
    if g.n_bays_x < 1 or g.n_bays_y < 1:
        errors.append("Number of bays must be positive.")
    if m.Ec_MPa <= 0:
        errors.append("Concrete elastic modulus must be positive.")
    if not (0.05 <= inp.factors.wall_cracked_factor <= 1.00):
        errors.append("Wall cracked factor must be between 0.05 and 1.00.")
    if not (0.05 <= inp.factors.column_cracked_factor <= 1.00):
        errors.append("Column cracked factor must be between 0.05 and 1.00.")
    if inp.drift_limit_ratio <= 0:
        errors.append("Drift limit ratio must be positive.")

    for o in inp.outriggers:
        if o.active:
            if not (1 <= o.story <= g.n_story):
                errors.append(f"Outrigger story {o.story} is outside the building height.")
            if o.depth_m <= 0 or o.chord_area_m2 <= 0 or o.diagonal_area_m2 <= 0:
                errors.append("Outrigger depth and member areas must be positive.")

    return errors


# ============================================================
# 3. BASIC GEOMETRY FUNCTIONS
# ============================================================

def get_zone(inp: BuildingInput, story: int) -> str:
    """Three-zone height division."""
    n = inp.geometry.n_story
    z1 = round(0.30 * n)
    z2 = round(0.70 * n)

    if story <= z1:
        return "Lower"
    if story <= z2:
        return "Middle"
    return "Upper"


def wall_count(inp: BuildingInput, story: int) -> int:
    z = get_zone(inp, story)
    if z == "Lower":
        return inp.lower_wall_count
    if z == "Middle":
        return inp.middle_wall_count
    return inp.upper_wall_count


def core_dimensions(inp: BuildingInput) -> Tuple[float, float, float, float]:
    """Return core outer X, outer Y, opening X, opening Y."""
    g = inp.geometry
    core_x = max(8.0, g.core_ratio_x * g.plan_x_m)
    core_y = max(8.0, g.core_ratio_y * g.plan_y_m)

    open_x = min(g.core_opening_ratio_x * g.plan_x_m, core_x - 2.0)
    open_y = min(g.core_opening_ratio_y * g.plan_y_m, core_y - 2.0)

    return core_x, core_y, open_x, open_y


def story_wall_thickness(inp: BuildingInput, story: int, core_scale: float) -> float:
    """Height-based wall thickness with zone reduction."""
    H = inp.geometry.height_m
    zone = get_zone(inp, story)

    zone_factor = {"Lower": 1.00, "Middle": 0.82, "Upper": 0.65}[zone]
    t = H / 180.0 * zone_factor * core_scale

    return float(np.clip(t, inp.limits.min_wall_t_m, inp.limits.max_wall_t_m))


def story_column_dimensions(inp: BuildingInput, story: int, column_scale: float) -> Tuple[float, float, float]:
    """Return interior, perimeter, corner column square dimensions."""
    zone = get_zone(inp, story)
    zone_factor = {"Lower": 1.00, "Middle": 0.86, "Upper": 0.72}[zone]

    base = 0.85 * zone_factor * column_scale
    base = float(np.clip(base, inp.limits.min_col_dim_m, inp.limits.max_col_dim_m))

    interior = base
    perimeter = float(np.clip(base * inp.perimeter_column_factor, inp.limits.min_col_dim_m, inp.limits.max_col_dim_m))
    corner = float(np.clip(base * inp.corner_column_factor, inp.limits.min_col_dim_m, inp.limits.max_col_dim_m))

    return interior, perimeter, corner


def story_slab_thickness(inp: BuildingInput, column_scale: float) -> float:
    """Preliminary slab thickness."""
    span = max(inp.geometry.bay_x_m, inp.geometry.bay_y_m)
    t = span / 30.0 * (0.95 + 0.10 * column_scale)
    return float(np.clip(t, inp.limits.min_slab_t_m, inp.limits.max_slab_t_m))


def story_beam_size(inp: BuildingInput, column_scale: float) -> Tuple[float, float]:
    """Preliminary beam size."""
    span = max(inp.geometry.bay_x_m, inp.geometry.bay_y_m)
    h = span / 12.0 * (0.95 + 0.10 * column_scale)
    h = float(np.clip(h, inp.limits.min_beam_h_m, inp.limits.max_beam_h_m))
    b = float(np.clip(0.45 * h, inp.limits.min_beam_b_m, inp.limits.max_beam_b_m))
    return b, h


def column_counts(inp: BuildingInput) -> Tuple[int, int, int]:
    """Return interior, perimeter, corner column counts."""
    g = inp.geometry
    total = (g.n_bays_x + 1) * (g.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (g.n_bays_x - 1) + 2 * (g.n_bays_y - 1))
    interior = max(0, total - perimeter - corner)
    return interior, perimeter, corner


# ============================================================
# 4. STIFFNESS MODEL
# ============================================================

def rect_I(b: float, h: float) -> float:
    """Second moment of area of a rectangle around centroidal axis."""
    return b * h**3 / 12.0


def core_effective_inertia(inp: BuildingInput, story: int, wall_t: float) -> Tuple[float, float]:
    """
    Approximate core effective inertia.

    Ix resists lateral displacement in Y direction.
    Iy resists lateral displacement in X direction.

    This is a preliminary closed-form wall-tube approximation.
    """
    core_x, core_y, _, _ = core_dimensions(inp)
    wc = wall_count(inp, story)

    # Basic rectangular tube approximation.
    Ix = 2 * (core_x * wall_t**3 / 12 + core_x * wall_t * (core_y / 2) ** 2)
    Ix += 2 * (wall_t * core_y**3 / 12)

    Iy = 2 * (core_y * wall_t**3 / 12 + core_y * wall_t * (core_x / 2) ** 2)
    Iy += 2 * (wall_t * core_x**3 / 12)

    # Internal wall contribution.
    if wc >= 6:
        Ix += 2 * rect_I(wall_t, 0.45 * core_y)
        Iy += 2 * (0.45 * core_x * wall_t**3 / 12 + 0.45 * core_x * wall_t * (0.22 * core_x) ** 2)

    if wc >= 8:
        Ix += 2 * (0.45 * core_y * wall_t**3 / 12 + 0.45 * core_y * wall_t * (0.22 * core_y) ** 2)
        Iy += 2 * rect_I(wall_t, 0.45 * core_x)

    return (
        inp.factors.wall_cracked_factor * Ix,
        inp.factors.wall_cracked_factor * Iy,
    )


def column_effective_inertia(inp: BuildingInput, interior: float, perimeter: float, corner: float) -> Tuple[float, float]:
    """Approximate total column group inertia for square columns."""
    ni, np_, nc = column_counts(inp)

    I_int = rect_I(interior, interior)
    I_per = rect_I(perimeter, perimeter)
    I_cor = rect_I(corner, corner)

    I_total = ni * I_int + np_ * I_per + nc * I_cor
    I_eff = inp.factors.column_cracked_factor * I_total

    return I_eff, I_eff


def story_lateral_stiffness_from_EI(EI: float, h: float) -> float:
    """
    Convert flexural stiffness to a preliminary storey spring.

    k_story ≈ 12 EI / h³

    This is a shear-building idealization and is acceptable for preliminary
    relative comparison, not final design.
    """
    return 12.0 * EI / max(h**3, 1e-9)


def outrigger_efficiency(system: OutriggerType) -> float:
    """Relative global stiffness efficiency."""
    if system == OutriggerType.BELT_TRUSS:
        return 1.00
    if system == OutriggerType.PIPE_BRACE:
        return 0.70
    return 0.00


def outrigger_material_factor(system: OutriggerType) -> float:
    """Relative steel penalty factor."""
    if system == OutriggerType.BELT_TRUSS:
        return 1.00
    if system == OutriggerType.PIPE_BRACE:
        return 1.15
    return 0.00


def outrigger_story_stiffness(inp: BuildingInput, story: int, outrigger_scale: float) -> Tuple[float, float, float]:
    """
    Calculate equivalent X/Y stiffness contribution from outriggers at a story.

    The real mechanism is core rotation restraint through perimeter column axial action.
    Here it is represented as an equivalent story-level stiffness contribution.
    """
    E = inp.material.Ec_N_m2
    g = inp.geometry
    core_x, core_y, _, _ = core_dimensions(inp)

    arm_x = max(0.5 * (g.plan_x_m - core_x), 1.0)
    arm_y = max(0.5 * (g.plan_y_m - core_y), 1.0)

    kx = 0.0
    ky = 0.0
    steel_kg = 0.0

    for o in inp.outriggers:
        if not o.active or o.system == OutriggerType.NONE or o.story != story:
            continue

        eta = outrigger_efficiency(o.system) * inp.factors.outrigger_connection_factor
        material_factor = outrigger_material_factor(o.system)

        A_chord = o.chord_area_m2 * outrigger_scale
        A_diag = o.diagonal_area_m2 * outrigger_scale

        Lx = sqrt(arm_x**2 + o.depth_m**2)
        Ly = sqrt(arm_y**2 + o.depth_m**2)

        # Axial path approximation.
        k_ax_x = 2 * E * A_chord / arm_x + 2 * E * A_diag / Lx
        k_ax_y = 2 * E * A_chord / arm_y + 2 * E * A_diag / Ly

        kx += eta * k_ax_x
        ky += eta * k_ax_y

        steel_volume = 2 * A_chord * (arm_x + arm_y) + 2 * A_diag * (Lx + Ly)
        steel_kg += steel_volume * RHO_STEEL * material_factor

    return kx, ky, steel_kg


# ============================================================
# 5. MASS AND QUANTITY MODEL
# ============================================================

def story_quantities_and_mass(
    inp: BuildingInput,
    wall_t: float,
    interior_col: float,
    perimeter_col: float,
    corner_col: float,
    slab_t: float,
    beam_b: float,
    beam_h: float,
    outrigger_steel_kg: float,
) -> Tuple[float, float, float, float]:
    """Return mass kg, weight kN, concrete m³, steel kg for one story."""
    g = inp.geometry
    loads = inp.loads

    A = g.floor_area_m2

    slab_vol = A * slab_t

    beam_line_count = g.n_bays_y * (g.n_bays_x + 1) + g.n_bays_x * (g.n_bays_y + 1)
    avg_span = 0.5 * (g.bay_x_m + g.bay_y_m)
    beam_vol = beam_line_count * avg_span * beam_b * beam_h

    ni, np_, nc = column_counts(inp)
    col_vol = (
        ni * interior_col**2 * g.story_height_m
        + np_ * perimeter_col**2 * g.story_height_m
        + nc * corner_col**2 * g.story_height_m
    )

    core_x, core_y, _, _ = core_dimensions(inp)
    wall_vol = 2 * (core_x + core_y) * wall_t * g.story_height_m

    concrete_m3 = slab_vol + beam_vol + col_vol + wall_vol

    rc_steel = (
        inp.slab_rebar_ratio * slab_vol * RHO_STEEL
        + inp.beam_rebar_ratio * beam_vol * RHO_STEEL
        + inp.column_rebar_ratio * col_vol * RHO_STEEL
        + inp.wall_rebar_ratio * wall_vol * RHO_STEEL
    )

    total_steel_kg = rc_steel + outrigger_steel_kg

    structural_weight_kN = concrete_m3 * GAMMA_RC + total_steel_kg * G / 1000

    superimposed_mass_weight_kN = (
        loads.DL_kN_m2
        + loads.finish_kN_m2
        + loads.live_load_mass_factor * loads.LL_kN_m2
    ) * A

    facade_weight_kN = loads.facade_line_kN_m * 2 * (g.plan_x_m + g.plan_y_m)

    weight_kN = structural_weight_kN + superimposed_mass_weight_kN + facade_weight_kN
    mass_kg = weight_kN * 1000 / G

    return mass_kg, weight_kN, concrete_m3, total_steel_kg


# ============================================================
# 6. STORY RESULTS
# ============================================================

def build_story_results(
    inp: BuildingInput,
    core_scale: float,
    column_scale: float,
    outrigger_scale: float,
) -> List[StoryResult]:

    E = inp.material.Ec_N_m2
    h = inp.geometry.story_height_m

    stories: List[StoryResult] = []

    for story in range(1, inp.geometry.n_story + 1):
        zone = get_zone(inp, story)
        wc = wall_count(inp, story)

        wt = story_wall_thickness(inp, story, core_scale)
        interior, perimeter, corner = story_column_dimensions(inp, story, column_scale)
        slab_t = story_slab_thickness(inp, column_scale)
        beam_b, beam_h = story_beam_size(inp, column_scale)

        Ix_core, Iy_core = core_effective_inertia(inp, story, wt)
        Ix_col, Iy_col = column_effective_inertia(inp, interior, perimeter, corner)

        # X translation is resisted by Iy. Y translation is resisted by Ix.
        kx_core = story_lateral_stiffness_from_EI(E * Iy_core, h)
        ky_core = story_lateral_stiffness_from_EI(E * Ix_core, h)

        kx_col = story_lateral_stiffness_from_EI(E * Iy_col, h)
        ky_col = story_lateral_stiffness_from_EI(E * Ix_col, h)

        kx_out, ky_out, out_steel = outrigger_story_stiffness(inp, story, outrigger_scale)

        # Coupling bonus only if outrigger exists at that story.
        if kx_out > 0:
            kx_out += 0.15 * min(kx_core, kx_col)
        if ky_out > 0:
            ky_out += 0.15 * min(ky_core, ky_col)

        mass, weight, concrete, steel = story_quantities_and_mass(
            inp,
            wt,
            interior,
            perimeter,
            corner,
            slab_t,
            beam_b,
            beam_h,
            out_steel,
        )

        stories.append(
            StoryResult(
                story=story,
                elevation_m=story * h,
                zone=zone,
                wall_t_m=wt,
                wall_count=wc,
                interior_col_m=interior,
                perimeter_col_m=perimeter,
                corner_col_m=corner,
                slab_t_m=slab_t,
                beam_b_m=beam_b,
                beam_h_m=beam_h,
                mass_kg=mass,
                weight_kN=weight,
                concrete_m3=concrete,
                steel_kg=steel,
                kx_core_N_m=kx_core,
                ky_core_N_m=ky_core,
                kx_columns_N_m=kx_col,
                ky_columns_N_m=ky_col,
                kx_outrigger_N_m=kx_out,
                ky_outrigger_N_m=ky_out,
                outrigger_steel_kg=out_steel,
            )
        )

    return stories


# ============================================================
# 7. MDOF MODAL ANALYSIS
# ============================================================

def assemble_mk(stories: List[StoryResult], direction: Direction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble M and K for a shear-building MDOF model.

    Equation:
        [K]{phi} = omega²[M]{phi}
    """
    n = len(stories)

    masses = np.array([s.mass_kg for s in stories], dtype=float)
    M = np.diag(masses)

    if direction == Direction.X:
        k_story = np.array([s.kx_total_N_m for s in stories], dtype=float)
    else:
        k_story = np.array([s.ky_total_N_m for s in stories], dtype=float)

    K = np.zeros((n, n), dtype=float)

    # Standard shear-building stiffness matrix.
    for i in range(n):
        k = k_story[i]
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

    # Solve M^-1 K phi = omega² phi.
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    keep = eigvals > 1e-8
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_modes = min(n_modes, len(eigvals))
    omegas = np.sqrt(eigvals[:n_modes])

    periods = [float(2 * pi / w) for w in omegas]
    freqs = [float(w / (2 * pi)) for w in omegas]

    total_mass = np.sum(np.diag(M)).item()
    ones = np.ones((len(stories), 1))

    mode_shapes = []
    eff_mass = []
    cum_eff = []
    cumulative = 0.0

    for i in range(n_modes):
        phi = eigvecs[:, i].reshape(-1, 1)

        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ ones) / denom).item()
        modal_mass = gamma**2 * denom
        ratio = modal_mass / total_mass

        cumulative += ratio

        ph = phi.flatten()
        if abs(ph[-1]) > 1e-12:
            ph = ph / ph[-1]
        if ph[-1] < 0:
            ph *= -1

        mode_shapes.append(ph)
        eff_mass.append(float(ratio))
        cum_eff.append(float(cumulative))

    return ModalResult(
        direction=direction,
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        effective_mass_ratio=eff_mass,
        cumulative_mass_ratio=cum_eff,
    )


# ============================================================
# 8. DRIFT
# ============================================================

def preliminary_base_shear(inp: BuildingInput, total_weight_kN: float) -> float:
    return inp.loads.preliminary_base_shear_coeff * total_weight_kN * 1000


def top_drift(stories: List[StoryResult], base_shear_N: float, direction: Direction) -> float:
    """Approximate top drift using triangular force distribution."""
    n = len(stories)
    heights = np.array([s.elevation_m for s in stories], dtype=float)
    force_shape = heights / max(np.sum(heights), 1e-9)
    lateral_forces = base_shear_N * force_shape

    story_shear = np.zeros(n)
    for i in range(n - 1, -1, -1):
        story_shear[i] = lateral_forces[i] + (story_shear[i + 1] if i < n - 1 else 0)

    if direction == Direction.X:
        k = np.array([s.kx_total_N_m for s in stories])
    else:
        k = np.array([s.ky_total_N_m for s in stories])

    story_drifts = story_shear / np.maximum(k, 1e-9)
    return float(np.sum(story_drifts))


# ============================================================
# 9. DESIGN EVALUATION AND ITERATION
# ============================================================

def evaluate(inp: BuildingInput, core_scale: float, column_scale: float, outrigger_scale: float) -> Dict:
    stories = build_story_results(inp, core_scale, column_scale, outrigger_scale)

    modal_x = solve_modal(stories, Direction.X, n_modes=5)
    modal_y = solve_modal(stories, Direction.Y, n_modes=5)

    tx = modal_x.periods_s[0]
    ty = modal_y.periods_s[0]

    if tx >= ty:
        gov_dir = Direction.X
        gov_T = tx
    else:
        gov_dir = Direction.Y
        gov_T = ty

    T_ref, T_target, T_upper = inp.period.compute(inp.geometry.height_m)

    total_weight = sum(s.weight_kN for s in stories)
    total_mass = sum(s.mass_kg for s in stories)
    total_concrete = sum(s.concrete_m3 for s in stories)
    total_steel = sum(s.steel_kg for s in stories)

    Vb = preliminary_base_shear(inp, total_weight)
    dx = top_drift(stories, Vb, Direction.X)
    dy = top_drift(stories, Vb, Direction.Y)

    return {
        "stories": stories,
        "modal_x": modal_x,
        "modal_y": modal_y,
        "gov_dir": gov_dir,
        "gov_T": gov_T,
        "T_ref": T_ref,
        "T_target": T_target,
        "T_upper": T_upper,
        "period_error": abs(gov_T - T_target) / max(T_target, 1e-9),
        "total_weight": total_weight,
        "total_mass": total_mass,
        "total_concrete": total_concrete,
        "total_steel": total_steel,
        "drift_x": dx,
        "drift_y": dy,
        "drift_x_ratio": dx / inp.geometry.height_m,
        "drift_y_ratio": dy / inp.geometry.height_m,
    }


def run_iteration(inp: BuildingInput, max_iter: int = 30, tolerance: float = 0.025) -> DesignResult:
    errors = validate_input(inp)
    if errors:
        raise ValueError("\n".join(errors))

    core_scale = 1.0
    column_scale = 1.0
    outrigger_scale = 1.0

    best_score = 1e99
    best = None
    logs = []

    for it in range(1, max_iter + 1):
        ev = evaluate(inp, core_scale, column_scale, outrigger_scale)

        drift_max = max(ev["drift_x_ratio"], ev["drift_y_ratio"])
        drift_over = max(drift_max / inp.drift_limit_ratio - 1.0, 0.0)
        period_over = max(ev["gov_T"] / ev["T_upper"] - 1.0, 0.0)

        score = (
            1200 * ev["period_error"] ** 2
            + 1800 * drift_over ** 2
            + 2000 * period_over ** 2
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
                "T target (s)": ev["T_target"],
                "T upper (s)": ev["T_upper"],
                "Period error (%)": 100 * ev["period_error"],
                "Governing dir": ev["gov_dir"].value,
                "Drift X": ev["drift_x_ratio"],
                "Drift Y": ev["drift_y_ratio"],
                "Weight (MN)": ev["total_weight"] / 1000,
                "Concrete (m³)": ev["total_concrete"],
                "Steel (t)": ev["total_steel"] / 1000,
            }
        )

        if score < best_score:
            best_score = score
            best = (core_scale, column_scale, outrigger_scale, ev)

        if (
            ev["period_error"] <= tolerance
            and drift_max <= inp.drift_limit_ratio
            and ev["gov_T"] <= ev["T_upper"]
        ):
            best = (core_scale, column_scale, outrigger_scale, ev)
            break

        # Period correction:
        # T ~ sqrt(M/K), therefore K_req/K_now = (T_now/T_target)^2
        stiffness_ratio = (ev["gov_T"] / max(ev["T_target"], 1e-9)) ** 2

        core_factor = stiffness_ratio ** (0.55 / 3.0)
        column_factor = stiffness_ratio ** (0.30 / 4.0)
        outrigger_factor = stiffness_ratio ** 0.15

        if drift_max > inp.drift_limit_ratio:
            drift_factor = min(1.20, (drift_max / inp.drift_limit_ratio) ** 0.20)
            core_factor *= drift_factor
            column_factor *= drift_factor
            outrigger_factor *= drift_factor

        damping = 0.65

        core_scale *= 1 + damping * (core_factor - 1)
        column_scale *= 1 + damping * (column_factor - 1)
        outrigger_scale *= 1 + damping * (outrigger_factor - 1)

        core_scale = float(np.clip(core_scale, 0.40, 2.50))
        column_scale = float(np.clip(column_scale, 0.40, 2.50))
        outrigger_scale = float(np.clip(outrigger_scale, 0.40, 3.00))

    if best is None:
        raise RuntimeError("No design result was produced.")

    core_scale, column_scale, outrigger_scale, ev = best

    messages = []
    messages.append("MDOF equation solved: [K]{phi} = omega²[M]{phi}.")
    messages.append("Period targeting equation: T_target = T_ref + beta(T_upper - T_ref).")
    messages.append("Outrigger is applied only at its physical story level.")
    messages.append("This is a preliminary framework, not a replacement for full 3D FEM design.")

    if ev["gov_T"] <= ev["T_upper"]:
        messages.append("Upper period check: OK.")
    else:
        messages.append("Upper period check: NOT OK.")

    if max(ev["drift_x_ratio"], ev["drift_y_ratio"]) <= inp.drift_limit_ratio:
        messages.append("Preliminary drift check: OK.")
    else:
        messages.append("Preliminary drift check: NOT OK.")

    return DesignResult(
        input=inp,
        stories=ev["stories"],
        modal_x=ev["modal_x"],
        modal_y=ev["modal_y"],
        T_ref_s=ev["T_ref"],
        T_target_s=ev["T_target"],
        T_upper_s=ev["T_upper"],
        governing_direction=ev["gov_dir"],
        governing_period_s=ev["gov_T"],
        period_error_percent=100 * ev["period_error"],
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
        iteration_df=pd.DataFrame(logs),
        messages=messages,
    )


# ============================================================
# 10. TABLES AND REPORTING
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
                "Kx core (MN/m)": s.kx_core_N_m / 1e6,
                "Kx columns (MN/m)": s.kx_columns_N_m / 1e6,
                "Kx outrigger (MN/m)": s.kx_outrigger_N_m / 1e6,
                "Kx total (MN/m)": s.kx_total_N_m / 1e6,
                "Ky core (MN/m)": s.ky_core_N_m / 1e6,
                "Ky columns (MN/m)": s.ky_columns_N_m / 1e6,
                "Ky outrigger (MN/m)": s.ky_outrigger_N_m / 1e6,
                "Ky total (MN/m)": s.ky_total_N_m / 1e6,
                "Mass (t)": s.mass_kg / 1000,
                "Concrete (m³)": s.concrete_m3,
                "Steel (kg)": s.steel_kg,
                "Outrigger steel (kg)": s.outrigger_steel_kg,
            }
        )
    return pd.DataFrame(rows)


def modal_table(modal: ModalResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mode": range(1, len(modal.periods_s) + 1),
            "Direction": modal.direction.value,
            "Period (s)": modal.periods_s,
            "Frequency (Hz)": modal.frequencies_hz,
            "Effective mass (%)": [100 * x for x in modal.effective_mass_ratio],
            "Cumulative mass (%)": [100 * x for x in modal.cumulative_mass_ratio],
        }
    )


def sustainability_table(res: DesignResult) -> pd.DataFrame:
    area_total = res.input.geometry.floor_area_m2 * res.input.geometry.n_story
    out_steel = sum(s.outrigger_steel_kg for s in res.stories)

    kx_out = sum(s.kx_outrigger_N_m for s in res.stories)
    ky_out = sum(s.ky_outrigger_N_m for s in res.stories)

    return pd.DataFrame(
        {
            "Indicator": [
                "Total concrete",
                "Total steel",
                "Concrete intensity",
                "Steel intensity",
                "Outrigger steel",
                "X stiffness gained per tonne outrigger steel",
                "Y stiffness gained per tonne outrigger steel",
            ],
            "Value": [
                res.total_concrete_m3,
                res.total_steel_kg,
                res.total_concrete_m3 / max(area_total, 1e-9),
                res.total_steel_kg / max(area_total, 1e-9),
                out_steel,
                (kx_out / 1e6) / max(out_steel / 1000, 1e-9),
                (ky_out / 1e6) / max(out_steel / 1000, 1e-9),
            ],
            "Unit": [
                "m³",
                "kg",
                "m³/m²",
                "kg/m²",
                "MN/m per tonne",
                "MN/m per tonne",
                "MN/m per tonne",
            ],
        }
    )


def build_report(res: DesignResult) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("PHD-DEFENSIBLE PRELIMINARY TOWER DESIGN REPORT")
    lines.append("=" * 90)
    lines.append(f"Application version: {APP_VERSION}")
    lines.append("")
    lines.append("1. MODEL PURPOSE")
    lines.append("-" * 90)
    lines.append(
        "This framework performs preliminary tower pre-design using a story-by-story "
        "MDOF shear-building idealization. It is suitable for concept comparison, "
        "target-period evaluation, and preliminary outrigger-system assessment."
    )
    lines.append("")
    lines.append("2. PERIOD TARGETING")
    lines.append("-" * 90)
    lines.append(f"T_ref    = {res.T_ref_s:.4f} s")
    lines.append(f"T_target = {res.T_target_s:.4f} s")
    lines.append(f"T_upper  = {res.T_upper_s:.4f} s")
    lines.append(f"T_MDOF governing = {res.governing_period_s:.4f} s")
    lines.append(f"Governing direction = {res.governing_direction.value}")
    lines.append(f"Period error = {res.period_error_percent:.2f} %")
    lines.append("")
    lines.append("3. DRIFT SCREENING")
    lines.append("-" * 90)
    lines.append(f"Top drift X = {res.drift_x_m:.5f} m")
    lines.append(f"Top drift Y = {res.drift_y_m:.5f} m")
    lines.append(f"Drift ratio X = {res.drift_x_ratio:.6f}")
    lines.append(f"Drift ratio Y = {res.drift_y_ratio:.6f}")
    lines.append(f"Allowable drift ratio = {res.input.drift_limit_ratio:.6f}")
    lines.append("")
    lines.append("4. MATERIAL QUANTITIES")
    lines.append("-" * 90)
    lines.append(f"Total weight = {res.total_weight_kN:,.1f} kN")
    lines.append(f"Total mass = {res.total_mass_kg:,.1f} kg")
    lines.append(f"Total concrete = {res.total_concrete_m3:,.1f} m³")
    lines.append(f"Total steel = {res.total_steel_kg:,.1f} kg")
    lines.append("")
    lines.append("5. FINAL SECTION SCALE FACTORS")
    lines.append("-" * 90)
    lines.append(f"Core scale = {res.core_scale:.3f}")
    lines.append(f"Column scale = {res.column_scale:.3f}")
    lines.append(f"Outrigger scale = {res.outrigger_scale:.3f}")
    lines.append("")
    lines.append("6. OUTRIGGER SYSTEMS")
    lines.append("-" * 90)
    if res.input.outriggers:
        for o in res.input.outriggers:
            lines.append(
                f"Story {o.story}: {o.system.value}, depth={o.depth_m:.2f} m, "
                f"A_chord={o.chord_area_m2:.4f} m², A_diag={o.diagonal_area_m2:.4f} m²"
            )
    else:
        lines.append("No outrigger system.")
    lines.append("")
    lines.append("7. ENGINEERING CONCLUSION")
    lines.append("-" * 90)
    lines.append(
        "For preliminary tower design, a belt-truss outrigger is usually more efficient "
        "than pipe bracing for global period and drift control, because it creates a clearer "
        "load path between core rotation and perimeter-column axial action. Pipe bracing may "
        "still be useful under architectural or constructability constraints, but its lower "
        "coupling efficiency should be verified by stiffness gain per unit steel."
    )
    lines.append("")
    lines.append("8. LIMITATIONS")
    lines.append("-" * 90)
    lines.append(
        "This framework is not a final design program. Final design must use calibrated "
        "3D analysis including torsion, diaphragm flexibility, P-Delta effects, wind and "
        "earthquake load combinations, foundation flexibility, and detailed member design."
    )
    lines.append("")
    lines.append("9. STATUS MESSAGES")
    lines.append("-" * 90)
    for msg in res.messages:
        lines.append(f"- {msg}")

    return "\n".join(lines)


# ============================================================
# 11. PLOTS
# ============================================================

def plot_stiffness(res: DesignResult, direction: Direction):
    df = story_table(res)

    fig, ax = plt.subplots(figsize=(8, 8))

    if direction == Direction.X:
        ax.plot(df["Kx core (MN/m)"], df["Story"], label="Core")
        ax.plot(df["Kx columns (MN/m)"], df["Story"], label="Columns")
        ax.plot(df["Kx outrigger (MN/m)"], df["Story"], label="Outrigger")
        ax.plot(df["Kx total (MN/m)"], df["Story"], linewidth=2.5, label="Total")
    else:
        ax.plot(df["Ky core (MN/m)"], df["Story"], label="Core")
        ax.plot(df["Ky columns (MN/m)"], df["Story"], label="Columns")
        ax.plot(df["Ky outrigger (MN/m)"], df["Story"], label="Outrigger")
        ax.plot(df["Ky total (MN/m)"], df["Story"], linewidth=2.5, label="Total")

    ax.set_xlabel("Story stiffness (MN/m)")
    ax.set_ylabel("Story")
    ax.set_title(f"Story stiffness profile - Direction {direction.value}")
    ax.grid(True, alpha=0.30)
    ax.legend()
    return fig


def plot_modes(res: DesignResult, modal: ModalResult):
    y = np.array([s.elevation_m for s in res.stories])
    n_modes = len(modal.mode_shapes)

    fig, axes = plt.subplots(1, n_modes, figsize=(16, 6))
    if n_modes == 1:
        axes = [axes]

    out_levels = [o.story for o in res.input.outriggers if o.active]

    for i in range(n_modes):
        ax = axes[i]
        phi = modal.mode_shapes[i]
        ax.plot(phi, y, linewidth=2)
        ax.scatter(phi, y, s=14)
        ax.axvline(0, linestyle="--", linewidth=0.8)

        for story in out_levels:
            ax.axhline(story * res.input.geometry.story_height_m, linestyle=":", alpha=0.6)

        ax.set_title(f"Mode {i+1}\nT={modal.periods_s[i]:.3f}s")
        ax.set_xlabel("Normalized displacement")
        if i == 0:
            ax.set_ylabel("Height (m)")
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Mode shapes - Direction {modal.direction.value}")
    fig.tight_layout()
    return fig


def plot_iteration(res: DesignResult):
    df = res.iteration_df

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(df["Iteration"], df["T governing (s)"], marker="o", label="T governing")
    axes[0, 0].plot(df["Iteration"], df["T target (s)"], linestyle="--", label="Target")
    axes[0, 0].plot(df["Iteration"], df["T upper (s)"], linestyle=":", label="Upper")
    axes[0, 0].set_title("Period convergence")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Period (s)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(df["Iteration"], df["Period error (%)"], marker="s")
    axes[0, 1].axhline(2.5, linestyle="--")
    axes[0, 1].set_title("Period error")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Error (%)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df["Iteration"], df["Core scale"], marker="o", label="Core")
    axes[1, 0].plot(df["Iteration"], df["Column scale"], marker="s", label="Column")
    axes[1, 0].plot(df["Iteration"], df["Outrigger scale"], marker="^", label="Outrigger")
    axes[1, 0].set_title("Section scale factors")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Scale")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(df["Iteration"], df["Steel (t)"], marker="d")
    axes[1, 1].set_title("Steel trend")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Steel (t)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_plan(res: DesignResult, story: int):
    g = res.input.geometry
    s = next(item for item in res.stories if item.story == story)

    core_x, core_y, open_x, open_y = core_dimensions(res.input)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot([0, g.plan_x_m, g.plan_x_m, 0, 0], [0, 0, g.plan_y_m, g.plan_y_m, 0], color="black")

    for i in range(g.n_bays_x + 1):
        x = i * g.bay_x_m
        ax.plot([x, x], [0, g.plan_y_m], color="#dddddd", linewidth=0.7)

    for j in range(g.n_bays_y + 1):
        y = j * g.bay_y_m
        ax.plot([0, g.plan_x_m], [y, y], color="#dddddd", linewidth=0.7)

    for i in range(g.n_bays_x + 1):
        for j in range(g.n_bays_y + 1):
            x = i * g.bay_x_m
            y = j * g.bay_y_m

            is_corner = (i in [0, g.n_bays_x]) and (j in [0, g.n_bays_y])
            is_perimeter = (i in [0, g.n_bays_x] or j in [0, g.n_bays_y]) and not is_corner

            if is_corner:
                dim = s.corner_col_m
            elif is_perimeter:
                dim = s.perimeter_col_m
            else:
                dim = s.interior_col_m

            ax.add_patch(
                plt.Rectangle(
                    (x - dim / 2, y - dim / 2),
                    dim,
                    dim,
                    facecolor="#8b0000",
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.85,
                )
            )

    cx0 = 0.5 * (g.plan_x_m - core_x)
    cy0 = 0.5 * (g.plan_y_m - core_y)
    ox0 = 0.5 * (g.plan_x_m - open_x)
    oy0 = 0.5 * (g.plan_y_m - open_y)

    ax.add_patch(plt.Rectangle((cx0, cy0), core_x, core_y, fill=False, edgecolor="#2e8b57", linewidth=2.5))
    ax.add_patch(plt.Rectangle((ox0, oy0), open_x, open_y, fill=False, edgecolor="#2e8b57", linewidth=1.2, linestyle="--"))

    active_out = [o for o in res.input.outriggers if o.active and o.story == story and o.system != OutriggerType.NONE]

    if active_out:
        mx = g.plan_x_m / 2
        my = g.plan_y_m / 2
        ax.plot([0, cx0], [my, my], color="#ff8c00", linewidth=4.5)
        ax.plot([cx0 + core_x, g.plan_x_m], [my, my], color="#ff8c00", linewidth=4.5)
        ax.plot([mx, mx], [0, cy0], color="#ff8c00", linewidth=4.5)
        ax.plot([mx, mx], [cy0 + core_y, g.plan_y_m], color="#ff8c00", linewidth=4.5)
        ax.text(mx, my, active_out[0].system.value, ha="center", va="center", fontsize=9)

    ax.set_title(
        f"Plan view - Story {story} | Zone: {s.zone}\n"
        f"Wall t={s.wall_t_m:.2f} m | Interior col={s.interior_col_m:.2f} m | Corner col={s.corner_col_m:.2f} m"
    )
    ax.set_aspect("equal")
    ax.set_xlim(-2, g.plan_x_m + 2)
    ax.set_ylim(-2, g.plan_y_m + 2)
    return fig


# ============================================================
# 12. STREAMLIT INTERFACE
# ============================================================

def read_inputs_from_sidebar() -> BuildingInput:
    st.sidebar.header("1. Geometry")

    g = Geometry(
        n_story=st.sidebar.number_input("Stories", 10, 120, 60),
        story_height_m=st.sidebar.number_input("Story height (m)", 2.8, 5.0, 3.2),
        plan_x_m=st.sidebar.number_input("Plan X (m)", 20.0, 200.0, 48.0),
        plan_y_m=st.sidebar.number_input("Plan Y (m)", 20.0, 200.0, 42.0),
        n_bays_x=st.sidebar.number_input("Bays X", 2, 20, 6),
        n_bays_y=st.sidebar.number_input("Bays Y", 2, 20, 6),
        core_ratio_x=st.sidebar.number_input("Core ratio X", 0.10, 0.45, 0.24),
        core_ratio_y=st.sidebar.number_input("Core ratio Y", 0.10, 0.45, 0.22),
        core_opening_ratio_x=st.sidebar.number_input("Core opening ratio X", 0.05, 0.35, 0.16),
        core_opening_ratio_y=st.sidebar.number_input("Core opening ratio Y", 0.05, 0.35, 0.14),
    )

    st.sidebar.header("2. Materials")
    material = Material(
        Ec_MPa=st.sidebar.number_input("Ec (MPa)", 20000.0, 60000.0, 34000.0),
        fck_MPa=st.sidebar.number_input("fck (MPa)", 25.0, 100.0, 60.0),
        fy_MPa=st.sidebar.number_input("fy (MPa)", 300.0, 700.0, 420.0),
    )

    st.sidebar.header("3. Loads")
    loads = Loads(
        DL_kN_m2=st.sidebar.number_input("DL (kN/m²)", 0.0, 15.0, 3.5),
        LL_kN_m2=st.sidebar.number_input("LL (kN/m²)", 0.0, 10.0, 2.5),
        finish_kN_m2=st.sidebar.number_input("Finish (kN/m²)", 0.0, 8.0, 1.5),
        live_load_mass_factor=st.sidebar.number_input("Live load mass factor", 0.0, 1.0, 0.25),
        facade_line_kN_m=st.sidebar.number_input("Facade line load (kN/m)", 0.0, 20.0, 1.2),
        preliminary_base_shear_coeff=st.sidebar.number_input("Prelim base shear coeff.", 0.001, 0.100, 0.015),
    )

    st.sidebar.header("4. Section limits")
    drift_den = st.sidebar.number_input("Drift denominator", 250.0, 2000.0, 500.0)

    limits = SectionLimits(
        min_wall_t_m=st.sidebar.number_input("Min wall t (m)", 0.20, 1.00, 0.30),
        max_wall_t_m=st.sidebar.number_input("Max wall t (m)", 0.40, 2.00, 1.20),
        min_col_dim_m=st.sidebar.number_input("Min col dim (m)", 0.40, 2.00, 0.70),
        max_col_dim_m=st.sidebar.number_input("Max col dim (m)", 0.80, 3.00, 1.80),
        min_slab_t_m=st.sidebar.number_input("Min slab t (m)", 0.15, 0.50, 0.22),
        max_slab_t_m=st.sidebar.number_input("Max slab t (m)", 0.20, 0.70, 0.40),
        min_beam_b_m=st.sidebar.number_input("Min beam b (m)", 0.25, 1.20, 0.40),
        max_beam_b_m=st.sidebar.number_input("Max beam b (m)", 0.40, 1.50, 1.00),
        min_beam_h_m=st.sidebar.number_input("Min beam h (m)", 0.50, 2.00, 0.75),
        max_beam_h_m=st.sidebar.number_input("Max beam h (m)", 0.80, 3.00, 1.80),
    )

    st.sidebar.header("5. Effective stiffness")
    factors = StiffnessFactors(
        wall_cracked_factor=st.sidebar.number_input("Wall cracked factor", 0.05, 1.00, 0.35),
        column_cracked_factor=st.sidebar.number_input("Column cracked factor", 0.05, 1.00, 0.70),
        outrigger_connection_factor=st.sidebar.number_input("Outrigger connection factor", 0.30, 1.00, 0.80),
    )

    st.sidebar.header("6. Period target")
    period = PeriodTarget(
        Ct=st.sidebar.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f"),
        x_exp=st.sidebar.number_input("x exponent", 0.10, 1.50, 0.75),
        upper_factor=st.sidebar.number_input("Upper period factor", 1.00, 2.00, 1.20),
        beta=st.sidebar.number_input("Target beta", 0.10, 0.95, 0.80),
    )

    st.sidebar.header("7. Structural layout")
    lower_wc = st.sidebar.number_input("Lower wall count", 4, 8, 8)
    middle_wc = st.sidebar.number_input("Middle wall count", 4, 8, 6)
    upper_wc = st.sidebar.number_input("Upper wall count", 4, 8, 4)
    perim_col_factor = st.sidebar.number_input("Perimeter column factor", 1.00, 2.00, 1.10)
    corner_col_factor = st.sidebar.number_input("Corner column factor", 1.00, 2.00, 1.30)

    st.sidebar.header("8. Outrigger systems")
    n_out = st.sidebar.number_input("Number of outriggers", 0, 5, 2)

    out_list: List[OutriggerInput] = []
    default_stories = [round(0.50 * g.n_story), round(0.70 * g.n_story), round(0.33 * g.n_story)]

    for i in range(int(n_out)):
        st.sidebar.markdown(f"**Outrigger {i+1}**")

        default_story = default_stories[i] if i < len(default_stories) else round((i + 1) * g.n_story / (int(n_out) + 1))

        story = st.sidebar.number_input(f"Outrigger story {i+1}", 1, int(g.n_story), int(default_story), key=f"out_story_{i}")

        system_name = st.sidebar.selectbox(
            f"Outrigger type {i+1}",
            [OutriggerType.BELT_TRUSS.value, OutriggerType.PIPE_BRACE.value],
            index=0,
            key=f"out_type_{i}",
        )

        depth = st.sidebar.number_input(f"Depth {i+1} (m)", 1.0, 8.0, 3.0, key=f"out_depth_{i}")
        A_chord = st.sidebar.number_input(f"Chord area {i+1} (m²)", 0.005, 0.500, 0.08, key=f"out_chord_{i}")
        A_diag = st.sidebar.number_input(f"Diagonal area {i+1} (m²)", 0.005, 0.500, 0.04, key=f"out_diag_{i}")

        out_list.append(
            OutriggerInput(
                story=int(story),
                system=OutriggerType(system_name),
                depth_m=float(depth),
                chord_area_m2=float(A_chord),
                diagonal_area_m2=float(A_diag),
                active=True,
            )
        )

    return BuildingInput(
        material=material,
        geometry=g,
        loads=loads,
        limits=limits,
        factors=factors,
        period=period,
        drift_limit_ratio=1 / float(drift_den),
        lower_wall_count=int(lower_wc),
        middle_wall_count=int(middle_wc),
        upper_wall_count=int(upper_wc),
        perimeter_column_factor=float(perim_col_factor),
        corner_column_factor=float(corner_col_factor),
        outriggers=out_list,
    )


def main():
    st.set_page_config(page_title="PhD Tower MDOF Pre-Design", layout="wide")

    st.title("PhD-Defensible Tower Pre-Design Framework")
    st.caption(APP_VERSION)

    st.markdown(
        """
        This app is organized as a transparent preliminary structural framework:
        **inputs → section stiffness → mass matrix → stiffness matrix → modal periods → target period check → sustainability indicators**.
        """
    )

    inp = read_inputs_from_sidebar()

    if "result_v8" not in st.session_state:
        st.session_state.result_v8 = None
    if "report_v8" not in st.session_state:
        st.session_state.report_v8 = ""

    if st.button("Run professional MDOF analysis", type="primary"):
        try:
            with st.spinner("Running MDOF analysis and section-stiffness iteration..."):
                res = run_iteration(inp)
                st.session_state.result_v8 = res
                st.session_state.report_v8 = build_report(res)
            st.success("Analysis completed.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    res = st.session_state.result_v8

    if res is None:
        st.info("Set inputs from the sidebar and run the analysis.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T ref (s)", f"{res.T_ref_s:.3f}")
    c2.metric("T target (s)", f"{res.T_target_s:.3f}")
    c3.metric("T MDOF gov. (s)", f"{res.governing_period_s:.3f}")
    c4.metric("Error (%)", f"{res.period_error_percent:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Gov. direction", res.governing_direction.value)
    c6.metric("Drift X", f"{res.drift_x_ratio:.5f}")
    c7.metric("Drift Y", f"{res.drift_y_ratio:.5f}")
    c8.metric("Weight (MN)", f"{res.total_weight_kN/1000:.1f}")

    tabs = st.tabs(
        [
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
        ]
    )

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
        st.dataframe(res.iteration_df, use_container_width=True, hide_index=True)

    with tabs[5]:
        selected_story = st.slider("Story for plan view", 1, res.input.geometry.n_story, max(1, res.input.geometry.n_story // 2))
        st.pyplot(plot_plan(res, selected_story), use_container_width=True)

    with tabs[6]:
        st.dataframe(story_table(res), use_container_width=True, hide_index=True)

    with tabs[7]:
        st.markdown("### X Direction")
        st.dataframe(modal_table(res.modal_x), use_container_width=True, hide_index=True)
        st.markdown("### Y Direction")
        st.dataframe(modal_table(res.modal_y), use_container_width=True, hide_index=True)

    with tabs[8]:
        st.dataframe(sustainability_table(res), use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Interpretation for defense:**  
            A sustainable structural option is not only the lightest one. It is the option that
            reaches the target period and drift requirements with the lowest reliable material demand
            and a rational load path.
            """
        )

    with tabs[9]:
        st.text_area("Generated report", st.session_state.report_v8, height=540)
        st.download_button(
            "Download report",
            data=st.session_state.report_v8.encode("utf-8"),
            file_name="phd_tower_mdof_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
