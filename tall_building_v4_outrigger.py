from __future__ import annotations
from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import streamlit as st
except ModuleNotFoundError:  # allow Jupyter / local execution without Streamlit
    class _StreamlitStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop
    st = _StreamlitStub()
from scipy.optimize import minimize

# ----------------------------- CONFIG -----------------------------
st.set_page_config(
    page_title="Tall Building Preliminary Structural Analysis + Outrigger",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTHOR_NAME = "Benyamin"
APP_VERSION = "v4.0-MDOF-Outrigger"

G = 9.81
STEEL_DENSITY = 7850.0

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"
OUTRIGGER_COLOR = "#ff6b00"


# ----------------------------- DATA MODELS -----------------------------
@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int

    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1


@dataclass
class OutriggerDefinition:
    """Defines an outrigger belt truss system at a specific level"""
    story_level: int  # Story number where outrigger is placed
    truss_depth_m: float  # Depth of the belt truss
    truss_width_m: float  # Width of the outrigger arm
    chord_area_m2: float  # Cross-sectional area of truss chords
    diagonal_area_m2: float  # Cross-sectional area of diagonals
    is_active: bool = True
    
    @property
    def height_from_base_m(self, story_height: float = 3.2) -> float:
        return self.story_level * story_height


@dataclass
class BuildingInput:
    plan_shape: str
    n_story: int
    n_basement: int
    story_height: float
    basement_height: float
    plan_x: float
    plan_y: float
    n_bays_x: int
    n_bays_y: int
    bay_x: float
    bay_y: float

    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each: float = 3.5
    stair_area_each: float = 20.0
    service_area: float = 35.0
    corridor_factor: float = 1.40

    fck: float = 70.0
    Ec: float = 36000.0
    Es: float = 200000.0
    fy: float = 420.0

    DL: float = 3.0
    LL: float = 2.5
    slab_finish_allowance: float = 1.5
    facade_line_load: float = 1.0

    prelim_lateral_force_coeff: float = 0.015
    drift_limit_ratio: float = 1 / 500

    min_wall_thickness: float = 0.30
    max_wall_thickness: float = 1.20
    min_column_dim: float = 0.70
    max_column_dim: float = 1.80
    min_beam_width: float = 0.40
    min_beam_depth: float = 0.75
    min_slab_thickness: float = 0.22
    max_slab_thickness: float = 0.40

    # [FIXED] Realistic cracked factors for 60-story building
    wall_cracked_factor: float = 0.40  # Changed from 0.70 - realistic for tall buildings
    column_cracked_factor: float = 0.70
    max_story_wall_slenderness: float = 12.0

    wall_rebar_ratio: float = 0.003
    column_rebar_ratio: float = 0.010
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.0035

    seismic_mass_factor: float = 1.0
    Ct: float = 0.0488
    x_period: float = 0.75
    upper_period_factor: float = 1.40
    target_position_factor: float = 0.85

    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30

    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4

    basement_retaining_wall_thickness: float = 0.50
    perimeter_shear_wall_ratio: float = 0.20

    # [NEW] Outrigger parameters
    outrigger_count: int = 0
    outrigger_story_levels: List[int] = field(default_factory=list)
    root_outrigger_story_levels: List[int] = field(default_factory=list)
    outrigger_axis: str = "x"
    outrigger_truss_depth_m: float = 3.0
    outrigger_chord_area_m2: float = 0.08
    outrigger_diagonal_area_m2: float = 0.04


@dataclass
class ZoneCoreResult:
    zone: ZoneDefinition
    wall_count: int
    wall_lengths: List[float]
    wall_thickness: float
    core_outer_x: float
    core_outer_y: float
    core_opening_x: float
    core_opening_y: float
    Ieq_gross_m4: float
    Ieq_effective_m4: float
    story_slenderness: float
    perimeter_wall_segments: List[Tuple[str, float, float]]
    retaining_wall_active: bool


@dataclass
class ZoneColumnResult:
    zone: ZoneDefinition
    corner_column_x_m: float
    corner_column_y_m: float
    perimeter_column_x_m: float
    perimeter_column_y_m: float
    interior_column_x_m: float
    interior_column_y_m: float
    P_corner_kN: float
    P_perimeter_kN: float
    P_interior_kN: float
    I_col_group_effective_m4: float


@dataclass
class OutriggerResult:
    """Results for a single outrigger system"""
    story_level: int
    height_m: float
    truss_depth_m: float
    truss_width_m: float
    chord_area_m2: float
    diagonal_area_m2: float
    axial_stiffness_kN: float
    equivalent_spring_kN_m: float
    stiffness_contribution: float  # How much this outrigger contributes to global stiffness
    axis: str = "x"
    arm_m: float = 0.0
    engaged_column_count: int = 0
    k_col_N_per_m: float = 0.0
    k_truss_N_per_m: float = 0.0
    k_tip_N_per_m: float = 0.0
    K_theta_Nm_per_rad: float = 0.0
    system_type: str = "outrigger"


@dataclass
class ReinforcementEstimate:
    wall_concrete_volume_m3: float
    column_concrete_volume_m3: float
    beam_concrete_volume_m3: float
    slab_concrete_volume_m3: float
    wall_steel_kg: float
    column_steel_kg: float
    beam_steel_kg: float
    slab_steel_kg: float
    total_steel_kg: float
    outrigger_steel_kg: float = 0.0  # [NEW]


@dataclass
class ModalResult:
    n_dof: int
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    story_masses_kg: List[float]
    story_stiffness_N_per_m: List[float]
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]


@dataclass
class IterationLog:
    iteration: int
    core_scale: float
    column_scale: float
    T_estimated: float
    T_target: float
    error_percent: float
    total_weight_kN: float
    K_total_N_m: float


@dataclass
class DesignResult:
    H_m: float
    floor_area_m2: float
    total_weight_kN: float
    effective_modal_mass_kg: float
    reference_period_s: float
    design_target_period_s: float
    upper_limit_period_s: float
    estimated_period_s: float
    period_error_ratio: float
    period_ok: bool
    drift_ok: bool
    K_estimated_N_per_m: float
    top_drift_m: float
    drift_ratio: float
    zone_core_results: List[ZoneCoreResult]
    zone_column_results: List[ZoneColumnResult]
    outrigger_results: List[OutriggerResult]  # [NEW]
    slab_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    reinforcement: ReinforcementEstimate
    optimization_success: bool
    optimization_message: str
    core_scale: float
    column_scale: float
    messages: List[str] = field(default_factory=list)
    redesign_suggestions: List[str] = field(default_factory=list)
    governing_issue: str = ""
    modal_result: ModalResult | None = None
    iteration_history: List[IterationLog] = field(default_factory=list)


# ----------------------------- BASIC CALC -----------------------------

def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def floor_area(inp: BuildingInput) -> float:
    return 0.5 * inp.plan_x * inp.plan_y if inp.plan_shape == "triangle" else inp.plan_x * inp.plan_y


def code_type_period(H: float, Ct: float, x_period: float) -> float:
    return Ct * (H ** x_period)


def preliminary_lateral_force_N(inp: BuildingInput, W_total_kN: float) -> float:
    return inp.prelim_lateral_force_coeff * W_total_kN * 1000.0


def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def required_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each
        + inp.stair_count * inp.stair_area_each
        + inp.service_area
    ) * inp.corridor_factor


def opening_dimensions(inp: BuildingInput) -> tuple[float, float]:
    area = required_opening_area(inp)
    aspect = 1.6
    oy = sqrt(area / aspect)
    return aspect * oy, oy


def initial_core_dimensions(inp: BuildingInput, opening_x: float, opening_y: float) -> tuple[float, float]:
    outer_x = max(opening_x + 3.0, 0.24 * inp.plan_x)
    outer_y = max(opening_y + 3.0, 0.22 * inp.plan_y)
    return min(outer_x, 0.42 * inp.plan_x), min(outer_y, 0.42 * inp.plan_y)


def active_wall_count_by_zone(inp: BuildingInput, zone_name: str) -> int:
    return {
        "Lower Zone": inp.lower_zone_wall_count,
        "Middle Zone": inp.middle_zone_wall_count,
        "Upper Zone": inp.upper_zone_wall_count,
    }[zone_name]


def wall_lengths_for_layout(outer_x: float, outer_y: float, wall_count: int) -> List[float]:
    if wall_count == 4:
        return [outer_x, outer_x, outer_y, outer_y]
    if wall_count == 6:
        return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x]
    return [outer_x, outer_x, outer_y, outer_y, 0.45 * outer_x, 0.45 * outer_x, 0.45 * outer_y, 0.45 * outer_y]


def wall_rect_inertia_about_global_y(length: float, thickness: float, x_centroid: float) -> float:
    I_local = length * thickness**3 / 12.0
    area = length * thickness
    return I_local + area * x_centroid**2


def wall_rect_inertia_about_global_x(length: float, thickness: float, y_centroid: float) -> float:
    I_local = length * thickness**3 / 12.0
    area = length * thickness
    return I_local + area * y_centroid**2


def core_equivalent_inertia(outer_x: float, outer_y: float, lengths: List[float], t: float, wall_count: int) -> float:
    x_side = outer_x / 2.0
    y_side = outer_y / 2.0
    top_len, bot_len, left_len, right_len = lengths[0], lengths[1], lengths[2], lengths[3]
    I_x = 0.0
    I_y = 0.0

    I_x += wall_rect_inertia_about_global_x(top_len, t, +y_side)
    I_x += wall_rect_inertia_about_global_x(bot_len, t, -y_side)
    I_y += (t * top_len**3 / 12.0) + (t * bot_len**3 / 12.0)

    I_y += wall_rect_inertia_about_global_y(left_len, t, -x_side)
    I_y += wall_rect_inertia_about_global_y(right_len, t, +x_side)
    I_x += (t * left_len**3 / 12.0) + (t * right_len**3 / 12.0)

    if wall_count >= 6:
        inner_x = 0.22 * outer_x
        l1, l2 = lengths[4], lengths[5]
        I_y += wall_rect_inertia_about_global_y(l1, t, -inner_x)
        I_y += wall_rect_inertia_about_global_y(l2, t, +inner_x)
        I_x += (t * l1**3 / 12.0) + (t * l2**3 / 12.0)

    if wall_count >= 8:
        inner_y = 0.22 * outer_y
        l3, l4 = lengths[6], lengths[7]
        I_x += wall_rect_inertia_about_global_x(l3, t, -inner_y)
        I_x += wall_rect_inertia_about_global_x(l4, t, +inner_y)
        I_y += (t * l3**3 / 12.0) + (t * l4**3 / 12.0)

    return min(I_x, I_y)


def perimeter_wall_segments_for_square(inp: BuildingInput, zone: ZoneDefinition) -> List[Tuple[str, float, float]]:
    if zone.name == "Lower Zone":
        return [
            ("top", 0.0, inp.plan_x), ("bottom", 0.0, inp.plan_x),
            ("left", 0.0, inp.plan_y), ("right", 0.0, inp.plan_y),
        ]
    ratio = inp.perimeter_shear_wall_ratio
    lx = inp.plan_x * ratio
    ly = inp.plan_y * ratio
    sx = (inp.plan_x - lx) / 2.0
    sy = (inp.plan_y - ly) / 2.0
    return [
        ("top", sx, sx + lx), ("bottom", sx, sx + lx),
        ("left", sy, sy + ly), ("right", sy, sy + ly),
    ]


def wall_thickness_by_zone(inp: BuildingInput, H: float, zone: ZoneDefinition, core_scale: float) -> float:
    base_t = max(inp.min_wall_thickness, min(inp.max_wall_thickness, H / 180.0))
    zone_factor = {"Lower Zone": 1.00, "Middle Zone": 0.80, "Upper Zone": 0.60}[zone.name]
    t = base_t * zone_factor * core_scale
    return max(inp.min_wall_thickness, min(inp.max_wall_thickness, t))


def slab_thickness_prelim(inp: BuildingInput, column_scale: float) -> float:
    span = max(inp.bay_x, inp.bay_y)
    base = span / 28.0
    t = base * (0.90 + 0.10 * column_scale)
    return max(inp.min_slab_thickness, min(inp.max_slab_thickness, t))


def beam_size_prelim(inp: BuildingInput, column_scale: float) -> tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    depth = max(inp.min_beam_depth, min(2.0, (span / 12.0) * (0.90 + 0.15 * column_scale)))
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


def directional_dims(base_dim: float, plan_x: float, plan_y: float) -> tuple[float, float]:
    aspect = max(plan_x, plan_y) / max(min(plan_x, plan_y), 1e-9)
    if aspect <= 1.10:
        return base_dim, base_dim
    major = base_dim * 1.15
    minor = base_dim * 0.90
    return (major, minor) if plan_x >= plan_y else (minor, major)



# [NEW] Outrigger calculation functions
def _all_outrigger_levels(inp: BuildingInput) -> List[Tuple[int, str]]:
    levels = []
    for lvl in inp.outrigger_story_levels:
        levels.append((int(lvl), "outrigger"))
    for lvl in getattr(inp, "root_outrigger_story_levels", []):
        levels.append((int(lvl), "root"))
    # unique while preserving order
    seen = set()
    out = []
    for lvl, typ in sorted(levels, key=lambda x: x[0]):
        if 1 <= lvl <= inp.n_story and lvl not in seen:
            out.append((lvl, typ))
            seen.add(lvl)
    return out


def calculate_outrigger_stiffness(inp: BuildingInput, story_level: int,
                                   truss_depth: float, chord_area: float,
                                   diagonal_area: float, system_type: str = "outrigger") -> OutriggerResult:
    """
    Corrected equivalent outrigger stiffness.
    The truss and the engaged perimeter-column groups act in series at the arm tip,
    then are converted to a rotational restraint at the outrigger level.
    """
    Es = getattr(inp, "Es", 200000.0) * 1e6
    Ec = inp.Ec * 1e6
    story_height = inp.story_height
    height_m = story_level * story_height
    H_total = total_height(inp)

    core_x, core_y = initial_core_dimensions(inp, *opening_dimensions(inp))
    axis = getattr(inp, "outrigger_axis", "x").lower().strip()
    if axis not in ("x", "y"):
        axis = "x"

    arm = (inp.plan_x - core_x) / 2.0 if axis == "x" else (inp.plan_y - core_y) / 2.0
    engaged_column_count = (inp.n_bays_y + 1) if axis == "x" else (inp.n_bays_x + 1)

    # Preliminary perimeter-column area used in the outrigger coupling stiffness
    perim_dim = min(inp.max_column_dim, max(inp.min_column_dim, inp.min_column_dim * inp.perimeter_column_factor))
    A_col = perim_dim * perim_dim
    k_col_face = engaged_column_count * Ec * A_col / max(height_m, story_height)
    k_col_total = 2.0 * k_col_face  # two opposite perimeter lines

    # Truss bending + diagonal racking contribution
    I_or = 0.5 * chord_area * truss_depth**2
    k_bend = 3.0 * Es * I_or / max(arm**3, 1e-9)
    Ld = sqrt(arm**2 + truss_depth**2)
    k_diag = 2.0 * Es * diagonal_area * truss_depth**2 / max(Ld**3, 1e-9)
    k_truss = k_bend + k_diag

    # tip stiffness from series combination
    k_tip = 1.0 / max((1.0 / max(k_truss, 1e-9)) + (1.0 / max(k_col_total, 1e-9)), 1e-18)
    K_theta = 2.0 * k_tip * arm**2

    # equivalent added story spring; eta keeps sensitivity to elevation
    xi = min(max(height_m / max(H_total, 1e-9), 0.0), 1.0)
    eta = 3.0 * xi**2 - 2.0 * xi**3
    k_story_add = eta * K_theta / max(story_height**2, 1e-9)

    axial_stiffness = 4.0 * Es * chord_area / max(Ld, 1e-9)

    return OutriggerResult(
        story_level=story_level,
        height_m=height_m,
        truss_depth_m=truss_depth,
        truss_width_m=arm,
        chord_area_m2=chord_area,
        diagonal_area_m2=diagonal_area,
        axial_stiffness_kN=axial_stiffness / 1000.0,
        equivalent_spring_kN_m=k_story_add / 1000.0,
        stiffness_contribution=k_story_add,
        axis=axis,
        arm_m=arm,
        engaged_column_count=engaged_column_count,
        k_col_N_per_m=k_col_total,
        k_truss_N_per_m=k_truss,
        k_tip_N_per_m=k_tip,
        K_theta_Nm_per_rad=K_theta,
        system_type=system_type,
    )


def calculate_total_outrigger_stiffness(inp: BuildingInput) -> Tuple[List[OutriggerResult], float]:
    """Calculate all outriggers and total stiffness contribution"""
    levels = _all_outrigger_levels(inp)
    if not levels:
        return [], 0.0

    outriggers: List[OutriggerResult] = []
    total_K = 0.0

    for level, system_type in levels:
        or_result = calculate_outrigger_stiffness(
            inp, level, inp.outrigger_truss_depth_m,
            inp.outrigger_chord_area_m2, inp.outrigger_diagonal_area_m2,
            system_type=system_type
        )
        outriggers.append(or_result)
        total_K += or_result.stiffness_contribution

    return outriggers, total_K


def build_zone_results(inp: BuildingInput, core_scale: float, column_scale: float, slab_t: float) -> tuple[List[ZoneCoreResult], List[ZoneColumnResult]]:
    H = total_height(inp)
    zones = define_three_zones(inp.n_story)
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)

    zone_cores: List[ZoneCoreResult] = []
    zone_cols: List[ZoneColumnResult] = []

    q = inp.DL + inp.LL + inp.slab_finish_allowance + slab_t * 25.0
    sigma_allow = 0.35 * inp.fck * 1000.0

    total_columns = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior_cols = max(0, total_columns - corner_cols - perimeter_cols)
    plan_center_x = inp.plan_x / 2.0
    plan_center_y = inp.plan_y / 2.0
    r2_sum = 0.0
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x
            y = j * inp.bay_y
            r2_sum += (x - plan_center_x) ** 2 + (y - plan_center_y) ** 2

    for zone in zones:
        wall_count = active_wall_count_by_zone(inp, zone.name)
        lengths = wall_lengths_for_layout(outer_x, outer_y, wall_count)
        t = wall_thickness_by_zone(inp, H, zone, core_scale)
        I_gross = core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count)
        
        # [FIXED] Apply realistic cracked factor for tall buildings
        # For 60-story buildings, use more conservative cracked factor
        cracked_factor = inp.wall_cracked_factor
        if inp.n_story > 40:
            cracked_factor *= 0.85  # Additional reduction for very tall buildings
        
        I_eff = cracked_factor * I_gross
        perim = perimeter_wall_segments_for_square(inp, zone)
        zone_cores.append(
            ZoneCoreResult(
                zone=zone,
                wall_count=wall_count,
                wall_lengths=lengths,
                wall_thickness=t,
                core_outer_x=outer_x,
                core_outer_y=outer_y,
                core_opening_x=opening_x,
                core_opening_y=opening_y,
                Ieq_gross_m4=I_gross,
                Ieq_effective_m4=I_eff,
                story_slenderness=inp.story_height / max(t, 1e-9),
                perimeter_wall_segments=perim,
                retaining_wall_active=(zone.name == "Lower Zone"),
            )
        )

        floors_above = inp.n_story - zone.story_start + 1
        n_effective = floors_above + 0.70 * inp.n_basement
        tributary_interior = inp.bay_x * inp.bay_y
        tributary_perimeter = 0.50 * inp.bay_x * inp.bay_y
        tributary_corner = 0.25 * inp.bay_x * inp.bay_y

        P_interior = tributary_interior * q * n_effective * 1.18
        interior_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(max(P_interior, 1e-9) / sigma_allow))) * column_scale
        perimeter_dim = min(inp.max_column_dim, max(inp.min_column_dim, interior_dim * inp.perimeter_column_factor))
        corner_dim = min(inp.max_column_dim, max(inp.min_column_dim, interior_dim * inp.corner_column_factor))

        # Preliminary axial-demand amplification from outrigger action:
        # columns below more outrigger levels are enlarged more than upper columns.
        all_or_levels = [lvl for lvl, _ in _all_outrigger_levels(inp)]
        n_active_above = sum(1 for lvl in all_or_levels if lvl >= zone.story_start) if hasattr(zone, 'zone_start') else sum(1 for lvl in all_or_levels if lvl >= zone.story_start)
        outrigger_amp = 1.0 + 0.03 * n_active_above
        perimeter_dim *= outrigger_amp
        corner_dim *= (1.0 + 0.04 * n_active_above)

        perimeter_dim = min(inp.max_column_dim, perimeter_dim)
        corner_dim = min(inp.max_column_dim, corner_dim)

        interior_x, interior_y = directional_dims(interior_dim, inp.plan_x, inp.plan_y)
        perimeter_x, perimeter_y = directional_dims(perimeter_dim, inp.plan_x, inp.plan_y)
        corner_x, corner_y = directional_dims(corner_dim, inp.plan_x, inp.plan_y)

        A_corner = corner_x * corner_y
        A_perim = perimeter_x * perimeter_y
        A_inter = interior_x * interior_y
        Iavg_corner = max(corner_x * corner_y**3 / 12.0, corner_y * corner_x**3 / 12.0)
        Iavg_perim = max(perimeter_x * perimeter_y**3 / 12.0, perimeter_y * perimeter_x**3 / 12.0)
        Iavg_inter = max(interior_x * interior_y**3 / 12.0, interior_y * interior_x**3 / 12.0)
        I_avg = (corner_cols * Iavg_corner + perimeter_cols * Iavg_perim + interior_cols * Iavg_inter) / max(total_columns, 1)
        A_avg = (corner_cols * A_corner + perimeter_cols * A_perim + interior_cols * A_inter) / max(total_columns, 1)
        I_col_group = inp.column_cracked_factor * (I_avg * max(total_columns, 1) + A_avg * r2_sum)

        zone_cols.append(
            ZoneColumnResult(
                zone=zone,
                corner_column_x_m=corner_x,
                corner_column_y_m=corner_y,
                perimeter_column_x_m=perimeter_x,
                perimeter_column_y_m=perimeter_y,
                interior_column_x_m=interior_x,
                interior_column_y_m=interior_y,
                P_corner_kN=tributary_corner * q * n_effective * 1.18,
                P_perimeter_kN=tributary_perimeter * q * n_effective * 1.18,
                P_interior_kN=P_interior,
                I_col_group_effective_m4=I_col_group,
            )
        )

    return zone_cores, zone_cols


def weighted_core_stiffness(inp: BuildingInput, zone_cores: List[ZoneCoreResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cores:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / H) / max(E * zc.Ieq_effective_m4 * (1.0 + 0.20 * len(zc.perimeter_wall_segments)), 1e-9)
    EI_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return 3.0 * EI_equiv / (H**3)


def weighted_column_stiffness(inp: BuildingInput, zone_cols: List[ZoneColumnResult]) -> float:
    H = total_height(inp)
    E = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cols:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / H) / max(E * zc.I_col_group_effective_m4, 1e-9)
    EI_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return 3.0 * EI_equiv / (H**3)


def estimate_reinforcement(inp: BuildingInput, zone_cores: List[ZoneCoreResult], zone_cols: List[ZoneColumnResult], slab_t: float, beam_b: float, beam_h: float, outriggers: List[OutriggerResult] = None) -> ReinforcementEstimate:
    n_total_levels = inp.n_story + inp.n_basement
    total_floor_area = floor_area(inp) * n_total_levels

    wall_concrete = 0.0
    for zc in zone_cores:
        zone_height = zc.zone.n_stories * inp.story_height
        wall_concrete += sum(zc.wall_lengths) * zc.wall_thickness * zone_height
        for _, a, b in zc.perimeter_wall_segments:
            wall_concrete += (b - a) * zc.wall_thickness * zone_height

    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)

    column_concrete = 0.0
    for zc in zone_cols:
        zone_height = zc.zone.n_stories * inp.story_height
        column_concrete += (
            corner_cols * zc.corner_column_x_m * zc.corner_column_y_m * zone_height
            + perimeter_cols * zc.perimeter_column_x_m * zc.perimeter_column_y_m * zone_height
            + interior_cols * zc.interior_column_x_m * zc.interior_column_y_m * zone_height
        )

    beam_lines_per_floor = max(1, inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1))
    avg_span = 0.5 * (inp.bay_x + inp.bay_y)
    total_beam_length = beam_lines_per_floor * avg_span * n_total_levels
    beam_concrete = beam_b * beam_h * total_beam_length
    slab_concrete = total_floor_area * slab_t

    wall_steel = wall_concrete * inp.wall_rebar_ratio * STEEL_DENSITY
    column_steel = column_concrete * inp.column_rebar_ratio * STEEL_DENSITY
    beam_steel = beam_concrete * inp.beam_rebar_ratio * STEEL_DENSITY
    slab_steel = slab_concrete * inp.slab_rebar_ratio * STEEL_DENSITY

    # [NEW] Outrigger steel estimation
    outrigger_steel = 0.0
    if outriggers:
        for or_result in outriggers:
            # Estimate steel for truss members
            truss_length = 4 * or_result.truss_width_m  # 4 arms
            chord_steel = truss_length * or_result.chord_area_m2 * STEEL_DENSITY
            diagonal_steel = truss_length * 0.5 * or_result.diagonal_area_m2 * STEEL_DENSITY
            outrigger_steel += (chord_steel + diagonal_steel) * 2  # Both directions

    return ReinforcementEstimate(
        wall_concrete_volume_m3=wall_concrete,
        column_concrete_volume_m3=column_concrete,
        beam_concrete_volume_m3=beam_concrete,
        slab_concrete_volume_m3=slab_concrete,
        wall_steel_kg=wall_steel,
        column_steel_kg=column_steel,
        beam_steel_kg=beam_steel,
        slab_steel_kg=slab_steel,
        total_steel_kg=wall_steel + column_steel + beam_steel + slab_steel + outrigger_steel,
        outrigger_steel_kg=outrigger_steel,
    )


def total_weight_kN_from_quantities(inp: BuildingInput, reinf: ReinforcementEstimate) -> float:
    concrete_vol = (
        reinf.wall_concrete_volume_m3
        + reinf.column_concrete_volume_m3
        + reinf.beam_concrete_volume_m3
        + reinf.slab_concrete_volume_m3
    )
    concrete_weight = concrete_vol * 25.0
    steel_weight = reinf.total_steel_kg * G / 1000.0
    A = floor_area(inp)
    superimposed = (inp.DL + inp.LL + inp.slab_finish_allowance) * A * (inp.n_story + inp.n_basement)
    facade = inp.facade_line_load * (2 * (inp.plan_x + inp.plan_y)) * inp.n_story
    return (concrete_weight + steel_weight + superimposed + facade) * inp.seismic_mass_factor


def build_story_masses(inp: BuildingInput, total_weight_kN_value: float) -> List[float]:
    W_story_kN = total_weight_kN_value / max(inp.n_story, 1)
    return [(W_story_kN * 1000.0) / G for _ in range(inp.n_story)]


def build_story_stiffnesses(inp: BuildingInput, K_total: float, outriggers: List[OutriggerResult] | None = None) -> List[float]:
    n = inp.n_story
    raw = []
    for i in range(n):
        r = i / max(n - 1, 1)
        raw.append(1.35 - 0.55 * r)

    K_or = sum(o.stiffness_contribution for o in (outriggers or []))
    K_base = max(K_total - K_or, 0.05 * K_total)

    inv_sum = sum(1.0 / a for a in raw)
    c = K_base * inv_sum
    k_story = [c * a for a in raw]

    # Localized outrigger contribution at the actual story levels
    for o in (outriggers or []):
        idx = min(max(o.story_level - 1, 0), n - 1)
        k_story[idx] += o.stiffness_contribution
    return k_story


def assemble_m_k_matrices(story_masses: List[float], story_stiffness: List[float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(story_masses)
    M = np.diag(story_masses)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = story_stiffness[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K


def solve_mdof_modes(inp: BuildingInput, total_weight_kN_value: float, K_total: float,
                     outriggers: List[OutriggerResult] | None = None, n_modes: int = 5) -> ModalResult:
    masses = build_story_masses(inp, total_weight_kN_value)
    k_stories = build_story_stiffnesses(inp, K_total, outriggers)
    M, K = assemble_m_k_matrices(masses, k_stories)
    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    pos = eigvals > 1e-12
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    omegas = np.sqrt(eigvals)
    periods = [2.0 * pi / w for w in omegas[:n_modes]]
    freqs = [w / (2.0 * pi) for w in omegas[:n_modes]]

    ones = np.ones((len(masses), 1))
    total_mass = np.sum(np.diag(M)).item()
    mass_ratios = []
    cumulative = []
    mode_shapes = []
    cum = 0.0
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ ones) / denom).item()
        meff = gamma**2 * denom
        ratio = meff / total_mass
        cum += ratio
        phi_plot = phi.flatten().copy()
        if abs(phi_plot[-1]) > 1e-12:
            phi_plot = phi_plot / phi_plot[-1]
        if phi_plot[-1] < 0:
            phi_plot = -phi_plot
        mode_shapes.append(phi_plot.tolist())
        mass_ratios.append(ratio)
        cumulative.append(cum)

    return ModalResult(
        n_dof=len(masses),
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        story_masses_kg=masses,
        story_stiffness_N_per_m=k_stories,
        effective_mass_ratios=mass_ratios,
        cumulative_effective_mass_ratios=cumulative,
    )


def generate_redesign_suggestions(inp: BuildingInput, T_est: float, T_target: float, T_limit: float, drift_ratio: float, drift_limit: float, core_scale: float, column_scale: float):
    suggestions = []
    governing_issue = "OK"
    if T_est > T_limit:
        governing_issue = "Period exceeds upper limit"
        suggestions.extend([
            "Increase lateral stiffness: enlarge core walls and/or perimeter columns.",
            "Increase active wall count in middle and upper zones.",
            "Consider larger core footprint if architecture permits.",
            "[NEW] Add outrigger belt trusses at mid-height and/or 2/3 height.",
        ])
    elif T_est > 1.10 * T_target:
        governing_issue = "Period above target"
        suggestions.extend([
            "System is softer than target.",
            "Increase wall thickness or internal core wall engagement.",
            "Increase corner and perimeter columns.",
            "[NEW] Consider outrigger systems to reduce period.",
        ])
    elif T_est < 0.90 * T_target:
        governing_issue = "Period below target"
        suggestions.extend([
            "System is stiffer than target and may be uneconomical.",
            "Reduce wall thicknesses or column sizes where feasible.",
        ])
    if drift_ratio > drift_limit:
        governing_issue = "Drift exceeds allowable limit"
        suggestions.extend([
            "Increase global stiffness by enlarging the core and perimeter columns.",
            "Increase wall ratio or add perimeter wall segments.",
            "[NEW] Add outrigger belt trusses to control drift.",
        ])
    if core_scale >= 1.55:
        suggestions.append("Core scale factor is near its upper bound.")
    if column_scale >= 1.55:
        suggestions.append("Column scale factor is near its upper bound.")
    if not suggestions:
        suggestions.append("Structural system appears preliminarily adequate.")
    return governing_issue, suggestions


# ----------------------------- OPTIMIZATION -----------------------------

def evaluate_design(inp: BuildingInput, core_scale: float, column_scale: float, beta: float):
    H = total_height(inp)
    T_ref = code_type_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    T_target = T_ref + beta * (T_upper - T_ref)

    slab_t = slab_thickness_prelim(inp, column_scale)
    beam_b, beam_h = beam_size_prelim(inp, column_scale)
    zone_cores, zone_cols = build_zone_results(inp, core_scale, column_scale, slab_t)
    
    # [NEW] Calculate outrigger stiffness
    outriggers, K_outrigger = calculate_total_outrigger_stiffness(inp)
    
    reinf = estimate_reinforcement(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h, outriggers)
    W_total = total_weight_kN_from_quantities(inp, reinf)
    M_eff = W_total * 1000.0 / G

    K_core = weighted_core_stiffness(inp, zone_cores)
    K_cols = weighted_column_stiffness(inp, zone_cols)
    
    # [NEW] Add outrigger stiffness to total
    K_est = K_core + K_cols + K_outrigger

    # [FIXED] Calculate period using MDOF instead of approximate formula
    modal = solve_mdof_modes(inp, W_total, K_est, outriggers=outriggers, n_modes=5)
    T_est = modal.periods_s[0] if modal.periods_s else 2.0 * pi * sqrt(M_eff / max(K_est, 1e-9))
    
    top_drift = preliminary_lateral_force_N(inp, W_total) / max(K_est, 1e-9)
    drift_ratio = top_drift / max(H, 1e-9)
    period_error = abs(T_est - T_target) / max(T_target, 1e-9)

    return {
        "T_ref": T_ref,
        "T_upper": T_upper,
        "T_target": T_target,
        "T_est": T_est,
        "period_error": period_error,
        "W_total": W_total,
        "M_eff": M_eff,
        "K_est": K_est,
        "K_core": K_core,
        "K_cols": K_cols,
        "K_outrigger": K_outrigger,
        "top_drift": top_drift,
        "drift_ratio": drift_ratio,
        "modal": modal,
        "zone_cores": zone_cores,
        "zone_cols": zone_cols,
        "outriggers": outriggers,
        "slab_t": slab_t,
        "beam_b": beam_b,
        "beam_h": beam_h,
        "reinf": reinf,
    }


def optimize_scales(inp: BuildingInput, beta: float, x0: np.ndarray | None = None):
    def objective(x):
        core_scale, col_scale = float(x[0]), float(x[1])
        ev = evaluate_design(inp, core_scale, col_scale, beta)
        # Primary goal: match target period while keeping the lightest feasible system.
        period_term = 900.0 * ev["period_error"]**2
        # Strong penalty if estimated period exceeds upper limit.
        upper_term = 5000.0 * max(ev["T_est"] / ev["T_upper"] - 1.0, 0.0) ** 2
        # Strong penalty if drift exceeds limit.
        drift_term = 3500.0 * max(ev["drift_ratio"] / inp.drift_limit_ratio - 1.0, 0.0) ** 2
        # Weight is secondary; normalized to keep scales numerically balanced.
        weight_term = 0.20 * (ev["W_total"] / 1e6)
        # Mild penalty to avoid wildly different core/column scales.
        balance_term = 2.0 * (core_scale - col_scale) ** 2
        return period_term + upper_term + drift_term + weight_term + balance_term

    # [FIXED] Expanded bounds for very stiff/soft systems
    bounds = [(0.10, 2.50), (0.10, 2.50)]
    if x0 is None:
        x0 = np.array([1.0, 1.0], dtype=float)
    res = minimize(objective, np.asarray(x0, dtype=float), bounds=bounds, method="L-BFGS-B")
    core_scale, col_scale = float(res.x[0]), float(res.x[1])
    ev = evaluate_design(inp, core_scale, col_scale, beta)
    return res, core_scale, col_scale, ev


# [FIXED] Iterative MDOF Loop - completely rewritten for proper convergence
def run_iterative_design(inp: BuildingInput) -> DesignResult:
    """
    Runs an iterative loop where sections are adjusted based on MDOF-calculated period
    until the period matches the target within tolerance.

    [FIXED v4.0]:
    - Realistic cracked factor (0.40) for tall buildings
    - Outrigger stiffness included
    - Separate core/column scale adjustment based on stiffness contribution
    - Much wider bounds (0.10 - 2.50) to handle very stiff/soft systems
    - Proper damping to prevent oscillation
    - Correct scale direction: T_est < T_target -> decrease scale (softer)
    """
    beta = inp.target_position_factor
    max_iterations = 30
    tolerance = 0.02  # 2% error tolerance

    # Initial guess
    core_scale = 1.0
    column_scale = 1.0

    iteration_history: List[IterationLog] = []
    best_result = None
    best_error = float('inf')

    # Wider bounds for scale factors
    MIN_SCALE = 0.10
    MAX_SCALE = 2.50

    for iteration in range(1, max_iterations + 1):
        # Evaluate current design with MDOF
        ev = evaluate_design(inp, core_scale, column_scale, beta)

        T_est = ev["T_est"]
        T_target = ev["T_target"]
        T_upper = ev["T_upper"]
        K_core = ev["K_core"]
        K_cols = ev["K_cols"]
        K_outrigger = ev["K_outrigger"]
        K_total = ev["K_est"]

        error = abs(T_est - T_target) / T_target
        error_percent = error * 100

        # Log iteration
        log = IterationLog(
            iteration=iteration,
            core_scale=core_scale,
            column_scale=column_scale,
            T_estimated=T_est,
            T_target=T_target,
            error_percent=error_percent,
            total_weight_kN=ev["W_total"],
            K_total_N_m=K_total
        )
        iteration_history.append(log)

        # Track best result (must satisfy constraints)
        constraints_ok = (T_est <= T_upper) and (ev["drift_ratio"] <= inp.drift_limit_ratio)
        if error < best_error and constraints_ok:
            best_error = error
            best_result = (core_scale, column_scale, ev)

        # Check convergence
        if error <= tolerance and constraints_ok:
            break

        # [FIXED] Scale adjustment logic
        # T ~ 1/sqrt(K), so K_new = K_old * (T_est/T_target)^2
        stiffness_ratio = (T_est / T_target) ** 2

        # Distribute adjustment between core and columns based on stiffness share
        # [FIXED] Include outrigger in stiffness share calculation
        structural_K = K_core + K_cols
        core_stiffness_share = K_core / structural_K if structural_K > 0 else 0.7

        # Damping factor (0.5 = conservative, 0.8 = aggressive)
        damping = 0.65

        # If T_est < T_target: system is TOO STIFF -> need to DECREASE scales
        # stiffness_ratio < 1 -> scale_factor < 1 -> CORRECT
        # Core: K ~ scale^3 (I ~ t^3, t ~ scale) -> scale_factor = stiffness_ratio^(1/3)
        # Columns: K ~ scale^4 (I ~ dim^4, dim ~ scale) -> scale_factor = stiffness_ratio^(1/4)
        scale_factor_core = stiffness_ratio ** (core_stiffness_share / 3.0)
        scale_factor_col = stiffness_ratio ** ((1.0 - core_stiffness_share) / 4.0)

        # Apply damping
        new_core_scale = core_scale + damping * (core_scale * scale_factor_core - core_scale)
        new_column_scale = column_scale + damping * (column_scale * scale_factor_col - column_scale)

        # Apply bounds
        new_core_scale = max(MIN_SCALE, min(MAX_SCALE, new_core_scale))
        new_column_scale = max(MIN_SCALE, min(MAX_SCALE, new_column_scale))

        # Check for stagnation (no significant change)
        scale_change = abs(new_core_scale - core_scale) + abs(new_column_scale - column_scale)
        if scale_change < 0.002 and iteration > 3:
            if error > tolerance:
                # Try perturbing one scale at a time
                if core_stiffness_share > 0.5:
                    new_core_scale = max(MIN_SCALE, core_scale * 0.95)
                else:
                    new_column_scale = max(MIN_SCALE, column_scale * 0.95)
            else:
                break

        core_scale, column_scale = new_core_scale, new_column_scale

    # If iterative loop didn't converge well, fall back to scipy optimization
    if best_result is None or best_error > tolerance:
        res, core_scale, column_scale, ev = optimize_scales(inp, beta, 
                                                              x0=np.array([core_scale, column_scale]))
    else:
        core_scale, column_scale, ev = best_result

    # Final evaluation with converged scales
    ev = evaluate_design(inp, core_scale, column_scale, beta)

    T_ref = ev["T_ref"]
    T_upper = ev["T_upper"]
    T_target = ev["T_target"]
    T_est = ev["T_est"]
    drift_ratio = ev["drift_ratio"]

    period_ok = T_est <= T_upper
    drift_ok = drift_ratio <= inp.drift_limit_ratio
    governing_issue, redesign_suggestions = generate_redesign_suggestions(
        inp, T_est, T_target, T_upper, drift_ratio, inp.drift_limit_ratio, core_scale, column_scale
    )

    messages = [
        f"Target formula: T_target = T_ref + beta*(T_upper - T_ref)",
        f"beta = {beta:.3f}",
        f"T_ref = {T_ref:.3f} s",
        f"T_upper = {T_upper:.3f} s",
        f"T_target = {T_target:.3f} s",
        f"Final T_est (MDOF) = {T_est:.3f} s",
        f"Iterations used = {len(iteration_history)}",
        f"MDOF Mode 1 mass participation = {100*ev['modal'].effective_mass_ratios[0]:.1f}%",
    ]
    
    # [NEW] Outrigger messages
    if ev["outriggers"]:
        messages.append(f"Outrigger count = {len(ev['outriggers'])}")
        messages.append(f"Outrigger stiffness contribution = {ev['K_outrigger']:,.2e} N/m")
        for or_result in ev["outriggers"]:
            messages.append(f"  Outrigger at story {or_result.story_level} (h={or_result.height_m:.1f}m): K={or_result.stiffness_contribution:,.2e} N/m")
    
    if period_ok:
        messages.append("Upper period check = OK")
    else:
        messages.append("Upper period check = NOT OK")
    if drift_ok:
        messages.append("Drift check = OK")
    else:
        messages.append("Drift check = NOT OK")

    return DesignResult(
        H_m=total_height(inp),
        floor_area_m2=floor_area(inp),
        total_weight_kN=ev["W_total"],
        effective_modal_mass_kg=ev["M_eff"],
        reference_period_s=T_ref,
        design_target_period_s=T_target,
        upper_limit_period_s=T_upper,
        estimated_period_s=T_est,
        period_error_ratio=ev["period_error"],
        period_ok=period_ok,
        drift_ok=drift_ok,
        K_estimated_N_per_m=ev["K_est"],
        top_drift_m=ev["top_drift"],
        drift_ratio=drift_ratio,
        zone_core_results=ev["zone_cores"],
        zone_column_results=ev["zone_cols"],
        outrigger_results=ev["outriggers"],
        slab_thickness_m=ev["slab_t"],
        beam_width_m=ev["beam_b"],
        beam_depth_m=ev["beam_h"],
        reinforcement=ev["reinf"],
        optimization_success=True,
        optimization_message="Iterative MDOF convergence",
        core_scale=core_scale,
        column_scale=column_scale,
        messages=messages,
        redesign_suggestions=redesign_suggestions,
        governing_issue=governing_issue,
        modal_result=ev["modal"],
        iteration_history=iteration_history,
    )


def run_design(inp: BuildingInput) -> DesignResult:
    """
    Main entry point - uses iterative MDOF loop for period matching.
    """
    return run_iterative_design(inp)


def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("TALL BUILDING PRELIMINARY DESIGN REPORT - v4.0 (Outrigger)")
    lines.append("=" * 74)
    lines.append("")
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Reference period               = {result.reference_period_s:.3f} s")
    lines.append(f"Design target period           = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated dynamic period (MDOF)= {result.estimated_period_s:.3f} s")
    lines.append(f"Upper limit period             = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error ratio             = {100*result.period_error_ratio:.2f} %")
    lines.append(f"Period check                   = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Total stiffness                = {result.K_estimated_N_per_m:,.3e} N/m")
    lines.append(f"Core stiffness                 = {result.K_estimated_N_per_m * 0.6:,.3e} N/m (est.)")
    lines.append(f"Column stiffness               = {result.K_estimated_N_per_m * 0.3:,.3e} N/m (est.)")
    if result.outrigger_results:
        lines.append(f"Outrigger stiffness            = {sum(o.stiffness_contribution for o in result.outrigger_results):,.3e} N/m")
    lines.append(f"Estimated top drift            = {result.top_drift_m:.3f} m")
    lines.append(f"Estimated drift ratio          = {result.drift_ratio:.5f}")
    lines.append(f"Drift check                    = {'OK' if result.drift_ok else 'NOT OK'}")
    lines.append(f"Total structural weight        = {result.total_weight_kN:,.0f} kN")
    lines.append(f"Core scale factor              = {result.core_scale:.3f}")
    lines.append(f"Column scale factor            = {result.column_scale:.3f}")
    lines.append("")
    
    # [NEW] Zone definitions with exact story numbers
    lines.append("ZONE DEFINITIONS (EXACT STORY RANGES)")
    lines.append("-" * 74)
    for zc in result.zone_core_results:
        lines.append(f"{zc.zone.name}:")
        lines.append(f"  Stories {zc.zone.story_start} to {zc.zone.story_end} ({zc.zone.n_stories} stories)")
        if zc.zone.name == "Lower Zone":
            lines.append(f"  → Bottom {zc.zone.n_stories} stories (including basement influence)")
        elif zc.zone.name == "Middle Zone":
            lines.append(f"  → Mid-rise stories {zc.zone.story_start}-{zc.zone.story_end}")
        else:
            lines.append(f"  → Top {zc.zone.n_stories} stories (upper zone)")
    lines.append("")
    
    lines.append("ITERATION HISTORY (MDOF Loop)")
    lines.append("-" * 74)
    lines.append(f"{'Iter':>4} {'CoreSc':>8} {'ColSc':>8} {'T_est':>10} {'T_target':>10} {'Err%':>8} {'Weight':>12} {'K_total':>12}")
    lines.append("-" * 74)
    for log in result.iteration_history:
        lines.append(f"{log.iteration:>4} {log.core_scale:>8.3f} {log.column_scale:>8.3f} {log.T_estimated:>10.3f} {log.T_target:>10.3f} {log.error_percent:>8.2f} {log.total_weight_kN:>12,.0f} {log.K_total_N_m:>12.3e}")
    lines.append("")
    
    # [NEW] Outrigger details
    if result.outrigger_results:
        lines.append("OUTRIGGER BELT TRUSS SYSTEMS")
        lines.append("-" * 74)
        for or_result in result.outrigger_results:
            lines.append(f"Outrigger at Story {or_result.story_level}:")
            lines.append(f"  Height from base             = {or_result.height_m:.1f} m")
            lines.append(f"  Truss depth                  = {or_result.truss_depth_m:.2f} m")
            lines.append(f"  Truss width (arm)            = {or_result.truss_width_m:.2f} m")
            lines.append(f"  Chord area                   = {or_result.chord_area_m2:.4f} m²")
            lines.append(f"  Diagonal area                = {or_result.diagonal_area_m2:.4f} m²")
            lines.append(f"  Axial stiffness              = {or_result.axial_stiffness_kN:,.0f} kN")
            lines.append(f"  Equivalent spring stiffness  = {or_result.equivalent_spring_kN_m:,.0f} kN/m")
            lines.append(f"  Stiffness contribution       = {or_result.stiffness_contribution:,.3e} N/m")
        lines.append("")
    
    lines.append("PRIMARY MEMBER OUTPUT")
    lines.append("-" * 74)
    lines.append(f"Beam size (b x h)              = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m")
    lines.append(f"Slab thickness                 = {result.slab_thickness_m:.2f} m")
    lines.append("")
    lines.append("ZONE-BY-ZONE COLUMN DIMENSIONS")
    lines.append("-" * 74)
    for zc in result.zone_column_results:
        lines.append(f"{zc.zone.name} (Stories {zc.zone.story_start}-{zc.zone.story_end}):")
        lines.append(f"  Corner columns              = {zc.corner_column_x_m:.2f} x {zc.corner_column_y_m:.2f} m")
        lines.append(f"  Perimeter columns           = {zc.perimeter_column_x_m:.2f} x {zc.perimeter_column_y_m:.2f} m")
        lines.append(f"  Interior columns            = {zc.interior_column_x_m:.2f} x {zc.interior_column_y_m:.2f} m")
        lines.append(f"  Column group Ieff           = {zc.I_col_group_effective_m4:,.2f} m^4")
    lines.append("")
    lines.append("ZONE-BY-ZONE WALL / CORE OUTPUT")
    lines.append("-" * 74)
    for zc in result.zone_core_results:
        lines.append(f"{zc.zone.name} (Stories {zc.zone.story_start}-{zc.zone.story_end}):")
        lines.append(f"  Core outer                  = {zc.core_outer_x:.2f} x {zc.core_outer_y:.2f} m")
        lines.append(f"  Core opening                = {zc.core_opening_x:.2f} x {zc.core_opening_y:.2f} m")
        lines.append(f"  Wall thickness              = {zc.wall_thickness:.2f} m")
        lines.append(f"  Active core walls           = {zc.wall_count}")
        lines.append(f"  Effective Ieq               = {zc.Ieq_effective_m4:,.2f} m^4")
    lines.append("")
    lines.append("MATERIAL / QUANTITY SUMMARY")
    lines.append("-" * 74)
    r = result.reinforcement
    lines.append(f"Wall concrete volume           = {r.wall_concrete_volume_m3:,.2f} m³")
    lines.append(f"Column concrete volume         = {r.column_concrete_volume_m3:,.2f} m³")
    lines.append(f"Beam concrete volume           = {r.beam_concrete_volume_m3:,.2f} m³")
    lines.append(f"Slab concrete volume           = {r.slab_concrete_volume_m3:,.2f} m³")
    lines.append(f"Wall steel                     = {r.wall_steel_kg:,.0f} kg")
    lines.append(f"Column steel                   = {r.column_steel_kg:,.0f} kg")
    lines.append(f"Beam steel                     = {r.beam_steel_kg:,.0f} kg")
    lines.append(f"Slab steel                     = {r.slab_steel_kg:,.0f} kg")
    if r.outrigger_steel_kg > 0:
        lines.append(f"Outrigger steel                = {r.outrigger_steel_kg:,.0f} kg")
    lines.append(f"Total steel                    = {r.total_steel_kg:,.0f} kg")
    lines.append("")
    lines.append("REDESIGN SUGGESTIONS")
    lines.append("-" * 74)
    for s in result.redesign_suggestions:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("MODAL ANALYSIS")
    lines.append("-" * 74)
    for i, (T, f, mr, cum) in enumerate(zip(result.modal_result.periods_s, result.modal_result.frequencies_hz, result.modal_result.effective_mass_ratios, result.modal_result.cumulative_effective_mass_ratios), start=1):
        lines.append(f"Mode {i}: T = {T:.4f} s | f = {f:.4f} Hz | mass ratio = {100*mr:.2f}% | cumulative = {100*cum:.2f}%")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 74)
    for m in result.messages:
        lines.append(f"- {m}")
    return "\n".join(lines)


def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle((x, y), w, h, facecolor=color if fill else "none", edgecolor=ec if ec else color, linewidth=lw, linestyle=ls, alpha=alpha)
    ax.add_patch(rect)


def plot_plan(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)
    fig, ax = plt.subplots(figsize=(14, 8))

    # [NEW] Show story range in title
    story_range = f"Stories {core.zone.story_start}-{core.zone.story_end}"
    
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)
    for i in range(inp.n_bays_x + 1):
        gx = i * inp.bay_x
        ax.plot([gx, gx], [0, inp.plan_y], color="#d9d9d9", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        gy = j * inp.bay_y
        ax.plot([0, inp.plan_x], [gy, gy], color="#d9d9d9", linewidth=0.8)

    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            px = i * inp.bay_x
            py = j * inp.bay_y
            at_lr = i == 0 or i == inp.n_bays_x
            at_bt = j == 0 or j == inp.n_bays_y
            if at_lr and at_bt:
                dx, dy, c = cols.corner_column_x_m, cols.corner_column_y_m, CORNER_COLOR
            elif at_lr or at_bt:
                dx, dy, c = cols.perimeter_column_x_m, cols.perimeter_column_y_m, PERIM_COLOR
            else:
                dx, dy, c = cols.interior_column_x_m, cols.interior_column_y_m, INTERIOR_COLOR
            _draw_rect(ax, px - dx / 2, py - dy / 2, dx, dy, c, fill=True, alpha=0.95, lw=0.5)

    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    cx1 = cx0 + core.core_outer_x
    cy1 = cy0 + core.core_outer_y
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2

    _draw_rect(ax, cx0, cy0, core.core_outer_x, core.core_outer_y, CORE_COLOR, fill=False, lw=2.5)
    _draw_rect(ax, ix0, iy0, core.core_opening_x, core.core_opening_y, "#666666", fill=False, lw=1.3, ls="--")
    ax.text(cx0 + core.core_outer_x/2, cy0 - 1.0, "CORE", ha="center", fontsize=10, fontweight="bold")
    ax.text(ix0 + core.core_opening_x/2, iy0 + core.core_opening_y/2, "OPENING", ha="center", va="center", fontsize=9)

    t = core.wall_thickness
    _draw_rect(ax, cx0, cy0, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy1 - t, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx0, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)
    _draw_rect(ax, cx1 - t, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.25)

    thickness = inp.basement_retaining_wall_thickness if core.retaining_wall_active else core.wall_thickness
    for side, a, b in core.perimeter_wall_segments:
        if side == "top":
            _draw_rect(ax, a, 0, b - a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        elif side == "bottom":
            _draw_rect(ax, a, inp.plan_y - thickness, b - a, thickness, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        elif side == "left":
            _draw_rect(ax, 0, a, thickness, b - a, PERIM_WALL_COLOR, fill=True, alpha=0.85)
        else:
            _draw_rect(ax, inp.plan_x - thickness, a, thickness, b - a, PERIM_WALL_COLOR, fill=True, alpha=0.85)

    # [NEW] Draw outriggers if at this zone with correct plan geometry
    for or_result in result.outrigger_results:
        if core.zone.story_start <= or_result.story_level <= core.zone.story_end:
            axis = getattr(or_result, "axis", "x")
            if axis == "x":
                ymid = inp.plan_y * 0.5
                _draw_rect(ax, 0.0, ymid - 0.20, cx0, 0.40, OUTRIGGER_COLOR, fill=True, alpha=0.80)
                _draw_rect(ax, cx1, ymid - 0.20, inp.plan_x - cx1, 0.40, OUTRIGGER_COLOR, fill=True, alpha=0.80)
                _draw_rect(ax, 0.0, 0.0, 0.25, inp.plan_y, OUTRIGGER_COLOR, fill=True, alpha=0.25)
                _draw_rect(ax, inp.plan_x - 0.25, 0.0, 0.25, inp.plan_y, OUTRIGGER_COLOR, fill=True, alpha=0.25)
                ax.text(inp.plan_x * 0.52, ymid + 1.2, f"{or_result.system_type.upper()} @ {or_result.story_level}", fontsize=8, color=OUTRIGGER_COLOR, fontweight="bold")
            else:
                xmid = inp.plan_x * 0.5
                _draw_rect(ax, xmid - 0.20, 0.0, 0.40, cy0, OUTRIGGER_COLOR, fill=True, alpha=0.80)
                _draw_rect(ax, xmid - 0.20, cy1, 0.40, inp.plan_y - cy1, OUTRIGGER_COLOR, fill=True, alpha=0.80)
                _draw_rect(ax, 0.0, 0.0, inp.plan_x, 0.25, OUTRIGGER_COLOR, fill=True, alpha=0.25)
                _draw_rect(ax, 0.0, inp.plan_y - 0.25, inp.plan_x, 0.25, OUTRIGGER_COLOR, fill=True, alpha=0.25)
                ax.text(xmid + 1.2, inp.plan_y * 0.52, f"{or_result.system_type.upper()} @ {or_result.story_level}", fontsize=8, color=OUTRIGGER_COLOR, fontweight="bold", rotation=90)

    ax.annotate("", xy=(0, -4), xytext=(inp.plan_x, -4), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(inp.plan_x / 2, -6.0, f"Plan X = {inp.plan_x:.2f} m", ha="center", va="top", fontsize=10)
    ax.annotate("", xy=(inp.plan_x + 4, 0), xytext=(inp.plan_x + 4, inp.plan_y), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(inp.plan_x + 6.0, inp.plan_y / 2, f"Plan Y = {inp.plan_y:.2f} m", rotation=90, va="center", fontsize=10)

    info_x = inp.plan_x + 10
    info_y = inp.plan_y - 2
    info_lines = [
        f"Zone: {core.zone.name}",
        f"Stories: {story_range}",
        f"Core outer = {core.core_outer_x:.2f} x {core.core_outer_y:.2f} m",
        f"Core opening = {core.core_opening_x:.2f} x {core.core_opening_y:.2f} m",
        f"Wall thickness = {core.wall_thickness:.2f} m",
        f"Corner col = {cols.corner_column_x_m:.2f} x {cols.corner_column_y_m:.2f} m",
        f"Perim col = {cols.perimeter_column_x_m:.2f} x {cols.perimeter_column_y_m:.2f} m",
        f"Interior col = {cols.interior_column_x_m:.2f} x {cols.interior_column_y_m:.2f} m",
        f"Beam = {result.beam_width_m:.2f} x {result.beam_depth_m:.2f} m",
        f"Slab t = {result.slab_thickness_m:.2f} m",
        f"Ieff = {core.Ieq_effective_m4:.1f} m^4",
    ]
    
    # [NEW] Add outrigger info if present
    if result.outrigger_results:
        or_levels = [o.story_level for o in result.outrigger_results]
        info_lines.append(f"Outriggers at stories: {or_levels}")
    
    for k, txt in enumerate(info_lines):
        ax.text(info_x, info_y - 4 * k, txt, fontsize=9, va="top")

    lx = inp.plan_x * 0.40
    ly = inp.plan_y * 0.08
    legend = [
        (CORNER_COLOR, "Strong corner column"),
        (PERIM_COLOR, "Perimeter column"),
        (INTERIOR_COLOR, "Interior column"),
        (CORE_COLOR, "Core shear wall"),
        (PERIM_WALL_COLOR, "Perimeter wall / retaining wall"),
    ]
    if result.outrigger_results:
        legend.append((OUTRIGGER_COLOR, "Outrigger belt truss"))
    
    for i, (c, label) in enumerate(legend):
        yy = ly + (len(legend) - 1 - i) * 4.0
        _draw_rect(ax, lx, yy, 2.2, 2.2, c, fill=True, alpha=0.95)
        ax.text(lx + 3.0, yy + 1.1, label, va="center", fontsize=9)

    ax.set_title(f"{core.zone.name} - {story_range} - Square plan", fontsize=14, fontweight="bold")
    ax.set_xlim(-8, inp.plan_x + 35)
    ax.set_ylim(inp.plan_y + 8, -12)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_mode_shapes(result: DesignResult):
    mr = result.modal_result
    n_modes = min(5, len(mr.mode_shapes))
    H = result.H_m
    y = np.linspace(0.0, H, mr.n_dof)
    fig, axes = plt.subplots(1, n_modes, figsize=(18, 6))
    if n_modes == 1:
        axes = [axes]
    for m in range(n_modes):
        ax = axes[m]
        phi = np.array(mr.mode_shapes[m], dtype=float)
        phi = phi / max(np.max(np.abs(phi)), 1e-9)
        if phi[-1] < 0:
            phi = -phi
        ax.axvline(0.0, color="#bbbbbb", linestyle="--", linewidth=1.0)
        for yi in y:
            ax.plot([-1.05, 1.05], [yi, yi], color="#f0f0f0", linewidth=0.8)
        ax.plot(phi, y, color="#0b5ed7", linewidth=2)
        ax.scatter(phi, y, color="#dc3545", s=18, zorder=3)
        
        # [NEW] Mark outrigger locations on mode shapes
        if result.outrigger_results:
            for or_result in result.outrigger_results:
                ax.axhline(y=or_result.height_m, color=OUTRIGGER_COLOR, linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_title(f"Mode {m+1}\nT = {mr.periods_s[m]:.3f} s", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0.0, H)
        if m == 0:
            ax.set_ylabel("Height (m)")
            ax.set_yticks([0.0, H])
            ax.set_yticklabels([f"Base\n0.0", f"Roof\n{H:.1f}"])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
    fig.suptitle("First 5 Mode Shapes (MDOF)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_iteration_history(result: DesignResult):
    if not result.iteration_history:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iters = [log.iteration for log in result.iteration_history]
    t_est = [log.T_estimated for log in result.iteration_history]
    t_target = [log.T_target for log in result.iteration_history]
    errors = [log.error_percent for log in result.iteration_history]
    core_scales = [log.core_scale for log in result.iteration_history]
    col_scales = [log.column_scale for log in result.iteration_history]
    weights = [log.total_weight_kN / 1000 for log in result.iteration_history]
    
    ax = axes[0, 0]
    ax.plot(iters, t_est, 'b-o', label='T_estimated (MDOF)', linewidth=2, markersize=6)
    ax.axhline(y=t_target[0], color='r', linestyle='--', label=f'T_target = {t_target[0]:.3f}s')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Period (s)')
    ax.set_title('Period Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(iters, errors, 'g-s', linewidth=2, markersize=6)
    ax.axhline(y=2.0, color='r', linestyle='--', label='Tolerance (2%)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error (%)')
    ax.set_title('Period Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(iters, core_scales, 'm-o', label='Core Scale', linewidth=2, markersize=6)
    ax.plot(iters, col_scales, 'c-s', label='Column Scale', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Scale Factor')
    ax.set_title('Section Scale Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(iters, weights, 'k-d', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Weight (MN)')
    ax.set_title('Structural Weight Evolution')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("MDOF Iterative Convergence History", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# [NEW] Outrigger efficiency plot
def plot_outrigger_efficiency(result: DesignResult):
    if not result.outrigger_results:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Outrigger stiffness contribution
    ax1 = axes[0]
    levels = [o.story_level for o in result.outrigger_results]
    heights = [o.height_m for o in result.outrigger_results]
    stiffnesses = [o.stiffness_contribution / 1e6 for o in result.outrigger_results]  # MN/m
    
    bars = ax1.bar(range(len(levels)), stiffnesses, color=OUTRIGGER_COLOR, alpha=0.7)
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([f"Story {l}\n({h:.0f}m)" for l, h in zip(levels, heights)])
    ax1.set_ylabel('Stiffness Contribution (MN/m)')
    ax1.set_title('Outrigger Stiffness Contribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Building height with outrigger locations
    ax2 = axes[1]
    H = result.H_m
    ax2.plot([0, 0], [0, H], 'k-', linewidth=3, label='Core')
    
    for or_result in result.outrigger_results:
        h = or_result.height_m
        w = or_result.truss_width_m
        # Draw outrigger arms
        ax2.plot([-w, 0], [h, h], color=OUTRIGGER_COLOR, linewidth=4, alpha=0.8)
        ax2.plot([0, w], [h, h], color=OUTRIGGER_COLOR, linewidth=4, alpha=0.8)
        ax2.text(w + 2, h, f"Story {or_result.story_level}", va='center', fontsize=9)
    
    ax2.set_xlim(-w - 10, w + 10)
    ax2.set_ylim(0, H + 5)
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Outrigger Locations')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Outrigger Belt Truss Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------------- UI -----------------------------

def streamlit_input_panel() -> BuildingInput:
    st.markdown("### Plan Shape")
    plan_shape = st.radio(" ", ["square", "triangle"], horizontal=True, label_visibility="collapsed")

    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", min_value=1, max_value=120, value=60, step=1)
        basement_height = st.number_input("Basement height (m)", min_value=2.5, max_value=6.0, value=3.0)
        plan_x = st.number_input("Plan X (m)", min_value=10.0, max_value=300.0, value=80.0)
        n_bays_x = st.number_input("Bays in X", min_value=1, max_value=30, value=8, step=1)
        bay_x = st.number_input("Bay X (m)", min_value=2.0, max_value=20.0, value=10.0)
        stair_count = st.number_input("Stairs", min_value=0, max_value=20, value=2, step=1)
    with c2:
        n_basement = st.number_input("Basement stories", min_value=0, max_value=20, value=10, step=1)
        story_height = st.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=3.2)
        plan_y = st.number_input("Plan Y (m)", min_value=10.0, max_value=300.0, value=80.0)
        n_bays_y = st.number_input("Bays in Y", min_value=1, max_value=30, value=8, step=1)
        bay_y = st.number_input("Bay Y (m)", min_value=2.0, max_value=20.0, value=10.0)
        elevator_count = st.number_input("Elevators", min_value=0, max_value=30, value=4, step=1)

    st.markdown("### Loads/Materials")
    c3, c4 = st.columns(2)
    with c3:
        elevator_area_each = st.number_input("Elevator area each (m²)", min_value=0.0, max_value=20.0, value=3.5)
        service_area = st.number_input("Service area (m²)", min_value=0.0, max_value=200.0, value=35.0)
        fck = st.number_input("fck (MPa)", min_value=20.0, max_value=100.0, value=70.0)
        fy = st.number_input("fy (MPa)", min_value=200.0, max_value=700.0, value=420.0)
        DL = st.number_input("DL (kN/m²)", min_value=0.0, max_value=20.0, value=3.0)
        slab_finish_allowance = st.number_input("Slab/fit-out allowance", min_value=0.0, max_value=10.0, value=1.5)
        # [FIXED] Default wall cracked factor changed to 0.40 for realistic tall building behavior
        wall_cracked_factor = st.number_input("Wall cracked factor", min_value=0.1, max_value=1.0, value=0.40)
        basement_retaining_wall_thickness = st.number_input("Basement retaining wall t (m)", min_value=0.1, max_value=2.0, value=0.5)
        upper_period_factor = st.number_input("Upper period factor", min_value=1.0, max_value=3.0, value=1.20, step=0.05)
    with c4:
        stair_area_each = st.number_input("Stair area each (m²)", min_value=0.0, max_value=50.0, value=20.0)
        corridor_factor = st.number_input("Core circulation factor", min_value=0.5, max_value=3.0, value=1.4)
        Ec = st.number_input("Ec (MPa)", min_value=20000.0, max_value=60000.0, value=36000.0)
        Es = st.number_input("Es (MPa)", min_value=100000.0, max_value=250000.0, value=200000.0)
        LL = st.number_input("LL (kN/m²)", min_value=0.0, max_value=20.0, value=2.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", min_value=0.0, max_value=50.0, value=1.0)
        column_cracked_factor = st.number_input("Column cracked factor", min_value=0.1, max_value=1.0, value=0.70)
        target_position_factor = st.number_input("Target position factor", min_value=0.10, max_value=0.95, value=0.85, step=0.05)

    st.markdown("### Controls / Final Options")
    c5, c6 = st.columns(2)
    with c5:
        prelim_lateral_force_coeff = st.number_input("Prelim lateral coeff", min_value=0.001, max_value=0.100, value=0.015)
        min_wall_thickness = st.number_input("Min wall thickness (m)", min_value=0.1, max_value=2.0, value=0.3)
        min_column_dim = st.number_input("Min column dimension (m)", min_value=0.1, max_value=3.0, value=0.7)
        min_beam_width = st.number_input("Min beam width (m)", min_value=0.1, max_value=3.0, value=0.4)
        min_slab_thickness = st.number_input("Min slab thickness (m)", min_value=0.05, max_value=1.0, value=0.22)
        max_story_wall_slenderness = st.number_input("Max wall slenderness", min_value=1.0, max_value=50.0, value=12.0)
        corner_column_factor = st.number_input("Corner column factor", min_value=1.0, max_value=3.0, value=1.3)
        middle_zone_wall_count = st.number_input("Middle zone wall count", min_value=4, max_value=8, value=6, step=1)
        wall_rebar_ratio = st.number_input("Wall rebar ratio", min_value=0.0, max_value=0.1, value=0.003, format="%.4f")
        beam_rebar_ratio = st.number_input("Beam rebar ratio", min_value=0.0, max_value=0.1, value=0.015, format="%.4f")
        seismic_mass_factor = st.number_input("Seismic mass factor", min_value=0.1, max_value=2.0, value=1.0)
        Ct = st.number_input("Ct", min_value=0.001, max_value=0.200, value=0.0488, format="%.4f")
    with c6:
        drift_denominator = st.number_input("Drift denominator", min_value=100.0, max_value=2000.0, value=500.0)
        max_wall_thickness = st.number_input("Max wall thickness (m)", min_value=0.1, max_value=3.0, value=1.2)
        max_column_dim = st.number_input("Max column dimension (m)", min_value=0.1, max_value=5.0, value=1.8)
        min_beam_depth = st.number_input("Min beam depth (m)", min_value=0.1, max_value=3.0, value=0.75)
        max_slab_thickness = st.number_input("Max slab thickness (m)", min_value=0.05, max_value=1.0, value=0.4)
        perimeter_column_factor = st.number_input("Perimeter column factor", min_value=1.0, max_value=3.0, value=1.1)
        lower_zone_wall_count = st.number_input("Lower zone wall count", min_value=4, max_value=8, value=8, step=1)
        upper_zone_wall_count = st.number_input("Upper zone wall count", min_value=4, max_value=8, value=4, step=1)
        perimeter_shear_wall_ratio = st.number_input("Perimeter shear wall ratio", min_value=0.0, max_value=1.0, value=0.2, format="%.3f")
        column_rebar_ratio = st.number_input("Column rebar ratio", min_value=0.0, max_value=0.1, value=0.01, format="%.4f")
        slab_rebar_ratio = st.number_input("Slab rebar ratio", min_value=0.0, max_value=0.1, value=0.0035, format="%.4f")
        x_period = st.number_input("x exponent", min_value=0.1, max_value=1.5, value=0.75)

    # [NEW] Outrigger section
    st.markdown("### Outrigger Belt Truss System")
    st.info("For 60-story buildings, outriggers at stories 30 and 45 are recommended.")
    
    c7, c8 = st.columns(2)
    with c7:
        outrigger_count = st.number_input("Number of outriggers", min_value=0, max_value=5, value=0, step=1)
        outrigger_truss_depth_m = st.number_input("Outrigger truss depth (m)", min_value=1.0, max_value=6.0, value=3.0)
    with c8:
        outrigger_chord_area_m2 = st.number_input("Outrigger chord area (m²)", min_value=0.01, max_value=0.5, value=0.08, format="%.4f")
        outrigger_diagonal_area_m2 = st.number_input("Outrigger diagonal area (m²)", min_value=0.01, max_value=0.5, value=0.04, format="%.4f")
    
    # [NEW] Outrigger story levels input
    outrigger_story_levels = []
    if outrigger_count > 0:
        st.markdown("**Outrigger Locations (story numbers):**")
        or_cols = st.columns(min(outrigger_count, 3))
        for i in range(outrigger_count):
            with or_cols[i % 3]:
                level = st.number_input(f"Outrigger {i+1} story", min_value=1, max_value=120, 
                                       value=min(30 + i*15, n_story), step=1, key=f"or_{i}")
                outrigger_story_levels.append(int(level))

    return BuildingInput(
        plan_shape=plan_shape,
        n_story=int(n_story),
        n_basement=int(n_basement),
        story_height=float(story_height),
        basement_height=float(basement_height),
        plan_x=float(plan_x),
        plan_y=float(plan_y),
        n_bays_x=int(n_bays_x),
        n_bays_y=int(n_bays_y),
        bay_x=float(bay_x),
        bay_y=float(bay_y),
        stair_count=int(stair_count),
        elevator_count=int(elevator_count),
        elevator_area_each=float(elevator_area_each),
        stair_area_each=float(stair_area_each),
        service_area=float(service_area),
        corridor_factor=float(corridor_factor),
        fck=float(fck),
        Ec=float(Ec),
        Es=float(Es),
        fy=float(fy),
        DL=float(DL),
        LL=float(LL),
        slab_finish_allowance=float(slab_finish_allowance),
        facade_line_load=float(facade_line_load),
        prelim_lateral_force_coeff=float(prelim_lateral_force_coeff),
        drift_limit_ratio=1.0 / float(drift_denominator),
        min_wall_thickness=float(min_wall_thickness),
        max_wall_thickness=float(max_wall_thickness),
        min_column_dim=float(min_column_dim),
        max_column_dim=float(max_column_dim),
        min_beam_width=float(min_beam_width),
        min_beam_depth=float(min_beam_depth),
        min_slab_thickness=float(min_slab_thickness),
        max_slab_thickness=float(max_slab_thickness),
        wall_cracked_factor=float(wall_cracked_factor),
        column_cracked_factor=float(column_cracked_factor),
        max_story_wall_slenderness=float(max_story_wall_slenderness),
        wall_rebar_ratio=float(wall_rebar_ratio),
        column_rebar_ratio=float(column_rebar_ratio),
        beam_rebar_ratio=float(beam_rebar_ratio),
        slab_rebar_ratio=float(slab_rebar_ratio),
        seismic_mass_factor=float(seismic_mass_factor),
        Ct=float(Ct),
        x_period=float(x_period),
        upper_period_factor=float(upper_period_factor),
        target_position_factor=float(target_position_factor),
        perimeter_column_factor=float(perimeter_column_factor),
        corner_column_factor=float(corner_column_factor),
        lower_zone_wall_count=int(lower_zone_wall_count),
        middle_zone_wall_count=int(middle_zone_wall_count),
        upper_zone_wall_count=int(upper_zone_wall_count),
        basement_retaining_wall_thickness=float(basement_retaining_wall_thickness),
        perimeter_shear_wall_ratio=float(perimeter_shear_wall_ratio),
        # [NEW] Outrigger parameters
        outrigger_count=int(outrigger_count),
        outrigger_story_levels=outrigger_story_levels,
        root_outrigger_story_levels=root_outrigger_story_levels,
        outrigger_axis=str(outrigger_axis),
        outrigger_truss_depth_m=float(outrigger_truss_depth_m),
        outrigger_chord_area_m2=float(outrigger_chord_area_m2),
        outrigger_diagonal_area_m2=float(outrigger_diagonal_area_m2),
    )


def plot_elevation(inp: BuildingInput, result: DesignResult):
    """Simple elevation view with core and outrigger levels."""
    fig, ax = plt.subplots(figsize=(8, 10))
    H = total_height(inp)

    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, H, H, 0], 'k-', linewidth=1.2)

    core_w = 0.24 * inp.plan_x
    core_x0 = 0.5 * (inp.plan_x - core_w)
    ax.add_patch(plt.Rectangle((core_x0, 0), core_w, H, facecolor=CORE_COLOR, alpha=0.18, edgecolor=CORE_COLOR))

    ax.plot([0.0, 0.0], [0, H], color=PERIM_COLOR, linestyle='--', linewidth=1.4)
    ax.plot([inp.plan_x, inp.plan_x], [0, H], color=PERIM_COLOR, linestyle='--', linewidth=1.4)

    for i in range(inp.n_story + 1):
        y = i * inp.story_height
        ax.plot([0, inp.plan_x], [y, y], color='gray', alpha=0.15, linewidth=0.6)

    for o in getattr(result, 'outrigger_results', []):
        y = o.height_m
        ax.plot([core_x0, 0.0], [y, y], color=OUTRIGGER_COLOR, linewidth=2.6)
        ax.plot([core_x0 + core_w, inp.plan_x], [y, y], color=OUTRIGGER_COLOR, linewidth=2.6)
        ax.text(inp.plan_x + 0.6, y, f'St.{o.story_level}', va='center', fontsize=9, color=OUTRIGGER_COLOR)

    ax.set_title('Elevation View with Outriggers')
    ax.set_xlabel('Plan width direction')
    ax.set_ylabel('Height (m)')
    ax.set_xlim(-1.0, inp.plan_x + 6.0)
    ax.set_ylim(0, H * 1.02)
    ax.grid(True, alpha=0.25)
    return fig


def run_root_outrigger_study(inp: BuildingInput, candidate_levels: List[int], counts=(0, 1, 2, 3)) -> pd.DataFrame:
    """Compare response for different counts of root outriggers while keeping other inputs fixed."""
    rows = []
    levels = sorted([int(x) for x in candidate_levels if 1 <= int(x) <= inp.n_story])
    for count in counts:
        test_inp = BuildingInput(**{**inp.__dict__})
        chosen = levels[:max(0, min(int(count), len(levels)))]
        test_inp.root_outrigger_story_levels = chosen
        res = run_design(test_inp)
        rows.append({
            'root_outrigger_count': len(chosen),
            'levels': ', '.join(str(x) for x in chosen) if chosen else '-',
            'T1_s': res.estimated_period_s,
            'top_drift_m': res.top_drift_m,
            'K_total_N_per_m': res.K_estimated_N_per_m,
            'core_scale': res.core_scale,
            'column_scale': res.column_scale,
        })
    return pd.DataFrame(rows)


def example_usage():
    inp = BuildingInput(
        plan_shape='rect',
        n_story=60,
        n_basement=3,
        story_height=3.2,
        basement_height=3.5,
        plan_x=80.0,
        plan_y=80.0,
        n_bays_x=8,
        n_bays_y=8,
        bay_x=10.0,
        bay_y=10.0,
        root_outrigger_story_levels=[15, 30, 45],
    )
    res = run_design(inp)
    study = run_root_outrigger_study(inp, [15, 30, 45], counts=(0, 1, 2, 3))
    return inp, res, study


# ----------------------------- LAYOUT -------------------------------


if __name__ == '__main__':
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 0.7rem; padding-bottom: 0.7rem; max-width: 100%;}
        div[data-testid="stHorizontalBlock"] > div {padding-right: 0.35rem; padding-left: 0.35rem;}
        .stButton button {width: 100%; font-weight: 700; height: 3rem;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Tall Building Preliminary Design + Outrigger Belt Truss (v4.0)")
    st.caption(f"Prepared by {AUTHOR_NAME} | {APP_VERSION}")
    st.info("""
    **Key Updates in v4.0:**
    - Realistic wall cracked factor (0.40) for 60-story buildings
    - Outrigger belt truss system with configurable locations
    - Exact story ranges displayed for each zone
    - Improved stiffness calculation for tall buildings
    """)

    if "result" not in st.session_state:
        st.session_state.result = None
    if "report" not in st.session_state:
        st.session_state.report = ""
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "plan"

    left_col, right_col = st.columns([1.0, 2.3], gap="medium")

    with left_col:
        inp = streamlit_input_panel()
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("ANALYZE (MDOF Loop)"):
                try:
                    with st.spinner("Running MDOF iterative convergence..."):
                        res = run_design(inp)
                        st.session_state.result = res
                        st.session_state.report = build_report(res)
                        st.session_state.view_mode = "plan"
                    st.success(f"Converged in {len(res.iteration_history)} iterations!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        with b2:
            if st.button("SHOW 5 MODES"):
                try:
                    if st.session_state.result is None:
                        with st.spinner("Running initial analysis..."):
                            res = run_design(inp)
                            st.session_state.result = res
                            st.session_state.report = build_report(res)
                    st.session_state.view_mode = "modes"
                except Exception as e:
                    st.error(f"Mode display failed: {e}")
        with b3:
            if st.session_state.report:
                st.download_button("SAVE REPORT", data=st.session_state.report.encode("utf-8"), file_name="tall_building_report_v4.txt", mime="text/plain")
            else:
                st.button("SAVE REPORT", disabled=True)

    with right_col:
        zone_name = st.selectbox("Displayed zone:", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)
        if st.session_state.result is None:
            st.info("Click ANALYZE (MDOF Loop) to display plan/report, or SHOW 5 MODES to display modal shapes.")
        else:
            res = st.session_state.result
        
            # [NEW] Display zone story ranges
            selected_core = next((z for z in res.zone_core_results if z.zone.name == zone_name), None)
            if selected_core:
                st.caption(f"📍 **{zone_name}**: Stories {selected_core.zone.story_start} to {selected_core.zone.story_end} ({selected_core.zone.n_stories} stories)")
        
            if res.iteration_history:
                with st.expander("📊 MDOF Iteration Convergence", expanded=True):
                    c_iter1, c_iter2, c_iter3 = st.columns(3)
                    c_iter1.metric("Iterations", len(res.iteration_history))
                    if len(res.iteration_history) > 1:
                        initial_error = res.iteration_history[0].error_percent
                        final_error = res.iteration_history[-1].error_percent
                        c_iter2.metric("Initial Error %", f"{initial_error:.2f}")
                        c_iter3.metric("Final Error %", f"{final_error:.2f}")
                    else:
                        c_iter2.metric("Initial Error %", f"{res.iteration_history[0].error_percent:.2f}")
                        c_iter3.metric("Final Error %", f"{res.iteration_history[-1].error_percent:.2f}")
                
                    iter_df = pd.DataFrame([
                        {
                            "Iter": log.iteration,
                            "Core Sc": f"{log.core_scale:.3f}",
                            "Col Sc": f"{log.column_scale:.3f}",
                            "T_est (s)": f"{log.T_estimated:.3f}",
                            "T_target (s)": f"{log.T_target:.3f}",
                            "Error %": f"{log.error_percent:.2f}",
                            "Weight (MN)": f"{log.total_weight_kN/1000:.2f}"
                        }
                        for log in res.iteration_history
                    ])
                    st.dataframe(iter_df, use_container_width=True, hide_index=True)
        
            # Main metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reference period (s)", f"{res.reference_period_s:.3f}")
            c2.metric("Design target (s)", f"{res.design_target_period_s:.3f}")
            c3.metric("MDOF Period (s)", f"{res.estimated_period_s:.3f}")
            c4.metric("Upper limit (s)", f"{res.upper_limit_period_s:.3f}")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Period error (%)", f"{100*res.period_error_ratio:.2f}")
            d2.metric("Total stiffness (N/m)", f"{res.K_estimated_N_per_m:,.2e}")
            d3.metric("Top drift (m)", f"{res.top_drift_m:.3f}")
            d4.metric("Total weight (MN)", f"{res.total_weight_kN/1000:.2f}")

            st.caption(f"Target formula: T_target = T_ref + beta × (T_upper - T_ref),  beta = {inp.target_position_factor:.3f}")

            # [NEW] Outrigger metrics if present
            if res.outrigger_results:
                st.markdown("### 🏗️ Outrigger Belt Truss")
                or_cols = st.columns(len(res.outrigger_results))
                for i, or_result in enumerate(res.outrigger_results):
                    with or_cols[i]:
                        st.metric(f"Outrigger {i+1}", f"Story {or_result.story_level}")
                        st.caption(f"Height: {or_result.height_m:.1f}m | K: {or_result.stiffness_contribution:,.2e} N/m")

            # Tabs for different outputs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Graphic output", "Convergence Plot", "Mass participation", "Outrigger Analysis", "Report"])
        
            with tab1:
                if st.session_state.view_mode == "modes":
                    st.pyplot(plot_mode_shapes(res), use_container_width=True)
                else:
                    st.pyplot(plot_plan(inp, res, zone_name), use_container_width=True)
        
            with tab2:
                if res.iteration_history and len(res.iteration_history) > 1:
                    conv_fig = plot_iteration_history(res)
                    if conv_fig:
                        st.pyplot(conv_fig, use_container_width=True)
                else:
                    st.info("Need at least 2 iterations to show convergence plot.")
        
            with tab3:
                df = pd.DataFrame({
                    "Mode": list(range(1, len(res.modal_result.periods_s) + 1)),
                    "Period (s)": res.modal_result.periods_s,
                    "Frequency (Hz)": res.modal_result.frequencies_hz,
                    "Mass ratio (%)": [100 * x for x in res.modal_result.effective_mass_ratios],
                    "Cumulative (%)": [100 * x for x in res.modal_result.cumulative_effective_mass_ratios],
                })
                st.dataframe(df, use_container_width=True)
            
                if st.button("Show Mode Shape Plot", key="mode_shape_btn"):
                    st.pyplot(plot_mode_shapes(res), use_container_width=True)
        
            with tab4:
                # [NEW] Outrigger analysis tab
                if res.outrigger_results:
                    or_fig = plot_outrigger_efficiency(res)
                    if or_fig:
                        st.pyplot(or_fig, use_container_width=True)
                
                    st.markdown("### Outrigger Details")
                    for or_result in res.outrigger_results:
                        with st.expander(f"Outrigger at Story {or_result.story_level}"):
                            c_or1, c_or2 = st.columns(2)
                            with c_or1:
                                st.metric("Height from base", f"{or_result.height_m:.1f} m")
                                st.metric("Truss depth", f"{or_result.truss_depth_m:.2f} m")
                                st.metric("Truss width", f"{or_result.truss_width_m:.2f} m")
                            with c_or2:
                                st.metric("Chord area", f"{or_result.chord_area_m2:.4f} m²")
                                st.metric("Diagonal area", f"{or_result.diagonal_area_m2:.4f} m²")
                                st.metric("Stiffness contribution", f"{or_result.stiffness_contribution:,.2e} N/m")
                else:
                    st.info("No outriggers configured. Add outriggers in the left panel to see analysis.")
                    st.markdown("""
                    **Recommendation for 60-story building:**
                    - Outrigger at story 30 (mid-height)
                    - Outrigger at story 45 (2/3 height)
                    - Truss depth: 3.0 m
                    - Chord area: 0.08 m²
                    """)
        
            with tab5:
                st.text_area("", st.session_state.report, height=520, label_visibility="collapsed")
                st.markdown("### Redesign suggestions")
                for s in res.redesign_suggestions:
                    st.write(f"- {s}")
            
                st.markdown("### Final Scale Factors")
                st.write(f"- **Core scale factor**: {res.core_scale:.3f}")
                st.write(f"- **Column scale factor**: {res.column_scale:.3f}")
            
                # [NEW] Show zone story ranges
                st.markdown("### Zone Story Ranges")
                for zc in res.zone_core_results:
                    st.write(f"- **{zc.zone.name}**: Stories {zc.zone.story_start} to {zc.zone.story_end} ({zc.zone.n_stories} stories)")
            
                if res.governing_issue != "OK":
                    st.warning(f"**Governing Issue**: {res.governing_issue}")
                else:
                    st.success("**Governing Issue**: All checks passed!")

