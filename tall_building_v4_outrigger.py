from __future__ import annotations
from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

G = 9.81
STEEL_DENSITY = 7850.0


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
class BuildingInput:
    plan_shape: str = "square"
    n_story: int = 60
    n_basement: int = 3
    story_height: float = 3.2
    basement_height: float = 3.6
    plan_x: float = 80.0
    plan_y: float = 80.0
    n_bays_x: int = 8
    n_bays_y: int = 8
    bay_x: float = 10.0
    bay_y: float = 10.0

    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each: float = 3.5
    stair_area_each: float = 20.0
    service_area: float = 35.0
    corridor_factor: float = 1.40

    fck: float = 70.0
    Ec: float = 36000.0  # MPa
    Es: float = 200000.0  # MPa
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

    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.70
    max_story_wall_slenderness: float = 12.0

    wall_rebar_ratio: float = 0.003
    column_rebar_ratio: float = 0.010
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.0035

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

    # Corrected outrigger controls
    outrigger_story_levels: List[int] = field(default_factory=list)
    root_outrigger_story_levels: List[int] = field(default_factory=list)
    outrigger_axis: str = "x"   # x or y
    outrigger_truss_depth_m: float = 3.0
    outrigger_chord_area_m2: float = 0.08
    outrigger_diagonal_area_m2: float = 0.04
    lock_sections_for_comparison: bool = True


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
    story_level: int
    axis: str
    height_m: float
    arm_m: float
    truss_depth_m: float
    chord_area_m2: float
    diagonal_area_m2: float
    engaged_column_count: int
    k_col_N_per_m: float
    k_truss_N_per_m: float
    k_tip_N_per_m: float
    K_theta_Nm_per_rad: float
    k_story_add_N_per_m: float
    equivalent_I_addition_m4: float


@dataclass
class ModalResult:
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    story_masses_kg: List[float]
    story_stiffness_N_per_m: List[float]


@dataclass
class DesignResult:
    total_weight_kN: float
    effective_modal_mass_kg: float
    reference_period_s: float
    design_target_period_s: float
    upper_limit_period_s: float
    estimated_period_s: float
    K_core_N_per_m: float
    K_cols_N_per_m: float
    K_outrigger_equiv_N_per_m: float
    K_estimated_N_per_m: float
    top_drift_m: float
    drift_ratio: float
    slab_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    zone_core_results: List[ZoneCoreResult]
    zone_column_results: List[ZoneColumnResult]
    outriggers: List[OutriggerResult]
    modal_result: ModalResult
    story_stiffnesses_N_per_m: List[float]
    messages: List[str] = field(default_factory=list)


# ----------------------------- BASICS -----------------------------
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


def opening_dimensions(inp: BuildingInput) -> Tuple[float, float]:
    area = required_opening_area(inp)
    aspect = 1.6
    oy = sqrt(area / aspect)
    return aspect * oy, oy


def initial_core_dimensions(inp: BuildingInput, opening_x: float, opening_y: float) -> Tuple[float, float]:
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


def beam_size_prelim(inp: BuildingInput, column_scale: float) -> Tuple[float, float]:
    span = max(inp.bay_x, inp.bay_y)
    depth = max(inp.min_beam_depth, min(2.0, (span / 12.0) * (0.90 + 0.15 * column_scale)))
    width = max(inp.min_beam_width, 0.45 * depth)
    return width, depth


def directional_dims(base_dim: float, plan_x: float, plan_y: float) -> Tuple[float, float]:
    aspect = max(plan_x, plan_y) / max(min(plan_x, plan_y), 1e-9)
    if aspect <= 1.10:
        return base_dim, base_dim
    major = base_dim * 1.15
    minor = base_dim * 0.90
    return (major, minor) if plan_x >= plan_y else (minor, major)


def build_zone_results(inp: BuildingInput, core_scale: float, column_scale: float, slab_t: float) -> Tuple[List[ZoneCoreResult], List[ZoneColumnResult]]:
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

        cracked_factor = inp.wall_cracked_factor * (0.85 if inp.n_story > 40 else 1.0)
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


def total_weight_kN_from_quantities(inp: BuildingInput, slab_t: float) -> float:
    area = floor_area(inp)
    n_total_levels = inp.n_story + inp.n_basement
    q = inp.DL + 0.25 * inp.LL + inp.slab_finish_allowance + 25.0 * slab_t
    facade = 2.0 * (inp.plan_x + inp.plan_y) * inp.story_height * inp.facade_line_load * n_total_levels
    return area * n_total_levels * q + facade


# ----------------------------- CORRECTED OUTRIGGER MODEL -----------------------------
def get_base_story_stiffnesses(inp: BuildingInput, K_total: float) -> np.ndarray:
    # same general logic as original app: stronger lower stories, softer upper stories
    n = inp.n_story
    if n == 1:
        return np.array([K_total], dtype=float)
    weights = np.linspace(1.35, 0.65, n)
    weights = weights / weights.mean()
    k = K_total * weights
    return k.astype(float)


def build_story_masses(inp: BuildingInput, total_weight_kN: float) -> np.ndarray:
    w_story = total_weight_kN * 1000.0 / inp.n_story
    return np.full(inp.n_story, w_story / G, dtype=float)


def assemble_story_spring_matrix(k_story: np.ndarray) -> np.ndarray:
    n = len(k_story)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += k_story[i]
        else:
            kij = k_story[i]
            K[i, i] += kij
            K[i - 1, i - 1] += kij
            K[i, i - 1] -= kij
            K[i - 1, i] -= kij
    return K


def assemble_mass_matrix(masses: np.ndarray) -> np.ndarray:
    return np.diag(masses)


def _select_zone(zone_results: List, story: int):
    for z in zone_results:
        if z.zone.story_start <= story <= z.zone.story_end:
            return z
    return zone_results[-1]


def _perimeter_column_area_for_story(inp: BuildingInput, zone_cols: List[ZoneColumnResult], story: int) -> float:
    z = _select_zone(zone_cols, story)
    return z.perimeter_column_x_m * z.perimeter_column_y_m


def _core_dims(inp: BuildingInput, zone_cores: List[ZoneCoreResult], story: int) -> Tuple[float, float]:
    z = _select_zone(zone_cores, story)
    return z.core_outer_x, z.core_outer_y


def calculate_corrected_outrigger(inp: BuildingInput, zone_cores: List[ZoneCoreResult], zone_cols: List[ZoneColumnResult], story_level: int, axis: str, truss_depth: float, chord_area: float, diagonal_area: float) -> OutriggerResult:
    height_m = story_level * inp.story_height
    core_x, core_y = _core_dims(inp, zone_cores, story_level)
    arm_x = max((inp.plan_x - core_x) / 2.0, 1e-6)
    arm_y = max((inp.plan_y - core_y) / 2.0, 1e-6)
    axis = axis.lower()
    arm = arm_x if axis == "x" else arm_y

    # engaged exterior column lines
    if axis == "x":
        engaged_column_count = 2 * (inp.n_bays_y + 1)  # left and right faces
    else:
        engaged_column_count = 2 * (inp.n_bays_x + 1)  # bottom and top faces

    Ecol = inp.Ec * 1e6
    Estr = inp.Es * 1e6
    Acol = _perimeter_column_area_for_story(inp, zone_cols, story_level)
    Lcol = max(height_m, inp.story_height)
    k_col = engaged_column_count * Ecol * Acol / Lcol

    # truss flexibility: bending of chords + diagonal action
    d = max(truss_depth, 1e-6)
    I_or = 0.5 * chord_area * d**2
    k_bend = 3.0 * Estr * I_or / max(arm**3, 1e-9)
    Ld = sqrt(arm**2 + d**2)
    k_diag = 2.0 * Estr * diagonal_area * d**2 / max(Ld**3, 1e-9)
    k_truss = k_bend + k_diag

    # series combination of truss and exterior columns at one outrigger level
    k_tip = 1.0 / (1.0 / max(k_truss, 1e-9) + 1.0 / max(k_col, 1e-9))
    K_theta = 2.0 * k_tip * arm**2

    # convert rotational restraint to equivalent local translational spring for story-shear model
    k_story_add = K_theta / max(inp.story_height**2, 1e-9)
    effective_I_addition = k_story_add * total_height(inp)**3 / max(3.0 * inp.Ec * 1e6, 1e-9)

    return OutriggerResult(
        story_level=story_level,
        axis=axis,
        height_m=height_m,
        arm_m=arm,
        truss_depth_m=truss_depth,
        chord_area_m2=chord_area,
        diagonal_area_m2=diagonal_area,
        engaged_column_count=engaged_column_count,
        k_col_N_per_m=k_col,
        k_truss_N_per_m=k_truss,
        k_tip_N_per_m=k_tip,
        K_theta_Nm_per_rad=K_theta,
        k_story_add_N_per_m=k_story_add,
        equivalent_I_addition_m4=effective_I_addition,
    )


def calculate_total_outrigger_stiffness_corrected(inp: BuildingInput, zone_cores: List[ZoneCoreResult], zone_cols: List[ZoneColumnResult]) -> Tuple[List[OutriggerResult], float, Dict[int, float]]:
    levels = sorted(set(inp.outrigger_story_levels + inp.root_outrigger_story_levels))
    if not levels:
        return [], 0.0, {}
    outriggers: List[OutriggerResult] = []
    by_story: Dict[int, float] = {}
    total = 0.0
    for level in levels:
        if 1 <= level <= inp.n_story:
            o = calculate_corrected_outrigger(
                inp, zone_cores, zone_cols, level, inp.outrigger_axis,
                inp.outrigger_truss_depth_m, inp.outrigger_chord_area_m2,
                inp.outrigger_diagonal_area_m2,
            )
            outriggers.append(o)
            by_story[level] = by_story.get(level, 0.0) + o.k_story_add_N_per_m
            total += o.k_story_add_N_per_m
    return outriggers, total, by_story


def solve_mdof_modes_corrected(inp: BuildingInput, total_weight_kN_value: float, K_base_total: float, story_additions: Optional[Dict[int, float]] = None, n_modes: int = 5) -> Tuple[ModalResult, np.ndarray, np.ndarray, np.ndarray]:
    masses = build_story_masses(inp, total_weight_kN_value)
    k_stories = get_base_story_stiffnesses(inp, K_base_total)
    if story_additions:
        for story, dk in story_additions.items():
            idx = int(story) - 1
            if 0 <= idx < len(k_stories):
                k_stories[idx] += dk

    M = assemble_mass_matrix(masses)
    K = assemble_story_spring_matrix(k_stories)
    eigvals, eigvecs = eigh(K, M)
    keep = eigvals > 1e-12
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    omegas = np.sqrt(eigvals[:n_modes])
    periods = [2.0 * pi / w for w in omegas]
    freqs = [w / (2.0 * pi) for w in omegas]
    mode_shapes = []
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].copy()
        phi /= np.max(np.abs(phi)) if np.max(np.abs(phi)) > 0 else 1.0
        mode_shapes.append(phi.tolist())

    modal = ModalResult(
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=mode_shapes,
        story_masses_kg=masses.tolist(),
        story_stiffness_N_per_m=k_stories.tolist(),
    )
    return modal, masses, k_stories, K


def evaluate_design(inp: BuildingInput, core_scale: float = 1.0, column_scale: float = 1.0) -> DesignResult:
    H = total_height(inp)
    T_ref = code_type_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    T_target = T_ref + inp.target_position_factor * (T_upper - T_ref)

    slab_t = slab_thickness_prelim(inp, column_scale)
    beam_b, beam_h = beam_size_prelim(inp, column_scale)
    zone_cores, zone_cols = build_zone_results(inp, core_scale, column_scale, slab_t)
    W_total = total_weight_kN_from_quantities(inp, slab_t)
    M_eff = W_total * 1000.0 / G

    K_core = weighted_core_stiffness(inp, zone_cores)
    K_cols = weighted_column_stiffness(inp, zone_cols)
    K_base = K_core + K_cols

    outriggers, K_outrigger_equiv, by_story = calculate_total_outrigger_stiffness_corrected(inp, zone_cores, zone_cols)
    modal, masses, k_story, Kmat = solve_mdof_modes_corrected(inp, W_total, K_base, by_story, n_modes=5)
    T_est = modal.periods_s[0]

    # static drift with same MDOF K matrix
    F = np.full(inp.n_story, preliminary_lateral_force_N(inp, W_total) / inp.n_story, dtype=float)
    u = np.linalg.solve(Kmat, F)
    top_drift = float(u[-1])
    drift_ratio = top_drift / H

    messages = [
        f"Base stiffness (core + columns) = {K_base:,.3e} N/m",
        f"Equivalent added outrigger stiffness = {K_outrigger_equiv:,.3e} N/m",
        f"T1 = {T_est:.3f} s",
        f"Top drift = {top_drift:.4f} m",
    ]
    if outriggers:
        for o in outriggers:
            messages.append(
                f"Outrigger story {o.story_level}: Kθ={o.K_theta_Nm_per_rad:,.3e} Nm/rad, Δk_story={o.k_story_add_N_per_m:,.3e} N/m"
            )

    return DesignResult(
        total_weight_kN=W_total,
        effective_modal_mass_kg=M_eff,
        reference_period_s=T_ref,
        design_target_period_s=T_target,
        upper_limit_period_s=T_upper,
        estimated_period_s=T_est,
        K_core_N_per_m=K_core,
        K_cols_N_per_m=K_cols,
        K_outrigger_equiv_N_per_m=K_outrigger_equiv,
        K_estimated_N_per_m=K_base + K_outrigger_equiv,
        top_drift_m=top_drift,
        drift_ratio=drift_ratio,
        slab_thickness_m=slab_t,
        beam_width_m=beam_b,
        beam_depth_m=beam_h,
        zone_core_results=zone_cores,
        zone_column_results=zone_cols,
        outriggers=outriggers,
        modal_result=modal,
        story_stiffnesses_N_per_m=k_story.tolist(),
        messages=messages,
    )


# ----------------------------- SENSITIVITY / ROOT OUTRIGGER STUDY -----------------------------
def run_root_outrigger_study(inp: BuildingInput, root_candidates: List[int], counts=(0, 1, 2, 3), keep_sections_locked: bool = True) -> pd.DataFrame:
    rows = []
    base_core_scale = 1.0
    base_col_scale = 1.0

    # locked-sections comparison: only outriggers change
    for c in counts:
        new_inp = BuildingInput(**{**inp.__dict__})
        new_inp.root_outrigger_story_levels = root_candidates[:c]
        new_inp.lock_sections_for_comparison = keep_sections_locked
        res = evaluate_design(new_inp, base_core_scale, base_col_scale)
        rows.append({
            "root_outrigger_count": c,
            "root_levels": str(root_candidates[:c]),
            "T1_s": res.estimated_period_s,
            "K_base_GN_per_m": (res.K_core_N_per_m + res.K_cols_N_per_m) / 1e9,
            "K_outrigger_GN_per_m": res.K_outrigger_equiv_N_per_m / 1e9,
            "K_total_GN_per_m": res.K_estimated_N_per_m / 1e9,
            "TopDrift_m": res.top_drift_m,
            "DriftRatio": res.drift_ratio,
        })
    return pd.DataFrame(rows)


# ----------------------------- PLOTTING -----------------------------
def plot_plan(inp: BuildingInput, result: DesignResult, story_level: Optional[int] = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)

    # grid / columns
    xs, ys = [], []
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            xs.append(i * inp.bay_x)
            ys.append(j * inp.bay_y)
    ax.scatter(xs, ys, s=20, label="Columns")

    core = _select_zone(result.zone_core_results, story_level or 1)
    cx0 = 0.5 * (inp.plan_x - core.core_outer_x)
    cy0 = 0.5 * (inp.plan_y - core.core_outer_y)
    rect = plt.Rectangle((cx0, cy0), core.core_outer_x, core.core_outer_y, fill=False, linewidth=2.5)
    ax.add_patch(rect)

    use_outriggers = result.outriggers if story_level is None else [o for o in result.outriggers if o.story_level == story_level]
    for idx, o in enumerate(use_outriggers):
        if o.axis == "x":
            y_mid = 0.5 * inp.plan_y
            ax.plot([cx0, 0.0], [y_mid, y_mid], linewidth=2.2)
            ax.plot([cx0 + core.core_outer_x, inp.plan_x], [y_mid, y_mid], linewidth=2.2)
            ax.plot([0.0, 0.0], [0.0, inp.plan_y], linestyle="--", linewidth=1.4)
            ax.plot([inp.plan_x, inp.plan_x], [0.0, inp.plan_y], linestyle="--", linewidth=1.4)
            ax.text(cx0 + core.core_outer_x + 0.5, y_mid + 1.0, f"OR @ Story {o.story_level}")
        else:
            x_mid = 0.5 * inp.plan_x
            ax.plot([x_mid, x_mid], [cy0, 0.0], linewidth=2.2)
            ax.plot([x_mid, x_mid], [cy0 + core.core_outer_y, inp.plan_y], linewidth=2.2)
            ax.plot([0.0, inp.plan_x], [0.0, 0.0], linestyle="--", linewidth=1.4)
            ax.plot([0.0, inp.plan_x], [inp.plan_y, inp.plan_y], linestyle="--", linewidth=1.4)
            ax.text(x_mid + 1.0, cy0 + core.core_outer_y + 0.5, f"OR @ Story {o.story_level}")

    ax.set_title("Corrected plan view: core-to-perimeter outrigger arms")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig, ax


def plot_elevation(inp: BuildingInput, result: DesignResult):
    fig, ax = plt.subplots(figsize=(8, 10))
    H = total_height(inp)
    ax.plot([0, 0], [0, H], color="black", linewidth=2.2, label="Core axis")
    ax.plot([1, 1], [0, H], color="gray", linewidth=1.8, label="Perimeter column line")
    for s in range(1, inp.n_story + 1):
        z = s * inp.story_height
        ax.plot([-0.15, 1.15], [z, z], color="lightgray", linewidth=0.5)
    for o in result.outriggers:
        z = o.story_level * inp.story_height
        ax.plot([0, 1], [z, z], linewidth=3.0)
        ax.text(1.05, z, f"Story {o.story_level}", va="center")
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(0, H)
    ax.set_title("Corrected elevation view of outriggers")
    ax.set_xlabel("Schematic width")
    ax.set_ylabel("Height (m)")
    return fig, ax


def plot_mode_shape(result: DesignResult, mode_index: int = 1):
    idx = mode_index - 1
    if idx >= len(result.modal_result.mode_shapes):
        raise ValueError("Requested mode is not available.")
    phi = np.array(result.modal_result.mode_shapes[idx])
    z = np.arange(1, len(phi) + 1) * (total_height(BuildingInput()) / len(phi))
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(phi, z, marker="o")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"Mode {mode_index} shape")
    ax.set_xlabel("Normalized displacement")
    ax.set_ylabel("Height (m)")
    return fig, ax


# ----------------------------- QUICK EXAMPLE -----------------------------
def example_usage() -> Tuple[DesignResult, pd.DataFrame]:
    inp = BuildingInput(
        outrigger_story_levels=[],
        root_outrigger_story_levels=[15, 30, 45],
        outrigger_axis="x",
        outrigger_truss_depth_m=3.0,
        outrigger_chord_area_m2=0.08,
        outrigger_diagonal_area_m2=0.04,
    )
    res = evaluate_design(inp)
    study = run_root_outrigger_study(inp, [15, 30, 45], counts=(0, 1, 2, 3))
    return res, study


if __name__ == "__main__":
    res, study = example_usage()
    print(study.to_string(index=False))
    print("\nT1 =", round(res.estimated_period_s, 4), "s")
    print("Top drift =", round(res.top_drift_m, 5), "m")
