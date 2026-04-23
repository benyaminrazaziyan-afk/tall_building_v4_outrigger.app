
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


G = 9.81
STEEL_DENSITY = 7850.0

APP_TITLE = "Tall Building Preliminary Design + Outrigger (Wall-Controlled)"
APP_VERSION = "v6.0-wall-outrigger-fix"

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"
OUTRIGGER_COLOR = "#ff6b00"


@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int

    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1


@dataclass
class OutriggerResult:
    story_level: int
    height_m: float
    truss_depth_m: float
    truss_width_m: float
    chord_area_m2: float
    diagonal_area_m2: float
    axial_stiffness_kN: float
    equivalent_spring_kN_m: float
    stiffness_contribution: float
    K_rotational_Nm_per_rad: float
    effective_I_addition_m4: float
    capped: bool = False


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
    outrigger_I_addition_m4: float = 0.0


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
    outrigger_steel_kg: float = 0.0


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
    K_core_N_m: float
    K_cols_N_m: float
    K_outrigger_N_m: float


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
    outrigger_results: List[OutriggerResult]
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
    modal_result: Optional[ModalResult] = None
    iteration_history: List[IterationLog] = field(default_factory=list)


@dataclass
class BuildingInput:
    plan_shape: str = "square"
    n_story: int = 60
    n_basement: int = 10
    story_height: float = 3.2
    basement_height: float = 3.0
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
    Ec: float = 36000.0
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

    wall_rebar_ratio: float = 0.003
    column_rebar_ratio: float = 0.010
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.0035

    seismic_mass_factor: float = 1.0
    Ct: float = 0.0488
    x_period: float = 0.75
    upper_period_factor: float = 1.20
    target_position_factor: float = 0.85

    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30

    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4

    basement_retaining_wall_thickness: float = 0.50
    perimeter_shear_wall_ratio: float = 0.20

    outrigger_count: int = 2
    outrigger_story_levels: List[int] = field(default_factory=lambda: [30, 45])
    outrigger_truss_depth_m: float = 3.0
    outrigger_chord_area_m2: float = 0.08
    outrigger_diagonal_area_m2: float = 0.04

    # new controls
    outrigger_core_influence: float = 0.90
    outrigger_column_influence: float = 0.05
    max_outrigger_share_of_core: float = 0.35


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
            ("top", 0.0, inp.plan_x),
            ("bottom", 0.0, inp.plan_x),
            ("left", 0.0, inp.plan_y),
            ("right", 0.0, inp.plan_y),
        ]
    ratio = inp.perimeter_shear_wall_ratio
    lx = inp.plan_x * ratio
    ly = inp.plan_y * ratio
    sx = (inp.plan_x - lx) / 2.0
    sy = (inp.plan_y - ly) / 2.0
    return [
        ("top", sx, sx + lx),
        ("bottom", sx, sx + lx),
        ("left", sy, sy + ly),
        ("right", sy, sy + ly),
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


def calculate_raw_outrigger_stiffness(inp: BuildingInput, story_level: int, truss_depth: float, chord_area: float, diagonal_area: float) -> OutriggerResult:
    E = inp.Ec * 1e6
    story_height = inp.story_height
    H_total = total_height(inp)
    height_m = story_level * story_height

    core_x, core_y = initial_core_dimensions(inp, *opening_dimensions(inp))
    truss_width_x = (inp.plan_x - core_x) / 2
    truss_width_y = (inp.plan_y - core_y) / 2
    truss_width = max(truss_width_x, truss_width_y)

    n_chords = 4
    chord_length = sqrt(truss_depth**2 + (truss_width / 2) ** 2)
    axial_stiffness = n_chords * E * chord_area / chord_length
    d_arm = truss_width
    K_rot = 2 * E * chord_area * d_arm**2 / max(truss_width, 1e-9)

    # moderated formula: no arbitrary *10
    height_ratio = height_m / H_total
    position_factor = 0.55 + 0.45 * (height_ratio * (2 - height_ratio))
    K_brace = (E * chord_area / chord_length) * (truss_width / max(story_height, 1e-9)) ** 2
    stiffness_contribution = 0.10 * K_brace * position_factor
    effective_I_addition = stiffness_contribution * (H_total**3) / (3 * E)

    return OutriggerResult(
        story_level=story_level,
        height_m=height_m,
        truss_depth_m=truss_depth,
        truss_width_m=truss_width,
        chord_area_m2=chord_area,
        diagonal_area_m2=diagonal_area,
        axial_stiffness_kN=axial_stiffness / 1000.0,
        equivalent_spring_kN_m=stiffness_contribution / 1000.0,
        stiffness_contribution=stiffness_contribution,
        K_rotational_Nm_per_rad=K_rot,
        effective_I_addition_m4=effective_I_addition,
        capped=False,
    )


def build_zone_results(inp: BuildingInput, core_scale: float, column_scale: float, slab_t: float, outriggers: Optional[List[OutriggerResult]] = None) -> tuple[List[ZoneCoreResult], List[ZoneColumnResult]]:
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

        zone_outrigger_I = 0.0
        if outriggers:
            for o in outriggers:
                if zone.story_start <= o.story_level <= zone.story_end:
                    zone_outrigger_I += o.effective_I_addition_m4

        I_eff_total = I_eff + zone_outrigger_I

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
                Ieq_effective_m4=I_eff_total,
                story_slenderness=inp.story_height / max(t, 1e-9),
                perimeter_wall_segments=perim,
                retaining_wall_active=(zone.name == "Lower Zone"),
                outrigger_I_addition_m4=zone_outrigger_I,
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


def calculate_total_outrigger_stiffness(inp: BuildingInput, base_core_K: float, base_col_K: float) -> tuple[List[OutriggerResult], float, float]:
    if inp.outrigger_count == 0 or not inp.outrigger_story_levels:
        return [], 0.0, 0.0

    outriggers: List[OutriggerResult] = []
    raw_total = 0.0

    for level in inp.outrigger_story_levels[:inp.outrigger_count]:
        if 1 <= level <= inp.n_story:
            o = calculate_raw_outrigger_stiffness(
                inp,
                level,
                inp.outrigger_truss_depth_m,
                inp.outrigger_chord_area_m2,
                inp.outrigger_diagonal_area_m2,
            )
            outriggers.append(o)
            raw_total += o.stiffness_contribution

    # cap total outrigger share relative to core
    max_total = inp.max_outrigger_share_of_core * base_core_K
    scale = 1.0
    capped = False
    if raw_total > max_total > 0:
        scale = max_total / raw_total
        capped = True

    total_core_bonus = 0.0
    total_col_bonus = 0.0
    for o in outriggers:
        o.stiffness_contribution *= scale
        o.equivalent_spring_kN_m = o.stiffness_contribution / 1000.0
        o.effective_I_addition_m4 *= scale
        o.capped = capped
        total_core_bonus += inp.outrigger_core_influence * o.stiffness_contribution
        total_col_bonus += inp.outrigger_column_influence * o.stiffness_contribution

    return outriggers, total_core_bonus, total_col_bonus


def estimate_reinforcement(inp: BuildingInput, zone_cores: List[ZoneCoreResult], zone_cols: List[ZoneColumnResult], slab_t: float, beam_b: float, beam_h: float, outriggers: Optional[List[OutriggerResult]] = None) -> ReinforcementEstimate:
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

    outrigger_steel = 0.0
    if outriggers:
        for o in outriggers:
            truss_length = 4 * o.truss_width_m
            chord_steel = truss_length * o.chord_area_m2 * STEEL_DENSITY
            diagonal_steel = truss_length * 0.5 * o.diagonal_area_m2 * STEEL_DENSITY
            outrigger_steel += (chord_steel + diagonal_steel) * 2

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


def build_story_stiffnesses(inp: BuildingInput, K_total: float) -> List[float]:
    n = inp.n_story
    raw = []
    for i in range(n):
        r = i / max(n - 1, 1)
        raw.append(1.35 - 0.55 * r)
    inv_sum = sum(1.0 / a for a in raw)
    c = K_total * inv_sum
    return [c * a for a in raw]


def assemble_m_k_with_outriggers(story_masses: List[float], story_stiffness: List[float], outriggers: List[OutriggerResult], story_height: float) -> tuple[np.ndarray, np.ndarray]:
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

    for o in outriggers:
        story_idx = o.story_level - 1
        if 0 <= story_idx < n:
            h = story_height
            # softened rotational contribution for stability
            K_rot_eff = 0.20 * o.K_rotational_Nm_per_rad / (h**2)
            K[story_idx, story_idx] += K_rot_eff
            if story_idx > 0:
                K[story_idx, story_idx - 1] -= K_rot_eff * 0.5
                K[story_idx - 1, story_idx] -= K_rot_eff * 0.5
                K[story_idx - 1, story_idx - 1] += K_rot_eff * 0.5
            if story_idx < n - 1:
                K[story_idx, story_idx + 1] -= K_rot_eff * 0.5
                K[story_idx + 1, story_idx] -= K_rot_eff * 0.5
                K[story_idx + 1, story_idx + 1] += K_rot_eff * 0.5

    return M, K


def solve_mdof_modes(inp: BuildingInput, total_weight_kN_value: float, K_total: float, outriggers: Optional[List[OutriggerResult]] = None, n_modes: int = 5) -> ModalResult:
    masses = build_story_masses(inp, total_weight_kN_value)
    k_stories = build_story_stiffnesses(inp, K_total)
    if outriggers:
        M, K = assemble_m_k_with_outriggers(masses, k_stories, outriggers, inp.story_height)
    else:
        M = np.diag(masses)
        K = np.zeros((len(masses), len(masses)), dtype=float)
        for i in range(len(masses)):
            ki = k_stories[i]
            if i == 0:
                K[i, i] += ki
            else:
                K[i, i] += ki
                K[i, i - 1] -= ki
                K[i - 1, i] -= ki
                K[i - 1, i - 1] += ki

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


def evaluate_design(inp: BuildingInput, core_scale: float, column_scale: float, beta: float) -> dict:
    H = total_height(inp)
    T_ref = code_type_period(H, inp.Ct, inp.x_period)
    T_upper = inp.upper_period_factor * T_ref
    T_target = T_ref + beta * (T_upper - T_ref)

    slab_t = slab_thickness_prelim(inp, column_scale)
    beam_b, beam_h = beam_size_prelim(inp, column_scale)

    # first pass without outrigger I-addition
    zone_cores0, zone_cols = build_zone_results(inp, core_scale, column_scale, slab_t, outriggers=None)
    K_core0 = weighted_core_stiffness(inp, zone_cores0)
    K_cols0 = weighted_column_stiffness(inp, zone_cols)

    outriggers, K_outrigger_core, K_outrigger_cols = calculate_total_outrigger_stiffness(inp, K_core0, K_cols0)

    # second pass: inject equivalent I into walls/core zones
    zone_cores, zone_cols = build_zone_results(inp, core_scale, column_scale, slab_t, outriggers=outriggers)
    K_core = weighted_core_stiffness(inp, zone_cores) + K_outrigger_core
    K_cols = weighted_column_stiffness(inp, zone_cols) + K_outrigger_cols
    K_est = K_core + K_cols

    reinf = estimate_reinforcement(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h, outriggers)
    W_total = total_weight_kN_from_quantities(inp, reinf)
    M_eff = W_total * 1000.0 / G

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
        "K_outrigger": K_outrigger_core + K_outrigger_cols,
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


def run_iterative_design(inp: BuildingInput) -> DesignResult:
    beta = inp.target_position_factor
    max_iterations = 30
    tolerance = 0.02

    core_scale = 1.0
    column_scale = 1.0
    iteration_history: List[IterationLog] = []

    best_result = None
    best_error = float("inf")

    MIN_SCALE = 0.25
    MAX_SCALE = 2.20

    for iteration in range(1, max_iterations + 1):
        ev = evaluate_design(inp, core_scale, column_scale, beta)

        T_est = ev["T_est"]
        T_target = ev["T_target"]
        T_upper = ev["T_upper"]
        K_core = ev["K_core"]
        K_cols = ev["K_cols"]
        K_outrigger = ev["K_outrigger"]
        K_total = ev["K_est"]

        error = abs(T_est - T_target) / T_target
        error_percent = error * 100.0

        iteration_history.append(
            IterationLog(
                iteration=iteration,
                core_scale=core_scale,
                column_scale=column_scale,
                T_estimated=T_est,
                T_target=T_target,
                error_percent=error_percent,
                total_weight_kN=ev["W_total"],
                K_total_N_m=K_total,
                K_core_N_m=K_core,
                K_cols_N_m=K_cols,
                K_outrigger_N_m=K_outrigger,
            )
        )

        constraints_ok = (T_est <= T_upper) and (ev["drift_ratio"] <= inp.drift_limit_ratio)
        if error < best_error and constraints_ok:
            best_error = error
            best_result = (core_scale, column_scale, ev)

        if error <= tolerance and constraints_ok:
            break

        stiffness_ratio = (T_est / T_target) ** 2
        structural_K = K_core + K_cols
        core_share = K_core / structural_K if structural_K > 0 else 0.75
        damping = 0.60

        # prefer wall/core changes; column changes much smaller when outriggers exist
        outrigger_bias = 0.15 if inp.outrigger_count > 0 else 1.0
        scale_factor_core = stiffness_ratio ** (core_share / 3.0)
        scale_factor_col = stiffness_ratio ** (((1.0 - core_share) * outrigger_bias) / 4.0)

        new_core_scale = core_scale + damping * (core_scale * scale_factor_core - core_scale)
        new_column_scale = column_scale + damping * 0.35 * (column_scale * scale_factor_col - column_scale)

        new_core_scale = max(MIN_SCALE, min(MAX_SCALE, new_core_scale))
        new_column_scale = max(MIN_SCALE, min(MAX_SCALE, new_column_scale))

        core_scale, column_scale = new_core_scale, new_column_scale

    if best_result is None:
        ev = evaluate_design(inp, core_scale, column_scale, beta)
    else:
        core_scale, column_scale, ev = best_result

    T_ref = ev["T_ref"]
    T_upper = ev["T_upper"]
    T_target = ev["T_target"]
    T_est = ev["T_est"]
    drift_ratio = ev["drift_ratio"]

    period_ok = T_est <= T_upper
    drift_ok = drift_ratio <= inp.drift_limit_ratio

    messages = [
        f"T_ref = {T_ref:.3f} s",
        f"T_target = {T_target:.3f} s",
        f"T_est = {T_est:.3f} s",
        f"K_core = {ev['K_core']:.3e} N/m",
        f"K_cols = {ev['K_cols']:.3e} N/m",
        f"K_outrigger(total used) = {ev['K_outrigger']:.3e} N/m",
    ]

    if ev["outriggers"]:
        capped_any = any(o.capped for o in ev["outriggers"])
        messages.append(f"Outrigger count = {len(ev['outriggers'])}")
        messages.append("Outrigger effect applied mainly to core/walls.")
        if capped_any:
            messages.append("Outrigger stiffness was capped to avoid unrealistic period collapse.")

    redesign_suggestions = []
    if not period_ok:
        redesign_suggestions.append("Increase wall/core stiffness or revise outrigger levels.")
    if not drift_ok:
        redesign_suggestions.append("Increase wall/core stiffness or add a second/third outrigger.")
    if not redesign_suggestions:
        redesign_suggestions.append("Preliminary response is within the adopted limits.")

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
        optimization_message="Iterative wall-controlled convergence",
        core_scale=core_scale,
        column_scale=column_scale,
        messages=messages,
        redesign_suggestions=redesign_suggestions,
        governing_issue="OK" if (period_ok and drift_ok) else "Check response",
        modal_result=ev["modal"],
        iteration_history=iteration_history,
    )


def run_design(inp: BuildingInput) -> DesignResult:
    return run_iterative_design(inp)


def result_summary_table(result: DesignResult) -> pd.DataFrame:
    return pd.DataFrame({
        "Parameter": [
            "H (m)", "Floor Area (m²)", "Weight (kN)", "T_ref (s)", "T_target (s)", "T_est (s)",
            "T_upper (s)", "Drift (m)", "Drift Ratio", "Beam b (m)", "Beam h (m)", "Slab t (m)"
        ],
        "Value": [
            result.H_m, result.floor_area_m2, result.total_weight_kN, result.reference_period_s,
            result.design_target_period_s, result.estimated_period_s, result.upper_limit_period_s,
            result.top_drift_m, result.drift_ratio, result.beam_width_m, result.beam_depth_m, result.slab_thickness_m
        ]
    })


def iteration_table(result: DesignResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "iteration": it.iteration,
        "core_scale": it.core_scale,
        "column_scale": it.column_scale,
        "T_estimated_s": it.T_estimated,
        "T_target_s": it.T_target,
        "error_percent": it.error_percent,
        "total_weight_kN": it.total_weight_kN,
        "K_total_N_m": it.K_total_N_m,
        "K_core_N_m": it.K_core_N_m,
        "K_cols_N_m": it.K_cols_N_m,
        "K_outrigger_N_m": it.K_outrigger_N_m,
    } for it in result.iteration_history])


def _draw_rect(ax, x, y, w, h, color, fill=True, alpha=1.0, lw=1.0, ls="-", ec=None):
    rect = plt.Rectangle((x, y), w, h, facecolor=color if fill else "none", edgecolor=ec if ec else color, linewidth=lw, linestyle=ls, alpha=alpha)
    ax.add_patch(rect)


def plot_plan(inp: BuildingInput, result: DesignResult, zone_name: str):
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)
    fig, ax = plt.subplots(figsize=(14, 8))
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
    t = core.wall_thickness
    _draw_rect(ax, cx0, cy0, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.28)
    _draw_rect(ax, cx0, cy1 - t, core.core_outer_x, t, CORE_COLOR, fill=True, alpha=0.28)
    _draw_rect(ax, cx0, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.28)
    _draw_rect(ax, cx1 - t, cy0, t, core.core_outer_y, CORE_COLOR, fill=True, alpha=0.28)

    # explicitly show perimeter walls in plan
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

    for o in result.outrigger_results:
        if core.zone.story_start <= o.story_level <= core.zone.story_end:
            or_y = inp.plan_y * 0.5
            or_x = inp.plan_x * 0.5
            _draw_rect(ax, cx1, or_y - 0.25, or_x - cx1, 0.5, OUTRIGGER_COLOR, fill=True, alpha=0.80)
            _draw_rect(ax, cx0 - (or_x - cx0), or_y - 0.25, or_x - cx0, 0.5, OUTRIGGER_COLOR, fill=True, alpha=0.80)
            _draw_rect(ax, or_x - 0.25, cy1, 0.5, or_y - cy1, OUTRIGGER_COLOR, fill=True, alpha=0.80)
            _draw_rect(ax, or_x - 0.25, cy0 - (or_y - cy0), 0.5, or_y - cy0, OUTRIGGER_COLOR, fill=True, alpha=0.80)
            ax.text(or_x + 2, or_y + 2, f"OUTRIGGER\nStory {o.story_level}", fontsize=8, color=OUTRIGGER_COLOR, fontweight="bold")

    ax.set_title(f"{zone_name} | {story_range} | plan")
    ax.set_xlim(-5, inp.plan_x + 28)
    ax.set_ylim(inp.plan_y + 5, -5)
    ax.set_aspect("equal")
    ax.axis("off")

    info = [
        f"Core outer = {core.core_outer_x:.2f} x {core.core_outer_y:.2f} m",
        f"Core opening = {core.core_opening_x:.2f} x {core.core_opening_y:.2f} m",
        f"Wall thickness = {core.wall_thickness:.2f} m",
        f"Core Ieff(total) = {core.Ieq_effective_m4:,.1f} m⁴",
        f"Outrigger I add = {core.outrigger_I_addition_m4:,.1f} m⁴",
    ]
    for i, text in enumerate(info):
        ax.text(inp.plan_x + 4, 6 + 4*i, text, fontsize=9)

    return fig


def plot_elevation(inp: BuildingInput, result: DesignResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 10))
    H = result.H_m
    building_w = 24.0
    core_w = 8.0
    center_x = 0.0

    # building outline
    ax.plot([-building_w/2, -building_w/2, building_w/2, building_w/2, -building_w/2],
            [0, H, H, 0, 0], color="black", lw=1.5)

    # zone walls / core representation
    for z in result.zone_core_results:
        y0 = (z.zone.story_start - 1) * inp.story_height
        y1 = z.zone.story_end * inp.story_height
        ax.add_patch(plt.Rectangle((-core_w/2, y0), core_w, y1 - y0, facecolor=CORE_COLOR, edgecolor=CORE_COLOR, alpha=0.22))
        ax.text(core_w/2 + 1.2, (y0 + y1)/2, f"{z.zone.name}\nt={z.wall_thickness:.2f}m\nIadd={z.outrigger_I_addition_m4:,.0f}", va="center", fontsize=9)

    # floor lines
    for i in range(inp.n_story + 1):
        y = i * inp.story_height
        ax.plot([-building_w/2, building_w/2], [y, y], color="#e5e5e5", lw=0.5)

    # outriggers in elevation
    for o in result.outrigger_results:
        y = o.height_m
        ax.plot([-building_w/2, -core_w/2], [y, y], color=OUTRIGGER_COLOR, lw=4)
        ax.plot([core_w/2, building_w/2], [y, y], color=OUTRIGGER_COLOR, lw=4)
        ax.text(building_w/2 + 1, y, f"Story {o.story_level}\nK={o.stiffness_contribution/1e6:.2f} MN/m", va="center", fontsize=8, color=OUTRIGGER_COLOR)

    ax.set_title("Elevation view - walls/core + outriggers")
    ax.set_xlabel("Schematic width")
    ax.set_ylabel("Height (m)")
    ax.set_xlim(-18, 22)
    ax.set_ylim(0, H + 5)
    ax.grid(True, alpha=0.15)
    return fig


def plot_mode_shapes(result: DesignResult, modes: int = 5) -> plt.Figure:
    modal = result.modal_result
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, modal.n_dof + 1)
    for i in range(min(modes, len(modal.mode_shapes))):
        ax.plot(modal.mode_shapes[i], y, marker="o", label=f"Mode {i+1} | T={modal.periods_s[i]:.3f}s")
    for o in result.outrigger_results:
        ax.axhline(y=o.story_level, color=OUTRIGGER_COLOR, linestyle=":", alpha=0.6)
    ax.set_xlabel("Normalized mode shape")
    ax.set_ylabel("Story")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_iteration_history(result: DesignResult) -> plt.Figure:
    df = iteration_table(result)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["iteration"], df["T_estimated_s"], marker="o", label="T_est")
    ax.plot(df["iteration"], df["T_target_s"], marker="s", label="T_target")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Period (s)")
    ax.set_title("Iteration history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def streamlit_input_panel() -> BuildingInput:
    st.sidebar.header("Input Data")

    plan_shape = st.sidebar.radio("Plan shape", ["square", "triangle"], horizontal=True)
    n_story = int(st.sidebar.number_input("Above-grade stories", min_value=1, max_value=120, value=60, step=1))
    n_basement = int(st.sidebar.number_input("Basement stories", min_value=0, max_value=20, value=10, step=1))
    story_height = float(st.sidebar.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=3.2))
    basement_height = float(st.sidebar.number_input("Basement height (m)", min_value=2.5, max_value=6.0, value=3.0))
    plan_x = float(st.sidebar.number_input("Plan X (m)", min_value=10.0, max_value=300.0, value=80.0))
    plan_y = float(st.sidebar.number_input("Plan Y (m)", min_value=10.0, max_value=300.0, value=80.0))
    n_bays_x = int(st.sidebar.number_input("Bays in X", min_value=1, max_value=30, value=8, step=1))
    n_bays_y = int(st.sidebar.number_input("Bays in Y", min_value=1, max_value=30, value=8, step=1))
    bay_x = float(st.sidebar.number_input("Bay X (m)", min_value=2.0, max_value=20.0, value=10.0))
    bay_y = float(st.sidebar.number_input("Bay Y (m)", min_value=2.0, max_value=20.0, value=10.0))

    st.sidebar.subheader("Core opening")
    stair_count = int(st.sidebar.number_input("Stairs", min_value=0, max_value=20, value=2, step=1))
    elevator_count = int(st.sidebar.number_input("Elevators", min_value=0, max_value=30, value=4, step=1))
    elevator_area_each = float(st.sidebar.number_input("Elevator area each (m²)", min_value=0.0, max_value=20.0, value=3.5))
    stair_area_each = float(st.sidebar.number_input("Stair area each (m²)", min_value=0.0, max_value=50.0, value=20.0))
    service_area = float(st.sidebar.number_input("Service area (m²)", min_value=0.0, max_value=200.0, value=35.0))

    st.sidebar.subheader("Materials / loads")
    fck = float(st.sidebar.number_input("fck (MPa)", min_value=20.0, max_value=100.0, value=70.0))
    Ec = float(st.sidebar.number_input("Ec (MPa)", min_value=15000.0, max_value=50000.0, value=36000.0))
    fy = float(st.sidebar.number_input("fy (MPa)", min_value=200.0, max_value=700.0, value=420.0))
    DL = float(st.sidebar.number_input("DL (kN/m²)", min_value=0.0, max_value=20.0, value=3.0))
    LL = float(st.sidebar.number_input("LL (kN/m²)", min_value=0.0, max_value=20.0, value=2.5))
    slab_finish_allowance = float(st.sidebar.number_input("Fit-out allowance (kN/m²)", min_value=0.0, max_value=10.0, value=1.5))

    st.sidebar.subheader("Outriggers")
    outrigger_count = int(st.sidebar.selectbox("Number of outriggers", [0, 1, 2, 3], index=2))
    suggested_levels = []
    if outrigger_count >= 1:
        suggested_levels.append(max(1, int(round(n_story * 0.50))))
    if outrigger_count >= 2:
        suggested_levels.append(max(1, int(round(n_story * 0.75))))
    if outrigger_count >= 3:
        suggested_levels.append(max(1, int(round(n_story * 0.90))))

    levels = []
    for i in range(outrigger_count):
        levels.append(int(st.sidebar.number_input(
            f"Outrigger level {i+1}",
            min_value=1,
            max_value=max(1, n_story),
            value=min(suggested_levels[i], n_story),
            step=1,
        )))

    outrigger_truss_depth_m = float(st.sidebar.number_input("Outrigger truss depth (m)", min_value=0.5, max_value=10.0, value=3.0))
    outrigger_chord_area_m2 = float(st.sidebar.number_input("Chord area (m²)", min_value=0.001, max_value=1.0, value=0.08, format="%.3f"))
    outrigger_diagonal_area_m2 = float(st.sidebar.number_input("Diagonal area (m²)", min_value=0.001, max_value=1.0, value=0.04, format="%.3f"))
    max_outrigger_share_of_core = float(st.sidebar.slider("Max outrigger share of core stiffness", min_value=0.05, max_value=0.60, value=0.35, step=0.05))
    outrigger_core_influence = float(st.sidebar.slider("Outrigger influence on core", min_value=0.50, max_value=1.00, value=0.90, step=0.05))
    outrigger_column_influence = float(st.sidebar.slider("Outrigger influence on columns", min_value=0.00, max_value=0.30, value=0.05, step=0.01))

    return BuildingInput(
        plan_shape=plan_shape,
        n_story=n_story,
        n_basement=n_basement,
        story_height=story_height,
        basement_height=basement_height,
        plan_x=plan_x,
        plan_y=plan_y,
        n_bays_x=n_bays_x,
        n_bays_y=n_bays_y,
        bay_x=bay_x,
        bay_y=bay_y,
        stair_count=stair_count,
        elevator_count=elevator_count,
        elevator_area_each=elevator_area_each,
        stair_area_each=stair_area_each,
        service_area=service_area,
        fck=fck,
        Ec=Ec,
        fy=fy,
        DL=DL,
        LL=LL,
        slab_finish_allowance=slab_finish_allowance,
        outrigger_count=outrigger_count,
        outrigger_story_levels=levels,
        outrigger_truss_depth_m=outrigger_truss_depth_m,
        outrigger_chord_area_m2=outrigger_chord_area_m2,
        outrigger_diagonal_area_m2=outrigger_diagonal_area_m2,
        max_outrigger_share_of_core=max_outrigger_share_of_core,
        outrigger_core_influence=outrigger_core_influence,
        outrigger_column_influence=outrigger_column_influence,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info("Outriggers are now applied mainly to wall/core behavior, with capped contribution to avoid unrealistic period collapse.")

    if "result" not in st.session_state:
        st.session_state.result = None

    inp = streamlit_input_panel()

    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("Analyze")
    with c2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.session_state.result = None
        st.rerun()

    if run_btn:
        try:
            with st.spinner("Running wall-controlled analysis..."):
                st.session_state.result = run_design(inp)
            st.success("Analysis completed.")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.result
    if result is None:
        st.warning("Enter input data and click Analyze.")
        return

    st.dataframe(result_summary_table(result), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("T_est (s)", f"{result.estimated_period_s:.3f}")
    m2.metric("T_target (s)", f"{result.design_target_period_s:.3f}")
    m3.metric("Drift ratio", f"{result.drift_ratio:.5f}")
    m4.metric("Weight (kN)", f"{result.total_weight_kN:,.0f}")

    for msg in result.messages:
        st.caption(msg)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan", "Elevation", "Modes", "Iteration", "Tables"])

    with tab1:
        zone_name = st.selectbox("Zone in plan", [z.zone.name for z in result.zone_core_results], index=0)
        st.pyplot(plot_plan(inp, result, zone_name), use_container_width=True)

    with tab2:
        st.pyplot(plot_elevation(inp, result), use_container_width=True)

    with tab3:
        st.pyplot(plot_mode_shapes(result, modes=5), use_container_width=True)

    with tab4:
        st.dataframe(iteration_table(result), use_container_width=True)
        st.pyplot(plot_iteration_history(result), use_container_width=True)

    with tab5:
        df_core = pd.DataFrame([{
            "Zone": z.zone.name,
            "Story start": z.zone.story_start,
            "Story end": z.zone.story_end,
            "Wall t (m)": z.wall_thickness,
            "Core Ieff total (m4)": z.Ieq_effective_m4,
            "Outrigger I add (m4)": z.outrigger_I_addition_m4,
        } for z in result.zone_core_results])
        st.subheader("Wall / core outputs")
        st.dataframe(df_core, use_container_width=True)

        df_outrigger = pd.DataFrame([{
            "Story": o.story_level,
            "Height (m)": o.height_m,
            "K used (N/m)": o.stiffness_contribution,
            "Krot (Nm/rad)": o.K_rotational_Nm_per_rad,
            "I add (m4)": o.effective_I_addition_m4,
            "Capped": o.capped,
        } for o in result.outrigger_results]) if result.outrigger_results else pd.DataFrame()

        st.subheader("Outrigger outputs")
        if len(df_outrigger):
            st.dataframe(df_outrigger, use_container_width=True)
        else:
            st.info("No outriggers defined.")

        st.subheader("Column outputs")
        df_cols = pd.DataFrame([{
            "Zone": z.zone.name,
            "Corner x (m)": z.corner_column_x_m,
            "Corner y (m)": z.corner_column_y_m,
            "Perimeter x (m)": z.perimeter_column_x_m,
            "Perimeter y (m)": z.perimeter_column_y_m,
            "Interior x (m)": z.interior_column_x_m,
            "Interior y (m)": z.interior_column_y_m,
        } for z in result.zone_column_results])
        st.dataframe(df_cols, use_container_width=True)


if __name__ == "__main__":
    main()
