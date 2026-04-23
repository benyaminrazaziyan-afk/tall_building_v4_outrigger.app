from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


G = 9.81
STEEL_DENSITY = 7850.0
APP_TITLE = "Tall Building Preliminary Design + Outrigger Belt Truss"
APP_VERSION = "v6.2-checked"


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
class StaticResponse:
    roof_displacement_m: float
    story_drifts_m: List[float]
    story_drift_ratios: List[float]
    max_story_drift_m: float
    max_story_drift_ratio: float
    displacements_m: List[float]
    story_forces_N: List[float]


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
    max_story_drift_m: float
    max_story_drift_ratio: float
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
    static_response: Optional[StaticResponse] = None


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
    seismic_zone_factor: float = 1.0
    importance_factor: float = 1.0
    behavior_factor: float = 4.0
    spectral_accel_short: float = 1.00
    spectral_accel_1s: float = 0.40
    outrigger_core_relief_max: float = 0.18
    outrigger_column_relief_max: float = 0.22
    outrigger_zone_wall_boost: float = 0.06
    outrigger_zone_column_boost: float = 0.08
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


# ----------------------------- ENGINE -----------------------------

def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def floor_area(inp: BuildingInput) -> float:
    if inp.plan_shape == "triangle":
        return 0.5 * inp.plan_x * inp.plan_y
    return inp.plan_x * inp.plan_y


def code_type_period(H: float, Ct: float, x_period: float) -> float:
    return Ct * (H ** x_period)


def effective_lateral_force_coeff(inp: BuildingInput) -> float:
    spectral_factor = 0.6 + 0.25 * inp.spectral_accel_short + 0.15 * inp.spectral_accel_1s
    coeff = (
        inp.prelim_lateral_force_coeff
        * inp.seismic_zone_factor
        * inp.importance_factor
        * spectral_factor
        / max(inp.behavior_factor, 1e-6)
    )
    return max(0.001, coeff)


def preliminary_lateral_force_N(inp: BuildingInput, W_total_kN: float) -> float:
    return effective_lateral_force_coeff(inp) * W_total_kN * 1000.0


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
    i_local = length * thickness**3 / 12.0
    area = length * thickness
    return i_local + area * x_centroid**2


def wall_rect_inertia_about_global_x(length: float, thickness: float, y_centroid: float) -> float:
    i_local = length * thickness**3 / 12.0
    area = length * thickness
    return i_local + area * y_centroid**2


def core_equivalent_inertia(outer_x: float, outer_y: float, lengths: List[float], t: float, wall_count: int) -> float:
    x_side = outer_x / 2.0
    y_side = outer_y / 2.0
    top_len, bot_len, left_len, right_len = lengths[0], lengths[1], lengths[2], lengths[3]
    i_x = 0.0
    i_y = 0.0

    i_x += wall_rect_inertia_about_global_x(top_len, t, +y_side)
    i_x += wall_rect_inertia_about_global_x(bot_len, t, -y_side)
    i_y += (t * top_len**3 / 12.0) + (t * bot_len**3 / 12.0)

    i_y += wall_rect_inertia_about_global_y(left_len, t, -x_side)
    i_y += wall_rect_inertia_about_global_y(right_len, t, +x_side)
    i_x += (t * left_len**3 / 12.0) + (t * right_len**3 / 12.0)

    if wall_count >= 6:
        inner_x = 0.22 * outer_x
        l1, l2 = lengths[4], lengths[5]
        i_y += wall_rect_inertia_about_global_y(l1, t, -inner_x)
        i_y += wall_rect_inertia_about_global_y(l2, t, +inner_x)
        i_x += (t * l1**3 / 12.0) + (t * l2**3 / 12.0)

    if wall_count >= 8:
        inner_y = 0.22 * outer_y
        l3, l4 = lengths[6], lengths[7]
        i_x += wall_rect_inertia_about_global_x(l3, t, -inner_y)
        i_x += wall_rect_inertia_about_global_x(l4, t, +inner_y)
        i_y += (t * l3**3 / 12.0) + (t * l4**3 / 12.0)

    return min(i_x, i_y)


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


def directional_dims(base_dim: float, plan_x: float, plan_y: float, min_dim: Optional[float] = None, max_dim: Optional[float] = None) -> tuple[float, float]:
    aspect = max(plan_x, plan_y) / max(min(plan_x, plan_y), 1e-9)
    if aspect <= 1.10:
        dx, dy = base_dim, base_dim
    else:
        major = base_dim * 1.15
        minor = base_dim * 0.90
        dx, dy = (major, minor) if plan_x >= plan_y else (minor, major)
    if min_dim is not None:
        dx = max(min_dim, dx)
        dy = max(min_dim, dy)
    if max_dim is not None:
        dx = min(max_dim, dx)
        dy = min(max_dim, dy)
    return dx, dy


def calculate_outrigger_stiffness(
    inp: BuildingInput,
    story_level: int,
    truss_depth: float,
    chord_area: float,
    diagonal_area: float,
) -> OutriggerResult:
    e_concrete = inp.Ec * 1e6
    e_steel = 200e9
    e_eff = max(e_concrete, e_steel)
    story_height = inp.story_height
    h_total = total_height(inp)
    height_m = story_level * story_height

    core_x, core_y = initial_core_dimensions(inp, *opening_dimensions(inp))
    truss_width_x = max((inp.plan_x - core_x) / 2.0, 1.0)
    truss_width_y = max((inp.plan_y - core_y) / 2.0, 1.0)
    truss_width = max(truss_width_x, truss_width_y)

    chord_length = max(truss_width, 1e-6)
    diagonal_length = sqrt(truss_width**2 + truss_depth**2)

    chord_k = 4.0 * e_steel * chord_area / chord_length
    diag_k = 4.0 * e_steel * diagonal_area * (truss_depth / diagonal_length) ** 2 / diagonal_length
    axial_stiffness = chord_k + diag_k

    lever_arm = truss_width
    k_rot = axial_stiffness * lever_arm**2

    height_ratio = max(0.15, min(0.95, height_m / max(h_total, 1e-9)))
    position_factor = 0.70 + 0.60 * height_ratio * (2.0 - height_ratio)
    k_eq = (k_rot / max(story_height**2, 1e-9)) * position_factor
    effective_i_addition = k_rot * (h_total**2) / (3.0 * e_eff)

    return OutriggerResult(
        story_level=story_level,
        height_m=height_m,
        truss_depth_m=truss_depth,
        truss_width_m=truss_width,
        chord_area_m2=chord_area,
        diagonal_area_m2=diagonal_area,
        axial_stiffness_kN=axial_stiffness / 1000.0,
        equivalent_spring_kN_m=k_eq / 1000.0,
        stiffness_contribution=k_eq,
        K_rotational_Nm_per_rad=k_rot,
        effective_I_addition_m4=effective_i_addition,
    )


def calculate_total_outrigger_stiffness(inp: BuildingInput) -> tuple[List[OutriggerResult], float]:
    if inp.outrigger_count == 0 or not inp.outrigger_story_levels:
        return [], 0.0

    outriggers: List[OutriggerResult] = []
    total_k = 0.0
    for level in inp.outrigger_story_levels[:inp.outrigger_count]:
        if 1 <= level <= inp.n_story:
            item = calculate_outrigger_stiffness(
                inp,
                level,
                inp.outrigger_truss_depth_m,
                inp.outrigger_chord_area_m2,
                inp.outrigger_diagonal_area_m2,
            )
            outriggers.append(item)
            total_k += item.stiffness_contribution
    return outriggers, total_k


def outrigger_story_density(inp: BuildingInput) -> float:
    if inp.n_story <= 0 or inp.outrigger_count <= 0:
        return 0.0
    valid = [lvl for lvl in inp.outrigger_story_levels[:inp.outrigger_count] if 1 <= lvl <= inp.n_story]
    return min(1.0, len(valid) / max(inp.n_story / 15.0, 1.0))


def zone_outrigger_count(inp: BuildingInput, zone: ZoneDefinition) -> int:
    return sum(1 for lvl in inp.outrigger_story_levels[:inp.outrigger_count] if zone.story_start <= lvl <= zone.story_end)


def outrigger_relief_factors(inp: BuildingInput) -> tuple[float, float]:
    density = outrigger_story_density(inp)
    core_relief = inp.outrigger_core_relief_max * density
    column_relief = inp.outrigger_column_relief_max * density
    return core_relief, column_relief


def build_zone_results(inp: BuildingInput, core_scale: float, column_scale: float, slab_t: float) -> tuple[List[ZoneCoreResult], List[ZoneColumnResult]]:
    h_total = total_height(inp)
    zones = define_three_zones(inp.n_story)
    opening_x, opening_y = opening_dimensions(inp)
    outer_x, outer_y = initial_core_dimensions(inp, opening_x, opening_y)

    zone_cores: List[ZoneCoreResult] = []
    zone_cols: List[ZoneColumnResult] = []

    core_relief, column_relief = outrigger_relief_factors(inp)
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

    w_total_est = q * floor_area(inp) * (inp.n_story + 0.7 * inp.n_basement)
    base_shear_est = effective_lateral_force_coeff(inp) * w_total_est
    overturning_est = base_shear_est * 0.67 * h_total

    for zone in zones:
        wall_count = active_wall_count_by_zone(inp, zone.name)
        lengths = wall_lengths_for_layout(outer_x, outer_y, wall_count)
        n_outriggers_in_zone = zone_outrigger_count(inp, zone)
        zone_core_scale = core_scale * (1.0 - core_relief) * (1.0 + inp.outrigger_zone_wall_boost * n_outriggers_in_zone)
        t = wall_thickness_by_zone(inp, h_total, zone, zone_core_scale)
        i_gross = core_equivalent_inertia(outer_x, outer_y, lengths, t, wall_count)

        cracked_factor = inp.wall_cracked_factor
        if inp.n_story > 40:
            cracked_factor *= 0.85
        i_eff = cracked_factor * i_gross

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
                Ieq_gross_m4=i_gross,
                Ieq_effective_m4=i_eff,
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

        p_interior_gravity = tributary_interior * q * n_effective * 1.18
        p_perimeter_gravity = tributary_perimeter * q * n_effective * 1.18
        p_corner_gravity = tributary_corner * q * n_effective * 1.18

        zone_height_ratio = 1.0 - (zone.story_start - 1) / max(inp.n_story, 1)
        zone_otm = overturning_est * zone_height_ratio
        lever = max(0.45 * max(inp.plan_x, inp.plan_y), 1.0)
        delta_p_corner = 0.35 * zone_otm / max(corner_cols * lever, 1e-9)
        delta_p_perimeter = 0.20 * zone_otm / max(max(perimeter_cols, 1) * lever, 1e-9)

        p_corner = p_corner_gravity + abs(delta_p_corner)
        p_perimeter = p_perimeter_gravity + abs(delta_p_perimeter)
        p_interior = p_interior_gravity + 0.05 * abs(delta_p_perimeter)

        zone_column_scale = column_scale * (1.0 - column_relief) * (1.0 + inp.outrigger_zone_column_boost * n_outriggers_in_zone)
        interior_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(max(p_interior, 1e-9) / sigma_allow))) * zone_column_scale
        perimeter_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(max(p_perimeter, 1e-9) / sigma_allow))) * max(inp.perimeter_column_factor, zone_column_scale)
        corner_dim = min(inp.max_column_dim, max(inp.min_column_dim, sqrt(max(p_corner, 1e-9) / sigma_allow))) * max(inp.corner_column_factor, zone_column_scale)

        interior_x, interior_y = directional_dims(interior_dim, inp.plan_x, inp.plan_y, inp.min_column_dim, inp.max_column_dim)
        perimeter_x, perimeter_y = directional_dims(perimeter_dim, inp.plan_x, inp.plan_y, inp.min_column_dim, inp.max_column_dim)
        corner_x, corner_y = directional_dims(corner_dim, inp.plan_x, inp.plan_y, inp.min_column_dim, inp.max_column_dim)

        a_corner = corner_x * corner_y
        a_perim = perimeter_x * perimeter_y
        a_inter = interior_x * interior_y

        iavg_corner = max(corner_x * corner_y**3 / 12.0, corner_y * corner_x**3 / 12.0)
        iavg_perim = max(perimeter_x * perimeter_y**3 / 12.0, perimeter_y * perimeter_x**3 / 12.0)
        iavg_inter = max(interior_x * interior_y**3 / 12.0, interior_y * interior_x**3 / 12.0)

        i_avg = (corner_cols * iavg_corner + perimeter_cols * iavg_perim + interior_cols * iavg_inter) / max(total_columns, 1)
        a_avg = (corner_cols * a_corner + perimeter_cols * a_perim + interior_cols * a_inter) / max(total_columns, 1)
        i_col_group = inp.column_cracked_factor * (i_avg * max(total_columns, 1) + a_avg * r2_sum)

        zone_cols.append(
            ZoneColumnResult(
                zone=zone,
                corner_column_x_m=corner_x,
                corner_column_y_m=corner_y,
                perimeter_column_x_m=perimeter_x,
                perimeter_column_y_m=perimeter_y,
                interior_column_x_m=interior_x,
                interior_column_y_m=interior_y,
                P_corner_kN=p_corner,
                P_perimeter_kN=p_perimeter,
                P_interior_kN=p_interior,
                I_col_group_effective_m4=i_col_group,
            )
        )

    return zone_cores, zone_cols


def weighted_core_stiffness(inp: BuildingInput, zone_cores: List[ZoneCoreResult]) -> float:
    h_total = total_height(inp)
    e_mod = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cores:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / h_total) / max(e_mod * zc.Ieq_effective_m4 * (1.0 + 0.20 * len(zc.perimeter_wall_segments)), 1e-9)
    ei_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return 3.0 * ei_equiv / (h_total**3)


def weighted_column_stiffness(inp: BuildingInput, zone_cols: List[ZoneColumnResult]) -> float:
    h_total = total_height(inp)
    e_mod = inp.Ec * 1e6
    total_flex_factor = 0.0
    for zc in zone_cols:
        hi = zc.zone.n_stories * inp.story_height
        total_flex_factor += (hi / h_total) / max(e_mod * zc.I_col_group_effective_m4, 1e-9)
    ei_equiv = 1.0 / max(total_flex_factor, 1e-18)
    return 3.0 * ei_equiv / (h_total**3)


def estimate_reinforcement(
    inp: BuildingInput,
    zone_cores: List[ZoneCoreResult],
    zone_cols: List[ZoneColumnResult],
    slab_t: float,
    beam_b: float,
    beam_h: float,
    outriggers: Optional[List[OutriggerResult]] = None,
) -> ReinforcementEstimate:
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
        for or_result in outriggers:
            truss_length = 4 * or_result.truss_width_m
            chord_steel = truss_length * or_result.chord_area_m2 * STEEL_DENSITY
            diagonal_steel = truss_length * 0.5 * or_result.diagonal_area_m2 * STEEL_DENSITY
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
    a_floor = floor_area(inp)
    superimposed = (inp.DL + inp.LL + inp.slab_finish_allowance) * a_floor * (inp.n_story + inp.n_basement)
    facade = inp.facade_line_load * (2 * (inp.plan_x + inp.plan_y)) * inp.n_story
    return (concrete_weight + steel_weight + superimposed + facade) * inp.seismic_mass_factor


def build_story_masses(inp: BuildingInput, total_weight_kN_value: float) -> List[float]:
    w_story_kN = total_weight_kN_value / max(inp.n_story, 1)
    masses = [(w_story_kN * 1000.0) / G for _ in range(inp.n_story)]
    if masses:
        masses[-1] *= 0.90
    return masses


def build_story_stiffnesses(inp: BuildingInput, k_structural: float) -> List[float]:
    """توزیع سختی طبقات: پایین سخت‌تر، بالا نرم‌تر."""
    n = inp.n_story
    raw = []
    for i in range(n):
        r = i / max(n - 1, 1)
        raw.append(1.35 - 0.55 * r)
    inv_sum = sum(1.0 / max(a, 1e-6) for a in raw)
    c = k_structural * inv_sum
    return [c * a for a in raw]


def assemble_m_k_with_outriggers(
    story_masses: List[float],
    story_stiffness: List[float],
    outriggers: List[OutriggerResult],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(story_masses)
    m_mat = np.diag(story_masses)
    k_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        ki = story_stiffness[i]
        if i == 0:
            k_mat[i, i] += ki
        else:
            k_mat[i, i] += ki
            k_mat[i, i - 1] -= ki
            k_mat[i - 1, i] -= ki
            k_mat[i - 1, i - 1] += ki

    # outrigger فقط در اینجا اضافه می‌شود؛ در K_total دوباره‌شماری نمی‌شود.
    for or_result in outriggers:
        idx = or_result.story_level - 1
        if 0 <= idx < n:
            k_or = or_result.stiffness_contribution
            k_mat[idx, idx] += k_or
            if idx > 0:
                k_mat[idx, idx - 1] -= 0.35 * k_or
                k_mat[idx - 1, idx] -= 0.35 * k_or
                k_mat[idx - 1, idx - 1] += 0.35 * k_or
            if idx < n - 1:
                k_mat[idx, idx + 1] -= 0.35 * k_or
                k_mat[idx + 1, idx] -= 0.35 * k_or
                k_mat[idx + 1, idx + 1] += 0.35 * k_or
    return m_mat, k_mat


def triangular_story_forces(inp: BuildingInput, total_weight_kN_value: float) -> np.ndarray:
    v_base = preliminary_lateral_force_N(inp, total_weight_kN_value)
    heights = np.arange(1, inp.n_story + 1, dtype=float) * inp.story_height
    coeff = heights / max(np.sum(heights), 1e-9)
    return v_base * coeff


def solve_static_response(inp: BuildingInput, k_mat: np.ndarray, total_weight_kN_value: float) -> StaticResponse:
    f_vec = triangular_story_forces(inp, total_weight_kN_value)
    u = np.linalg.solve(k_mat, f_vec)
    u = np.real_if_close(u)
    story_drifts = []
    for i in range(len(u)):
        if i == 0:
            story_drifts.append(float(u[i]))
        else:
            story_drifts.append(float(u[i] - u[i - 1]))
    ratios = [d / max(inp.story_height, 1e-9) for d in story_drifts]
    return StaticResponse(
        roof_displacement_m=float(u[-1]),
        story_drifts_m=story_drifts,
        story_drift_ratios=ratios,
        max_story_drift_m=max(abs(x) for x in story_drifts),
        max_story_drift_ratio=max(abs(x) for x in ratios),
        displacements_m=[float(x) for x in u.tolist()],
        story_forces_N=[float(x) for x in f_vec.tolist()],
    )


def solve_mdof_modes(
    inp: BuildingInput,
    total_weight_kN_value: float,
    k_structural: float,
    outriggers: Optional[List[OutriggerResult]] = None,
    n_modes: int = 5,
) -> tuple[ModalResult, np.ndarray, np.ndarray, StaticResponse]:
    masses = build_story_masses(inp, total_weight_kN_value)
    k_stories = build_story_stiffnesses(inp, k_structural)

    if outriggers:
        m_mat, k_mat = assemble_m_k_with_outriggers(masses, k_stories, outriggers)
    else:
        m_mat = np.diag(masses)
        k_mat = np.zeros((len(masses), len(masses)), dtype=float)
        for i in range(len(masses)):
            ki = k_stories[i]
            if i == 0:
                k_mat[i, i] += ki
            else:
                k_mat[i, i] += ki
                k_mat[i, i - 1] -= ki
                k_mat[i - 1, i] -= ki
                k_mat[i - 1, i - 1] += ki

    a_mat = np.linalg.solve(m_mat, k_mat)
    eigvals, eigvecs = np.linalg.eig(a_mat)
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
    total_mass = np.sum(np.diag(m_mat)).item()

    mass_ratios = []
    cumulative = []
    mode_shapes = []
    cum = 0.0

    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = (phi.T @ m_mat @ phi).item()
        gamma = ((phi.T @ m_mat @ ones) / max(denom, 1e-12)).item()
        meff = gamma**2 * denom
        ratio = meff / max(total_mass, 1e-12)
        cum += ratio

        phi_plot = phi.flatten().copy()
        max_abs = np.max(np.abs(phi_plot))
        if max_abs > 1e-12:
            phi_plot = phi_plot / max_abs
        if phi_plot[-1] < 0:
            phi_plot = -phi_plot

        mode_shapes.append(phi_plot.tolist())
        mass_ratios.append(ratio)
        cumulative.append(cum)

    static_response = solve_static_response(inp, k_mat, total_weight_kN_value)

    return (
        ModalResult(
            n_dof=len(masses),
            periods_s=periods,
            frequencies_hz=freqs,
            mode_shapes=mode_shapes,
            story_masses_kg=masses,
            story_stiffness_N_per_m=k_stories,
            effective_mass_ratios=mass_ratios,
            cumulative_effective_mass_ratios=cumulative,
        ),
        m_mat,
        k_mat,
        static_response,
    )


def generate_redesign_suggestions(
    inp: BuildingInput,
    t_est: float,
    t_target: float,
    t_limit: float,
    roof_drift_ratio: float,
    drift_limit: float,
    story_drift_ratio: float,
    core_scale: float,
    column_scale: float,
) -> tuple[str, List[str]]:
    suggestions: List[str] = []
    governing_issue = "OK"

    if t_est > t_limit:
        governing_issue = "Period exceeds upper limit"
        suggestions.extend([
            "Increase lateral stiffness by enlarging core walls and perimeter columns.",
            "Increase active wall count in middle and upper zones.",
            "Consider larger core footprint if architecture permits.",
            "Add or strengthen outriggers.",
        ])
    elif t_est > 1.10 * t_target:
        governing_issue = "Period above target"
        suggestions.extend([
            "System is softer than target.",
            "Increase wall thickness or internal core wall engagement.",
            "Increase corner and perimeter columns.",
            "Consider adding outriggers.",
        ])
    elif t_est < 0.90 * t_target:
        governing_issue = "Period below target"
        suggestions.extend([
            "System is stiffer than target and may be uneconomical.",
            "Reduce wall thicknesses or column sizes where feasible.",
        ])

    if roof_drift_ratio > drift_limit or story_drift_ratio > drift_limit:
        governing_issue = "Drift exceeds allowable limit"
        suggestions.extend([
            "Increase global stiffness by enlarging core and perimeter columns.",
            "Increase wall ratio or add perimeter wall segments.",
            "Add outriggers to control drift.",
        ])

    if core_scale >= 1.55:
        suggestions.append("Core scale factor is near its upper bound.")
    if column_scale >= 1.55:
        suggestions.append("Column scale factor is near its upper bound.")
    if not suggestions:
        suggestions.append("Structural system appears preliminarily adequate.")

    return governing_issue, suggestions


def evaluate_design(inp: BuildingInput, core_scale: float, column_scale: float, beta: float) -> dict:
    h_total = total_height(inp)
    t_ref = code_type_period(h_total, inp.Ct, inp.x_period)
    t_upper = inp.upper_period_factor * t_ref
    t_target = t_ref + beta * (t_upper - t_ref)

    slab_t = slab_thickness_prelim(inp, column_scale)
    beam_b, beam_h = beam_size_prelim(inp, column_scale)
    zone_cores, zone_cols = build_zone_results(inp, core_scale, column_scale, slab_t)

    outriggers, k_outrigger_nominal = calculate_total_outrigger_stiffness(inp)
    reinf = estimate_reinforcement(inp, zone_cores, zone_cols, slab_t, beam_b, beam_h, outriggers)
    w_total = total_weight_kN_from_quantities(inp, reinf)
    m_eff = w_total * 1000.0 / G

    k_core = weighted_core_stiffness(inp, zone_cores)
    k_cols = weighted_column_stiffness(inp, zone_cols)
    k_structural = k_core + k_cols

    modal, m_mat, k_mat, static_response = solve_mdof_modes(inp, w_total, k_structural, outriggers=outriggers, n_modes=5)
    t_est = modal.periods_s[0] if modal.periods_s else 2.0 * pi * sqrt(m_eff / max(k_structural, 1e-9))
    roof_drift = static_response.roof_displacement_m
    roof_drift_ratio = roof_drift / max(h_total, 1e-9)
    period_error = abs(t_est - t_target) / max(t_target, 1e-9)

    k_total_effective = preliminary_lateral_force_N(inp, w_total) / max(roof_drift, 1e-9)

    return {
        "T_ref": t_ref,
        "T_upper": t_upper,
        "T_target": t_target,
        "T_est": t_est,
        "period_error": period_error,
        "W_total": w_total,
        "M_eff": m_eff,
        "K_est": k_total_effective,
        "K_core": k_core,
        "K_cols": k_cols,
        "K_outrigger": k_outrigger_nominal,
        "top_drift": roof_drift,
        "drift_ratio": roof_drift_ratio,
        "modal": modal,
        "zone_cores": zone_cores,
        "zone_cols": zone_cols,
        "outriggers": outriggers,
        "slab_t": slab_t,
        "beam_b": beam_b,
        "beam_h": beam_h,
        "reinf": reinf,
        "k_matrix": k_mat,
        "m_matrix": m_mat,
        "static_response": static_response,
        "k_structural": k_structural,
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

    min_scale = 0.35
    max_scale = 2.25

    for iteration in range(1, max_iterations + 1):
        ev = evaluate_design(inp, core_scale, column_scale, beta)

        t_est = ev["T_est"]
        t_target = ev["T_target"]
        t_upper = ev["T_upper"]
        k_core = ev["K_core"]
        k_cols = ev["K_cols"]
        k_outrigger = ev["K_outrigger"]
        k_total = ev["K_est"]
        static_response = ev["static_response"]

        error = abs(t_est - t_target) / max(t_target, 1e-9)
        error_percent = error * 100.0

        iteration_history.append(
            IterationLog(
                iteration=iteration,
                core_scale=core_scale,
                column_scale=column_scale,
                T_estimated=t_est,
                T_target=t_target,
                error_percent=error_percent,
                total_weight_kN=ev["W_total"],
                K_total_N_m=k_total,
                K_core_N_m=k_core,
                K_cols_N_m=k_cols,
                K_outrigger_N_m=k_outrigger,
            )
        )

        constraints_ok = (t_est <= t_upper) and (static_response.max_story_drift_ratio <= inp.drift_limit_ratio)
        if error < best_error and constraints_ok:
            best_error = error
            best_result = (core_scale, column_scale, ev)

        if error <= tolerance and constraints_ok:
            break

        stiffness_ratio = (t_est / max(t_target, 1e-9)) ** 2
        structural_k = k_core + k_cols
        core_share = k_core / structural_k if structural_k > 0 else 0.65
        damping = 0.55

        scale_factor_core = stiffness_ratio ** (core_share / 3.0)
        scale_factor_col = stiffness_ratio ** ((1.0 - core_share) / 4.0)

        new_core_scale = core_scale + damping * (core_scale * scale_factor_core - core_scale)
        new_column_scale = column_scale + damping * (column_scale * scale_factor_col - column_scale)

        if static_response.max_story_drift_ratio > inp.drift_limit_ratio:
            new_core_scale *= 1.04
            new_column_scale *= 1.03

        new_core_scale = max(min_scale, min(max_scale, new_core_scale))
        new_column_scale = max(min_scale, min(max_scale, new_column_scale))

        scale_change = abs(new_core_scale - core_scale) + abs(new_column_scale - column_scale)
        if scale_change < 0.002 and iteration > 3:
            if error > tolerance:
                if core_share > 0.5:
                    new_core_scale = max(min_scale, core_scale * 0.96)
                else:
                    new_column_scale = max(min_scale, column_scale * 0.96)
            else:
                break

        core_scale, column_scale = new_core_scale, new_column_scale

    if best_result is None:
        ev = evaluate_design(inp, core_scale, column_scale, beta)
    else:
        core_scale, column_scale, ev = best_result

    t_ref = ev["T_ref"]
    t_upper = ev["T_upper"]
    t_target = ev["T_target"]
    t_est = ev["T_est"]
    drift_ratio = ev["drift_ratio"]
    static_response = ev["static_response"]

    period_ok = t_est <= t_upper
    drift_ok = static_response.max_story_drift_ratio <= inp.drift_limit_ratio
    governing_issue, redesign_suggestions = generate_redesign_suggestions(
        inp,
        t_est,
        t_target,
        t_upper,
        drift_ratio,
        inp.drift_limit_ratio,
        static_response.max_story_drift_ratio,
        core_scale,
        column_scale,
    )

    messages = [
        "Target formula: T_target = T_ref + beta*(T_upper - T_ref)",
        f"beta = {beta:.3f}",
        f"T_ref = {t_ref:.3f} s",
        f"T_upper = {t_upper:.3f} s",
        f"T_target = {t_target:.3f} s",
        f"Final T_est (MDOF) = {t_est:.3f} s",
        f"Iterations used = {len(iteration_history)}",
        f"Mode 1 effective mass = {100 * ev['modal'].effective_mass_ratios[0]:.1f}%",
        f"Max story drift ratio = {static_response.max_story_drift_ratio:.5f}",
    ]

    if ev["outriggers"]:
        core_relief, column_relief = outrigger_relief_factors(inp)
        messages.append(f"Outrigger count = {len(ev['outriggers'])}")
        messages.append(f"Nominal outrigger stiffness contribution = {ev['K_outrigger']:.2e} N/m")
        messages.append(f"Core demand relief used = {100.0 * core_relief:.1f}%")
        messages.append(f"Column demand relief used = {100.0 * column_relief:.1f}%")

    return DesignResult(
        H_m=total_height(inp),
        floor_area_m2=floor_area(inp),
        total_weight_kN=ev["W_total"],
        effective_modal_mass_kg=ev["M_eff"],
        reference_period_s=t_ref,
        design_target_period_s=t_target,
        upper_limit_period_s=t_upper,
        estimated_period_s=t_est,
        period_error_ratio=ev["period_error"],
        period_ok=period_ok,
        drift_ok=drift_ok,
        K_estimated_N_per_m=ev["K_est"],
        top_drift_m=ev["top_drift"],
        drift_ratio=drift_ratio,
        max_story_drift_m=static_response.max_story_drift_m,
        max_story_drift_ratio=static_response.max_story_drift_ratio,
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
        static_response=static_response,
    )


def run_design(inp: BuildingInput) -> DesignResult:
    return run_iterative_design(inp)


def clone_input_without_outriggers(inp: BuildingInput) -> BuildingInput:
    return replace(
        inp,
        outrigger_count=0,
        outrigger_story_levels=[],
    )


def global_comparison_table(base_result: DesignResult, outrigger_result: DesignResult) -> pd.DataFrame:
    rows = [
        ("T_est (s)", base_result.estimated_period_s, outrigger_result.estimated_period_s),
        ("Roof Drift (m)", base_result.top_drift_m, outrigger_result.top_drift_m),
        ("Roof Drift Ratio", base_result.drift_ratio, outrigger_result.drift_ratio),
        ("Max Story Drift Ratio", base_result.max_story_drift_ratio, outrigger_result.max_story_drift_ratio),
        ("Effective Lateral Stiffness (N/m)", base_result.K_estimated_N_per_m, outrigger_result.K_estimated_N_per_m),
        ("Weight (kN)", base_result.total_weight_kN, outrigger_result.total_weight_kN),
        ("Core Scale", base_result.core_scale, outrigger_result.core_scale),
        ("Column Scale", base_result.column_scale, outrigger_result.column_scale),
    ]
    data = []
    for name, before, after in rows:
        delta = after - before
        pct = 100.0 * delta / abs(before) if abs(before) > 1e-12 else np.nan
        data.append({
            "Parameter": name,
            "Without Outrigger": before,
            "With Outrigger": after,
            "Delta": delta,
            "Delta %": pct,
        })
    return pd.DataFrame(data)


def zone_comparison_table(base_result: DesignResult, outrigger_result: DesignResult) -> pd.DataFrame:
    core_before = {z.zone.name: z for z in base_result.zone_core_results}
    core_after = {z.zone.name: z for z in outrigger_result.zone_core_results}
    col_before = {z.zone.name: z for z in base_result.zone_column_results}
    col_after = {z.zone.name: z for z in outrigger_result.zone_column_results}

    rows = []
    for zone_name in ["Lower Zone", "Middle Zone", "Upper Zone"]:
        cb = core_before[zone_name]
        ca = core_after[zone_name]
        pb = col_before[zone_name]
        pa = col_after[zone_name]

        def pct_change(a, b):
            return 100.0 * (b - a) / abs(a) if abs(a) > 1e-12 else np.nan

        rows.append({
            "Zone": zone_name,
            "Wall t before (m)": cb.wall_thickness,
            "Wall t after (m)": ca.wall_thickness,
            "Wall Δ %": pct_change(cb.wall_thickness, ca.wall_thickness),
            "Core Ieff before (m4)": cb.Ieq_effective_m4,
            "Core Ieff after (m4)": ca.Ieq_effective_m4,
            "Core Ieff Δ %": pct_change(cb.Ieq_effective_m4, ca.Ieq_effective_m4),
            "Corner col before (m)": max(pb.corner_column_x_m, pb.corner_column_y_m),
            "Corner col after (m)": max(pa.corner_column_x_m, pa.corner_column_y_m),
            "Corner Δ %": pct_change(max(pb.corner_column_x_m, pb.corner_column_y_m), max(pa.corner_column_x_m, pa.corner_column_y_m)),
            "Perimeter col before (m)": max(pb.perimeter_column_x_m, pb.perimeter_column_y_m),
            "Perimeter col after (m)": max(pa.perimeter_column_x_m, pa.perimeter_column_y_m),
            "Perimeter Δ %": pct_change(max(pb.perimeter_column_x_m, pb.perimeter_column_y_m), max(pa.perimeter_column_x_m, pa.perimeter_column_y_m)),
            "Interior col before (m)": max(pb.interior_column_x_m, pb.interior_column_y_m),
            "Interior col after (m)": max(pa.interior_column_x_m, pa.interior_column_y_m),
            "Interior Δ %": pct_change(max(pb.interior_column_x_m, pb.interior_column_y_m), max(pa.interior_column_x_m, pa.interior_column_y_m)),
        })
    return pd.DataFrame(rows)


# ----------------------------- REPORTS / TABLES -----------------------------

def build_report(result: DesignResult) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append(f"{APP_TITLE} - {APP_VERSION}")
    lines.append("=" * 74)
    lines.append("")
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 74)
    lines.append(f"Reference period               = {result.reference_period_s:.3f} s")
    lines.append(f"Design target period           = {result.design_target_period_s:.3f} s")
    lines.append(f"Estimated dynamic period       = {result.estimated_period_s:.3f} s")
    lines.append(f"Upper limit period             = {result.upper_limit_period_s:.3f} s")
    lines.append(f"Period error ratio             = {100 * result.period_error_ratio:.2f} %")
    lines.append(f"Period check                   = {'OK' if result.period_ok else 'NOT OK'}")
    lines.append(f"Effective lateral stiffness    = {result.K_estimated_N_per_m:,.3e} N/m")
    lines.append(f"Estimated roof drift           = {result.top_drift_m:.3f} m")
    lines.append(f"Estimated roof drift ratio     = {result.drift_ratio:.5f}")
    lines.append(f"Max story drift                = {result.max_story_drift_m:.4f} m")
    lines.append(f"Max story drift ratio          = {result.max_story_drift_ratio:.5f}")
    lines.append(f"Drift check                    = {'OK' if result.drift_ok else 'NOT OK'}")
    lines.append(f"Total structural weight        = {result.total_weight_kN:,.0f} kN")
    lines.append(f"Core scale factor              = {result.core_scale:.3f}")
    lines.append(f"Column scale factor            = {result.column_scale:.3f}")
    lines.append("")
    lines.append("OUTRIGGERS")
    lines.append("-" * 74)
    if result.outrigger_results:
        for o in result.outrigger_results:
            lines.append(
                f"Story {o.story_level:>3} | h = {o.height_m:>6.1f} m | "
                f"K = {o.stiffness_contribution:>10.3e} N/m | "
                f"Krot = {o.K_rotational_Nm_per_rad:>10.3e} Nm/rad"
            )
    else:
        lines.append("No outriggers defined.")
    lines.append("")
    lines.append("ZONE-BY-ZONE CORE")
    lines.append("-" * 74)
    for z in result.zone_core_results:
        lines.append(
            f"{z.zone.name:12s} ({z.zone.story_start:>2}-{z.zone.story_end:<2}) | "
            f"t = {z.wall_thickness:.2f} m | Ieff = {z.Ieq_effective_m4:,.2f} m^4 | walls = {z.wall_count}"
        )
    lines.append("")
    lines.append("ZONE-BY-ZONE COLUMNS")
    lines.append("-" * 74)
    for z in result.zone_column_results:
        lines.append(
            f"{z.zone.name:12s} ({z.zone.story_start:>2}-{z.zone.story_end:<2}) | "
            f"Corner = {z.corner_column_x_m:.2f}x{z.corner_column_y_m:.2f} m | "
            f"Perim = {z.perimeter_column_x_m:.2f}x{z.perimeter_column_y_m:.2f} m | "
            f"Interior = {z.interior_column_x_m:.2f}x{z.interior_column_y_m:.2f} m"
        )
    lines.append("")
    lines.append("MESSAGES")
    lines.append("-" * 74)
    for m in result.messages:
        lines.append(f"- {m}")
    return "\n".join(lines)


def result_summary_table(result: DesignResult) -> pd.DataFrame:
    return pd.DataFrame({
        "Parameter": [
            "H (m)", "Floor Area (m²)", "Weight (kN)", "T_ref (s)", "T_target (s)", "T_est (s)",
            "T_upper (s)", "Roof Drift (m)", "Roof Drift Ratio", "Max Story Drift (m)",
            "Max Story Drift Ratio", "Beam b (m)", "Beam h (m)", "Slab t (m)"
        ],
        "Value": [
            result.H_m, result.floor_area_m2, result.total_weight_kN, result.reference_period_s,
            result.design_target_period_s, result.estimated_period_s, result.upper_limit_period_s,
            result.top_drift_m, result.drift_ratio, result.max_story_drift_m,
            result.max_story_drift_ratio, result.beam_width_m, result.beam_depth_m, result.slab_thickness_m
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


def story_drift_table(result: DesignResult) -> pd.DataFrame:
    if result.static_response is None:
        return pd.DataFrame()
    return pd.DataFrame({
        "Story": np.arange(1, len(result.static_response.story_drifts_m) + 1),
        "Story Drift (m)": result.static_response.story_drifts_m,
        "Story Drift Ratio": result.static_response.story_drift_ratios,
        "Displacement (m)": result.static_response.displacements_m,
        "Story Force (N)": result.static_response.story_forces_N,
    })


# ----------------------------- PLOTS -----------------------------

def plot_mode_shapes(result: DesignResult, modes: int = 3) -> plt.Figure:
    modal = result.modal_result
    if modal is None:
        raise ValueError("Modal result is not available.")

    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, modal.n_dof + 1)

    for i in range(min(modes, len(modal.mode_shapes))):
        ax.plot(modal.mode_shapes[i], y, marker="o", label=f"Mode {i+1} | T={modal.periods_s[i]:.3f}s")

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


def plot_story_drifts(result: DesignResult) -> plt.Figure:
    if result.static_response is None:
        raise ValueError("Static response is not available.")
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.static_response.story_drift_ratios) + 1)
    ax.plot(result.static_response.story_drift_ratios, y, marker="o")
    ax.axvline(result.max_story_drift_ratio, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Story drift ratio")
    ax.set_ylabel("Story")
    ax.set_title("Story drift ratios")
    ax.grid(True, alpha=0.3)
    return fig


def plot_plan(result: DesignResult, inp: BuildingInput, zone_name: str = "Lower Zone") -> plt.Figure:
    core = next(z for z in result.zone_core_results if z.zone.name == zone_name)
    cols = next(z for z in result.zone_column_results if z.zone.name == zone_name)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.5)

    for i in range(inp.n_bays_x + 1):
        gx = i * inp.bay_x
        ax.plot([gx, gx], [0, inp.plan_y], color="#cccccc", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        gy = j * inp.bay_y
        ax.plot([0, inp.plan_x], [gy, gy], color="#cccccc", linewidth=0.8)

    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            px = i * inp.bay_x
            py = j * inp.bay_y
            at_lr = i == 0 or i == inp.n_bays_x
            at_bt = j == 0 or j == inp.n_bays_y
            if at_lr and at_bt:
                dx, dy, c = cols.corner_column_x_m, cols.corner_column_y_m, "#8b0000"
            elif at_lr or at_bt:
                dx, dy, c = cols.perimeter_column_x_m, cols.perimeter_column_y_m, "#cc5500"
            else:
                dx, dy, c = cols.interior_column_x_m, cols.interior_column_y_m, "#4444aa"
            rect = plt.Rectangle((px - dx / 2, py - dy / 2), dx, dy, facecolor=c, edgecolor=c, alpha=0.9)
            ax.add_patch(rect)

    cx0 = (inp.plan_x - core.core_outer_x) / 2
    cy0 = (inp.plan_y - core.core_outer_y) / 2
    cx1 = cx0 + core.core_outer_x
    cy1 = cy0 + core.core_outer_y
    ix0 = (inp.plan_x - core.core_opening_x) / 2
    iy0 = (inp.plan_y - core.core_opening_y) / 2

    t = core.wall_thickness
    wall_color = "#2e8b57"
    ax.add_patch(plt.Rectangle((cx0, cy0), core.core_outer_x, core.core_outer_y, fill=False, edgecolor=wall_color, linewidth=2.5))
    ax.add_patch(plt.Rectangle((ix0, iy0), core.core_opening_x, core.core_opening_y, fill=False, edgecolor="#666666", linewidth=1.5, linestyle="--"))

    for x, y, w, h in [
        (cx0, cy0, core.core_outer_x, t),
        (cx0, cy1 - t, core.core_outer_x, t),
        (cx0, cy0, t, core.core_outer_y),
        (cx1 - t, cy0, t, core.core_outer_y),
    ]:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=wall_color, edgecolor=wall_color, alpha=0.3))

    for or_result in result.outrigger_results:
        if core.zone.story_start <= or_result.story_level <= core.zone.story_end:
            arm = or_result.truss_width_m
            dep = or_result.truss_depth_m
            center_x = inp.plan_x / 2
            center_y = inp.plan_y / 2
            outrigger_color = "#ff6b00"
            ax.add_patch(plt.Rectangle((cx0 - arm, center_y - dep / 2), arm, dep, facecolor=outrigger_color, edgecolor=outrigger_color, alpha=0.7))
            ax.add_patch(plt.Rectangle((cx1, center_y - dep / 2), arm, dep, facecolor=outrigger_color, edgecolor=outrigger_color, alpha=0.7))
            ax.add_patch(plt.Rectangle((center_x - dep / 2, cy0 - arm), dep, arm, facecolor=outrigger_color, edgecolor=outrigger_color, alpha=0.7))
            ax.add_patch(plt.Rectangle((center_x - dep / 2, cy1), dep, arm, facecolor=outrigger_color, edgecolor=outrigger_color, alpha=0.7))

    ax.set_aspect("equal")
    ax.set_xlim(-5, inp.plan_x + 5)
    ax.set_ylim(-5, inp.plan_y + 5)
    ax.set_title(f"Plan view - {zone_name}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig


def plot_elevation(result: DesignResult, inp: BuildingInput) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 10))
    h_total = total_height(inp)

    ax.add_patch(plt.Rectangle((0, 0), inp.plan_x, h_total, fill=False, edgecolor="black", linewidth=1.5))

    for i in range(inp.n_story + 1):
        y = i * inp.story_height
        ax.plot([0, inp.plan_x], [y, y], color="#d9d9d9", linewidth=0.8)

    x_positions = np.linspace(0, inp.plan_x, inp.n_bays_x + 1)
    lower_cols = next(z for z in result.zone_column_results if z.zone.name == "Lower Zone")
    for idx, x in enumerate(x_positions):
        if idx == 0 or idx == len(x_positions) - 1:
            w = lower_cols.corner_column_x_m
            color = "#8b0000"
        elif idx == 1 or idx == len(x_positions) - 2:
            w = lower_cols.perimeter_column_x_m
            color = "#cc5500"
        else:
            w = lower_cols.interior_column_x_m
            color = "#4444aa"
        ax.add_patch(plt.Rectangle((x - w / 2, 0), w, h_total, facecolor=color, edgecolor=color, alpha=0.18))

    core = next(z for z in result.zone_core_results if z.zone.name == "Lower Zone")
    cx0 = (inp.plan_x - core.core_outer_x) / 2.0
    ax.add_patch(plt.Rectangle((cx0, 0), core.core_outer_x, h_total, facecolor="#2e8b57", edgecolor="#2e8b57", alpha=0.20))

    for o in result.outrigger_results:
        y = o.story_level * inp.story_height
        ax.add_patch(plt.Rectangle((0, y - o.truss_depth_m / 2), inp.plan_x, o.truss_depth_m, facecolor="#ff6b00", edgecolor="#ff6b00", alpha=0.35))
        ax.text(inp.plan_x + 1.0, y, f"L{o.story_level}", va="center")

    ax.set_xlim(-2, inp.plan_x + 8)
    ax.set_ylim(0, h_total + 5)
    ax.set_xlabel("Building width / plan projection (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Elevation view with core, columns, and outriggers")
    ax.grid(True, alpha=0.2)
    return fig


# ----------------------------- STREAMLIT UI -----------------------------

def streamlit_input_panel() -> BuildingInput:
    st.sidebar.header("Input Data")

    plan_shape = st.sidebar.radio("Plan shape", ["square", "triangle"], horizontal=True)
    if plan_shape == "triangle":
        st.sidebar.warning("Triangle option currently affects area only. Plot and framing layout remain orthogonal.")

    st.sidebar.subheader("Geometry")
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

    st.sidebar.subheader("Vertical circulation / core opening")
    stair_count = int(st.sidebar.number_input("Stairs", min_value=0, max_value=20, value=2, step=1))
    elevator_count = int(st.sidebar.number_input("Elevators", min_value=0, max_value=30, value=4, step=1))
    elevator_area_each = float(st.sidebar.number_input("Elevator area each (m²)", min_value=0.0, max_value=20.0, value=3.5))
    stair_area_each = float(st.sidebar.number_input("Stair area each (m²)", min_value=0.0, max_value=50.0, value=20.0))
    service_area = float(st.sidebar.number_input("Service area (m²)", min_value=0.0, max_value=200.0, value=35.0))

    st.sidebar.subheader("Materials / loads")
    fck = float(st.sidebar.number_input("fck (MPa)", min_value=20.0, max_value=100.0, value=70.0))
    ec = float(st.sidebar.number_input("Ec (MPa)", min_value=15000.0, max_value=50000.0, value=36000.0))
    fy = float(st.sidebar.number_input("fy (MPa)", min_value=200.0, max_value=700.0, value=420.0))
    dl = float(st.sidebar.number_input("DL (kN/m²)", min_value=0.0, max_value=20.0, value=3.0))
    ll = float(st.sidebar.number_input("LL (kN/m²)", min_value=0.0, max_value=20.0, value=2.5))
    slab_finish_allowance = float(st.sidebar.number_input("Slab/fit-out allowance (kN/m²)", min_value=0.0, max_value=10.0, value=1.5))

    st.sidebar.subheader("Cracked section / serviceability")
    wall_cracked_factor = float(st.sidebar.number_input("Wall cracked factor", min_value=0.10, max_value=1.00, value=0.40))
    column_cracked_factor = float(st.sidebar.number_input("Column cracked factor", min_value=0.10, max_value=1.00, value=0.70))
    drift_limit_ratio = float(st.sidebar.number_input("Allowable drift ratio", min_value=0.0005, max_value=0.0200, value=1 / 500, format="%.4f"))

    st.sidebar.subheader("Seismic input")
    prelim_lateral_force_coeff = float(st.sidebar.number_input("Base lateral coefficient", min_value=0.001, max_value=0.300, value=0.015, format="%.3f"))
    seismic_zone_factor = float(st.sidebar.number_input("Seismic zone factor", min_value=0.20, max_value=3.00, value=1.00, format="%.2f"))
    importance_factor = float(st.sidebar.number_input("Importance factor I", min_value=0.50, max_value=2.00, value=1.00, format="%.2f"))
    behavior_factor = float(st.sidebar.number_input("Behavior / reduction factor R", min_value=1.00, max_value=10.00, value=4.00, format="%.2f"))
    spectral_accel_short = float(st.sidebar.number_input("Short-period spectral accel SDS", min_value=0.10, max_value=3.00, value=1.00, format="%.2f"))
    spectral_accel_1s = float(st.sidebar.number_input("1-sec spectral accel SD1", min_value=0.05, max_value=2.00, value=0.40, format="%.2f"))
    seismic_mass_factor = float(st.sidebar.number_input("Seismic mass factor", min_value=0.50, max_value=1.50, value=1.00, format="%.2f"))
    ct = float(st.sidebar.number_input("Ct", min_value=0.0050, max_value=0.2000, value=0.0488, format="%.4f"))
    x_period = float(st.sidebar.number_input("x exponent", min_value=0.50, max_value=1.20, value=0.75, format="%.2f"))
    upper_period_factor = float(st.sidebar.number_input("Upper period factor", min_value=1.00, max_value=2.00, value=1.20, format="%.2f"))
    target_position_factor = float(st.sidebar.number_input("Target position factor β", min_value=0.00, max_value=1.00, value=0.85, format="%.2f"))

    st.sidebar.subheader("Outriggers")
    outrigger_count = int(st.sidebar.selectbox("Number of outriggers", [0, 1, 2, 3], index=2))

    suggested_levels = []
    if outrigger_count >= 1:
        suggested_levels.append(max(1, int(round(n_story * 0.45))))
    if outrigger_count >= 2:
        suggested_levels.append(max(1, int(round(n_story * 0.70))))
    if outrigger_count >= 3:
        suggested_levels.append(max(1, int(round(n_story * 0.88))))

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
    outrigger_core_relief_max = float(st.sidebar.number_input("Max core relief from outriggers", min_value=0.00, max_value=0.50, value=0.18, format="%.2f"))
    outrigger_column_relief_max = float(st.sidebar.number_input("Max column relief from outriggers", min_value=0.00, max_value=0.50, value=0.22, format="%.2f"))
    outrigger_zone_wall_boost = float(st.sidebar.number_input("Local wall demand boost at outrigger zones", min_value=0.00, max_value=0.30, value=0.06, format="%.2f"))
    outrigger_zone_column_boost = float(st.sidebar.number_input("Local column demand boost at outrigger zones", min_value=0.00, max_value=0.30, value=0.08, format="%.2f"))

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
        Ec=ec,
        fy=fy,
        DL=dl,
        LL=ll,
        slab_finish_allowance=slab_finish_allowance,
        wall_cracked_factor=wall_cracked_factor,
        column_cracked_factor=column_cracked_factor,
        prelim_lateral_force_coeff=prelim_lateral_force_coeff,
        drift_limit_ratio=drift_limit_ratio,
        seismic_zone_factor=seismic_zone_factor,
        importance_factor=importance_factor,
        behavior_factor=behavior_factor,
        spectral_accel_short=spectral_accel_short,
        spectral_accel_1s=spectral_accel_1s,
        seismic_mass_factor=seismic_mass_factor,
        Ct=ct,
        x_period=x_period,
        upper_period_factor=upper_period_factor,
        target_position_factor=target_position_factor,
        outrigger_count=outrigger_count,
        outrigger_story_levels=levels,
        outrigger_truss_depth_m=outrigger_truss_depth_m,
        outrigger_chord_area_m2=outrigger_chord_area_m2,
        outrigger_diagonal_area_m2=outrigger_diagonal_area_m2,
        outrigger_core_relief_max=outrigger_core_relief_max,
        outrigger_column_relief_max=outrigger_column_relief_max,
        outrigger_zone_wall_boost=outrigger_zone_wall_boost,
        outrigger_zone_column_boost=outrigger_zone_column_boost,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info(
        "Engineering fixes applied: corrected story stiffness trend, removed outrigger double counting, "
        "added static drift solution from K·u=F, and exposed cracked-section, seismic, and outrigger-sizing inputs."
    )

    if "result" not in st.session_state:
        st.session_state.result = None
    if "report" not in st.session_state:
        st.session_state.report = ""
    if "baseline_result" not in st.session_state:
        st.session_state.baseline_result = None

    inp = streamlit_input_panel()

    col1, col2, col3 = st.columns(3)
    with col1:
        analyze = st.button("Analyze")
    with col2:
        show_modes = st.button("Show 5 Modes")
    with col3:
        clear_btn = st.button("Clear Results")

    if clear_btn:
        st.session_state.result = None
        st.session_state.report = ""
        st.session_state.baseline_result = None
        st.rerun()

    if analyze:
        try:
            with st.spinner("Running iterative MDOF analysis..."):
                base_inp = clone_input_without_outriggers(inp)
                base_res = run_design(base_inp)
                res = run_design(inp)
                st.session_state.baseline_result = base_res
                st.session_state.result = res
                st.session_state.report = build_report(res)
            st.success(f"Analysis completed in {len(res.iteration_history)} iterations.")
        except Exception as exc:
            st.exception(exc)

    if show_modes and st.session_state.result is None:
        try:
            with st.spinner("Running analysis first..."):
                base_inp = clone_input_without_outriggers(inp)
                base_res = run_design(base_inp)
                res = run_design(inp)
                st.session_state.baseline_result = base_res
                st.session_state.result = res
                st.session_state.report = build_report(res)
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.result
    if result is None:
        st.warning("Enter data and click Analyze.")
        return

    baseline_result = st.session_state.baseline_result

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Summary", "Comparison", "Iteration", "Modes", "Plan", "Elevation", "Report"
    ])

    with tab1:
        st.subheader("Summary")
        st.dataframe(result_summary_table(result), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T_est (s)", f"{result.estimated_period_s:.3f}")
        c2.metric("T_target (s)", f"{result.design_target_period_s:.3f}")
        c3.metric("Max story drift ratio", f"{result.max_story_drift_ratio:.5f}")
        c4.metric("Weight (kN)", f"{result.total_weight_kN:,.0f}")

        st.subheader("Story drift table")
        st.dataframe(story_drift_table(result), use_container_width=True)
        st.pyplot(plot_story_drifts(result))

        if result.redesign_suggestions:
            st.subheader("Redesign suggestions")
            for item in result.redesign_suggestions:
                st.write(f"- {item}")

    with tab2:
        st.subheader("Before / after outrigger comparison")
        if baseline_result is None:
            st.info("Run analysis to generate the comparison table.")
        else:
            st.markdown("**Global comparison**")
            st.dataframe(global_comparison_table(baseline_result, result), use_container_width=True)

            st.markdown("**Zone-by-zone member comparison**")
            st.dataframe(zone_comparison_table(baseline_result, result), use_container_width=True)

    with tab3:
        st.subheader("Iteration history")
        df_iter = iteration_table(result)
        st.dataframe(df_iter, use_container_width=True)
        st.pyplot(plot_iteration_history(result))
        csv_iter = df_iter.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download iteration CSV",
            data=csv_iter,
            file_name="iteration_history.csv",
            mime="text/csv",
        )

    with tab4:
        st.subheader("Mode shapes")
        st.pyplot(plot_mode_shapes(result, modes=5))

    with tab5:
        st.subheader("Plan view")
        zone_options = [z.zone.name for z in result.zone_core_results]
        zone_name = st.selectbox("Zone", zone_options, index=0)
        st.pyplot(plot_plan(result, inp, zone_name=zone_name))

    with tab6:
        st.subheader("Elevation view")
        st.pyplot(plot_elevation(result, inp))

    with tab7:
        st.subheader("Report")
        st.text_area("Text report", st.session_state.report, height=500)
        st.download_button(
            "Download report as TXT",
            data=st.session_state.report,
            file_name="tall_building_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
