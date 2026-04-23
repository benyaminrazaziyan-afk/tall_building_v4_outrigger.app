from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Tall Building Rational Analysis + Braced Outrigger"
APP_VERSION = "v8.2-rebuilt"
G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3
STEEL_DENSITY = 7850.0  # kg/m3
KAPPA = 1.2


@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int

    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1


@dataclass
class ZoneMemberInput:
    wall_thickness_m: float
    beam_width_m: float
    beam_depth_m: float
    slab_thickness_m: float
    corner_col_x_m: float
    corner_col_y_m: float
    perimeter_col_x_m: float
    perimeter_col_y_m: float
    interior_col_x_m: float
    interior_col_y_m: float
    wall_count: int = 8


@dataclass
class PerimeterWallInput:
    thickness_m: float
    top_length_m: float
    bottom_length_m: float
    left_length_m: float
    right_length_m: float


@dataclass
class RetainingWallInput:
    enabled: bool = True
    thickness_m: float = 0.35
    top_length_m: float = 0.0
    bottom_length_m: float = 0.0
    left_length_m: float = 0.0
    right_length_m: float = 0.0

    def normalized(self, plan_x: float, plan_y: float) -> "RetainingWallInput":
        return RetainingWallInput(
            enabled=self.enabled,
            thickness_m=self.thickness_m,
            top_length_m=plan_x if self.top_length_m <= 0 else min(self.top_length_m, plan_x),
            bottom_length_m=plan_x if self.bottom_length_m <= 0 else min(self.bottom_length_m, plan_x),
            left_length_m=plan_y if self.left_length_m <= 0 else min(self.left_length_m, plan_y),
            right_length_m=plan_y if self.right_length_m <= 0 else min(self.right_length_m, plan_y),
        )



@dataclass
class BuildingInput:
    n_story: int = 60
    n_basement: int = 2
    story_height_m: float = 3.2
    basement_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6
    bay_x_m: float = 8.0
    bay_y_m: float = 7.0

    core_outer_x_m: float = 18.0
    core_outer_y_m: float = 15.0
    core_opening_x_m: float = 12.0
    core_opening_y_m: float = 10.5

    Ec_mpa: float = 34000.0
    Es_mpa: float = 200000.0
    nu_concrete: float = 0.20

    dl_kn_m2: float = 3.0
    ll_kn_m2: float = 2.0
    superimposed_dead_kn_m2: float = 1.5
    facade_line_load_kn_m: float = 1.0
    live_load_mass_factor: float = 0.30
    seismic_base_shear_coeff: float = 0.05
    drift_limit_ratio: float = 1.0 / 500.0

    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.60
    beam_cracked_factor: float = 0.35
    slab_cracked_factor: float = 0.25
    retaining_wall_cracked_factor: float = 0.50

    Ct: float = 0.0488
    x_period: float = 0.75
    Cu: float = 1.4

    lower_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.50, 0.45, 0.85, 0.24, 1.00, 1.00, 0.90, 0.90, 0.80, 0.80, 8))
    middle_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.40, 0.40, 0.75, 0.22, 0.90, 0.90, 0.80, 0.80, 0.70, 0.70, 6))
    upper_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.30, 0.35, 0.65, 0.20, 0.75, 0.75, 0.65, 0.65, 0.55, 0.55, 4))

    lower_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.25, 14.0, 14.0, 12.0, 12.0))
    middle_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.22, 10.0, 10.0, 8.0, 8.0))
    upper_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.18, 6.0, 6.0, 5.0, 5.0))
    retaining_wall: RetainingWallInput = field(default_factory=RetainingWallInput)

    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each_m2: float = 3.5
    stair_area_each_m2: float = 20.0
    service_area_m2: float = 35.0
    corridor_factor: float = 1.35

    outrigger_count: int = 2
    outrigger_story_levels: List[int] = field(default_factory=lambda: [28, 42])
    brace_outer_diameter_mm: float = 355.6
    brace_thickness_mm: float = 16.0
    braces_per_side: int = 2
    outrigger_depth_m: float = 3.0
    brace_effective_length_factor: float = 1.0
    brace_buckling_reduction: float = 0.85
    target_period_s: float = 3.80


def nearest_grid_index(value: float, spacing: float, max_bays: int) -> int:
    idx = int(round(value / max(spacing, 1e-12)))
    return max(0, min(max_bays, idx))


def auto_brace_anchor_columns(inp: "BuildingInput") -> Dict[str, Tuple[float, float]]:
    cx0 = 0.5 * (inp.plan_x_m - inp.core_outer_x_m)
    cx1 = cx0 + inp.core_outer_x_m
    cy0 = 0.5 * (inp.plan_y_m - inp.core_outer_y_m)
    cy1 = cy0 + inp.core_outer_y_m
    x_mid = 0.5 * inp.plan_x_m
    y_mid = 0.5 * inp.plan_y_m
    left_col_x = nearest_grid_index(cx0 / inp.bay_x_m, 1.0, inp.n_bays_x) * inp.bay_x_m
    right_col_x = nearest_grid_index(cx1 / inp.bay_x_m, 1.0, inp.n_bays_x) * inp.bay_x_m
    bottom_col_y = nearest_grid_index(cy0 / inp.bay_y_m, 1.0, inp.n_bays_y) * inp.bay_y_m
    top_col_y = nearest_grid_index(cy1 / inp.bay_y_m, 1.0, inp.n_bays_y) * inp.bay_y_m
    return {
        "left": ((cx0, y_mid), (0.0, y_mid if inp.n_bays_y % 2 == 0 else nearest_grid_index(y_mid/inp.bay_y_m,1.0,inp.n_bays_y)*inp.bay_y_m)),
        "right": ((cx1, y_mid), (inp.plan_x_m, y_mid if inp.n_bays_y % 2 == 0 else nearest_grid_index(y_mid/inp.bay_y_m,1.0,inp.n_bays_y)*inp.bay_y_m)),
        "bottom": ((x_mid, cy0), (x_mid if inp.n_bays_x % 2 == 0 else nearest_grid_index(x_mid/inp.bay_x_m,1.0,inp.n_bays_x)*inp.bay_x_m, 0.0)),
        "top": ((x_mid, cy1), (x_mid if inp.n_bays_x % 2 == 0 else nearest_grid_index(x_mid/inp.bay_x_m,1.0,inp.n_bays_x)*inp.bay_x_m, inp.plan_y_m)),
    }


def target_scale_factor(current: float, target: float, exponent: float, low: float, high: float) -> float:
    if current <= 1e-9 or target <= 1e-9:
        return 1.0
    s = (current / target) ** exponent
    return max(low, min(high, s))


def scale_zone_member(z, wall_scale: float, frame_scale: float, slab_scale: float):
    from dataclasses import replace
    return replace(
        z,
        wall_thickness_m=max(0.20, min(1.50, z.wall_thickness_m * wall_scale)),
        beam_width_m=max(0.25, min(1.20, z.beam_width_m * frame_scale)),
        beam_depth_m=max(0.40, min(1.80, z.beam_depth_m * frame_scale)),
        slab_thickness_m=max(0.16, min(0.45, z.slab_thickness_m * slab_scale)),
        corner_col_x_m=max(0.35, min(2.00, z.corner_col_x_m * frame_scale)),
        corner_col_y_m=max(0.35, min(2.00, z.corner_col_y_m * frame_scale)),
        perimeter_col_x_m=max(0.35, min(2.00, z.perimeter_col_x_m * frame_scale)),
        perimeter_col_y_m=max(0.35, min(2.00, z.perimeter_col_y_m * frame_scale)),
        interior_col_x_m=max(0.30, min(2.00, z.interior_col_x_m * frame_scale)),
        interior_col_y_m=max(0.30, min(2.00, z.interior_col_y_m * frame_scale)),
    )


@dataclass
class OutriggerResult:
    story_level: int
    brace_length_m: float
    brace_area_m2: float
    brace_radius_gyration_m: float
    slenderness: float
    axial_stiffness_n: float
    effective_story_stiffness_n_m: float
    steel_weight_kg: float


@dataclass
class ZoneStiffnessResult:
    zone_name: str
    story_start: int
    story_end: int
    wall_count: int
    wall_t_m: float
    beam_b_m: float
    beam_h_m: float
    slab_t_m: float
    perimeter_wall_t_m: float
    corner_col_m: str
    perimeter_col_m: str
    interior_col_m: str
    k_core_story_n_m: float
    k_perimeter_wall_story_n_m: float
    k_frame_story_n_m: float
    k_total_story_n_m: float


@dataclass
class AnalysisResult:
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    story_stiffness_n_m: List[float]
    story_masses_kg: List[float]
    floor_displacements_m: List[float]
    story_drifts_m: List[float]
    story_drift_ratios: List[float]
    story_elevations_m: List[float]
    lateral_forces_n: List[float]
    zone_results: List[ZoneStiffnessResult]
    outriggers: List[OutriggerResult]
    summary_table: pd.DataFrame
    story_table: pd.DataFrame
    zone_table: pd.DataFrame
    outrigger_table: pd.DataFrame
    total_weight_kn: float
    total_mass_kg: float
    base_shear_kn: float
    T_code_s: float
    T_upper_s: float
    basement_wall_report: str


def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def zone_input(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    return {"Lower Zone": inp.lower_zone, "Middle Zone": inp.middle_zone, "Upper Zone": inp.upper_zone}[zone_name]


def perimeter_input(inp: BuildingInput, zone_name: str) -> PerimeterWallInput:
    return {"Lower Zone": inp.lower_perimeter_walls, "Middle Zone": inp.middle_perimeter_walls, "Upper Zone": inp.upper_perimeter_walls}[zone_name]


def gross_floor_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m


def concrete_shear_modulus(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    return E / (2 * (1 + inp.nu_concrete))


def rect_ix(b: float, h: float) -> float:
    return b * h ** 3 / 12.0


def rect_iy(b: float, h: float) -> float:
    return h * b ** 3 / 12.0


def circular_hollow_area(do: float, t: float) -> float:
    di = max(do - 2 * t, 1e-6)
    return pi / 4.0 * (do ** 2 - di ** 2)


def circular_hollow_rg(do: float, t: float) -> float:
    di = max(do - 2 * t, 1e-6)
    I = pi / 64.0 * (do ** 4 - di ** 4)
    A = circular_hollow_area(do, t)
    return sqrt(I / max(A, 1e-12))


def wall_lengths_for_core(inp: BuildingInput, wall_count: int) -> List[float]:
    ox, oy = inp.core_outer_x_m, inp.core_outer_y_m
    if wall_count == 4:
        return [ox, ox, oy, oy]
    if wall_count == 6:
        return [ox, ox, oy, oy, 0.45 * ox, 0.45 * ox]
    return [ox, ox, oy, oy, 0.45 * ox, 0.45 * ox, 0.45 * oy, 0.45 * oy]


def core_min_inertia(inp: BuildingInput, t: float, wall_count: int) -> float:
    ox, oy = inp.core_outer_x_m, inp.core_outer_y_m
    lengths = wall_lengths_for_core(inp, wall_count)
    x_side, y_side = ox / 2.0, oy / 2.0
    Ix = 0.0
    Iy = 0.0
    for L in lengths[:2]:
        Ix += rect_ix(L, t) + L * t * y_side ** 2
        Iy += rect_iy(L, t)
    for L in lengths[2:4]:
        Iy += rect_iy(t, L) + L * t * x_side ** 2
        Ix += rect_ix(t, L)
    if wall_count >= 6:
        inner_x = 0.22 * ox
        for sign, L in zip([-1, 1], lengths[4:6]):
            Iy += rect_iy(t, L) + L * t * (sign * inner_x) ** 2
            Ix += rect_ix(t, L)
    if wall_count >= 8:
        inner_y = 0.22 * oy
        for sign, L in zip([-1, 1], lengths[6:8]):
            Ix += rect_ix(L, t) + L * t * (sign * inner_y) ** 2
            Iy += rect_iy(L, t)
    return min(Ix, Iy)


def total_cols(inp: BuildingInput) -> Tuple[int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return corner, perimeter, interior




def story_core_component(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus(inp)
    h = inp.story_height_m
    Ieff = inp.wall_cracked_factor * core_min_inertia(inp, z.wall_thickness_m, z.wall_count)
    kb = 12 * E * Ieff / max(h ** 3, 1e-12)
    area = sum(L * z.wall_thickness_m for L in wall_lengths_for_core(inp, z.wall_count))
    ks = KAPPA * Gc * area / max(h, 1e-12)
    k_story = 1.0 / max(1.0 / max(kb, 1e-12) + 1.0 / max(ks, 1e-12), 1e-18)
    return 0.018 * k_story


def story_perimeter_wall_component(inp: BuildingInput, p: PerimeterWallInput) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus(inp)
    h = inp.story_height_m
    t = p.thickness_m
    if t <= 0:
        return 0.0
    Ix = rect_ix(p.top_length_m, t) + p.top_length_m * t * (inp.plan_y_m / 2.0) ** 2
    Ix += rect_ix(p.bottom_length_m, t) + p.bottom_length_m * t * (inp.plan_y_m / 2.0) ** 2
    Iy = rect_iy(t, p.left_length_m) + p.left_length_m * t * (inp.plan_x_m / 2.0) ** 2
    Iy += rect_iy(t, p.right_length_m) + p.right_length_m * t * (inp.plan_x_m / 2.0) ** 2
    Ieff = inp.wall_cracked_factor * min(Ix, Iy)
    area = t * (p.top_length_m + p.bottom_length_m + p.left_length_m + p.right_length_m)
    kb = 12 * E * Ieff / max(h ** 3, 1e-12)
    ks = KAPPA * Gc * area / max(h, 1e-12)
    k_story = 1.0 / max(1.0 / max(kb, 1e-12) + 1.0 / max(ks, 1e-12), 1e-18)
    return 0.015 * k_story


def story_frame_component(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    corner, perimeter, interior = total_cols(inp)
    I_corner = min(rect_ix(z.corner_col_x_m, z.corner_col_y_m), rect_iy(z.corner_col_x_m, z.corner_col_y_m))
    I_per = min(rect_ix(z.perimeter_col_x_m, z.perimeter_col_y_m), rect_iy(z.perimeter_col_x_m, z.perimeter_col_y_m))
    I_int = min(rect_ix(z.interior_col_x_m, z.interior_col_y_m), rect_iy(z.interior_col_x_m, z.interior_col_y_m))
    Itot = inp.column_cracked_factor * (corner * I_corner + perimeter * I_per + interior * I_int)
    k_col = 12 * E * Itot / max(h ** 3, 1e-12)
    Ib = inp.beam_cracked_factor * rect_ix(z.beam_width_m, z.beam_depth_m)
    n_beams = inp.n_bays_x * (inp.n_bays_y + 1) + inp.n_bays_y * (inp.n_bays_x + 1)
    beam_ratio = n_beams * Ib / max(Itot / max(inp.column_cracked_factor, 1e-12), 1e-12)
    amplifier = 1.0 + min(0.45, 0.02 * beam_ratio)
    return 0.020 * k_col * amplifier

def floor_weight_kn(inp: BuildingInput, z: ZoneMemberInput) -> float:
    area = gross_floor_area(inp)
    slab_self = z.slab_thickness_m * CONCRETE_UNIT_WEIGHT
    total_beam_len = inp.n_bays_x * (inp.n_bays_y + 1) * inp.bay_x_m + inp.n_bays_y * (inp.n_bays_x + 1) * inp.bay_y_m
    beam_self = z.beam_width_m * z.beam_depth_m * CONCRETE_UNIT_WEIGHT * total_beam_len / max(area, 1e-12)
    facade = inp.facade_line_load_kn_m * 2 * (inp.plan_x_m + inp.plan_y_m) / max(area, 1e-12)
    q = inp.dl_kn_m2 + inp.superimposed_dead_kn_m2 + slab_self + beam_self + inp.live_load_mass_factor * inp.ll_kn_m2 + facade
    return area * q


def zone_flexibility_weights(inp: BuildingInput) -> Tuple[List[ZoneDefinition], Dict[str, float]]:
    zones = define_three_zones(inp.n_story)
    H = inp.n_story * inp.story_height_m
    weights = {}
    zbar_prev = 0.0
    for z in zones:
        z1 = (z.story_start - 1) * inp.story_height_m
        z2 = z.story_end * inp.story_height_m
        # weight proportional to contribution to cantilever curvature energy
        w = max((z2 ** 3 - z1 ** 3) / max(H ** 3, 1e-12), 1e-9)
        weights[z.name] = w
        zbar_prev = z2
    return zones, weights


def global_core_stiffness(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus(inp)
    H = inp.n_story * inp.story_height_m
    zones, weights = zone_flexibility_weights(inp)
    flex_b = 0.0
    flex_s = 0.0
    for z in zones:
        zm = zone_input(inp, z.name)
        I = inp.wall_cracked_factor * core_min_inertia(inp, zm.wall_thickness_m, zm.wall_count)
        A = sum(L * zm.wall_thickness_m for L in wall_lengths_for_core(inp, zm.wall_count))
        flex_b += weights[z.name] / max(E * I, 1e-12)
        flex_s += weights[z.name] / max(KAPPA * Gc * A, 1e-12)
    delta_per_unit = H ** 3 / 3.0 * flex_b + H * flex_s
    return 1.0 / max(delta_per_unit, 1e-18)


def global_perimeter_wall_stiffness(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus(inp)
    H = inp.n_story * inp.story_height_m
    zones, weights = zone_flexibility_weights(inp)
    flex_b = 0.0
    flex_s = 0.0
    for z in zones:
        p = perimeter_input(inp, z.name)
        if p.thickness_m <= 0:
            continue
        t = p.thickness_m
        lengths = [p.top_length_m, p.bottom_length_m, p.left_length_m, p.right_length_m]
        Ix = rect_ix(p.top_length_m, t) + p.top_length_m * t * (inp.plan_y_m / 2.0) ** 2
        Ix += rect_ix(p.bottom_length_m, t) + p.bottom_length_m * t * (inp.plan_y_m / 2.0) ** 2
        Iy = rect_iy(t, p.left_length_m) + p.left_length_m * t * (inp.plan_x_m / 2.0) ** 2
        Iy += rect_iy(t, p.right_length_m) + p.right_length_m * t * (inp.plan_x_m / 2.0) ** 2
        I = inp.wall_cracked_factor * min(Ix, Iy)
        A = t * sum(lengths)
        flex_b += weights[z.name] / max(E * I, 1e-12)
        flex_s += weights[z.name] / max(KAPPA * Gc * A, 1e-12)
    delta_per_unit = H ** 3 / 3.0 * flex_b + H * flex_s
    return 1.0 / max(delta_per_unit, 1e-18)


def global_frame_stiffness(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    H = inp.n_story * inp.story_height_m
    corner, perimeter, interior = total_cols(inp)
    zones, weights = zone_flexibility_weights(inp)
    flex = 0.0
    beam_restraint_sum = 0.0
    for z in zones:
        zm = zone_input(inp, z.name)
        Icol = (
            corner * min(rect_ix(zm.corner_col_x_m, zm.corner_col_y_m), rect_iy(zm.corner_col_x_m, zm.corner_col_y_m))
            + perimeter * min(rect_ix(zm.perimeter_col_x_m, zm.perimeter_col_y_m), rect_iy(zm.perimeter_col_x_m, zm.perimeter_col_y_m))
            + interior * min(rect_ix(zm.interior_col_x_m, zm.interior_col_y_m), rect_iy(zm.interior_col_x_m, zm.interior_col_y_m))
        )
        Icol *= inp.column_cracked_factor
        Ib = inp.beam_cracked_factor * rect_ix(zm.beam_width_m, zm.beam_depth_m)
        # beam-column restraint ratio moderates frame stiffness rather than adding a separate huge spring
        n_beams = inp.n_bays_x * (inp.n_bays_y + 1) + inp.n_bays_y * (inp.n_bays_x + 1)
        beam_restraint = n_beams * Ib / max((corner + perimeter + interior) * Icol / max(inp.column_cracked_factor,1e-9), 1e-12)
        beam_restraint_sum += weights[z.name] * beam_restraint
        flex += weights[z.name] / max(E * Icol, 1e-12)
    delta_per_unit = H ** 3 / 3.0 * flex
    k_bare = 1.0 / max(delta_per_unit, 1e-18)
    frame_amplifier = 1.0 + min(0.60, 0.08 * beam_restraint_sum)
    return k_bare * frame_amplifier


def retaining_wall_support_stiffness(inp: BuildingInput) -> Tuple[float, str]:
    rw = inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m)
    if not rw.enabled or inp.n_basement <= 0:
        return 0.0, "Basement retaining wall not included."
    E = inp.Ec_mpa * 1e6
    H = inp.n_basement * inp.basement_height_m
    t = rw.thickness_m
    Ix = rect_ix(rw.top_length_m, t) + rw.top_length_m * t * (inp.plan_y_m / 2.0) ** 2
    Ix += rect_ix(rw.bottom_length_m, t) + rw.bottom_length_m * t * (inp.plan_y_m / 2.0) ** 2
    Iy = rect_iy(t, rw.left_length_m) + rw.left_length_m * t * (inp.plan_x_m / 2.0) ** 2
    Iy += rect_iy(t, rw.right_length_m) + rw.right_length_m * t * (inp.plan_x_m / 2.0) ** 2
    I = inp.retaining_wall_cracked_factor * min(Ix, Iy)
    # use a capped base spring to avoid unrealistically locking the structure
    k = min(12 * E * I / max(H ** 3, 1e-12), 5.0e8)
    msg = (
        f"Basement retaining wall included as base spring: t={t:.2f} m, "
        f"active lengths={rw.top_length_m:.1f}/{rw.bottom_length_m:.1f}/{rw.left_length_m:.1f}/{rw.right_length_m:.1f} m."
    )
    return k, msg


def brace_arm_length(inp: BuildingInput) -> float:
    return 0.5 * max(inp.plan_x_m - inp.core_outer_x_m, inp.plan_y_m - inp.core_outer_y_m)


def outrigger_results(inp: BuildingInput) -> List[OutriggerResult]:
    if inp.outrigger_count <= 0:
        return []
    levels = [lv for lv in inp.outrigger_story_levels[: inp.outrigger_count] if 1 <= lv <= inp.n_story]
    if not levels:
        return []
    do = inp.brace_outer_diameter_mm / 1000.0
    t = inp.brace_thickness_mm / 1000.0
    A = circular_hollow_area(do, t)
    rg = circular_hollow_rg(do, t)
    E = inp.Es_mpa * 1e6
    arm = brace_arm_length(inp)
    depth = inp.outrigger_depth_m
    H = inp.n_story * inp.story_height_m
    results = []
    for level in levels:
        Lb = sqrt(arm ** 2 + depth ** 2)
        axial_k = E * A / max(Lb, 1e-12)
        slenderness = inp.brace_effective_length_factor * Lb / max(rg, 1e-12)
        k_diag = axial_k * (depth / max(Lb, 1e-12)) ** 2
        # effective lateral stiffness contribution derived from brace geometry and outrigger arm
        k_story = 4.0 * inp.braces_per_side * inp.brace_buckling_reduction * k_diag
        # cap outrigger story stiffness to avoid nonphysical period collapse
        k_story = min(k_story, 1.2e9)
        steel_weight = 4.0 * inp.braces_per_side * Lb * A * STEEL_DENSITY
        results.append(OutriggerResult(level, Lb, A, rg, slenderness, axial_k, k_story, steel_weight))
    return results


def build_story_vectors(inp: BuildingInput, zone_results: List[ZoneStiffnessResult], outriggers: List[OutriggerResult]) -> Tuple[List[float], List[float], List[float]]:
    stiffness = [0.0] * inp.n_story
    masses = [0.0] * inp.n_story
    elevations = [i * inp.story_height_m for i in range(1, inp.n_story + 1)]
    for zr in zone_results:
        z = zone_input(inp, zr.zone_name)
        fw = floor_weight_kn(inp, z) * 1000.0 / G
        for s in range(zr.story_start, zr.story_end + 1):
            stiffness[s - 1] = zr.k_total_story_n_m
            masses[s - 1] = fw
    for o in outriggers:
        stiffness[o.story_level - 1] += o.effective_story_stiffness_n_m
        masses[o.story_level - 1] += o.steel_weight_kg
    return stiffness, masses, elevations


def assemble_m_k(masses: List[float], stiffnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(masses)
    M = np.diag(masses)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        k = stiffnesses[i]
        if i == 0:
            K[i, i] += k
        else:
            K[i, i] += k
            K[i, i - 1] -= k
            K[i - 1, i] -= k
            K[i - 1, i - 1] += k
    return M, K


def solve_modes(masses: List[float], stiffnesses: List[float], n_modes: int = 5) -> Tuple[List[float], List[float], List[List[float]]]:
    M, K = assemble_m_k(masses, stiffnesses)
    vals, vecs = np.linalg.eig(np.linalg.solve(M, K))
    vals = np.real(vals)
    vecs = np.real(vecs)
    mask = vals > 1e-12
    vals = vals[mask]
    vecs = vecs[:, mask]
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    omegas = np.sqrt(vals[:n_modes])
    periods = [2 * pi / w for w in omegas]
    freqs = [w / (2 * pi) for w in omegas]
    shapes = []
    for i in range(min(n_modes, vecs.shape[1])):
        phi = vecs[:, i].copy()
        phi /= max(np.max(np.abs(phi)), 1e-12)
        if phi[-1] < 0:
            phi = -phi
        shapes.append(phi.tolist())
    return periods, freqs, shapes


def equivalent_lateral_force_distribution(weights_kn: List[float], elevations_m: List[float], base_shear_kn: float) -> List[float]:
    wxh = [w * h for w, h in zip(weights_kn, elevations_m)]
    denom = max(sum(wxh), 1e-12)
    return [base_shear_kn * v / denom * 1000.0 for v in wxh]


def solve_static_response(stiffnesses: List[float], lateral_forces_n: List[float], story_height: float) -> Tuple[List[float], List[float], List[float]]:
    _, K = assemble_m_k([1.0] * len(stiffnesses), stiffnesses)
    u = np.linalg.solve(K, np.array(lateral_forces_n, dtype=float))
    disp = u.tolist()
    drifts = [disp[0]] + [disp[i] - disp[i - 1] for i in range(1, len(disp))]
    drift_ratios = [d / story_height for d in drifts]
    return disp, drifts, drift_ratios


def build_zone_results(inp: BuildingInput) -> List[ZoneStiffnessResult]:
    zones = define_three_zones(inp.n_story)
    results: List[ZoneStiffnessResult] = []
    for z in zones:
        zm = zone_input(inp, z.name)
        pm = perimeter_input(inp, z.name)
        k_core_story = story_core_component(inp, zm)
        k_per_story = story_perimeter_wall_component(inp, pm)
        k_frame_story = story_frame_component(inp, zm)
        k_total_story = k_core_story + k_per_story + k_frame_story
        results.append(ZoneStiffnessResult(
            zone_name=z.name,
            story_start=z.story_start,
            story_end=z.story_end,
            wall_count=zm.wall_count,
            wall_t_m=zm.wall_thickness_m,
            beam_b_m=zm.beam_width_m,
            beam_h_m=zm.beam_depth_m,
            slab_t_m=zm.slab_thickness_m,
            perimeter_wall_t_m=pm.thickness_m,
            corner_col_m=f"{zm.corner_col_x_m:.2f}×{zm.corner_col_y_m:.2f}",
            perimeter_col_m=f"{zm.perimeter_col_x_m:.2f}×{zm.perimeter_col_y_m:.2f}",
            interior_col_m=f"{zm.interior_col_x_m:.2f}×{zm.interior_col_y_m:.2f}",
            k_core_story_n_m=k_core_story,
            k_perimeter_wall_story_n_m=k_per_story,
            k_frame_story_n_m=k_frame_story,
            k_total_story_n_m=k_total_story,
        ))
    return results


def validate_input(inp: BuildingInput) -> Tuple[bool, str]:
    if abs(inp.n_bays_x * inp.bay_x_m - inp.plan_x_m) > 1e-6:
        return False, "Plan X must equal n_bays_x * bay_x."
    if abs(inp.n_bays_y * inp.bay_y_m - inp.plan_y_m) > 1e-6:
        return False, "Plan Y must equal n_bays_y * bay_y."
    if inp.core_opening_x_m >= inp.core_outer_x_m or inp.core_opening_y_m >= inp.core_outer_y_m:
        return False, "Core opening must be smaller than core outer dimensions."
    return True, "OK"


def analyze_once(inp: BuildingInput) -> AnalysisResult:
    ok, msg = validate_input(inp)
    if not ok:
        raise ValueError(msg)
    zone_results = build_zone_results(inp)
    outr = outrigger_results(inp)
    stiffness, masses, elevations = build_story_vectors(inp, zone_results, outr)
    k_ret, basement_report = retaining_wall_support_stiffness(inp)
    if k_ret > 0:
        stiffness[0] += k_ret
    periods, freqs, shapes = solve_modes(masses, stiffness, n_modes=5)
    weights_kn = [m * G / 1000.0 for m in masses]
    total_weight_kn = float(sum(weights_kn))
    total_mass_kg = float(sum(masses))
    base_shear_kn = inp.seismic_base_shear_coeff * total_weight_kn
    lateral_forces = equivalent_lateral_force_distribution(weights_kn, elevations, base_shear_kn)
    disp, drifts, drift_ratios = solve_static_response(stiffness, lateral_forces, inp.story_height_m)
    T_code = inp.Ct * (inp.n_story * inp.story_height_m) ** inp.x_period
    T_upper = inp.Cu * T_code

    zone_table = pd.DataFrame([{
        "Zone": zr.zone_name,
        "Stories": f"{zr.story_start}-{zr.story_end}",
        "Wall count": zr.wall_count,
        "Core wall t (m)": zr.wall_t_m,
        "Perimeter wall t (m)": zr.perimeter_wall_t_m,
        "Beam b×h (m)": f"{zr.beam_b_m:.2f}×{zr.beam_h_m:.2f}",
        "Slab t (m)": zr.slab_t_m,
        "Corner col (m)": zr.corner_col_m,
        "Perimeter col (m)": zr.perimeter_col_m,
        "Interior col (m)": zr.interior_col_m,
        "K core/story (N/m)": zr.k_core_story_n_m,
        "K perim walls/story (N/m)": zr.k_perimeter_wall_story_n_m,
        "K frame/story (N/m)": zr.k_frame_story_n_m,
        "K total/story (N/m)": zr.k_total_story_n_m,
    } for zr in zone_results])

    story_rows = []
    zone_map = {s: zr for zr in zone_results for s in range(zr.story_start, zr.story_end + 1)}
    outr_map = {o.story_level: o for o in outr}
    for i in range(inp.n_story, 0, -1):
        zr = zone_map[i]
        story_rows.append({
            "Story": i,
            "Elevation (m)": elevations[i - 1],
            "Zone": zr.zone_name,
            "Core wall t (m)": zr.wall_t_m,
            "Perimeter wall t (m)": zr.perimeter_wall_t_m,
            "Beam width (m)": zr.beam_b_m,
            "Beam depth (m)": zr.beam_h_m,
            "Slab t (m)": zr.slab_t_m,
            "Corner col (m)": zr.corner_col_m,
            "Perimeter col (m)": zr.perimeter_col_m,
            "Interior col (m)": zr.interior_col_m,
            "Story stiffness (N/m)": stiffness[i - 1],
            "Mass (kg)": masses[i - 1],
            "Floor displacement (m)": disp[i - 1],
            "Story drift (m)": drifts[i - 1],
            "Story drift ratio": drift_ratios[i - 1],
            "Braced story": "Yes" if i in outr_map else "No",
        })
    story_table = pd.DataFrame(story_rows)

    outrigger_table = pd.DataFrame([{
        "Story": o.story_level,
        "Brace length (m)": o.brace_length_m,
        "CHS area (m²)": o.brace_area_m2,
        "CHS r (m)": o.brace_radius_gyration_m,
        "KL/r": o.slenderness,
        "EA/L (N)": o.axial_stiffness_n,
        "Added story stiffness (N/m)": o.effective_story_stiffness_n_m,
        "Steel weight (kg)": o.steel_weight_kg,
    } for o in outr])

    summary_table = pd.DataFrame({
        "Parameter": [
            "Total weight (kN)", "Base shear (kN)", "T1 from eigen analysis (s)",
            "Design target period used for resizing only (s)",
            "Code period T_code = Ct.H^x (s)", "Upper period T_upper = Cu.T_code (s)",
            "Roof displacement (m)", "Max story drift (m)", "Max drift ratio"
        ],
        "Value": [
            total_weight_kn, base_shear_kn, periods[0], inp.target_period_s, T_code, T_upper,
            max(disp), max(drifts), max(drift_ratios)
        ]
    })

    return AnalysisResult(periods, freqs, shapes, stiffness, masses, disp, drifts, drift_ratios, elevations, lateral_forces,
                          zone_results, outr, summary_table, story_table, zone_table, outrigger_table,
                          total_weight_kn, total_mass_kg, base_shear_kn, T_code, T_upper, basement_report)


def auto_resize_for_target(inp: BuildingInput, max_iter: int = 12):
    from dataclasses import replace
    work = replace(inp)
    # deep-ish copy zone objects
    work.lower_zone = replace(inp.lower_zone)
    work.middle_zone = replace(inp.middle_zone)
    work.upper_zone = replace(inp.upper_zone)
    best = analyze_once(work)
    best_err = abs(best.periods_s[0] - work.target_period_s)
    for _ in range(max_iter):
        res = analyze_once(work)
        T1 = res.periods_s[0]
        err = abs(T1 - work.target_period_s) / max(work.target_period_s, 1e-9)
        if err < best_err:
            best = res
            best_err = err
        if err < 0.04:
            return work, res
        wall_scale = target_scale_factor(T1, work.target_period_s, 0.30, 0.85, 1.18)
        frame_scale = target_scale_factor(T1, work.target_period_s, 0.24, 0.87, 1.16)
        slab_scale = target_scale_factor(T1, work.target_period_s, 0.08, 0.95, 1.05)
        work.lower_zone = scale_zone_member(work.lower_zone, wall_scale, frame_scale, slab_scale)
        work.middle_zone = scale_zone_member(work.middle_zone, wall_scale, frame_scale, slab_scale)
        work.upper_zone = scale_zone_member(work.upper_zone, wall_scale, frame_scale, slab_scale)
    return work, best


def analyze(inp: BuildingInput) -> AnalysisResult:
    sized_inp, result = auto_resize_for_target(inp)
    inp.lower_zone = sized_inp.lower_zone
    inp.middle_zone = sized_inp.middle_zone
    inp.upper_zone = sized_inp.upper_zone
    return analyze_once(inp)


def plot_plan(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black", lw=1.5)
    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#cfcfcf", lw=0.8)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#cfcfcf", lw=0.8)
    
    # columns as points
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            ax.plot(i * inp.bay_x_m, j * inp.bay_y_m, marker='s', color='#5b5b5b', markersize=4)

    # core
    cx0 = 0.5 * (inp.plan_x_m - inp.core_outer_x_m)
    cy0 = 0.5 * (inp.plan_y_m - inp.core_outer_y_m)
    ix0 = 0.5 * (inp.plan_x_m - inp.core_opening_x_m)
    iy0 = 0.5 * (inp.plan_y_m - inp.core_opening_y_m)
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core_outer_x_m, inp.core_outer_y_m, fill=False, ec='#1f77b4', lw=2.5))
    ax.add_patch(plt.Rectangle((ix0, iy0), inp.core_opening_x_m, inp.core_opening_y_m, fill=False, ec='#1f77b4', ls='--', lw=1.2))

    # perimeter walls lower zone shown
    p = inp.lower_perimeter_walls
    t = p.thickness_m
    sx = 0.5 * (inp.plan_x_m - p.top_length_m)
    bx = 0.5 * (inp.plan_x_m - p.bottom_length_m)
    ly = 0.5 * (inp.plan_y_m - p.left_length_m)
    ry = 0.5 * (inp.plan_y_m - p.right_length_m)
    for x, y, w, h in [(sx, inp.plan_y_m - t, p.top_length_m, t), (bx, 0, p.bottom_length_m, t), (0, ly, t, p.left_length_m), (inp.plan_x_m - t, ry, t, p.right_length_m)]:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc='#8dd3c7', ec='#2c7f75', alpha=0.7))

    # automatic brace lines from core to perimeter columns/walls
    anchors = auto_brace_anchor_columns(inp)
    brace_color = '#d62728'
    for _key, (p0, p1) in anchors.items():
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=brace_color, lw=2.6)
        ax.scatter([p1[0]], [p1[1]], color=brace_color, s=24, zorder=5)

    if inp.n_basement > 0 and inp.retaining_wall.enabled:
        ax.add_patch(plt.Rectangle((-0.5, -0.5), inp.plan_x_m + 1.0, inp.plan_y_m + 1.0, fill=False, ec='#6a3d9a', lw=2, ls=':'))

    ax.set_title("Plan view\nRed X: braced bays, Green: perimeter walls, Purple dotted: retaining wall line")
    ax.set_aspect('equal')
    ax.set_xlim(-2, inp.plan_x_m + 2)
    ax.set_ylim(-2, inp.plan_y_m + 2)
    return fig


def plot_elevation(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 11))
    left, right = 0.0, inp.plan_x_m
    # columns
    ax.plot([left, left], [0, inp.n_story * inp.story_height_m], color='black', lw=2)
    ax.plot([right, right], [0, inp.n_story * inp.story_height_m], color='black', lw=2)
    for _, row in result.story_table.iterrows():
        y = (row['Story'] - 1) * inp.story_height_m
        beam_h = row['Beam depth (m)']
        ax.plot([left, right], [y, y], color='#1f78b4', lw=1.0 + 2.2 * beam_h / max(result.story_table['Beam depth (m)']))
    for o in result.outriggers:
        y = (o.story_level - 1) * inp.story_height_m
        ax.plot([left, right], [y, y], color='#d62728', lw=3.0)
        ax.plot([left, right / 2], [y, y + inp.outrigger_depth_m], color='#d62728', lw=2)
        ax.plot([right, right / 2], [y, y + inp.outrigger_depth_m], color='#d62728', lw=2)
    if inp.n_basement > 0 and inp.retaining_wall.enabled:
        ax.add_patch(plt.Rectangle((left - 0.6, -inp.n_basement * inp.basement_height_m), (right - left) + 1.2, inp.n_basement * inp.basement_height_m, fill=False, ec='#6a3d9a', lw=2))
    ax.set_title("Elevation view\nBlue: beams by story, Red: outrigger levels, Purple: retaining wall extent")
    ax.set_xlabel("Building width (schematic)")
    ax.set_ylabel("Elevation (m)")
    ax.set_xlim(-2, right + 2)
    ax.set_ylim(-inp.n_basement * inp.basement_height_m - 2, inp.n_story * inp.story_height_m + 2)
    ax.grid(True, alpha=0.2)
    return fig


def plot_mode_shape(result: AnalysisResult, mode_index: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(1, len(result.mode_shapes[mode_index]) + 1)
    ax.plot(result.mode_shapes[mode_index], y, marker='o')
    ax.set_xlabel('Normalized displacement')
    ax.set_ylabel('Story')
    ax.set_title(f"Mode {mode_index + 1} shape | T = {result.periods_s[mode_index]:.3f} s")
    ax.grid(True, alpha=0.3)
    return fig


def build_report(result: AnalysisResult) -> str:
    lines = []
    lines.append('=' * 90)
    lines.append(f"{APP_TITLE} - {APP_VERSION}")
    lines.append('=' * 90)
    lines.append('')
    lines.append('GLOBAL RESPONSE')
    lines.append('-' * 90)
    lines.append(f"T1 from eigen analysis                = {result.periods_s[0]:.3f} s")
    lines.append(f"Code period T_code = Ct*H^x          = {result.T_code_s:.3f} s")
    lines.append(f"Upper code period T_upper = Cu*Tcode = {result.T_upper_s:.3f} s")
    lines.append(f"Total weight                           = {result.total_weight_kn:,.1f} kN")
    lines.append(f"Base shear                             = {result.base_shear_kn:,.1f} kN")
    lines.append(f"Roof displacement                      = {max(result.floor_displacements_m):.4f} m")
    lines.append(f"Max story drift                        = {max(result.story_drifts_m):.4f} m")
    lines.append(f"Max story drift ratio                  = {max(result.story_drift_ratios):.6f}")
    lines.append('')
    lines.append('BASEMENT')
    lines.append('-' * 90)
    lines.append(result.basement_wall_report)
    lines.append('')
    lines.append('OUTRIGGERS')
    lines.append('-' * 90)
    if result.outriggers:
        for o in result.outriggers:
            lines.append(f"Story {o.story_level}: L={o.brace_length_m:.2f} m | KL/r={o.slenderness:.1f} | added K={o.effective_story_stiffness_n_m:.3e} N/m")
    else:
        lines.append('No outriggers defined.')
    return '\n'.join(lines)


def zone_block(label: str, defaults: ZoneMemberInput) -> ZoneMemberInput:
    st.sidebar.markdown(f"### {label} zone members")
    wall_t = float(st.sidebar.number_input(f"{label} wall thickness (m)", 0.15, 2.00, defaults.wall_thickness_m, 0.05, format="%.2f"))
    beam_b = float(st.sidebar.number_input(f"{label} beam width (m)", 0.20, 2.00, defaults.beam_width_m, 0.05, format="%.2f"))
    beam_h = float(st.sidebar.number_input(f"{label} beam depth (m)", 0.30, 3.00, defaults.beam_depth_m, 0.05, format="%.2f"))
    slab_t = float(st.sidebar.number_input(f"{label} slab thickness (m)", 0.10, 0.80, defaults.slab_thickness_m, 0.01, format="%.2f"))
    cxx = float(st.sidebar.number_input(f"{label} corner col X (m)", 0.30, 3.00, defaults.corner_col_x_m, 0.05, format="%.2f"))
    cxy = float(st.sidebar.number_input(f"{label} corner col Y (m)", 0.30, 3.00, defaults.corner_col_y_m, 0.05, format="%.2f"))
    pxx = float(st.sidebar.number_input(f"{label} perimeter col X (m)", 0.30, 3.00, defaults.perimeter_col_x_m, 0.05, format="%.2f"))
    pxy = float(st.sidebar.number_input(f"{label} perimeter col Y (m)", 0.30, 3.00, defaults.perimeter_col_y_m, 0.05, format="%.2f"))
    ixx = float(st.sidebar.number_input(f"{label} interior col X (m)", 0.30, 3.00, defaults.interior_col_x_m, 0.05, format="%.2f"))
    ixy = float(st.sidebar.number_input(f"{label} interior col Y (m)", 0.30, 3.00, defaults.interior_col_y_m, 0.05, format="%.2f"))
    wc = int(st.sidebar.number_input(f"{label} core wall count", 4, 8, defaults.wall_count, 1))
    return ZoneMemberInput(wall_t, beam_b, beam_h, slab_t, cxx, cxy, pxx, pxy, ixx, ixy, wc)


def perimeter_block(label: str, defaults: PerimeterWallInput, plan_x: float, plan_y: float) -> PerimeterWallInput:
    st.sidebar.markdown(f"### {label} perimeter walls")
    t = float(st.sidebar.number_input(f"{label} perimeter wall thickness (m)", 0.0, 1.50, defaults.thickness_m, 0.01, format="%.2f"))
    top = float(st.sidebar.number_input(f"{label} top wall length (m)", 0.0, plan_x, defaults.top_length_m, 0.5, format="%.1f"))
    bottom = float(st.sidebar.number_input(f"{label} bottom wall length (m)", 0.0, plan_x, defaults.bottom_length_m, 0.5, format="%.1f"))
    left = float(st.sidebar.number_input(f"{label} left wall length (m)", 0.0, plan_y, defaults.left_length_m, 0.5, format="%.1f"))
    right = float(st.sidebar.number_input(f"{label} right wall length (m)", 0.0, plan_y, defaults.right_length_m, 0.5, format="%.1f"))
    return PerimeterWallInput(t, top, bottom, left, right)


def parse_int_list(text: str, max_bay: int) -> List[int]:
    if not text.strip():
        return []
    vals = []
    for part in text.split(','):
        try:
            v = int(part.strip())
            if 1 <= v <= max_bay:
                vals.append(v)
        except Exception:
            pass
    return sorted(list(set(vals)))


def streamlit_input_panel() -> BuildingInput:
    d = BuildingInput()
    st.sidebar.header('Input Data')
    st.sidebar.subheader('Geometry')
    n_story = int(st.sidebar.number_input('Above-grade stories', 1, 120, d.n_story, 1))
    n_basement = int(st.sidebar.number_input('Basement stories', 0, 20, d.n_basement, 1))
    story_height_m = float(st.sidebar.number_input('Story height (m)', 2.5, 6.0, d.story_height_m, 0.1, format='%.2f'))
    basement_height_m = float(st.sidebar.number_input('Basement height (m)', 2.5, 6.0, d.basement_height_m, 0.1, format='%.2f'))
    plan_x_m = float(st.sidebar.number_input('Plan X (m)', 10.0, 300.0, d.plan_x_m, 1.0, format='%.2f'))
    plan_y_m = float(st.sidebar.number_input('Plan Y (m)', 10.0, 300.0, d.plan_y_m, 1.0, format='%.2f'))
    n_bays_x = int(st.sidebar.number_input('Bays in X', 1, 30, d.n_bays_x, 1))
    n_bays_y = int(st.sidebar.number_input('Bays in Y', 1, 30, d.n_bays_y, 1))
    bay_x_m = float(st.sidebar.number_input('Bay X (m)', 2.0, 20.0, d.bay_x_m, 0.5, format='%.2f'))
    bay_y_m = float(st.sidebar.number_input('Bay Y (m)', 2.0, 20.0, d.bay_y_m, 0.5, format='%.2f'))

    st.sidebar.subheader('Core')
    core_outer_x_m = float(st.sidebar.number_input('Core outer X (m)', 4.0, 100.0, d.core_outer_x_m, 0.5, format='%.2f'))
    core_outer_y_m = float(st.sidebar.number_input('Core outer Y (m)', 4.0, 100.0, d.core_outer_y_m, 0.5, format='%.2f'))
    core_opening_x_m = float(st.sidebar.number_input('Core opening X (m)', 2.0, 100.0, d.core_opening_x_m, 0.5, format='%.2f'))
    core_opening_y_m = float(st.sidebar.number_input('Core opening Y (m)', 2.0, 100.0, d.core_opening_y_m, 0.5, format='%.2f'))

    st.sidebar.subheader('Material / loads')
    Ec_mpa = float(st.sidebar.number_input('Concrete E (MPa)', 15000.0, 50000.0, d.Ec_mpa, 500.0, format='%.0f'))
    Es_mpa = float(st.sidebar.number_input('Steel E (MPa)', 150000.0, 220000.0, d.Es_mpa, 1000.0, format='%.0f'))
    dl_kn_m2 = float(st.sidebar.number_input('DL (kN/m²)', 0.0, 20.0, d.dl_kn_m2, 0.1, format='%.2f'))
    ll_kn_m2 = float(st.sidebar.number_input('LL (kN/m²)', 0.0, 20.0, d.ll_kn_m2, 0.1, format='%.2f'))
    superimposed_dead_kn_m2 = float(st.sidebar.number_input('Superimposed DL (kN/m²)', 0.0, 20.0, d.superimposed_dead_kn_m2, 0.1, format='%.2f'))
    facade_line_load_kn_m = float(st.sidebar.number_input('Facade line load (kN/m)', 0.0, 20.0, d.facade_line_load_kn_m, 0.1, format='%.2f'))
    live_load_mass_factor = float(st.sidebar.number_input('Live load mass factor', 0.0, 1.0, d.live_load_mass_factor, 0.05, format='%.2f'))
    seismic_base_shear_coeff = float(st.sidebar.number_input('Base shear coefficient', 0.01, 0.50, d.seismic_base_shear_coeff, 0.005, format='%.3f'))
    drift_limit_ratio = float(st.sidebar.number_input('Drift limit ratio', 0.0005, 0.01, d.drift_limit_ratio, 0.0005, format='%.4f'))

    st.sidebar.subheader('Cracked factors')
    wall_cracked_factor = float(st.sidebar.number_input('Wall cracked factor', 0.10, 1.00, d.wall_cracked_factor, 0.05, format='%.2f'))
    column_cracked_factor = float(st.sidebar.number_input('Column cracked factor', 0.10, 1.00, d.column_cracked_factor, 0.05, format='%.2f'))
    beam_cracked_factor = float(st.sidebar.number_input('Beam cracked factor', 0.10, 1.00, d.beam_cracked_factor, 0.05, format='%.2f'))
    slab_cracked_factor = float(st.sidebar.number_input('Slab cracked factor', 0.10, 1.00, d.slab_cracked_factor, 0.05, format='%.2f'))
    retaining_wall_cracked_factor = float(st.sidebar.number_input('Retaining wall cracked factor', 0.10, 1.00, d.retaining_wall_cracked_factor, 0.05, format='%.2f'))

    lower_zone = zone_block('Lower', d.lower_zone)
    middle_zone = zone_block('Middle', d.middle_zone)
    upper_zone = zone_block('Upper', d.upper_zone)

    lower_perimeter_walls = perimeter_block('Lower', d.lower_perimeter_walls, plan_x_m, plan_y_m)
    middle_perimeter_walls = perimeter_block('Middle', d.middle_perimeter_walls, plan_x_m, plan_y_m)
    upper_perimeter_walls = perimeter_block('Upper', d.upper_perimeter_walls, plan_x_m, plan_y_m)

    st.sidebar.subheader('Basement retaining wall')
    rw_enabled = st.sidebar.checkbox('Retaining wall enabled', value=d.retaining_wall.enabled)
    rw_t = float(st.sidebar.number_input('Retaining wall thickness (m)', 0.0, 1.5, d.retaining_wall.thickness_m, 0.01, format='%.2f'))

    st.sidebar.subheader('Outriggers / CHS braces')
    outrigger_count = int(st.sidebar.selectbox('Number of outriggers', [0,1,2,3], index=d.outrigger_count))
    levels = []
    for i in range(outrigger_count):
        default_level = d.outrigger_story_levels[i] if i < len(d.outrigger_story_levels) else max(1, int((i + 1) * n_story / (outrigger_count + 1)))
        levels.append(int(st.sidebar.number_input(f'Outrigger story level {i+1}', 1, n_story, default_level, 1)))
    brace_outer_diameter_mm = float(st.sidebar.number_input('Brace CHS outer diameter (mm)', 76.0, 1200.0, d.brace_outer_diameter_mm, 10.0, format='%.1f'))
    brace_thickness_mm = float(st.sidebar.number_input('Brace CHS thickness (mm)', 4.0, 80.0, d.brace_thickness_mm, 1.0, format='%.1f'))
    braces_per_side = int(st.sidebar.number_input('Braces per side', 1, 6, d.braces_per_side, 1))
    outrigger_depth_m = float(st.sidebar.number_input('Outrigger depth (m)', 0.5, 8.0, d.outrigger_depth_m, 0.1, format='%.2f'))
    brace_effective_length_factor = float(st.sidebar.number_input('Brace K factor', 0.5, 2.0, d.brace_effective_length_factor, 0.05, format='%.2f'))
    brace_buckling_reduction = float(st.sidebar.number_input('Brace buckling reduction', 0.10, 1.00, d.brace_buckling_reduction, 0.05, format='%.2f'))

    target_period_s = float(st.sidebar.number_input('Target first-mode period for auto-sizing (s)', 0.50, 15.00, d.target_period_s, 0.10, format='%.2f'))

    return BuildingInput(
        n_story=n_story, n_basement=n_basement, story_height_m=story_height_m, basement_height_m=basement_height_m,
        plan_x_m=plan_x_m, plan_y_m=plan_y_m, n_bays_x=n_bays_x, n_bays_y=n_bays_y, bay_x_m=bay_x_m, bay_y_m=bay_y_m,
        core_outer_x_m=core_outer_x_m, core_outer_y_m=core_outer_y_m, core_opening_x_m=core_opening_x_m, core_opening_y_m=core_opening_y_m,
        Ec_mpa=Ec_mpa, Es_mpa=Es_mpa, dl_kn_m2=dl_kn_m2, ll_kn_m2=ll_kn_m2, superimposed_dead_kn_m2=superimposed_dead_kn_m2,
        facade_line_load_kn_m=facade_line_load_kn_m, live_load_mass_factor=live_load_mass_factor,
        seismic_base_shear_coeff=seismic_base_shear_coeff, drift_limit_ratio=drift_limit_ratio,
        wall_cracked_factor=wall_cracked_factor, column_cracked_factor=column_cracked_factor,
        beam_cracked_factor=beam_cracked_factor, slab_cracked_factor=slab_cracked_factor,
        retaining_wall_cracked_factor=retaining_wall_cracked_factor,
        lower_zone=lower_zone, middle_zone=middle_zone, upper_zone=upper_zone,
        lower_perimeter_walls=lower_perimeter_walls, middle_perimeter_walls=middle_perimeter_walls, upper_perimeter_walls=upper_perimeter_walls,
        retaining_wall=RetainingWallInput(enabled=rw_enabled, thickness_m=rw_t),
        outrigger_count=outrigger_count, outrigger_story_levels=levels,
        brace_outer_diameter_mm=brace_outer_diameter_mm, brace_thickness_mm=brace_thickness_mm,
        braces_per_side=braces_per_side, outrigger_depth_m=outrigger_depth_m,
        brace_effective_length_factor=brace_effective_length_factor, brace_buckling_reduction=brace_buckling_reduction,
        target_period_s=target_period_s,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info('This version rebuilds stiffness assembly to avoid unrealistically low periods. T1 is always an analysis output, not a sizing target.')
    inp = streamlit_input_panel()
    if st.button('Analyze', use_container_width=True):
        st.session_state['result'] = analyze(inp)
        st.session_state['report'] = build_report(st.session_state['result'])
    result: AnalysisResult | None = st.session_state.get('result')
    if result is None:
        st.warning('Enter data and click Analyze.')
        return
    tabs = st.tabs(['Summary', 'Stories', 'Zones', 'Outriggers', 'Plan', 'Elevation', 'Modes', 'Report'])
    with tabs[0]:
        st.dataframe(result.summary_table, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('T1 (s)', f'{result.periods_s[0]:.3f}')
        c2.metric('Base shear (kN)', f'{result.base_shear_kn:,.0f}')
        c3.metric('Roof disp. (m)', f'{max(result.floor_displacements_m):.3f}')
        c4.metric('Max drift ratio', f'{max(result.story_drift_ratios):.5f}')
        st.write(result.basement_wall_report)
    with tabs[1]:
        st.dataframe(result.story_table, use_container_width=True, height=650)
    with tabs[2]:
        st.dataframe(result.zone_table, use_container_width=True)
    with tabs[3]:
        st.dataframe(result.outrigger_table, use_container_width=True)
    with tabs[4]:
        st.pyplot(plot_plan(inp, result))
    with tabs[5]:
        st.pyplot(plot_elevation(inp, result))
    with tabs[6]:
        mode_count = min(5, len(result.mode_shapes))
        mode_index = st.selectbox('Mode', list(range(1, mode_count + 1)), index=0) - 1
        st.pyplot(plot_mode_shape(result, mode_index))
    with tabs[7]:
        report = st.session_state.get('report', '')
        st.text_area('Text report', report, height=500)
        st.download_button('Download report as TXT', report, 'tall_building_report.txt', 'text/plain')


if __name__ == '__main__':
    main()
