
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Tall Building Rational Analysis + Braced Outrigger"
APP_VERSION = "v8.1-layout-basement"

G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3
STEEL_DENSITY = 7850.0  # kg/m3


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
    thickness_m: float = 0.40
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
class BracedBayLayout:
    top_x_bays: List[int] = field(default_factory=lambda: [2, 5])
    bottom_x_bays: List[int] = field(default_factory=lambda: [2, 5])
    left_y_bays: List[int] = field(default_factory=lambda: [2, 5])
    right_y_bays: List[int] = field(default_factory=lambda: [2, 5])


@dataclass
class BuildingInput:
    # geometry
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

    # core
    core_outer_x_m: float = 18.0
    core_outer_y_m: float = 15.0
    core_opening_x_m: float = 12.0
    core_opening_y_m: float = 10.5

    # materials
    Ec_mpa: float = 34000.0
    Es_mpa: float = 200000.0
    nu_concrete: float = 0.20
    fy_mpa: float = 420.0

    # loads
    dl_kn_m2: float = 3.0
    ll_kn_m2: float = 2.0
    superimposed_dead_kn_m2: float = 1.5
    facade_line_load_kn_m: float = 1.0
    live_load_mass_factor: float = 0.30
    seismic_base_shear_coeff: float = 0.08
    drift_limit_ratio: float = 1.0 / 500.0

    # cracked factors
    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.70
    beam_cracked_factor: float = 0.35
    slab_cracked_factor: float = 0.25
    retaining_wall_cracked_factor: float = 0.50

    # report-only code period parameters
    Ct: float = 0.0488
    x_period: float = 0.75
    Cu: float = 1.4

    # zoning/member sizes
    lower_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(
        wall_thickness_m=0.60, beam_width_m=0.45, beam_depth_m=0.85, slab_thickness_m=0.24,
        corner_col_x_m=1.20, corner_col_y_m=1.20, perimeter_col_x_m=1.00, perimeter_col_y_m=1.00,
        interior_col_x_m=0.90, interior_col_y_m=0.90, wall_count=8
    ))
    middle_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(
        wall_thickness_m=0.45, beam_width_m=0.40, beam_depth_m=0.75, slab_thickness_m=0.22,
        corner_col_x_m=1.00, corner_col_y_m=1.00, perimeter_col_x_m=0.85, perimeter_col_y_m=0.85,
        interior_col_x_m=0.75, interior_col_y_m=0.75, wall_count=6
    ))
    upper_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(
        wall_thickness_m=0.35, beam_width_m=0.35, beam_depth_m=0.65, slab_thickness_m=0.20,
        corner_col_x_m=0.85, corner_col_y_m=0.85, perimeter_col_x_m=0.75, perimeter_col_y_m=0.75,
        interior_col_x_m=0.65, interior_col_y_m=0.65, wall_count=4
    ))

    lower_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.30, 24.0, 24.0, 21.0, 21.0))
    middle_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.25, 12.0, 12.0, 10.5, 10.5))
    upper_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.20, 8.0, 8.0, 7.0, 7.0))
    retaining_wall: RetainingWallInput = field(default_factory=RetainingWallInput)

    # service/core opening checks
    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each_m2: float = 3.5
    stair_area_each_m2: float = 20.0
    service_area_m2: float = 35.0
    corridor_factor: float = 1.35

    # outriggers
    outrigger_count: int = 2
    outrigger_story_levels: List[int] = field(default_factory=lambda: [28, 42])
    brace_outer_diameter_mm: float = 355.6
    brace_thickness_mm: float = 16.0
    braces_per_side: int = 2
    outrigger_depth_m: float = 3.0
    brace_effective_length_factor: float = 1.0
    brace_buckling_reduction: float = 0.85
    braced_layout: BracedBayLayout = field(default_factory=BracedBayLayout)


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
    k_core_n_m: float
    k_perimeter_wall_n_m: float
    k_column_n_m: float
    k_beam_n_m: float
    k_diaphragm_n_m: float
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


# ----------------------------- HELPERS -----------------------------

def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def zone_input(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    return {
        "Lower Zone": inp.lower_zone,
        "Middle Zone": inp.middle_zone,
        "Upper Zone": inp.upper_zone,
    }[zone_name]


def perimeter_input(inp: BuildingInput, zone_name: str) -> PerimeterWallInput:
    return {
        "Lower Zone": inp.lower_perimeter_walls,
        "Middle Zone": inp.middle_perimeter_walls,
        "Upper Zone": inp.upper_perimeter_walls,
    }[zone_name]


def required_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each_m2
        + inp.stair_count * inp.stair_area_each_m2
        + inp.service_area_m2
    ) * inp.corridor_factor


def validate_input(inp: BuildingInput) -> Tuple[bool, str]:
    if inp.core_opening_x_m >= inp.core_outer_x_m or inp.core_opening_y_m >= inp.core_outer_y_m:
        return False, "Core opening must be smaller than core outer dimensions."
    if required_opening_area(inp) > inp.core_opening_x_m * inp.core_opening_y_m:
        return False, "Core opening area is smaller than required stair/elevator/service area."
    if abs(inp.n_bays_x * inp.bay_x_m - inp.plan_x_m) > 1e-6:
        return False, "Plan X must equal n_bays_x × bay_x."
    if abs(inp.n_bays_y * inp.bay_y_m - inp.plan_y_m) > 1e-6:
        return False, "Plan Y must equal n_bays_y × bay_y."
    return True, ""


def gross_floor_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m


def rect_ix(b: float, h: float) -> float:
    return b * h**3 / 12.0


def rect_iy(b: float, h: float) -> float:
    return h * b**3 / 12.0


def circular_hollow_area(do_m: float, t_m: float) -> float:
    di = max(do_m - 2 * t_m, 1e-9)
    return pi * (do_m**2 - di**2) / 4.0


def circular_hollow_inertia(do_m: float, t_m: float) -> float:
    di = max(do_m - 2 * t_m, 1e-9)
    return pi * (do_m**4 - di**4) / 64.0


def circular_hollow_rg(do_m: float, t_m: float) -> float:
    a = circular_hollow_area(do_m, t_m)
    i = circular_hollow_inertia(do_m, t_m)
    return sqrt(i / max(a, 1e-12))


def concrete_shear_modulus(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    return E / (2.0 * (1.0 + inp.nu_concrete))


def wall_lengths_for_core(inp: BuildingInput, wall_count: int) -> List[float]:
    ox = inp.core_outer_x_m
    oy = inp.core_outer_y_m
    if wall_count == 4:
        return [ox, ox, oy, oy]
    if wall_count == 6:
        return [ox, ox, oy, oy, 0.45 * ox, 0.45 * ox]
    return [ox, ox, oy, oy, 0.45 * ox, 0.45 * ox, 0.45 * oy, 0.45 * oy]


def core_min_inertia(inp: BuildingInput, t: float, wall_count: int) -> float:
    lengths = wall_lengths_for_core(inp, wall_count)
    ox = inp.core_outer_x_m
    oy = inp.core_outer_y_m
    x_side = ox / 2.0
    y_side = oy / 2.0
    Ix = 0.0
    Iy = 0.0

    for L in lengths[0:2]:
        Ix += rect_ix(L, t) + L * t * y_side**2
        Iy += rect_iy(L, t)
    for L in lengths[2:4]:
        Iy += rect_iy(t, L) + L * t * x_side**2
        Ix += rect_ix(t, L)
    if wall_count >= 6:
        inner_x = 0.22 * ox
        for sgn, L in zip([-1, 1], lengths[4:6]):
            off = sgn * inner_x
            Iy += rect_iy(t, L) + L * t * off**2
            Ix += rect_ix(t, L)
    if wall_count >= 8:
        inner_y = 0.22 * oy
        for sgn, L in zip([-1, 1], lengths[6:8]):
            off = sgn * inner_y
            Ix += rect_ix(L, t) + L * t * off**2
            Iy += rect_iy(L, t)

    return min(Ix, Iy)


def total_cols(inp: BuildingInput) -> Tuple[int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return corner, perimeter, interior


def perimeter_wall_story_stiffness(inp: BuildingInput, p: PerimeterWallInput) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus(inp)
    h = inp.story_height_m
    t = p.thickness_m
    lengths = [p.top_length_m, p.bottom_length_m, p.left_length_m, p.right_length_m]
    if t <= 0 or max(lengths) <= 0:
        return 0.0

    Ix = rect_ix(p.top_length_m, t) + p.top_length_m * t * (inp.plan_y_m / 2.0) ** 2
    Ix += rect_ix(p.bottom_length_m, t) + p.bottom_length_m * t * (inp.plan_y_m / 2.0) ** 2
    Iy = rect_iy(t, p.left_length_m) + p.left_length_m * t * (inp.plan_x_m / 2.0) ** 2
    Iy += rect_iy(t, p.right_length_m) + p.right_length_m * t * (inp.plan_x_m / 2.0) ** 2
    Ieff = inp.wall_cracked_factor * min(Ix, Iy)

    kb = 12 * E * Ieff / max(h**3, 1e-12)
    area = t * sum(lengths)
    ks = 1.2 * Gc * area / max(h, 1e-12)
    return 1.0 / max(1.0 / max(kb, 1e-12) + 1.0 / max(ks, 1e-12), 1e-18)


def core_story_stiffness(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus(inp)
    h = inp.story_height_m
    Ieff = inp.wall_cracked_factor * core_min_inertia(inp, z.wall_thickness_m, z.wall_count)
    kb = 12 * E * Ieff / max(h**3, 1e-12)
    area = sum(L * z.wall_thickness_m for L in wall_lengths_for_core(inp, z.wall_count))
    ks = 1.2 * Gc * area / max(h, 1e-12)
    return 1.0 / max(1.0 / max(kb, 1e-12) + 1.0 / max(ks, 1e-12), 1e-18)


def column_story_stiffness(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    corner, perimeter, interior = total_cols(inp)
    I_corner = min(rect_ix(z.corner_col_x_m, z.corner_col_y_m), rect_iy(z.corner_col_x_m, z.corner_col_y_m))
    I_peri = min(rect_ix(z.perimeter_col_x_m, z.perimeter_col_y_m), rect_iy(z.perimeter_col_x_m, z.perimeter_col_y_m))
    I_inter = min(rect_ix(z.interior_col_x_m, z.interior_col_y_m), rect_iy(z.interior_col_x_m, z.interior_col_y_m))
    Itot = inp.column_cracked_factor * (corner * I_corner + perimeter * I_peri + interior * I_inter)
    return 12 * E * Itot / max(h**3, 1e-12)


def beam_story_stiffness(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    Ib = inp.beam_cracked_factor * rect_ix(z.beam_width_m, z.beam_depth_m)
    n_beams_x = inp.n_bays_x * (inp.n_bays_y + 1)
    n_beams_y = inp.n_bays_y * (inp.n_bays_x + 1)
    kx = n_beams_x * 12 * E * Ib / max(inp.bay_x_m**3, 1e-12)
    ky = n_beams_y * 12 * E * Ib / max(inp.bay_y_m**3, 1e-12)
    # beam contribution reduced because floor beams mainly provide frame restraint, not direct shear-wall action
    return 0.35 * (kx + ky)


def diaphragm_stiffness(inp: BuildingInput, slab_t: float) -> float:
    E = inp.Ec_mpa * 1e6
    A = inp.slab_cracked_factor * slab_t * gross_floor_area(inp)
    L = max(inp.plan_x_m, inp.plan_y_m)
    return E * A / max(L, 1e-12)


def floor_weight_kn(inp: BuildingInput, z: ZoneMemberInput) -> float:
    area = gross_floor_area(inp)
    slab_self = z.slab_thickness_m * CONCRETE_UNIT_WEIGHT
    total_beam_len = inp.n_bays_x * (inp.n_bays_y + 1) * inp.bay_x_m + inp.n_bays_y * (inp.n_bays_x + 1) * inp.bay_y_m
    beam_self = z.beam_width_m * z.beam_depth_m * CONCRETE_UNIT_WEIGHT * total_beam_len / max(area, 1e-12)
    facade = inp.facade_line_load_kn_m * 2 * (inp.plan_x_m + inp.plan_y_m) / max(area, 1e-12)
    q = inp.dl_kn_m2 + inp.superimposed_dead_kn_m2 + slab_self + beam_self + inp.live_load_mass_factor * inp.ll_kn_m2 + facade
    return area * q


def build_zone_results(inp: BuildingInput) -> List[ZoneStiffnessResult]:
    results = []
    for zone in define_three_zones(inp.n_story):
        z = zone_input(inp, zone.name)
        p = perimeter_input(inp, zone.name)
        k_core = core_story_stiffness(inp, z)
        k_per = perimeter_wall_story_stiffness(inp, p)
        k_col = column_story_stiffness(inp, z)
        k_beam = beam_story_stiffness(inp, z)
        k_d = diaphragm_stiffness(inp, z.slab_thickness_m)
        k_total = k_core + k_per + k_col + k_beam
        results.append(ZoneStiffnessResult(
            zone_name=zone.name, story_start=zone.story_start, story_end=zone.story_end,
            wall_count=z.wall_count, wall_t_m=z.wall_thickness_m,
            beam_b_m=z.beam_width_m, beam_h_m=z.beam_depth_m, slab_t_m=z.slab_thickness_m,
            perimeter_wall_t_m=p.thickness_m,
            k_core_n_m=k_core, k_perimeter_wall_n_m=k_per, k_column_n_m=k_col,
            k_beam_n_m=k_beam, k_diaphragm_n_m=k_d, k_total_story_n_m=k_total
        ))
    return results


def brace_arm_length(inp: BuildingInput) -> float:
    return 0.5 * max(inp.plan_x_m - inp.core_outer_x_m, inp.plan_y_m - inp.core_outer_y_m)


def outrigger_results(inp: BuildingInput, zone_results: List[ZoneStiffnessResult]) -> List[OutriggerResult]:
    if inp.outrigger_count <= 0:
        return []
    levels = [lv for lv in inp.outrigger_story_levels[:inp.outrigger_count] if 1 <= lv <= inp.n_story]
    if not levels:
        return []

    do = inp.brace_outer_diameter_mm / 1000.0
    t = inp.brace_thickness_mm / 1000.0
    area = circular_hollow_area(do, t)
    rg = circular_hollow_rg(do, t)
    E = inp.Es_mpa * 1e6
    arm = brace_arm_length(inp)
    depth = inp.outrigger_depth_m

    # diaphragm stiffness at corresponding story limits transfer
    k_d_map = {}
    for zr in zone_results:
        for s in range(zr.story_start, zr.story_end + 1):
            k_d_map[s] = zr.k_diaphragm_n_m

    out = []
    for lv in levels:
        Lb = sqrt(arm**2 + depth**2)
        axial_k = E * area / max(Lb, 1e-12)
        slenderness = inp.brace_effective_length_factor * Lb / max(rg, 1e-12)
        k_diag = axial_k * (depth / max(Lb, 1e-12)) ** 2
        k_story = 4.0 * inp.braces_per_side * inp.brace_buckling_reduction * k_diag
        k_eff = 1.0 / max(1.0 / max(k_story, 1e-12) + 1.0 / max(k_d_map.get(lv, 1e12), 1e-12), 1e-18)
        steel_weight = 4.0 * inp.braces_per_side * Lb * area * STEEL_DENSITY
        out.append(OutriggerResult(
            story_level=lv,
            brace_length_m=Lb,
            brace_area_m2=area,
            brace_radius_gyration_m=rg,
            slenderness=slenderness,
            axial_stiffness_n=axial_k,
            effective_story_stiffness_n_m=k_eff,
            steel_weight_kg=steel_weight,
        ))
    return out


def build_story_vectors(inp: BuildingInput, zone_results: List[ZoneStiffnessResult], outriggers: List[OutriggerResult]) -> Tuple[List[float], List[float], List[float]]:
    k_by_story: Dict[int, float] = {}
    m_by_story: Dict[int, float] = {}
    elev = []
    for zr in zone_results:
        z = zone_input(inp, zr.zone_name)
        fw = floor_weight_kn(inp, z)
        for s in range(zr.story_start, zr.story_end + 1):
            k_by_story[s] = zr.k_total_story_n_m
            m_by_story[s] = fw * 1000.0 / G
            elev.append(s * inp.story_height_m)

    for outr in outriggers:
        k_by_story[outr.story_level] = k_by_story.get(outr.story_level, 0.0) + outr.effective_story_stiffness_n_m
        m_by_story[outr.story_level] = m_by_story.get(outr.story_level, 0.0) + outr.steel_weight_kg

    stiffness = [k_by_story[i] for i in range(1, inp.n_story + 1)]
    masses = [m_by_story[i] for i in range(1, inp.n_story + 1)]
    elevations = [i * inp.story_height_m for i in range(1, inp.n_story + 1)]
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
    pos = vals > 1e-12
    vals = vals[pos]
    vecs = vecs[:, pos]
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


def solve_static_response(stiffnesses: List[float], lateral_forces_n: List[float], story_height: float) -> Tuple[List[float], List[float], List[float]]:
    _, K = assemble_m_k([1.0] * len(stiffnesses), stiffnesses)
    u = np.linalg.solve(K, np.array(lateral_forces_n))
    disp = u.tolist()
    drifts = [disp[0]] + [disp[i] - disp[i - 1] for i in range(1, len(disp))]
    drift_ratios = [d / story_height for d in drifts]
    return disp, drifts, drift_ratios


def equivalent_lateral_force_distribution(weights_kn: List[float], elevations_m: List[float], base_shear_kn: float) -> List[float]:
    wxh = [w * h for w, h in zip(weights_kn, elevations_m)]
    denom = max(sum(wxh), 1e-12)
    return [base_shear_kn * x / denom * 1000.0 for x in wxh]


def retaining_wall_support_stiffness(inp: BuildingInput) -> Tuple[float, str]:
    rw = inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m)
    if not rw.enabled or inp.n_basement <= 0:
        return 0.0, "Basement retaining wall not included."
    E = inp.Ec_mpa * 1e6
    h = inp.basement_height_m
    t = rw.thickness_m
    lengths = [rw.top_length_m, rw.bottom_length_m, rw.left_length_m, rw.right_length_m]
    I = min(
        rect_ix(rw.top_length_m, t) + rw.top_length_m * t * (inp.plan_y_m / 2.0) ** 2
        + rect_ix(rw.bottom_length_m, t) + rw.bottom_length_m * t * (inp.plan_y_m / 2.0) ** 2,
        rect_iy(t, rw.left_length_m) + rw.left_length_m * t * (inp.plan_x_m / 2.0) ** 2
        + rect_iy(t, rw.right_length_m) + rw.right_length_m * t * (inp.plan_x_m / 2.0) ** 2
    )
    k = 12 * E * inp.retaining_wall_cracked_factor * I / max((inp.n_basement * h) ** 3, 1e-12)
    desc = (
        f"Basement retaining walls included as base support spring. "
        f"t={t:.2f} m, active lengths(top/bottom/left/right)="
        f"{rw.top_length_m:.1f}/{rw.bottom_length_m:.1f}/{rw.left_length_m:.1f}/{rw.right_length_m:.1f} m."
    )
    return k, desc


def analyze(inp: BuildingInput) -> AnalysisResult:
    ok, msg = validate_input(inp)
    if not ok:
        raise ValueError(msg)

    zones = build_zone_results(inp)
    outr = outrigger_results(inp, zones)
    stiffness, masses, elevations = build_story_vectors(inp, zones, outr)

    # add basement retaining wall as base spring to first story stiffness
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

    zone_rows = []
    for zr in zones:
        zone_rows.append({
            "Zone": zr.zone_name,
            "Stories": f"{zr.story_start}-{zr.story_end}",
            "Wall count": zr.wall_count,
            "Core wall t (m)": zr.wall_t_m,
            "Perimeter wall t (m)": zr.perimeter_wall_t_m,
            "Beam b×h (m)": f"{zr.beam_b_m:.2f}×{zr.beam_h_m:.2f}",
            "Slab t (m)": zr.slab_t_m,
            "K core (N/m)": zr.k_core_n_m,
            "K perimeter walls (N/m)": zr.k_perimeter_wall_n_m,
            "K columns (N/m)": zr.k_column_n_m,
            "K beams (N/m)": zr.k_beam_n_m,
            "K diaphragm (N/m)": zr.k_diaphragm_n_m,
            "K total/story (N/m)": zr.k_total_story_n_m,
        })
    zone_table = pd.DataFrame(zone_rows)

    story_rows = []
    zone_by_story = {}
    for zr in zones:
        for s in range(zr.story_start, zr.story_end + 1):
            zone_by_story[s] = zr
    outr_levels = {o.story_level for o in outr}
    for i in range(inp.n_story):
        s = i + 1
        zr = zone_by_story[s]
        story_rows.append({
            "Story": s,
            "Elevation (m)": elevations[i],
            "Zone": zr.zone_name,
            "Core wall t (m)": zr.wall_t_m,
            "Perimeter wall t (m)": zr.perimeter_wall_t_m,
            "Beam width (m)": zr.beam_b_m,
            "Beam depth (m)": zr.beam_h_m,
            "Slab t (m)": zr.slab_t_m,
            "Outrigger": "Yes" if s in outr_levels else "No",
            "Story stiffness (N/m)": stiffness[i],
            "Story mass (kg)": masses[i],
            "Lateral force (N)": lateral_forces[i],
            "Floor displacement (m)": disp[i],
            "Story drift (m)": drifts[i],
            "Story drift ratio": drift_ratios[i],
        })
    story_table = pd.DataFrame(story_rows)

    outr_rows = []
    for o in outr:
        outr_rows.append({
            "Story": o.story_level,
            "Brace length (m)": o.brace_length_m,
            "Brace area (m²)": o.brace_area_m2,
            "r (m)": o.brace_radius_gyration_m,
            "KL/r": o.slenderness,
            "Axial stiffness EA/L (N)": o.axial_stiffness_n,
            "Effective story stiffness (N/m)": o.effective_story_stiffness_n_m,
            "Steel weight (kg)": o.steel_weight_kg,
        })
    outrigger_table = pd.DataFrame(outr_rows)

    summary_table = pd.DataFrame({
        "Parameter": [
            "Total weight (kN)", "Base shear (kN)", "T1 from eigen analysis (s)",
            "Code period T_code = Ct.H^x (s)", "Upper period T_upper = Cu.T_code (s)",
            "Max roof displacement (m)", "Max story drift (m)", "Max story drift ratio", "Drift limit ratio"
        ],
        "Value": [
            total_weight_kn, base_shear_kn, periods[0], T_code, T_upper,
            max(disp), max(drifts), max(drift_ratios), inp.drift_limit_ratio
        ]
    })

    return AnalysisResult(
        periods_s=periods, frequencies_hz=freqs, mode_shapes=shapes,
        story_stiffness_n_m=stiffness, story_masses_kg=masses,
        floor_displacements_m=disp, story_drifts_m=drifts, story_drift_ratios=drift_ratios,
        story_elevations_m=elevations, lateral_forces_n=lateral_forces,
        zone_results=zones, outriggers=outr,
        summary_table=summary_table, story_table=story_table, zone_table=zone_table,
        outrigger_table=outrigger_table, total_weight_kn=total_weight_kn,
        total_mass_kg=total_mass_kg, base_shear_kn=base_shear_kn,
        T_code_s=T_code, T_upper_s=T_upper, basement_wall_report=basement_report
    )


# ----------------------------- PLOTS -----------------------------

def side_wall_segments(plan_length: float, active_length: float) -> Tuple[float, float]:
    active = min(active_length, plan_length)
    start = 0.5 * (plan_length - active)
    return start, start + active


def parse_bay_list(text: str, max_bay: int) -> List[int]:
    out = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        try:
            v = int(item)
        except ValueError:
            continue
        if 1 <= v <= max_bay and v not in out:
            out.append(v)
    return out


def plot_plan(inp: BuildingInput, zone_name: str) -> plt.Figure:
    z = zone_input(inp, zone_name)
    p = perimeter_input(inp, zone_name)
    rw = inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black", linewidth=1.5)

    # grid
    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#dddddd", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#dddddd", linewidth=0.8)

    # columns
    corner, perimeter_cols, interior = total_cols(inp)
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x_m
            y = j * inp.bay_y_m
            at_edge_x = i in (0, inp.n_bays_x)
            at_edge_y = j in (0, inp.n_bays_y)
            if at_edge_x and at_edge_y:
                dx, dy, c = z.corner_col_x_m, z.corner_col_y_m, "#7f1d1d"
            elif at_edge_x or at_edge_y:
                dx, dy, c = z.perimeter_col_x_m, z.perimeter_col_y_m, "#b45309"
            else:
                dx, dy, c = z.interior_col_x_m, z.interior_col_y_m, "#1d4ed8"
            ax.add_patch(plt.Rectangle((x - dx/2, y - dy/2), dx, dy, facecolor=c, edgecolor=c, alpha=0.8))

    # core
    cx0 = 0.5 * (inp.plan_x_m - inp.core_outer_x_m)
    cy0 = 0.5 * (inp.plan_y_m - inp.core_outer_y_m)
    ix0 = 0.5 * (inp.plan_x_m - inp.core_opening_x_m)
    iy0 = 0.5 * (inp.plan_y_m - inp.core_opening_y_m)
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core_outer_x_m, inp.core_outer_y_m, fill=False, edgecolor="#15803d", linewidth=2.2))
    ax.add_patch(plt.Rectangle((ix0, iy0), inp.core_opening_x_m, inp.core_opening_y_m, fill=False, edgecolor="#64748b", linewidth=1.4, linestyle="--"))
    t = z.wall_thickness_m
    for x, y, w, h in [
        (cx0, cy0, inp.core_outer_x_m, t), (cx0, cy0 + inp.core_outer_y_m - t, inp.core_outer_x_m, t),
        (cx0, cy0, t, inp.core_outer_y_m), (cx0 + inp.core_outer_x_m - t, cy0, t, inp.core_outer_y_m)
    ]:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#22c55e", edgecolor="#15803d", alpha=0.35))

    # perimeter lateral walls
    pt0, pt1 = side_wall_segments(inp.plan_x_m, p.top_length_m)
    pb0, pb1 = side_wall_segments(inp.plan_x_m, p.bottom_length_m)
    pl0, pl1 = side_wall_segments(inp.plan_y_m, p.left_length_m)
    pr0, pr1 = side_wall_segments(inp.plan_y_m, p.right_length_m)
    ax.add_patch(plt.Rectangle((pt0, inp.plan_y_m - p.thickness_m), pt1 - pt0, p.thickness_m, facecolor="#10b981", edgecolor="#047857", alpha=0.45))
    ax.add_patch(plt.Rectangle((pb0, 0), pb1 - pb0, p.thickness_m, facecolor="#10b981", edgecolor="#047857", alpha=0.45))
    ax.add_patch(plt.Rectangle((0, pl0), p.thickness_m, pl1 - pl0, facecolor="#10b981", edgecolor="#047857", alpha=0.45))
    ax.add_patch(plt.Rectangle((inp.plan_x_m - p.thickness_m, pr0), p.thickness_m, pr1 - pr0, facecolor="#10b981", edgecolor="#047857", alpha=0.45))

    # basement retaining wall overlay
    if rw.enabled:
        rt0, rt1 = side_wall_segments(inp.plan_x_m, rw.top_length_m)
        rb0, rb1 = side_wall_segments(inp.plan_x_m, rw.bottom_length_m)
        rl0, rl1 = side_wall_segments(inp.plan_y_m, rw.left_length_m)
        rr0, rr1 = side_wall_segments(inp.plan_y_m, rw.right_length_m)
        ax.plot([rt0, rt1], [inp.plan_y_m + 0.7, inp.plan_y_m + 0.7], color="#7c3aed", linewidth=4)
        ax.plot([rb0, rb1], [-0.7, -0.7], color="#7c3aed", linewidth=4)
        ax.plot([-0.7, -0.7], [rl0, rl1], color="#7c3aed", linewidth=4)
        ax.plot([inp.plan_x_m + 0.7, inp.plan_x_m + 0.7], [rr0, rr1], color="#7c3aed", linewidth=4)

    # braced bays locations
    layout = inp.braced_layout
    brace_color = "#dc2626"
    for bay in layout.top_x_bays:
        x0 = (bay - 1) * inp.bay_x_m
        x1 = bay * inp.bay_x_m
        y0 = cy0 + inp.core_outer_y_m
        y1 = inp.plan_y_m
        ax.plot([x0, x1], [y0, y1], color=brace_color, linewidth=2.2)
        ax.plot([x0, x1], [y1, y0], color=brace_color, linewidth=2.2)
    for bay in layout.bottom_x_bays:
        x0 = (bay - 1) * inp.bay_x_m
        x1 = bay * inp.bay_x_m
        y0 = 0
        y1 = cy0
        ax.plot([x0, x1], [y0, y1], color=brace_color, linewidth=2.2)
        ax.plot([x0, x1], [y1, y0], color=brace_color, linewidth=2.2)
    for bay in layout.left_y_bays:
        y0 = (bay - 1) * inp.bay_y_m
        y1 = bay * inp.bay_y_m
        x0 = 0
        x1 = cx0
        ax.plot([x0, x1], [y0, y1], color=brace_color, linewidth=2.2)
        ax.plot([x0, x1], [y1, y0], color=brace_color, linewidth=2.2)
    for bay in layout.right_y_bays:
        y0 = (bay - 1) * inp.bay_y_m
        y1 = bay * inp.bay_y_m
        x0 = cx0 + inp.core_outer_x_m
        x1 = inp.plan_x_m
        ax.plot([x0, x1], [y0, y1], color=brace_color, linewidth=2.2)
        ax.plot([x0, x1], [y1, y0], color=brace_color, linewidth=2.2)

    ax.set_aspect("equal")
    ax.set_xlim(-2.0, inp.plan_x_m + 2.0)
    ax.set_ylim(-2.0, inp.plan_y_m + 2.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Plan view - {zone_name}\nGreen: walls, Red X: braced bays, Purple lines: basement retaining walls")
    ax.grid(False)
    return fig


def plot_elevation(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 11))
    total_height = inp.n_story * inp.story_height_m
    width = max(inp.plan_x_m, inp.plan_y_m) * 0.18

    # frame lines
    left = 0.0
    right = width
    ax.plot([left, left], [0, total_height], color="black", linewidth=1.5)
    ax.plot([right, right], [0, total_height], color="black", linewidth=1.5)

    # story beams and zone-dependent beam sizes
    for _, row in result.story_table.iterrows():
        y = row["Elevation (m)"]
        beam_h = row["Beam depth (m)"]
        line_w = 1.2 + 3.0 * beam_h / max(result.story_table["Beam depth (m)"])
        ax.plot([left, right], [y, y], color="#2563eb", linewidth=line_w)
        ax.text(right + 0.4, y, f"S{int(row['Story'])}  b×h={row['Beam width (m)']:.2f}×{beam_h:.2f}  slab={row['Slab t (m)']:.2f}", fontsize=7, va="center")

    # outriggers
    for o in result.outriggers:
        y = o.story_level * inp.story_height_m
        ax.plot([left, right], [y, y], color="#dc2626", linewidth=4)
        ax.plot([left, right], [y - inp.outrigger_depth_m / 2, y + inp.outrigger_depth_m / 2], color="#dc2626", linewidth=2)
        ax.plot([left, right], [y + inp.outrigger_depth_m / 2, y - inp.outrigger_depth_m / 2], color="#dc2626", linewidth=2)
        ax.text(left - 0.4, y, f"Outrigger @ S{o.story_level}", ha="right", va="center", color="#dc2626", fontsize=8)

    # basement retaining walls
    if inp.retaining_wall.enabled and inp.n_basement > 0:
        bas_depth = inp.n_basement * inp.basement_height_m
        ax.add_patch(plt.Rectangle((left - 0.25, -bas_depth), 0.25, bas_depth, facecolor="#7c3aed", alpha=0.5))
        ax.add_patch(plt.Rectangle((right, -bas_depth), 0.25, bas_depth, facecolor="#7c3aed", alpha=0.5))
        ax.text(right + 0.6, -0.5 * bas_depth, f"Basement retaining walls\n{inp.n_basement} basements", fontsize=8, va="center", color="#6d28d9")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_ylim(-(inp.n_basement * inp.basement_height_m + 1.0), total_height + 2.0)
    ax.set_xlim(-2.5, right + 5.2)
    ax.set_xlabel("Schematic width")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Elevation view\nBlue: beams by story, Red: outrigger braced levels, Purple: basement retaining walls")
    return fig


def plot_mode_shapes(result: AnalysisResult, n_modes: int = 3) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 9))
    y = result.story_elevations_m
    for i in range(min(n_modes, len(result.mode_shapes))):
        ax.plot(result.mode_shapes[i], y, marker="o", label=f"Mode {i+1} | T={result.periods_s[i]:.3f} s")
    ax.set_xlabel("Normalized mode shape")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_drift(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(result.story_drift_ratios, result.story_elevations_m, marker="o")
    ax.set_xlabel("Story drift ratio")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Story drift ratio profile")
    ax.grid(True, alpha=0.3)
    return fig


# ----------------------------- REPORT -----------------------------

def build_report(inp: BuildingInput, result: AnalysisResult) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append(f"{APP_TITLE} - {APP_VERSION}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 88)
    lines.append(f"T1 from eigen analysis                = {result.periods_s[0]:.3f} s")
    lines.append(f"Code period T_code = Ct*H^x          = {result.T_code_s:.3f} s")
    lines.append(f"Upper code period T_upper = Cu*Tcode = {result.T_upper_s:.3f} s")
    lines.append("NOTE: T_code and T_upper are report/check values only. They are NOT used to size members.")
    lines.append(f"Total structural weight              = {result.total_weight_kn:,.1f} kN")
    lines.append(f"Equivalent base shear                = {result.base_shear_kn:,.1f} kN")
    lines.append(f"Roof displacement                    = {max(result.floor_displacements_m):.4f} m")
    lines.append(f"Max story drift                      = {max(result.story_drifts_m):.4f} m")
    lines.append(f"Max story drift ratio                = {max(result.story_drift_ratios):.6f}")
    lines.append(f"Drift limit ratio                    = {inp.drift_limit_ratio:.6f}")
    lines.append("")
    lines.append("BASEMENT RETAINING WALL")
    lines.append("-" * 88)
    lines.append(result.basement_wall_report)
    lines.append("")
    lines.append("ZONE MEMBER INPUTS USED IN ANALYSIS")
    lines.append("-" * 88)
    for _, row in result.zone_table.iterrows():
        lines.append(
            f"{row['Zone']:12s} | Stories {row['Stories']:>7s} | "
            f"Core wall t = {row['Core wall t (m)']:.2f} m | Perimeter wall t = {row['Perimeter wall t (m)']:.2f} m | "
            f"Beam b×h = {row['Beam b×h (m)']} | Slab t = {row['Slab t (m)']:.2f} m"
        )
    lines.append("")
    lines.append("OUTRIGGER / BRACED-BAY LAYOUT")
    lines.append("-" * 88)
    lines.append(f"Top X braced bays    = {inp.braced_layout.top_x_bays}")
    lines.append(f"Bottom X braced bays = {inp.braced_layout.bottom_x_bays}")
    lines.append(f"Left Y braced bays   = {inp.braced_layout.left_y_bays}")
    lines.append(f"Right Y braced bays  = {inp.braced_layout.right_y_bays}")
    lines.append("")
    if result.outriggers:
        for o in result.outriggers:
            lines.append(
                f"Story {o.story_level:>3d} | CHS A = {o.brace_area_m2:.4f} m² | KL/r = {o.slenderness:.1f} | "
                f"Keff = {o.effective_story_stiffness_n_m:.3e} N/m"
            )
    else:
        lines.append("No outriggers defined.")
    return "\n".join(lines)


# ----------------------------- STREAMLIT UI -----------------------------

def zone_block(label: str, defaults: ZoneMemberInput) -> ZoneMemberInput:
    st.sidebar.markdown(f"**{label} zone members**")
    wall_t = float(st.sidebar.number_input(f"{label} core wall thickness (m)", min_value=0.10, max_value=2.00, value=defaults.wall_thickness_m, step=0.05, format="%.2f"))
    wall_count = int(st.sidebar.selectbox(f"{label} core wall count", [4, 6, 8], index=[4, 6, 8].index(defaults.wall_count)))
    beam_b = float(st.sidebar.number_input(f"{label} beam width (m)", min_value=0.20, max_value=2.00, value=defaults.beam_width_m, step=0.05, format="%.2f"))
    beam_h = float(st.sidebar.number_input(f"{label} beam depth (m)", min_value=0.30, max_value=3.00, value=defaults.beam_depth_m, step=0.05, format="%.2f"))
    slab_t = float(st.sidebar.number_input(f"{label} slab thickness (m)", min_value=0.10, max_value=0.80, value=defaults.slab_thickness_m, step=0.01, format="%.2f"))
    cxx = float(st.sidebar.number_input(f"{label} corner column X (m)", min_value=0.20, max_value=3.00, value=defaults.corner_col_x_m, step=0.05, format="%.2f"))
    cxy = float(st.sidebar.number_input(f"{label} corner column Y (m)", min_value=0.20, max_value=3.00, value=defaults.corner_col_y_m, step=0.05, format="%.2f"))
    pxx = float(st.sidebar.number_input(f"{label} perimeter column X (m)", min_value=0.20, max_value=3.00, value=defaults.perimeter_col_x_m, step=0.05, format="%.2f"))
    pxy = float(st.sidebar.number_input(f"{label} perimeter column Y (m)", min_value=0.20, max_value=3.00, value=defaults.perimeter_col_y_m, step=0.05, format="%.2f"))
    ixx = float(st.sidebar.number_input(f"{label} interior column X (m)", min_value=0.20, max_value=3.00, value=defaults.interior_col_x_m, step=0.05, format="%.2f"))
    ixy = float(st.sidebar.number_input(f"{label} interior column Y (m)", min_value=0.20, max_value=3.00, value=defaults.interior_col_y_m, step=0.05, format="%.2f"))
    return ZoneMemberInput(wall_t, beam_b, beam_h, slab_t, cxx, cxy, pxx, pxy, ixx, ixy, wall_count)


def perimeter_wall_block(label: str, defaults: PerimeterWallInput, plan_x: float, plan_y: float) -> PerimeterWallInput:
    st.sidebar.markdown(f"**{label} perimeter lateral walls**")
    t = float(st.sidebar.number_input(f"{label} perimeter wall thickness (m)", min_value=0.0, max_value=1.50, value=defaults.thickness_m, step=0.05, format="%.2f"))
    top = float(st.sidebar.number_input(f"{label} top wall active length (m)", min_value=0.0, max_value=plan_x, value=min(defaults.top_length_m, plan_x), step=0.5))
    bottom = float(st.sidebar.number_input(f"{label} bottom wall active length (m)", min_value=0.0, max_value=plan_x, value=min(defaults.bottom_length_m, plan_x), step=0.5))
    left = float(st.sidebar.number_input(f"{label} left wall active length (m)", min_value=0.0, max_value=plan_y, value=min(defaults.left_length_m, plan_y), step=0.5))
    right = float(st.sidebar.number_input(f"{label} right wall active length (m)", min_value=0.0, max_value=plan_y, value=min(defaults.right_length_m, plan_y), step=0.5))
    return PerimeterWallInput(t, top, bottom, left, right)


def streamlit_input_panel() -> BuildingInput:
    st.sidebar.header("Input Data")
    default = BuildingInput()

    st.sidebar.subheader("Geometry")
    n_story = int(st.sidebar.number_input("Above-grade stories", min_value=1, max_value=120, value=default.n_story, step=1))
    n_basement = int(st.sidebar.number_input("Basement stories", min_value=0, max_value=10, value=default.n_basement, step=1))
    story_height_m = float(st.sidebar.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=default.story_height_m, step=0.1))
    basement_height_m = float(st.sidebar.number_input("Basement height (m)", min_value=2.5, max_value=6.0, value=default.basement_height_m, step=0.1))
    n_bays_x = int(st.sidebar.number_input("Bays in X", min_value=1, max_value=20, value=default.n_bays_x, step=1))
    n_bays_y = int(st.sidebar.number_input("Bays in Y", min_value=1, max_value=20, value=default.n_bays_y, step=1))
    bay_x_m = float(st.sidebar.number_input("Bay X (m)", min_value=2.0, max_value=20.0, value=default.bay_x_m, step=0.1))
    bay_y_m = float(st.sidebar.number_input("Bay Y (m)", min_value=2.0, max_value=20.0, value=default.bay_y_m, step=0.1))
    plan_x_m = n_bays_x * bay_x_m
    plan_y_m = n_bays_y * bay_y_m
    st.sidebar.caption(f"Plan dimensions are derived: X={plan_x_m:.2f} m, Y={plan_y_m:.2f} m")

    st.sidebar.subheader("Core geometry")
    core_outer_x_m = float(st.sidebar.number_input("Core outer X (m)", min_value=2.0, max_value=max(4.0, plan_x_m), value=min(default.core_outer_x_m, plan_x_m - 2.0), step=0.1))
    core_outer_y_m = float(st.sidebar.number_input("Core outer Y (m)", min_value=2.0, max_value=max(4.0, plan_y_m), value=min(default.core_outer_y_m, plan_y_m - 2.0), step=0.1))
    core_opening_x_m = float(st.sidebar.number_input("Core opening X (m)", min_value=1.0, max_value=max(1.0, core_outer_x_m - 0.5), value=min(default.core_opening_x_m, core_outer_x_m - 0.5), step=0.1))
    core_opening_y_m = float(st.sidebar.number_input("Core opening Y (m)", min_value=1.0, max_value=max(1.0, core_outer_y_m - 0.5), value=min(default.core_opening_y_m, core_outer_y_m - 0.5), step=0.1))

    st.sidebar.subheader("Materials / cracked sections")
    Ec_mpa = float(st.sidebar.number_input("Ec concrete (MPa)", min_value=15000.0, max_value=50000.0, value=default.Ec_mpa, step=500.0))
    Es_mpa = float(st.sidebar.number_input("Es steel (MPa)", min_value=150000.0, max_value=250000.0, value=default.Es_mpa, step=1000.0))
    fy_mpa = float(st.sidebar.number_input("fy steel (MPa)", min_value=200.0, max_value=700.0, value=default.fy_mpa, step=5.0))
    wall_cracked_factor = float(st.sidebar.number_input("Wall cracked factor", min_value=0.10, max_value=1.00, value=default.wall_cracked_factor, step=0.05, format="%.2f"))
    column_cracked_factor = float(st.sidebar.number_input("Column cracked factor", min_value=0.10, max_value=1.00, value=default.column_cracked_factor, step=0.05, format="%.2f"))
    beam_cracked_factor = float(st.sidebar.number_input("Beam cracked factor", min_value=0.10, max_value=1.00, value=default.beam_cracked_factor, step=0.05, format="%.2f"))
    slab_cracked_factor = float(st.sidebar.number_input("Slab cracked factor", min_value=0.10, max_value=1.00, value=default.slab_cracked_factor, step=0.05, format="%.2f"))
    retaining_wall_cracked_factor = float(st.sidebar.number_input("Retaining wall cracked factor", min_value=0.10, max_value=1.00, value=default.retaining_wall_cracked_factor, step=0.05, format="%.2f"))

    st.sidebar.subheader("Loads / seismic")
    dl_kn_m2 = float(st.sidebar.number_input("Dead load DL (kN/m²)", min_value=0.0, max_value=20.0, value=default.dl_kn_m2, step=0.1))
    ll_kn_m2 = float(st.sidebar.number_input("Live load LL (kN/m²)", min_value=0.0, max_value=20.0, value=default.ll_kn_m2, step=0.1))
    superimposed_dead_kn_m2 = float(st.sidebar.number_input("Superimposed dead (kN/m²)", min_value=0.0, max_value=20.0, value=default.superimposed_dead_kn_m2, step=0.1))
    facade_line_load_kn_m = float(st.sidebar.number_input("Facade line load (kN/m)", min_value=0.0, max_value=20.0, value=default.facade_line_load_kn_m, step=0.1))
    live_load_mass_factor = float(st.sidebar.number_input("Live load mass factor", min_value=0.0, max_value=1.0, value=default.live_load_mass_factor, step=0.05, format="%.2f"))
    seismic_base_shear_coeff = float(st.sidebar.number_input("Base shear coefficient", min_value=0.01, max_value=0.50, value=default.seismic_base_shear_coeff, step=0.005, format="%.3f"))
    drift_limit_ratio = float(st.sidebar.number_input("Drift limit ratio", min_value=0.0005, max_value=0.02, value=default.drift_limit_ratio, step=0.0005, format="%.4f"))
    Ct = float(st.sidebar.number_input("Code Ct (report only)", min_value=0.01, max_value=0.20, value=default.Ct, step=0.001, format="%.4f"))
    x_period = float(st.sidebar.number_input("Code x exponent (report only)", min_value=0.50, max_value=1.00, value=default.x_period, step=0.01, format="%.2f"))
    Cu = float(st.sidebar.number_input("Code Cu factor (report only)", min_value=1.00, max_value=2.00, value=default.Cu, step=0.05, format="%.2f"))

    st.sidebar.subheader("Core opening demand")
    stair_count = int(st.sidebar.number_input("Number of stairs", min_value=0, max_value=20, value=default.stair_count, step=1))
    elevator_count = int(st.sidebar.number_input("Number of elevators", min_value=0, max_value=40, value=default.elevator_count, step=1))
    elevator_area_each_m2 = float(st.sidebar.number_input("Elevator area each (m²)", min_value=0.0, max_value=20.0, value=default.elevator_area_each_m2, step=0.1))
    stair_area_each_m2 = float(st.sidebar.number_input("Stair area each (m²)", min_value=0.0, max_value=50.0, value=default.stair_area_each_m2, step=0.5))
    service_area_m2 = float(st.sidebar.number_input("Service area (m²)", min_value=0.0, max_value=100.0, value=default.service_area_m2, step=0.5))
    corridor_factor = float(st.sidebar.number_input("Corridor factor", min_value=1.0, max_value=2.0, value=default.corridor_factor, step=0.05, format="%.2f"))

    st.sidebar.subheader("Zone member sizes")
    lower_zone = zone_block("Lower", default.lower_zone)
    middle_zone = zone_block("Middle", default.middle_zone)
    upper_zone = zone_block("Upper", default.upper_zone)

    st.sidebar.subheader("Perimeter lateral walls")
    lower_perimeter_walls = perimeter_wall_block("Lower", default.lower_perimeter_walls, plan_x_m, plan_y_m)
    middle_perimeter_walls = perimeter_wall_block("Middle", default.middle_perimeter_walls, plan_x_m, plan_y_m)
    upper_perimeter_walls = perimeter_wall_block("Upper", default.upper_perimeter_walls, plan_x_m, plan_y_m)

    st.sidebar.subheader("Basement retaining wall")
    rw_enabled = st.sidebar.checkbox("Include basement retaining wall", value=default.retaining_wall.enabled)
    rw_t = float(st.sidebar.number_input("Retaining wall thickness (m)", min_value=0.0, max_value=1.50, value=default.retaining_wall.thickness_m, step=0.05, format="%.2f"))
    rw_top = float(st.sidebar.number_input("Retaining top active length (m)", min_value=0.0, max_value=plan_x_m, value=plan_x_m, step=0.5))
    rw_bottom = float(st.sidebar.number_input("Retaining bottom active length (m)", min_value=0.0, max_value=plan_x_m, value=plan_x_m, step=0.5))
    rw_left = float(st.sidebar.number_input("Retaining left active length (m)", min_value=0.0, max_value=plan_y_m, value=plan_y_m, step=0.5))
    rw_right = float(st.sidebar.number_input("Retaining right active length (m)", min_value=0.0, max_value=plan_y_m, value=plan_y_m, step=0.5))
    retaining_wall = RetainingWallInput(rw_enabled, rw_t, rw_top, rw_bottom, rw_left, rw_right)

    st.sidebar.subheader("Steel braced outriggers (CHS)")
    outrigger_count = int(st.sidebar.selectbox("Number of outrigger levels", [0, 1, 2, 3], index=2))
    suggested = []
    if outrigger_count >= 1:
        suggested.append(max(1, int(round(n_story * 0.45))))
    if outrigger_count >= 2:
        suggested.append(max(1, int(round(n_story * 0.70))))
    if outrigger_count >= 3:
        suggested.append(max(1, int(round(n_story * 0.85))))
    levels = []
    for i in range(outrigger_count):
        levels.append(int(st.sidebar.number_input(f"Outrigger level {i+1}", min_value=1, max_value=n_story, value=min(suggested[i], n_story), step=1)))
    brace_outer_diameter_mm = float(st.sidebar.number_input("CHS outer diameter (mm)", min_value=100.0, max_value=2000.0, value=default.brace_outer_diameter_mm, step=10.0))
    brace_thickness_mm = float(st.sidebar.number_input("CHS thickness (mm)", min_value=4.0, max_value=80.0, value=default.brace_thickness_mm, step=1.0))
    braces_per_side = int(st.sidebar.number_input("Braces per side", min_value=1, max_value=8, value=default.braces_per_side, step=1))
    outrigger_depth_m = float(st.sidebar.number_input("Outrigger brace depth (m)", min_value=0.5, max_value=10.0, value=default.outrigger_depth_m, step=0.1))
    brace_effective_length_factor = float(st.sidebar.number_input("Brace K factor", min_value=0.5, max_value=2.0, value=default.brace_effective_length_factor, step=0.05, format="%.2f"))
    brace_buckling_reduction = float(st.sidebar.number_input("Brace buckling reduction", min_value=0.10, max_value=1.00, value=default.brace_buckling_reduction, step=0.05, format="%.2f"))

    st.sidebar.subheader("Braced bays in plan")
    top_x = parse_bay_list(st.sidebar.text_input("Top side X-bays (comma-separated)", "2,5"), n_bays_x)
    bottom_x = parse_bay_list(st.sidebar.text_input("Bottom side X-bays (comma-separated)", "2,5"), n_bays_x)
    left_y = parse_bay_list(st.sidebar.text_input("Left side Y-bays (comma-separated)", "2,5"), n_bays_y)
    right_y = parse_bay_list(st.sidebar.text_input("Right side Y-bays (comma-separated)", "2,5"), n_bays_y)
    braced_layout = BracedBayLayout(top_x, bottom_x, left_y, right_y)

    return BuildingInput(
        n_story=n_story, n_basement=n_basement, story_height_m=story_height_m, basement_height_m=basement_height_m,
        plan_x_m=plan_x_m, plan_y_m=plan_y_m, n_bays_x=n_bays_x, n_bays_y=n_bays_y, bay_x_m=bay_x_m, bay_y_m=bay_y_m,
        core_outer_x_m=core_outer_x_m, core_outer_y_m=core_outer_y_m, core_opening_x_m=core_opening_x_m, core_opening_y_m=core_opening_y_m,
        Ec_mpa=Ec_mpa, Es_mpa=Es_mpa, fy_mpa=fy_mpa,
        dl_kn_m2=dl_kn_m2, ll_kn_m2=ll_kn_m2, superimposed_dead_kn_m2=superimposed_dead_kn_m2, facade_line_load_kn_m=facade_line_load_kn_m,
        live_load_mass_factor=live_load_mass_factor, seismic_base_shear_coeff=seismic_base_shear_coeff, drift_limit_ratio=drift_limit_ratio,
        wall_cracked_factor=wall_cracked_factor, column_cracked_factor=column_cracked_factor, beam_cracked_factor=beam_cracked_factor,
        slab_cracked_factor=slab_cracked_factor, retaining_wall_cracked_factor=retaining_wall_cracked_factor,
        Ct=Ct, x_period=x_period, Cu=Cu,
        lower_zone=lower_zone, middle_zone=middle_zone, upper_zone=upper_zone,
        lower_perimeter_walls=lower_perimeter_walls, middle_perimeter_walls=middle_perimeter_walls, upper_perimeter_walls=upper_perimeter_walls,
        retaining_wall=retaining_wall,
        stair_count=stair_count, elevator_count=elevator_count, elevator_area_each_m2=elevator_area_each_m2,
        stair_area_each_m2=stair_area_each_m2, service_area_m2=service_area_m2, corridor_factor=corridor_factor,
        outrigger_count=outrigger_count, outrigger_story_levels=levels,
        brace_outer_diameter_mm=brace_outer_diameter_mm, brace_thickness_mm=brace_thickness_mm, braces_per_side=braces_per_side,
        outrigger_depth_m=outrigger_depth_m, brace_effective_length_factor=brace_effective_length_factor,
        brace_buckling_reduction=brace_buckling_reduction, braced_layout=braced_layout
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info(
        "This version removes period-driven member sizing. Member sizes are entered directly by zone. "
        "T1 is obtained from eigen analysis of the assembled mass-stiffness model. "
        "Code T_code and T_upper are shown only as report/check values."
    )

    inp = streamlit_input_panel()

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.report = ""

    c1, c2 = st.columns([1, 1])
    with c1:
        analyze_btn = st.button("Analyze", use_container_width=True)
    with c2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.result = None
        st.session_state.report = ""
        st.rerun()

    if analyze_btn:
        try:
            with st.spinner("Running rational analysis..."):
                result = analyze(inp)
                st.session_state.result = result
                st.session_state.report = build_report(inp, result)
            st.success("Analysis completed.")
        except Exception as exc:
            st.exception(exc)

    result: AnalysisResult | None = st.session_state.result
    if result is None:
        st.warning("Enter inputs and click Analyze.")
        return

    tabs = st.tabs(["Summary", "Story Table", "Zones", "Outriggers", "Plan", "Elevation", "Modes", "Drift", "Report"])

    with tabs[0]:
        st.subheader("Summary")
        st.dataframe(result.summary_table, use_container_width=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("T1 (s)", f"{result.periods_s[0]:.3f}")
        mc2.metric("Base shear (kN)", f"{result.base_shear_kn:,.0f}")
        mc3.metric("Roof disp. (m)", f"{max(result.floor_displacements_m):.4f}")
        mc4.metric("Max drift ratio", f"{max(result.story_drift_ratios):.5f}")
        st.write(result.basement_wall_report)

    with tabs[1]:
        st.subheader("Story-by-story member sizes and response")
        st.dataframe(result.story_table, use_container_width=True)
        csv_story = result.story_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download story table CSV", data=csv_story, file_name="story_table_v81.csv", mime="text/csv")

    with tabs[2]:
        st.subheader("Zone stiffness table")
        st.dataframe(result.zone_table, use_container_width=True)

    with tabs[3]:
        st.subheader("Outrigger CHS results")
        st.dataframe(result.outrigger_table if not result.outrigger_table.empty else pd.DataFrame({"Info": ["No outriggers defined."]}), use_container_width=True)

    with tabs[4]:
        st.subheader("Plan view")
        zone_name = st.selectbox("Zone for plan view", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)
        st.pyplot(plot_plan(inp, zone_name))

    with tabs[5]:
        st.subheader("Elevation view")
        st.pyplot(plot_elevation(inp, result))

    with tabs[6]:
        st.subheader("Mode shapes")
        st.pyplot(plot_mode_shapes(result, n_modes=3))

    with tabs[7]:
        st.subheader("Drift profile")
        st.pyplot(plot_drift(result))

    with tabs[8]:
        st.subheader("Report")
        st.text_area("Text report", st.session_state.report, height=650)
        st.download_button("Download report as TXT", data=st.session_state.report, file_name="tall_building_report_v81.txt", mime="text/plain")


if __name__ == "__main__":
    main()
