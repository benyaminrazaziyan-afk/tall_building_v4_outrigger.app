
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m^3
STEEL_DENSITY = 7850.0  # kg/m^3
APP_TITLE = "Tall Building Rational Analysis + Steel Braced Outrigger"
APP_VERSION = "v8.0-professional"


# --------------------------- DATA MODELS ---------------------------

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
    corner_col_x_m: float
    corner_col_y_m: float
    perimeter_col_x_m: float
    perimeter_col_y_m: float
    interior_col_x_m: float
    interior_col_y_m: float


@dataclass
class BuildingInput:
    # Geometry
    n_story: int = 60
    n_basement: int = 0
    story_height_m: float = 3.2
    basement_height_m: float = 3.0
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6
    bay_x_m: float = 8.0
    bay_y_m: float = 7.0

    # Core geometry
    core_outer_x_m: float = 18.0
    core_outer_y_m: float = 15.0
    core_opening_x_m: float = 12.0
    core_opening_y_m: float = 10.5
    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4

    # Materials
    fck_mpa: float = 50.0
    Ec_mpa: float = 34000.0
    fy_mpa: float = 420.0
    Es_mpa: float = 200000.0
    nu_concrete: float = 0.20

    # Loads
    dl_kn_m2: float = 3.0
    ll_kn_m2: float = 2.0
    superimposed_dead_kn_m2: float = 1.5
    facade_line_load_kn_m: float = 1.0
    live_load_mass_factor: float = 0.30
    seismic_base_shear_coeff: float = 0.08

    # Cracked section factors
    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.70
    beam_cracked_factor: float = 0.35
    slab_cracked_factor: float = 0.25

    # Floor system
    slab_thickness_m: float = 0.22
    beam_width_m: float = 0.45
    beam_depth_m: float = 0.80

    # Zone member sizes
    lower_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.60, 1.20, 1.20, 1.00, 1.00, 0.90, 0.90))
    middle_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.45, 1.00, 1.00, 0.85, 0.85, 0.75, 0.75))
    upper_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.35, 0.85, 0.85, 0.75, 0.75, 0.65, 0.65))

    # Circulation / services
    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each_m2: float = 3.5
    stair_area_each_m2: float = 20.0
    service_area_m2: float = 35.0
    corridor_factor: float = 1.35

    # Outriggers
    outrigger_count: int = 0
    outrigger_story_levels: List[int] = field(default_factory=list)
    brace_outer_diameter_mm: float = 355.6
    brace_thickness_mm: float = 16.0
    braces_per_side: int = 2
    outrigger_depth_m: float = 3.0
    brace_effective_length_factor: float = 1.0
    brace_buckling_reduction: float = 0.85

    # Optional limits / reporting
    drift_limit_ratio: float = 1.0 / 500.0


@dataclass
class OutriggerResult:
    story_level: int
    arm_m: float
    brace_length_m: float
    brace_area_m2: float
    brace_radius_gyration_m: float
    slenderness: float
    axial_stiffness_n: float
    lateral_stiffness_n_m: float
    diaphragm_limited_stiffness_n_m: float
    steel_weight_kg: float


@dataclass
class ZoneStiffnessResult:
    zone_name: str
    story_start: int
    story_end: int
    wall_count: int
    wall_thickness_m: float
    wall_story_stiffness_n_m: float
    column_story_stiffness_n_m: float
    beam_story_stiffness_n_m: float
    diaphragm_stiffness_n_m: float
    total_story_stiffness_n_m: float


@dataclass
class AnalysisResult:
    story_stiffness_n_m: List[float]
    story_mass_kg: List[float]
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    lateral_forces_n: List[float]
    floor_displacements_m: List[float]
    story_drifts_m: List[float]
    story_drift_ratios: List[float]
    base_shear_kn: float
    total_weight_kn: float
    total_mass_kg: float
    k_wall_total_n_m: float
    k_column_total_n_m: float
    k_beam_total_n_m: float
    k_outrigger_total_n_m: float
    k_diaphragm_median_n_m: float
    outriggers: List[OutriggerResult]
    zone_results: List[ZoneStiffnessResult]
    zone_table: pd.DataFrame
    story_table: pd.DataFrame
    outrigger_table: pd.DataFrame
    summary_table: pd.DataFrame


# --------------------------- BASIC HELPERS ---------------------------

def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def get_zone_input(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    mapping = {
        "Lower Zone": inp.lower_zone,
        "Middle Zone": inp.middle_zone,
        "Upper Zone": inp.upper_zone,
    }
    return mapping[zone_name]


def get_wall_count(inp: BuildingInput, zone_name: str) -> int:
    mapping = {
        "Lower Zone": inp.lower_zone_wall_count,
        "Middle Zone": inp.middle_zone_wall_count,
        "Upper Zone": inp.upper_zone_wall_count,
    }
    return mapping[zone_name]


def gross_floor_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m


def required_service_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each_m2
        + inp.stair_count * inp.stair_area_each_m2
        + inp.service_area_m2
    ) * inp.corridor_factor


def core_geometry_is_valid(inp: BuildingInput) -> Tuple[bool, str]:
    if inp.core_opening_x_m >= inp.core_outer_x_m or inp.core_opening_y_m >= inp.core_outer_y_m:
        return False, "Core opening must be smaller than the core outer dimensions."
    if required_service_opening_area(inp) > inp.core_opening_x_m * inp.core_opening_y_m:
        return False, "Core opening area is smaller than required elevator/stair/service area."
    return True, ""


def rect_inertia_x(b: float, h: float) -> float:
    return b * h**3 / 12.0


def rect_inertia_y(b: float, h: float) -> float:
    return h * b**3 / 12.0


def circular_hollow_area(do_m: float, t_m: float) -> float:
    di = max(do_m - 2.0 * t_m, 1e-6)
    return (pi / 4.0) * (do_m**2 - di**2)


def circular_hollow_inertia(do_m: float, t_m: float) -> float:
    di = max(do_m - 2.0 * t_m, 1e-6)
    return (pi / 64.0) * (do_m**4 - di**4)


def circular_hollow_radius_of_gyration(do_m: float, t_m: float) -> float:
    area = circular_hollow_area(do_m, t_m)
    I = circular_hollow_inertia(do_m, t_m)
    return sqrt(I / max(area, 1e-12))


def concrete_shear_modulus_pa(inp: BuildingInput) -> float:
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


def core_min_inertia_about_principal(inp: BuildingInput, t: float, wall_count: int) -> float:
    lengths = wall_lengths_for_core(inp, wall_count)
    ox = inp.core_outer_x_m
    oy = inp.core_outer_y_m
    x_side = ox / 2.0
    y_side = oy / 2.0
    Ix = 0.0
    Iy = 0.0

    # top/bottom walls
    for L in lengths[0:2]:
        Ix += rect_inertia_x(L, t) + L * t * (y_side**2)
        Iy += rect_inertia_y(L, t)

    # left/right walls
    for L in lengths[2:4]:
        Iy += rect_inertia_y(t, L) + L * t * (x_side**2)
        Ix += rect_inertia_x(t, L)

    if wall_count >= 6:
        inner_x = 0.22 * ox
        for offset, L in zip([-inner_x, inner_x], lengths[4:6]):
            Iy += rect_inertia_y(t, L) + L * t * offset**2
            Ix += rect_inertia_x(t, L)

    if wall_count >= 8:
        inner_y = 0.22 * oy
        for offset, L in zip([-inner_y, inner_y], lengths[6:8]):
            Ix += rect_inertia_x(L, t) + L * t * offset**2
            Iy += rect_inertia_y(L, t)

    return min(Ix, Iy)


def wall_story_stiffness_n_m(inp: BuildingInput, wall_count: int, t: float) -> float:
    E = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus_pa(inp)
    h = inp.story_height_m
    Ieff = inp.wall_cracked_factor * core_min_inertia_about_principal(inp, t, wall_count)

    total_wall_area = sum(L * t for L in wall_lengths_for_core(inp, wall_count))
    kb = 12.0 * E * Ieff / max(h**3, 1e-12)

    kappa = 1.2
    ks = kappa * Gc * total_wall_area / max(h, 1e-12)

    return 1.0 / max((1.0 / max(kb, 1e-12)) + (1.0 / max(ks, 1e-12)), 1e-18)


def frame_story_stiffness_from_columns(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)

    I_corner = min(rect_inertia_x(z.corner_col_x_m, z.corner_col_y_m), rect_inertia_y(z.corner_col_x_m, z.corner_col_y_m))
    I_perim = min(rect_inertia_x(z.perimeter_col_x_m, z.perimeter_col_y_m), rect_inertia_y(z.perimeter_col_x_m, z.perimeter_col_y_m))
    I_inter = min(rect_inertia_x(z.interior_col_x_m, z.interior_col_y_m), rect_inertia_y(z.interior_col_x_m, z.interior_col_y_m))

    I_total = inp.column_cracked_factor * (
        corner_cols * I_corner + perimeter_cols * I_perim + interior_cols * I_inter
    )
    return 12.0 * E * I_total / max(h**3, 1e-12)


def beam_rotational_restraint_factor(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    Ib = inp.beam_cracked_factor * rect_inertia_x(inp.beam_width_m, inp.beam_depth_m)

    n_beams_x = inp.n_bays_x * (inp.n_bays_y + 1)
    n_beams_y = inp.n_bays_y * (inp.n_bays_x + 1)
    kb = n_beams_x * (4.0 * E * Ib / max(inp.bay_x_m, 1e-12)) + n_beams_y * (4.0 * E * Ib / max(inp.bay_y_m, 1e-12))

    total_cols = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner_cols = 4
    perimeter_cols = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)

    Ic = rect_inertia_x(z.corner_col_x_m, z.corner_col_y_m)
    Ip = rect_inertia_x(z.perimeter_col_x_m, z.perimeter_col_y_m)
    Ii = rect_inertia_x(z.interior_col_x_m, z.interior_col_y_m)
    kc = corner_cols * (4.0 * E * inp.column_cracked_factor * Ic / max(h, 1e-12))
    kc += perimeter_cols * (4.0 * E * inp.column_cracked_factor * Ip / max(h, 1e-12))
    kc += interior_cols * (4.0 * E * inp.column_cracked_factor * Ii / max(h, 1e-12))

    return kb / max(kb + kc, 1e-12)


def beam_story_stiffness_n_m(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    Ib = inp.beam_cracked_factor * rect_inertia_x(inp.beam_width_m, inp.beam_depth_m)
    n_beams_x = inp.n_bays_x * (inp.n_bays_y + 1)
    n_beams_y = inp.n_bays_y * (inp.n_bays_x + 1)
    kx = n_beams_x * 12.0 * E * Ib / max(inp.bay_x_m**3, 1e-12)
    ky = n_beams_y * 12.0 * E * Ib / max(inp.bay_y_m**3, 1e-12)
    restraint = beam_rotational_restraint_factor(inp, z)
    return (kx + ky) * restraint * (inp.story_height_m / max(0.5 * (inp.bay_x_m + inp.bay_y_m), 1e-12)) ** 2


def diaphragm_stiffness_n_m(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    A = inp.slab_cracked_factor * inp.slab_thickness_m * gross_floor_area(inp)
    L = max(inp.plan_x_m, inp.plan_y_m)
    return E * A / max(L, 1e-12)


def floor_weight_kn(inp: BuildingInput) -> float:
    area = gross_floor_area(inp)
    slab_self = inp.slab_thickness_m * CONCRETE_UNIT_WEIGHT
    beam_length = inp.n_bays_x * (inp.n_bays_y + 1) * inp.bay_x_m + inp.n_bays_y * (inp.n_bays_x + 1) * inp.bay_y_m
    beam_self = inp.beam_width_m * inp.beam_depth_m * CONCRETE_UNIT_WEIGHT * beam_length / max(area, 1e-12)
    gravity = inp.dl_kn_m2 + inp.superimposed_dead_kn_m2 + slab_self + beam_self + inp.live_load_mass_factor * inp.ll_kn_m2
    facade = inp.facade_line_load_kn_m * 2.0 * (inp.plan_x_m + inp.plan_y_m) / max(area, 1e-12)
    return area * (gravity + facade)


def floor_mass_kg(inp: BuildingInput) -> float:
    return floor_weight_kn(inp) * 1000.0 / G


def build_zone_stiffness_results(inp: BuildingInput) -> List[ZoneStiffnessResult]:
    results: List[ZoneStiffnessResult] = []
    kd = diaphragm_stiffness_n_m(inp)
    for zone in define_three_zones(inp.n_story):
        z = get_zone_input(inp, zone.name)
        wc = get_wall_count(inp, zone.name)
        kw = wall_story_stiffness_n_m(inp, wc, z.wall_thickness_m)
        kc = frame_story_stiffness_from_columns(inp, z)
        kb = beam_story_stiffness_n_m(inp, z)
        ktotal = kw + kc + kb
        results.append(
            ZoneStiffnessResult(
                zone_name=zone.name,
                story_start=zone.story_start,
                story_end=zone.story_end,
                wall_count=wc,
                wall_thickness_m=z.wall_thickness_m,
                wall_story_stiffness_n_m=kw,
                column_story_stiffness_n_m=kc,
                beam_story_stiffness_n_m=kb,
                diaphragm_stiffness_n_m=kd,
                total_story_stiffness_n_m=ktotal,
            )
        )
    return results


# --------------------------- OUTRIGGERS ---------------------------

def brace_arm_length_m(inp: BuildingInput) -> float:
    return 0.5 * max(inp.plan_x_m - inp.core_outer_x_m, inp.plan_y_m - inp.core_outer_y_m)


def outrigger_results(inp: BuildingInput) -> List[OutriggerResult]:
    if inp.outrigger_count <= 0:
        return []

    levels = [lvl for lvl in inp.outrigger_story_levels[: inp.outrigger_count] if 1 <= lvl <= inp.n_story]
    if not levels:
        return []

    do = inp.brace_outer_diameter_mm / 1000.0
    t = inp.brace_thickness_mm / 1000.0
    area = circular_hollow_area(do, t)
    rg = circular_hollow_radius_of_gyration(do, t)
    E = inp.Es_mpa * 1e6
    arm = brace_arm_length_m(inp)
    depth = inp.outrigger_depth_m
    diaphragm_k = diaphragm_stiffness_n_m(inp)

    out: List[OutriggerResult] = []
    for level in levels:
        Lb = sqrt(arm**2 + depth**2)
        axial_k = E * area / max(Lb, 1e-12)
        slenderness = inp.brace_effective_length_factor * Lb / max(rg, 1e-12)

        # Per braced side: equivalent horizontal stiffness from diagonal action
        k_one_brace = axial_k * (depth / max(Lb, 1e-12)) ** 2
        k_story = 4.0 * inp.braces_per_side * inp.brace_buckling_reduction * k_one_brace

        # Diaphragm limits effectiveness of force transfer from core to perimeter
        k_eff = 1.0 / max((1.0 / max(k_story, 1e-12)) + (1.0 / max(diaphragm_k, 1e-12)), 1e-18)

        steel_len_total = 4.0 * inp.braces_per_side * Lb
        steel_weight = steel_len_total * area * STEEL_DENSITY

        out.append(
            OutriggerResult(
                story_level=level,
                arm_m=arm,
                brace_length_m=Lb,
                brace_area_m2=area,
                brace_radius_gyration_m=rg,
                slenderness=slenderness,
                axial_stiffness_n=axial_k,
                lateral_stiffness_n_m=k_story,
                diaphragm_limited_stiffness_n_m=k_eff,
                steel_weight_kg=steel_weight,
            )
        )
    return out


# --------------------------- ANALYSIS ---------------------------

def build_story_stiffness_vector(inp: BuildingInput, zone_results: List[ZoneStiffnessResult], outriggers: List[OutriggerResult]) -> List[float]:
    by_story = {}
    for zr in zone_results:
        for s in range(zr.story_start, zr.story_end + 1):
            by_story[s] = zr.total_story_stiffness_n_m

    for outr in outriggers:
        by_story[outr.story_level] = by_story.get(outr.story_level, 0.0) + outr.diaphragm_limited_stiffness_n_m

    return [by_story[i] for i in range(1, inp.n_story + 1)]


def build_story_mass_vector(inp: BuildingInput, outriggers: List[OutriggerResult]) -> List[float]:
    m = [floor_mass_kg(inp) for _ in range(inp.n_story)]
    by_story_steel = {o.story_level: o.steel_weight_kg for o in outriggers}
    for i in range(inp.n_story):
        if (i + 1) in by_story_steel:
            m[i] += by_story_steel[i + 1]
    return m


def assemble_m_k(masses: List[float], stiffnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(masses)
    M = np.diag(np.array(masses, dtype=float))
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = stiffnesses[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K


def solve_modes(M: np.ndarray, K: np.ndarray, n_modes: int = 5) -> Tuple[List[float], List[float], List[List[float]]]:
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    mask = eigvals > 1e-12
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    omegas = np.sqrt(eigvals[:n_modes])
    periods = [2.0 * pi / max(w, 1e-12) for w in omegas]
    freqs = [w / (2.0 * pi) for w in omegas]

    shapes = []
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].copy()
        phi = phi / max(np.max(np.abs(phi)), 1e-12)
        if phi[-1] < 0:
            phi = -phi
        shapes.append(phi.tolist())
    return periods, freqs, shapes


def lateral_force_distribution_n(inp: BuildingInput, masses: List[float], total_weight_kn: float) -> np.ndarray:
    V = inp.seismic_base_shear_coeff * total_weight_kn * 1000.0
    heights = np.arange(1, inp.n_story + 1, dtype=float) * inp.story_height_m
    w = np.array(masses) * G
    coeff = w * heights
    F = V * coeff / max(np.sum(coeff), 1e-12)
    return F


def solve_static_displacements(K: np.ndarray, F: np.ndarray, story_height_m: float) -> Tuple[List[float], List[float], List[float]]:
    u = np.linalg.solve(K, F)
    drifts = np.zeros_like(u)
    drifts[0] = u[0]
    drifts[1:] = u[1:] - u[:-1]
    drift_ratios = drifts / max(story_height_m, 1e-12)
    return u.tolist(), drifts.tolist(), drift_ratios.tolist()


def analyze(inp: BuildingInput) -> AnalysisResult:
    valid, msg = core_geometry_is_valid(inp)
    if not valid:
        raise ValueError(msg)

    zone_results = build_zone_stiffness_results(inp)
    outrs = outrigger_results(inp)
    k_story = build_story_stiffness_vector(inp, zone_results, outrs)
    m_story = build_story_mass_vector(inp, outrs)

    total_weight_kn = sum(m_story) * G / 1000.0
    total_mass_kg = sum(m_story)

    M, K = assemble_m_k(m_story, k_story)
    periods, freqs, shapes = solve_modes(M, K, n_modes=5)

    F = lateral_force_distribution_n(inp, m_story, total_weight_kn)
    u, drifts, drift_ratios = solve_static_displacements(K, F, inp.story_height_m)

    k_wall_total = sum(z.wall_story_stiffness_n_m * z.wall_count * z.total_story_stiffness_n_m / max(z.total_story_stiffness_n_m, 1e-12) * z.n_stories if False else 0 for z in [])
    # direct totals by story counts
    wall_total = 0.0
    col_total = 0.0
    beam_total = 0.0
    for z in zone_results:
        n = z.story_end - z.story_start + 1
        wall_total += z.wall_story_stiffness_n_m * n
        col_total += z.column_story_stiffness_n_m * n
        beam_total += z.beam_story_stiffness_n_m * n

    k_outrigger_total = sum(o.diaphragm_limited_stiffness_n_m for o in outrs)
    kd_median = float(np.median([z.diaphragm_stiffness_n_m for z in zone_results])) if zone_results else 0.0

    zone_table = pd.DataFrame([{
        "Zone": z.zone_name,
        "Stories": f"{z.story_start}-{z.story_end}",
        "Wall count": z.wall_count,
        "Wall thickness (m)": z.wall_thickness_m,
        "Wall K/story (N/m)": z.wall_story_stiffness_n_m,
        "Column K/story (N/m)": z.column_story_stiffness_n_m,
        "Beam K/story (N/m)": z.beam_story_stiffness_n_m,
        "Diaphragm K (N/m)": z.diaphragm_stiffness_n_m,
        "Total K/story (N/m)": z.total_story_stiffness_n_m,
    } for z in zone_results])

    story_table = pd.DataFrame({
        "Story": np.arange(1, inp.n_story + 1),
        "Mass (kg)": m_story,
        "Stiffness (N/m)": k_story,
        "Lateral force (N)": F.tolist(),
        "Floor displacement (m)": u,
        "Story drift (m)": drifts,
        "Story drift ratio": drift_ratios,
    })

    if outrs:
        outrigger_table = pd.DataFrame([{
            "Story": o.story_level,
            "Brace arm (m)": o.arm_m,
            "Brace length (m)": o.brace_length_m,
            "CHS area (m²)": o.brace_area_m2,
            "Radius of gyration (m)": o.brace_radius_gyration_m,
            "KL/r": o.slenderness,
            "Axial stiffness EA/L (N)": o.axial_stiffness_n,
            "Brace lateral stiffness (N/m)": o.lateral_stiffness_n_m,
            "Effective outrigger stiffness (N/m)": o.diaphragm_limited_stiffness_n_m,
            "Steel weight (kg)": o.steel_weight_kg,
        } for o in outrs])
    else:
        outrigger_table = pd.DataFrame(columns=[
            "Story", "Brace arm (m)", "Brace length (m)", "CHS area (m²)", "Radius of gyration (m)",
            "KL/r", "Axial stiffness EA/L (N)", "Brace lateral stiffness (N/m)",
            "Effective outrigger stiffness (N/m)", "Steel weight (kg)"
        ])

    summary_table = pd.DataFrame({
        "Parameter": [
            "Total weight (kN)",
            "Base shear (kN)",
            "Period T1 (s)",
            "Frequency f1 (Hz)",
            "Roof displacement (m)",
            "Max story drift (m)",
            "Max story drift ratio",
            "Total wall stiffness contribution (N/m-story sum)",
            "Total column stiffness contribution (N/m-story sum)",
            "Total beam stiffness contribution (N/m-story sum)",
            "Total effective outrigger stiffness (N/m)",
            "Median diaphragm in-plane stiffness (N/m)",
        ],
        "Value": [
            total_weight_kn,
            inp.seismic_base_shear_coeff * total_weight_kn,
            periods[0] if periods else np.nan,
            freqs[0] if freqs else np.nan,
            u[-1] if u else np.nan,
            max(drifts) if drifts else np.nan,
            max(drift_ratios) if drift_ratios else np.nan,
            wall_total,
            col_total,
            beam_total,
            k_outrigger_total,
            kd_median,
        ]
    })

    return AnalysisResult(
        story_stiffness_n_m=k_story,
        story_mass_kg=m_story,
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=shapes,
        lateral_forces_n=F.tolist(),
        floor_displacements_m=u,
        story_drifts_m=drifts,
        story_drift_ratios=drift_ratios,
        base_shear_kn=inp.seismic_base_shear_coeff * total_weight_kn,
        total_weight_kn=total_weight_kn,
        total_mass_kg=total_mass_kg,
        k_wall_total_n_m=wall_total,
        k_column_total_n_m=col_total,
        k_beam_total_n_m=beam_total,
        k_outrigger_total_n_m=k_outrigger_total,
        k_diaphragm_median_n_m=kd_median,
        outriggers=outrs,
        zone_results=zone_results,
        zone_table=zone_table,
        story_table=story_table,
        outrigger_table=outrigger_table,
        summary_table=summary_table,
    )


# --------------------------- PLOTTING ---------------------------

def plot_mode_shapes(result: AnalysisResult, n_modes: int = 3) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.story_mass_kg) + 1)
    for i in range(min(n_modes, len(result.mode_shapes))):
        ax.plot(result.mode_shapes[i], y, marker="o", label=f"Mode {i + 1} | T={result.periods_s[i]:.3f} s")
    ax.set_xlabel("Normalized mode shape")
    ax.set_ylabel("Story")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_story_response(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.floor_displacements_m) + 1)
    ax.plot(result.floor_displacements_m, y, marker="o", label="Floor displacement")
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel("Story")
    ax.set_title("Static floor displacements")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_story_drifts(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.story_drifts_m) + 1)
    ax.plot(result.story_drift_ratios, y, marker="o", label="Story drift ratio")
    ax.set_xlabel("Drift ratio")
    ax.set_ylabel("Story")
    ax.set_title("Story drift ratios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_plan(inp: BuildingInput, result: AnalysisResult, zone_name: str) -> plt.Figure:
    zone = next(z for z in result.zone_results if z.zone_name == zone_name)
    z_in = get_zone_input(inp, zone_name)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black")

    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#cccccc", linewidth=0.7)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#cccccc", linewidth=0.7)

    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    cy0 = (inp.plan_y_m - inp.core_outer_y_m) / 2.0
    ix0 = (inp.plan_x_m - inp.core_opening_x_m) / 2.0
    iy0 = (inp.plan_y_m - inp.core_opening_y_m) / 2.0

    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core_outer_x_m, inp.core_outer_y_m, fill=False, edgecolor="green", linewidth=2.0))
    ax.add_patch(plt.Rectangle((ix0, iy0), inp.core_opening_x_m, inp.core_opening_y_m, fill=False, edgecolor="gray", linestyle="--"))
    t = z_in.wall_thickness_m
    for x, y, w, h in [
        (cx0, cy0, inp.core_outer_x_m, t),
        (cx0, cy0 + inp.core_outer_y_m - t, inp.core_outer_x_m, t),
        (cx0, cy0, t, inp.core_outer_y_m),
        (cx0 + inp.core_outer_x_m - t, cy0, t, inp.core_outer_y_m),
    ]:
        ax.add_patch(plt.Rectangle((x, y), w, h, color="green", alpha=0.35))

    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x_m
            y = j * inp.bay_y_m
            at_x = i in (0, inp.n_bays_x)
            at_y = j in (0, inp.n_bays_y)
            if at_x and at_y:
                dx, dy, c = z_in.corner_col_x_m, z_in.corner_col_y_m, "#8b0000"
            elif at_x or at_y:
                dx, dy, c = z_in.perimeter_col_x_m, z_in.perimeter_col_y_m, "#cc5500"
            else:
                dx, dy, c = z_in.interior_col_x_m, z_in.interior_col_y_m, "#3366aa"
            ax.add_patch(plt.Rectangle((x - dx / 2.0, y - dy / 2.0), dx, dy, facecolor=c, edgecolor=c, alpha=0.85))

    if inp.outrigger_count > 0:
        arm = brace_arm_length_m(inp)
        depth = inp.outrigger_depth_m
        center_x = inp.plan_x_m / 2.0
        center_y = inp.plan_y_m / 2.0
        ax.add_patch(plt.Rectangle((cx0 - arm, center_y - depth / 2.0), arm, depth, color="#ff8c00", alpha=0.60))
        ax.add_patch(plt.Rectangle((cx0 + inp.core_outer_x_m, center_y - depth / 2.0), arm, depth, color="#ff8c00", alpha=0.60))

    ax.set_aspect("equal")
    ax.set_xlim(-2, inp.plan_x_m + 2)
    ax.set_ylim(-2, inp.plan_y_m + 2)
    ax.set_title(f"Plan view - {zone_name}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig


def plot_elevation(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    H = inp.n_story * inp.story_height_m
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, H, H, 0], color="black")
    for i in range(inp.n_story + 1):
        y = i * inp.story_height_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#dddddd", linewidth=0.7)

    # simple core band
    core_x0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    ax.add_patch(plt.Rectangle((core_x0, 0), inp.core_outer_x_m, H, edgecolor="green", fill=False, linewidth=2.0))

    for o in result.outriggers:
        y = o.story_level * inp.story_height_m
        ax.plot([core_x0, 0], [y, y], color="#ff8c00", linewidth=4)
        ax.plot([core_x0 + inp.core_outer_x_m, inp.plan_x_m], [y, y], color="#ff8c00", linewidth=4)

    ax.set_xlim(-1, inp.plan_x_m + 1)
    ax.set_ylim(0, H + inp.story_height_m)
    ax.set_xlabel("Plan width direction (schematic)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Elevation schematic")
    return fig


# --------------------------- REPORTING ---------------------------

def build_text_report(inp: BuildingInput, result: AnalysisResult) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"{APP_TITLE} - {APP_VERSION}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("MODEL BASIS")
    lines.append("-" * 78)
    lines.append("1) No empirical target-period tuning is used.")
    lines.append("2) Story stiffness is assembled from explicit wall, column, beam, and outrigger member stiffness.")
    lines.append("3) Cracked section factors are user inputs for walls, columns, beams, and slabs.")
    lines.append("4) Outriggers are modeled as steel CHS braced systems, diaphragm-limited in force transfer.")
    lines.append("")
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 78)
    lines.append(f"Total weight                           = {result.total_weight_kn:,.1f} kN")
    lines.append(f"Base shear                             = {result.base_shear_kn:,.1f} kN")
    lines.append(f"Fundamental period T1                  = {result.periods_s[0]:.3f} s")
    lines.append(f"Fundamental frequency f1               = {result.frequencies_hz[0]:.3f} Hz")
    lines.append(f"Roof displacement                      = {result.floor_displacements_m[-1]:.4f} m")
    lines.append(f"Maximum story drift                    = {max(result.story_drifts_m):.5f} m")
    lines.append(f"Maximum story drift ratio              = {max(result.story_drift_ratios):.6f}")
    lines.append(f"Drift limit ratio input                = {inp.drift_limit_ratio:.6f}")
    lines.append(f"Drift check                            = {'OK' if max(result.story_drift_ratios) <= inp.drift_limit_ratio else 'NOT OK'}")
    lines.append("")
    lines.append("SYSTEM STIFFNESS CONTRIBUTIONS")
    lines.append("-" * 78)
    lines.append(f"Wall story-stiffness sum               = {result.k_wall_total_n_m:,.3e} N/m")
    lines.append(f"Column story-stiffness sum             = {result.k_column_total_n_m:,.3e} N/m")
    lines.append(f"Beam story-stiffness sum               = {result.k_beam_total_n_m:,.3e} N/m")
    lines.append(f"Effective outrigger stiffness sum      = {result.k_outrigger_total_n_m:,.3e} N/m")
    lines.append(f"Median diaphragm in-plane stiffness    = {result.k_diaphragm_median_n_m:,.3e} N/m")
    lines.append("")
    lines.append("ZONE SUMMARY")
    lines.append("-" * 78)
    for z in result.zone_results:
        lines.append(
            f"{z.zone_name:12s} | stories {z.story_start:>2}-{z.story_end:<2} | "
            f"wall t = {z.wall_thickness_m:.3f} m | "
            f"K_wall = {z.wall_story_stiffness_n_m:.3e} | "
            f"K_col = {z.column_story_stiffness_n_m:.3e} | "
            f"K_beam = {z.beam_story_stiffness_n_m:.3e}"
        )
    lines.append("")
    lines.append("OUTRIGGER SUMMARY")
    lines.append("-" * 78)
    if result.outriggers:
        for o in result.outriggers:
            lines.append(
                f"Story {o.story_level:>3} | Lb = {o.brace_length_m:6.3f} m | "
                f"A = {o.brace_area_m2:8.5f} m² | KL/r = {o.slenderness:8.2f} | "
                f"K_eff = {o.diaphragm_limited_stiffness_n_m:10.3e} N/m"
            )
    else:
        lines.append("No outriggers defined.")
    return "\n".join(lines)


# --------------------------- STREAMLIT UI ---------------------------

def zone_block(label: str, defaults: ZoneMemberInput) -> ZoneMemberInput:
    st.sidebar.markdown(f"**{label} zone members**")
    wall_t = st.sidebar.number_input(f"{label} wall thickness (m)", min_value=0.15, max_value=2.00, value=float(defaults.wall_thickness_m), step=0.01, format="%.2f")
    cxx = st.sidebar.number_input(f"{label} corner column X (m)", min_value=0.20, max_value=3.00, value=float(defaults.corner_col_x_m), step=0.05, format="%.2f")
    cxy = st.sidebar.number_input(f"{label} corner column Y (m)", min_value=0.20, max_value=3.00, value=float(defaults.corner_col_y_m), step=0.05, format="%.2f")
    pxx = st.sidebar.number_input(f"{label} perimeter column X (m)", min_value=0.20, max_value=3.00, value=float(defaults.perimeter_col_x_m), step=0.05, format="%.2f")
    pxy = st.sidebar.number_input(f"{label} perimeter column Y (m)", min_value=0.20, max_value=3.00, value=float(defaults.perimeter_col_y_m), step=0.05, format="%.2f")
    ixx = st.sidebar.number_input(f"{label} interior column X (m)", min_value=0.20, max_value=3.00, value=float(defaults.interior_col_x_m), step=0.05, format="%.2f")
    ixy = st.sidebar.number_input(f"{label} interior column Y (m)", min_value=0.20, max_value=3.00, value=float(defaults.interior_col_y_m), step=0.05, format="%.2f")
    return ZoneMemberInput(wall_t, cxx, cxy, pxx, pxy, ixx, ixy)


def streamlit_input_panel() -> BuildingInput:
    st.sidebar.header("Input Data")
    default = BuildingInput()

    st.sidebar.subheader("Geometry")
    n_story = int(st.sidebar.number_input("Above-grade stories", min_value=1, max_value=120, value=default.n_story, step=1))
    n_basement = int(st.sidebar.number_input("Basement stories", min_value=0, max_value=20, value=default.n_basement, step=1))
    story_height_m = float(st.sidebar.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=default.story_height_m, step=0.1))
    basement_height_m = float(st.sidebar.number_input("Basement height (m)", min_value=2.5, max_value=6.0, value=default.basement_height_m, step=0.1))
    plan_x_m = float(st.sidebar.number_input("Plan X (m)", min_value=10.0, max_value=300.0, value=default.plan_x_m, step=0.5))
    plan_y_m = float(st.sidebar.number_input("Plan Y (m)", min_value=10.0, max_value=300.0, value=default.plan_y_m, step=0.5))
    n_bays_x = int(st.sidebar.number_input("Bays in X", min_value=1, max_value=40, value=default.n_bays_x, step=1))
    n_bays_y = int(st.sidebar.number_input("Bays in Y", min_value=1, max_value=40, value=default.n_bays_y, step=1))
    bay_x_m = float(st.sidebar.number_input("Bay X (m)", min_value=2.0, max_value=20.0, value=default.bay_x_m, step=0.1))
    bay_y_m = float(st.sidebar.number_input("Bay Y (m)", min_value=2.0, max_value=20.0, value=default.bay_y_m, step=0.1))

    st.sidebar.subheader("Core geometry")
    core_outer_x_m = float(st.sidebar.number_input("Core outer X (m)", min_value=2.0, max_value=100.0, value=default.core_outer_x_m, step=0.1))
    core_outer_y_m = float(st.sidebar.number_input("Core outer Y (m)", min_value=2.0, max_value=100.0, value=default.core_outer_y_m, step=0.1))
    core_opening_x_m = float(st.sidebar.number_input("Core opening X (m)", min_value=1.0, max_value=100.0, value=default.core_opening_x_m, step=0.1))
    core_opening_y_m = float(st.sidebar.number_input("Core opening Y (m)", min_value=1.0, max_value=100.0, value=default.core_opening_y_m, step=0.1))
    lower_zone_wall_count = int(st.sidebar.selectbox("Lower zone wall count", [4, 6, 8], index=2))
    middle_zone_wall_count = int(st.sidebar.selectbox("Middle zone wall count", [4, 6, 8], index=1))
    upper_zone_wall_count = int(st.sidebar.selectbox("Upper zone wall count", [4, 6, 8], index=0))

    st.sidebar.subheader("Materials")
    fck_mpa = float(st.sidebar.number_input("fck (MPa)", min_value=20.0, max_value=100.0, value=default.fck_mpa, step=1.0))
    Ec_mpa = float(st.sidebar.number_input("Ec (MPa)", min_value=15000.0, max_value=50000.0, value=default.Ec_mpa, step=500.0))
    fy_mpa = float(st.sidebar.number_input("fy steel (MPa)", min_value=200.0, max_value=700.0, value=default.fy_mpa, step=5.0))
    Es_mpa = float(st.sidebar.number_input("Es steel (MPa)", min_value=150000.0, max_value=250000.0, value=default.Es_mpa, step=1000.0))

    st.sidebar.subheader("Loads")
    dl_kn_m2 = float(st.sidebar.number_input("Dead load DL (kN/m²)", min_value=0.0, max_value=20.0, value=default.dl_kn_m2, step=0.1))
    ll_kn_m2 = float(st.sidebar.number_input("Live load LL (kN/m²)", min_value=0.0, max_value=20.0, value=default.ll_kn_m2, step=0.1))
    superimposed_dead_kn_m2 = float(st.sidebar.number_input("Superimposed dead (kN/m²)", min_value=0.0, max_value=20.0, value=default.superimposed_dead_kn_m2, step=0.1))
    facade_line_load_kn_m = float(st.sidebar.number_input("Facade line load (kN/m)", min_value=0.0, max_value=20.0, value=default.facade_line_load_kn_m, step=0.1))
    live_load_mass_factor = float(st.sidebar.number_input("Live load mass factor", min_value=0.0, max_value=1.0, value=default.live_load_mass_factor, step=0.05, format="%.2f"))
    seismic_base_shear_coeff = float(st.sidebar.number_input("Base shear coefficient", min_value=0.01, max_value=0.50, value=default.seismic_base_shear_coeff, step=0.005, format="%.3f"))
    drift_limit_ratio = float(st.sidebar.number_input("Drift limit ratio", min_value=0.0005, max_value=0.0200, value=default.drift_limit_ratio, step=0.0005, format="%.4f"))

    st.sidebar.subheader("Cracked section factors")
    wall_cracked_factor = float(st.sidebar.number_input("Wall cracked factor", min_value=0.10, max_value=1.00, value=default.wall_cracked_factor, step=0.05, format="%.2f"))
    column_cracked_factor = float(st.sidebar.number_input("Column cracked factor", min_value=0.10, max_value=1.00, value=default.column_cracked_factor, step=0.05, format="%.2f"))
    beam_cracked_factor = float(st.sidebar.number_input("Beam cracked factor", min_value=0.10, max_value=1.00, value=default.beam_cracked_factor, step=0.05, format="%.2f"))
    slab_cracked_factor = float(st.sidebar.number_input("Slab cracked factor", min_value=0.10, max_value=1.00, value=default.slab_cracked_factor, step=0.05, format="%.2f"))

    st.sidebar.subheader("Floor system")
    slab_thickness_m = float(st.sidebar.number_input("Slab thickness (m)", min_value=0.10, max_value=0.80, value=default.slab_thickness_m, step=0.01, format="%.2f"))
    beam_width_m = float(st.sidebar.number_input("Beam width (m)", min_value=0.20, max_value=2.00, value=default.beam_width_m, step=0.05, format="%.2f"))
    beam_depth_m = float(st.sidebar.number_input("Beam depth (m)", min_value=0.30, max_value=3.00, value=default.beam_depth_m, step=0.05, format="%.2f"))

    st.sidebar.subheader("Circulation / services")
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

    st.sidebar.subheader("Steel braced outriggers (CHS)")
    outrigger_count = int(st.sidebar.selectbox("Number of outriggers", [0, 1, 2, 3], index=0))
    suggested_levels = []
    if outrigger_count >= 1:
        suggested_levels.append(max(1, int(round(n_story * 0.45))))
    if outrigger_count >= 2:
        suggested_levels.append(max(1, int(round(n_story * 0.65))))
    if outrigger_count >= 3:
        suggested_levels.append(max(1, int(round(n_story * 0.82))))
    outrigger_story_levels: List[int] = []
    for i in range(outrigger_count):
        level = int(st.sidebar.number_input(
            f"Outrigger level {i + 1}",
            min_value=1,
            max_value=max(1, n_story),
            value=min(suggested_levels[i], n_story),
            step=1,
        ))
        outrigger_story_levels.append(level)

    brace_outer_diameter_mm = float(st.sidebar.number_input("CHS outer diameter (mm)", min_value=100.0, max_value=2000.0, value=default.brace_outer_diameter_mm, step=10.0))
    brace_thickness_mm = float(st.sidebar.number_input("CHS thickness (mm)", min_value=4.0, max_value=80.0, value=default.brace_thickness_mm, step=1.0))
    braces_per_side = int(st.sidebar.number_input("Braces per side", min_value=1, max_value=8, value=default.braces_per_side, step=1))
    outrigger_depth_m = float(st.sidebar.number_input("Outrigger brace depth (m)", min_value=0.5, max_value=10.0, value=default.outrigger_depth_m, step=0.1))
    brace_effective_length_factor = float(st.sidebar.number_input("Brace K factor", min_value=0.5, max_value=2.0, value=default.brace_effective_length_factor, step=0.05, format="%.2f"))
    brace_buckling_reduction = float(st.sidebar.number_input("Brace buckling reduction", min_value=0.10, max_value=1.00, value=default.brace_buckling_reduction, step=0.05, format="%.2f"))

    return BuildingInput(
        n_story=n_story,
        n_basement=n_basement,
        story_height_m=story_height_m,
        basement_height_m=basement_height_m,
        plan_x_m=plan_x_m,
        plan_y_m=plan_y_m,
        n_bays_x=n_bays_x,
        n_bays_y=n_bays_y,
        bay_x_m=bay_x_m,
        bay_y_m=bay_y_m,
        core_outer_x_m=core_outer_x_m,
        core_outer_y_m=core_outer_y_m,
        core_opening_x_m=core_opening_x_m,
        core_opening_y_m=core_opening_y_m,
        lower_zone_wall_count=lower_zone_wall_count,
        middle_zone_wall_count=middle_zone_wall_count,
        upper_zone_wall_count=upper_zone_wall_count,
        fck_mpa=fck_mpa,
        Ec_mpa=Ec_mpa,
        fy_mpa=fy_mpa,
        Es_mpa=Es_mpa,
        dl_kn_m2=dl_kn_m2,
        ll_kn_m2=ll_kn_m2,
        superimposed_dead_kn_m2=superimposed_dead_kn_m2,
        facade_line_load_kn_m=facade_line_load_kn_m,
        live_load_mass_factor=live_load_mass_factor,
        seismic_base_shear_coeff=seismic_base_shear_coeff,
        drift_limit_ratio=drift_limit_ratio,
        wall_cracked_factor=wall_cracked_factor,
        column_cracked_factor=column_cracked_factor,
        beam_cracked_factor=beam_cracked_factor,
        slab_cracked_factor=slab_cracked_factor,
        slab_thickness_m=slab_thickness_m,
        beam_width_m=beam_width_m,
        beam_depth_m=beam_depth_m,
        lower_zone=lower_zone,
        middle_zone=middle_zone,
        upper_zone=upper_zone,
        stair_count=stair_count,
        elevator_count=elevator_count,
        elevator_area_each_m2=elevator_area_each_m2,
        stair_area_each_m2=stair_area_each_m2,
        service_area_m2=service_area_m2,
        corridor_factor=corridor_factor,
        outrigger_count=outrigger_count,
        outrigger_story_levels=outrigger_story_levels,
        brace_outer_diameter_mm=brace_outer_diameter_mm,
        brace_thickness_mm=brace_thickness_mm,
        braces_per_side=braces_per_side,
        outrigger_depth_m=outrigger_depth_m,
        brace_effective_length_factor=brace_effective_length_factor,
        brace_buckling_reduction=brace_buckling_reduction,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info(
        "This version removes empirical target-period tuning. "
        "The analysis is based on explicit wall, column, beam, slab diaphragm, and steel CHS-braced outrigger stiffness inputs."
    )

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "input_data" not in st.session_state:
        st.session_state.input_data = None
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""

    inp = streamlit_input_panel()

    c1, c2 = st.columns(2)
    analyze_btn = c1.button("Analyze", use_container_width=True)
    clear_btn = c2.button("Clear results", use_container_width=True)

    if clear_btn:
        st.session_state.analysis_result = None
        st.session_state.input_data = None
        st.session_state.report_text = ""
        st.rerun()

    if analyze_btn:
        try:
            with st.spinner("Running rational analysis..."):
                result = analyze(inp)
                st.session_state.analysis_result = result
                st.session_state.input_data = inp
                st.session_state.report_text = build_text_report(inp, result)
            st.success("Analysis completed.")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.analysis_result
    saved_inp = st.session_state.input_data
    if result is None or saved_inp is None:
        st.warning("Enter inputs and click Analyze.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Summary", "Story Tables", "Mode Shapes", "Plan / Elevation", "Outriggers", "Report"
    ])

    with tab1:
        st.subheader("Summary")
        st.dataframe(result.summary_table, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("T1 (s)", f"{result.periods_s[0]:.3f}" if result.periods_s else "-")
        m2.metric("Roof disp. (m)", f"{result.floor_displacements_m[-1]:.4f}")
        m3.metric("Max drift ratio", f"{max(result.story_drift_ratios):.6f}")
        m4.metric("Base shear (kN)", f"{result.base_shear_kn:,.1f}")

        st.subheader("Zone stiffness table")
        st.dataframe(result.zone_table, use_container_width=True)

    with tab2:
        st.subheader("Story response table")
        st.dataframe(result.story_table, use_container_width=True)
        st.pyplot(plot_story_response(result))
        st.pyplot(plot_story_drifts(result))

        csv_story = result.story_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download story table CSV", data=csv_story, file_name="story_response.csv", mime="text/csv")

    with tab3:
        st.subheader("Mode shapes")
        st.pyplot(plot_mode_shapes(result, n_modes=5))

    with tab4:
        st.subheader("Plan and elevation")
        zone_names = [z.zone_name for z in result.zone_results]
        selected_zone = st.selectbox("Plan zone", zone_names, index=0)
        st.pyplot(plot_plan(saved_inp, result, selected_zone))
        st.pyplot(plot_elevation(saved_inp, result))

    with tab5:
        st.subheader("Steel CHS outriggers")
        if result.outriggers:
            st.dataframe(result.outrigger_table, use_container_width=True)
        else:
            st.info("No outriggers defined in the current run.")

    with tab6:
        st.subheader("Text report")
        st.text_area("Report", st.session_state.report_text, height=500)
        st.download_button(
            "Download report TXT",
            data=st.session_state.report_text,
            file_name="tall_building_rational_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
