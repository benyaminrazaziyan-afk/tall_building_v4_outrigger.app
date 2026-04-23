from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Tall Building Rational Analysis + Auto Braced Outrigger"
APP_VERSION = "v9.0-thesis-rebuild"
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
    fy_mpa: float = 355.0

    dl_kn_m2: float = 3.0
    ll_kn_m2: float = 2.0
    superimposed_dead_kn_m2: float = 1.5
    facade_line_load_kn_m: float = 1.0
    live_load_mass_factor: float = 0.30
    seismic_base_shear_coeff: float = 0.045
    drift_limit_ratio: float = 1.0 / 500.0

    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.60
    beam_cracked_factor: float = 0.35
    slab_cracked_factor: float = 0.25
    perimeter_wall_cracked_factor: float = 0.45
    retaining_wall_cracked_factor: float = 0.50

    Ct: float = 0.0488
    x_period: float = 0.75
    Cu: float = 1.40
    auto_size: bool = True

    lower_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.55, 0.45, 0.80, 0.24, 1.00, 1.00, 0.90, 0.90, 0.80, 0.80, 8))
    middle_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.45, 0.40, 0.70, 0.22, 0.90, 0.90, 0.80, 0.80, 0.70, 0.70, 6))
    upper_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.35, 0.35, 0.60, 0.20, 0.75, 0.75, 0.65, 0.65, 0.55, 0.55, 4))

    lower_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.28, 14.0, 14.0, 12.0, 12.0))
    middle_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.24, 10.0, 10.0, 8.0, 8.0))
    upper_perimeter_walls: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.20, 6.0, 6.0, 5.0, 5.0))
    retaining_wall: RetainingWallInput = field(default_factory=RetainingWallInput)

    outrigger_story_levels: List[int] = field(default_factory=lambda: [30, 45])
    outrigger_depth_m: float = 3.0
    chs_outer_diameter_mm: float = 508.0
    chs_thickness_mm: float = 16.0
    brace_k_factor: float = 1.0
    brace_buckling_reduction: float = 0.85
    enable_x_outrigger: bool = True
    enable_y_outrigger: bool = True


@dataclass
class ZoneResult:
    zone: ZoneDefinition
    member: ZoneMemberInput
    perimeter_wall: PerimeterWallInput
    I_core_eff_m4: float
    K_core_story_N_m: float
    K_columns_story_N_m: float
    K_beams_story_N_m: float
    K_perimeter_walls_story_N_m: float
    K_slab_story_N_m: float
    K_zone_story_N_m: float


@dataclass
class OutriggerResult:
    story: int
    enabled_x: bool
    enabled_y: bool
    brace_area_m2: float
    brace_slenderness: float
    brace_length_m: float
    brace_axial_stiffness_N_m: float
    equivalent_story_stiffness_N_m: float
    generated_brace_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]


@dataclass
class ModalResult:
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[np.ndarray]
    story_displacements_m: np.ndarray
    story_drifts_m: np.ndarray
    story_drift_ratios: np.ndarray
    story_stiffness_N_m: np.ndarray
    story_masses_kg: np.ndarray


@dataclass
class AnalysisResult:
    inp: BuildingInput
    zones: List[ZoneDefinition]
    zone_results: List[ZoneResult]
    outriggers: List[OutriggerResult]
    modal: ModalResult
    story_table: pd.DataFrame
    zone_table: pd.DataFrame
    outrigger_table: pd.DataFrame
    summary_table: pd.DataFrame
    total_weight_kN: float
    T_code_s: float
    T_upper_s: float
    notes: List[str]


# ----------------------------- UTILITIES -----------------------------

def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, int(round(0.30 * n_story)))
    z2 = max(z1 + 1, int(round(0.70 * n_story)))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def floor_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m


def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height_m


def code_period(inp: BuildingInput) -> float:
    return inp.Ct * (total_height(inp) ** inp.x_period)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def zone_member_by_name(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    return {
        "Lower Zone": inp.lower_zone,
        "Middle Zone": inp.middle_zone,
        "Upper Zone": inp.upper_zone,
    }[zone_name]


def zone_perimeter_wall_by_name(inp: BuildingInput, zone_name: str) -> PerimeterWallInput:
    return {
        "Lower Zone": inp.lower_perimeter_walls,
        "Middle Zone": inp.middle_perimeter_walls,
        "Upper Zone": inp.upper_perimeter_walls,
    }[zone_name]


def zone_of_story(zones: List[ZoneDefinition], story: int) -> ZoneDefinition:
    for z in zones:
        if z.story_start <= story <= z.story_end:
            return z
    return zones[-1]


def chs_properties(do_mm: float, t_mm: float) -> Tuple[float, float]:
    do = max(do_mm, 1.0) / 1000.0
    t = clamp(t_mm / 1000.0, 0.001, do / 2.5)
    di = max(0.0, do - 2.0 * t)
    area = pi / 4.0 * (do**2 - di**2)
    I = pi / 64.0 * (do**4 - di**4)
    r = sqrt(I / max(area, 1e-12))
    return area, r


def rectangular_I_major(b: float, h: float) -> float:
    return b * h**3 / 12.0


def perimeter_wall_inertia(length: float, thickness: float, offset: float) -> float:
    area = length * thickness
    i_local = length * thickness**3 / 12.0
    return i_local + area * offset**2


def core_equivalent_I(inp: BuildingInput, member: ZoneMemberInput) -> float:
    ox = inp.core_outer_x_m
    oy = inp.core_outer_y_m
    t = member.wall_thickness_m
    wc = member.wall_count
    lengths = [ox, ox, oy, oy]
    if wc >= 6:
        lengths.extend([0.45 * ox, 0.45 * ox])
    if wc >= 8:
        lengths.extend([0.45 * oy, 0.45 * oy])

    x_side = ox / 2.0
    y_side = oy / 2.0
    I_x = 0.0
    I_y = 0.0

    I_x += perimeter_wall_inertia(lengths[0], t, +y_side)
    I_x += perimeter_wall_inertia(lengths[1], t, -y_side)
    I_y += t * lengths[0] ** 3 / 12.0 + t * lengths[1] ** 3 / 12.0

    I_y += perimeter_wall_inertia(lengths[2], t, -x_side)
    I_y += perimeter_wall_inertia(lengths[3], t, +x_side)
    I_x += t * lengths[2] ** 3 / 12.0 + t * lengths[3] ** 3 / 12.0

    if wc >= 6:
        inner_x = 0.22 * ox
        for length, offset in [(lengths[4], -inner_x), (lengths[5], inner_x)]:
            I_y += perimeter_wall_inertia(length, t, offset)
            I_x += t * length**3 / 12.0

    if wc >= 8:
        inner_y = 0.22 * oy
        for length, offset in [(lengths[6], -inner_y), (lengths[7], inner_y)]:
            I_x += perimeter_wall_inertia(length, t, offset)
            I_y += t * length**3 / 12.0

    return min(I_x, I_y)


def perimeter_wall_story_stiffness(inp: BuildingInput, pw: PerimeterWallInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    t = pw.thickness_m
    k = 0.0
    lengths_offsets = [
        (pw.top_length_m, inp.plan_y_m / 2.0),
        (pw.bottom_length_m, inp.plan_y_m / 2.0),
        (pw.left_length_m, inp.plan_x_m / 2.0),
        (pw.right_length_m, inp.plan_x_m / 2.0),
    ]
    for L, off in lengths_offsets:
        I = perimeter_wall_inertia(L, t, off)
        k += 12.0 * E * inp.perimeter_wall_cracked_factor * I / max(h**3, 1e-9)
    return k


def column_group_story_stiffness(inp: BuildingInput, member: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    n_total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    n_corner = 4
    n_perim = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    n_inner = max(0, n_total - n_corner - n_perim)

    Ic = max(rectangular_I_major(member.corner_col_x_m, member.corner_col_y_m), rectangular_I_major(member.corner_col_y_m, member.corner_col_x_m))
    Ip = max(rectangular_I_major(member.perimeter_col_x_m, member.perimeter_col_y_m), rectangular_I_major(member.perimeter_col_y_m, member.perimeter_col_x_m))
    Ii = max(rectangular_I_major(member.interior_col_x_m, member.interior_col_y_m), rectangular_I_major(member.interior_col_y_m, member.interior_col_x_m))
    I_sum = n_corner * Ic + n_perim * Ip + n_inner * Ii
    return 12.0 * E * inp.column_cracked_factor * I_sum / max(h**3, 1e-9)


def beam_group_story_stiffness(inp: BuildingInput, member: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    Lx = inp.bay_x_m
    Ly = inp.bay_y_m
    Ix = rectangular_I_major(member.beam_width_m, member.beam_depth_m)
    Iy = Ix
    n_beams_x = inp.n_bays_x * (inp.n_bays_y + 1)
    n_beams_y = inp.n_bays_y * (inp.n_bays_x + 1)
    kx = n_beams_x * 12.0 * E * inp.beam_cracked_factor * Ix / max(Lx**3, 1e-9)
    ky = n_beams_y * 12.0 * E * inp.beam_cracked_factor * Iy / max(Ly**3, 1e-9)
    return 0.12 * (kx + ky)


def slab_story_stiffness(inp: BuildingInput, member: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    t = member.slab_thickness_m
    A = floor_area(inp)
    span = max(inp.plan_x_m, inp.plan_y_m)
    I_eq = A * t**2 / 8.0
    return 0.015 * E * inp.slab_cracked_factor * I_eq / max(span, 1e-9)


def core_story_stiffness(inp: BuildingInput, member: ZoneMemberInput) -> Tuple[float, float]:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    I = core_equivalent_I(inp, member)
    k = 12.0 * E * inp.wall_cracked_factor * I / max(h**3, 1e-9)
    return I, k


def retaining_base_spring(inp: BuildingInput) -> float:
    rw = inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m)
    if not rw.enabled or inp.n_basement <= 0:
        return 0.0
    E = inp.Ec_mpa * 1e6
    h = max(inp.basement_height_m, 1e-6)
    terms = [
        perimeter_wall_inertia(rw.top_length_m, rw.thickness_m, inp.plan_y_m / 2.0),
        perimeter_wall_inertia(rw.bottom_length_m, rw.thickness_m, inp.plan_y_m / 2.0),
        perimeter_wall_inertia(rw.left_length_m, rw.thickness_m, inp.plan_x_m / 2.0),
        perimeter_wall_inertia(rw.right_length_m, rw.thickness_m, inp.plan_x_m / 2.0),
    ]
    return inp.n_basement * 12.0 * E * inp.retaining_wall_cracked_factor * sum(terms) / max(h**3, 1e-9)


def auto_generated_brace_lines(inp: BuildingInput) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    cy0 = (inp.plan_y_m - inp.core_outer_y_m) / 2.0
    cx1 = cx0 + inp.core_outer_x_m
    cy1 = cy0 + inp.core_outer_y_m
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if inp.enable_x_outrigger:
        lines.append(((cx0, inp.plan_y_m / 2.0), (0.0, inp.plan_y_m / 2.0)))
        lines.append(((cx1, inp.plan_y_m / 2.0), (inp.plan_x_m, inp.plan_y_m / 2.0)))
    if inp.enable_y_outrigger:
        lines.append(((inp.plan_x_m / 2.0, cy0), (inp.plan_x_m / 2.0, 0.0)))
        lines.append(((inp.plan_x_m / 2.0, cy1), (inp.plan_x_m / 2.0, inp.plan_y_m)))
    return lines


def outrigger_story_stiffness(inp: BuildingInput) -> OutriggerResult:
    E = inp.Es_mpa * 1e6
    A, r = chs_properties(inp.chs_outer_diameter_mm, inp.chs_thickness_mm)
    lines = auto_generated_brace_lines(inp)
    if not lines:
        return OutriggerResult(0, False, False, A, 0.0, 0.0, 0.0, 0.0, [])
    lengths = []
    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        lengths.append(sqrt(dx * dx + dy * dy + inp.outrigger_depth_m**2))
    L = float(np.mean(lengths))
    slenderness = inp.brace_k_factor * L / max(r, 1e-9)
    phi_b = clamp(inp.brace_buckling_reduction, 0.10, 1.0)
    axial_k = phi_b * len(lines) * E * A / max(L, 1e-9)
    depth_factor = max(inp.outrigger_depth_m, 0.5) / max(inp.story_height_m, 1e-9)
    eq_k = axial_k * (0.35 + 0.25 * depth_factor)
    return OutriggerResult(0, inp.enable_x_outrigger, inp.enable_y_outrigger, A, slenderness, L, axial_k, eq_k, lines)


def build_zone_results(inp: BuildingInput, zones: List[ZoneDefinition]) -> List[ZoneResult]:
    out: List[ZoneResult] = []
    for z in zones:
        member = zone_member_by_name(inp, z.name)
        pw = zone_perimeter_wall_by_name(inp, z.name)
        I_core, k_core = core_story_stiffness(inp, member)
        k_cols = column_group_story_stiffness(inp, member)
        k_beams = beam_group_story_stiffness(inp, member)
        k_pw = perimeter_wall_story_stiffness(inp, pw)
        k_slab = slab_story_stiffness(inp, member)
        out.append(
            ZoneResult(
                zone=z,
                member=member,
                perimeter_wall=pw,
                I_core_eff_m4=I_core * inp.wall_cracked_factor,
                K_core_story_N_m=k_core,
                K_columns_story_N_m=k_cols,
                K_beams_story_N_m=k_beams,
                K_perimeter_walls_story_N_m=k_pw,
                K_slab_story_N_m=k_slab,
                K_zone_story_N_m=k_core + k_cols + k_beams + k_pw + k_slab,
            )
        )
    return out


def story_mass_kg(inp: BuildingInput, slab_t: float) -> float:
    A = floor_area(inp)
    floor_dead = (inp.dl_kn_m2 + inp.superimposed_dead_kn_m2 + slab_t * CONCRETE_UNIT_WEIGHT) * A
    floor_live = inp.ll_kn_m2 * inp.live_load_mass_factor * A
    facade = inp.facade_line_load_kn_m * (2.0 * (inp.plan_x_m + inp.plan_y_m))
    W = floor_dead + floor_live + facade
    return W * 1000.0 / G


def build_story_arrays(inp: BuildingInput, zones: List[ZoneDefinition], zone_results: List[ZoneResult]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    k_stories = np.zeros(inp.n_story, dtype=float)
    m_stories = np.zeros(inp.n_story, dtype=float)
    rows = []
    zmap = {zr.zone.name: zr for zr in zone_results}
    ort = outrigger_story_stiffness(inp)
    outrigger_levels = sorted([s for s in inp.outrigger_story_levels if 1 <= s <= inp.n_story])
    base_spring = retaining_base_spring(inp)

    for s in range(1, inp.n_story + 1):
        z = zone_of_story(zones, s)
        zr = zmap[z.name]
        k = zr.K_zone_story_N_m
        if s in outrigger_levels:
            k += ort.equivalent_story_stiffness_N_m
        if s == 1:
            k += base_spring
        k_stories[s - 1] = k
        m = story_mass_kg(inp, zr.member.slab_thickness_m)
        m_stories[s - 1] = m
        rows.append(
            {
                "Story": s,
                "Zone": z.name,
                "Brace level": "Yes" if s in outrigger_levels else "No",
                "Wall t (m)": zr.member.wall_thickness_m,
                "Perimeter wall t (m)": zr.perimeter_wall.thickness_m,
                "Beam b (m)": zr.member.beam_width_m,
                "Beam h (m)": zr.member.beam_depth_m,
                "Slab t (m)": zr.member.slab_thickness_m,
                "Corner col (m)": f"{zr.member.corner_col_x_m:.2f}x{zr.member.corner_col_y_m:.2f}",
                "Perimeter col (m)": f"{zr.member.perimeter_col_x_m:.2f}x{zr.member.perimeter_col_y_m:.2f}",
                "Interior col (m)": f"{zr.member.interior_col_x_m:.2f}x{zr.member.interior_col_y_m:.2f}",
                "Story stiffness (GN/m)": k / 1e9,
                "Story mass (t)": m / 1000.0,
            }
        )
    return k_stories, m_stories, pd.DataFrame(rows)


def modal_analysis(inp: BuildingInput, k_stories: np.ndarray, m_stories: np.ndarray) -> ModalResult:
    n = len(k_stories)
    M = np.diag(m_stories)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = k_stories[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki

    evals, evecs = np.linalg.eig(np.linalg.solve(M, K))
    evals = np.real(evals)
    evecs = np.real(evecs)
    mask = evals > 1e-12
    evals = evals[mask]
    evecs = evecs[:, mask]
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    omegas = np.sqrt(evals)
    periods = [2.0 * pi / w for w in omegas[:5]]
    freqs = [w / (2.0 * pi) for w in omegas[:5]]
    modes = []
    for i in range(min(5, evecs.shape[1])):
        phi = evecs[:, i].copy()
        phi /= max(np.max(np.abs(phi)), 1e-12)
        if phi[-1] < 0:
            phi *= -1.0
        modes.append(phi)

    F = np.ones(n) * (inp.seismic_base_shear_coeff * np.sum(m_stories) * G / n)
    u = np.linalg.solve(K, F)
    drifts = np.diff(np.r_[0.0, u])
    drift_ratios = drifts / inp.story_height_m
    return ModalResult(periods, freqs, modes, u, drifts, drift_ratios, k_stories, m_stories)


def scale_zone_member(member: ZoneMemberInput, factor: float) -> ZoneMemberInput:
    f = factor
    return ZoneMemberInput(
        wall_thickness_m=clamp(member.wall_thickness_m * f, 0.25, 1.20),
        beam_width_m=clamp(member.beam_width_m * (0.92 + 0.08 * f), 0.30, 0.80),
        beam_depth_m=clamp(member.beam_depth_m * f, 0.45, 0.40 * 3.2),
        slab_thickness_m=clamp(member.slab_thickness_m * (0.95 + 0.05 * f), 0.18, 0.35),
        corner_col_x_m=clamp(member.corner_col_x_m * f, 0.50, 1.60),
        corner_col_y_m=clamp(member.corner_col_y_m * f, 0.50, 1.60),
        perimeter_col_x_m=clamp(member.perimeter_col_x_m * f, 0.45, 1.50),
        perimeter_col_y_m=clamp(member.perimeter_col_y_m * f, 0.45, 1.50),
        interior_col_x_m=clamp(member.interior_col_x_m * f, 0.40, 1.40),
        interior_col_y_m=clamp(member.interior_col_y_m * f, 0.40, 1.40),
        wall_count=member.wall_count,
    )


def scale_perimeter_wall(pw: PerimeterWallInput, factor: float) -> PerimeterWallInput:
    return PerimeterWallInput(
        thickness_m=clamp(pw.thickness_m * factor, 0.18, 0.60),
        top_length_m=pw.top_length_m,
        bottom_length_m=pw.bottom_length_m,
        left_length_m=pw.left_length_m,
        right_length_m=pw.right_length_m,
    )


def auto_size_input(inp: BuildingInput) -> BuildingInput:
    if not inp.auto_size:
        return inp
    work = replace(inp)
    zones = define_three_zones(inp.n_story)
    T_code = code_period(inp)
    T_limit = inp.Cu * T_code

    for _ in range(10):
        zone_results = build_zone_results(work, zones)
        k_stories, m_stories, _ = build_story_arrays(work, zones, zone_results)
        modal = modal_analysis(work, k_stories, m_stories)
        T1 = modal.periods_s[0]
        max_drift = float(np.max(np.abs(modal.story_drift_ratios)))

        need_stiffer = (T1 > 0.98 * T_limit) or (max_drift > inp.drift_limit_ratio)
        need_softer = (T1 < 0.68 * T_code) and (max_drift < 0.55 * inp.drift_limit_ratio)
        if not need_stiffer and not need_softer:
            break

        if need_stiffer:
            factor = clamp((T1 / max(0.90 * T_limit, 1e-6)) ** 0.25, 1.02, 1.12)
        else:
            factor = clamp((T1 / max(0.78 * T_code, 1e-6)) ** 0.25, 0.92, 0.98)

        work.lower_zone = scale_zone_member(work.lower_zone, factor)
        work.middle_zone = scale_zone_member(work.middle_zone, factor)
        work.upper_zone = scale_zone_member(work.upper_zone, factor)
        work.lower_perimeter_walls = scale_perimeter_wall(work.lower_perimeter_walls, factor)
        work.middle_perimeter_walls = scale_perimeter_wall(work.middle_perimeter_walls, factor)
        work.upper_perimeter_walls = scale_perimeter_wall(work.upper_perimeter_walls, factor)
    return work


def analyze(inp: BuildingInput) -> AnalysisResult:
    sized = auto_size_input(inp)
    zones = define_three_zones(sized.n_story)
    zone_results = build_zone_results(sized, zones)
    k_stories, m_stories, story_df = build_story_arrays(sized, zones, zone_results)
    modal = modal_analysis(sized, k_stories, m_stories)

    ort_proto = outrigger_story_stiffness(sized)
    outriggers = []
    for s in sorted([v for v in sized.outrigger_story_levels if 1 <= v <= sized.n_story]):
        outriggers.append(replace(ort_proto, story=s))

    total_weight_kN = float(np.sum(m_stories) * G / 1000.0)
    T_code = code_period(sized)
    T_upper = sized.Cu * T_code

    zone_rows = []
    for zr in zone_results:
        zone_rows.append(
            {
                "Zone": zr.zone.name,
                "Stories": f"{zr.zone.story_start}-{zr.zone.story_end}",
                "Core size (m)": f"{sized.core_outer_x_m:.1f}x{sized.core_outer_y_m:.1f}",
                "Core opening (m)": f"{sized.core_opening_x_m:.1f}x{sized.core_opening_y_m:.1f}",
                "Core wall t (m)": zr.member.wall_thickness_m,
                "Perimeter wall t (m)": zr.perimeter_wall.thickness_m,
                "Beam b x h (m)": f"{zr.member.beam_width_m:.2f}x{zr.member.beam_depth_m:.2f}",
                "Slab t (m)": zr.member.slab_thickness_m,
                "Corner col (m)": f"{zr.member.corner_col_x_m:.2f}x{zr.member.corner_col_y_m:.2f}",
                "Perimeter col (m)": f"{zr.member.perimeter_col_x_m:.2f}x{zr.member.perimeter_col_y_m:.2f}",
                "Interior col (m)": f"{zr.member.interior_col_x_m:.2f}x{zr.member.interior_col_y_m:.2f}",
                "K_core (GN/m)": zr.K_core_story_N_m / 1e9,
                "K_cols (GN/m)": zr.K_columns_story_N_m / 1e9,
                "K_beams (GN/m)": zr.K_beams_story_N_m / 1e9,
                "K_perim walls (GN/m)": zr.K_perimeter_walls_story_N_m / 1e9,
                "K_slab (GN/m)": zr.K_slab_story_N_m / 1e9,
                "K_zone total (GN/m)": zr.K_zone_story_N_m / 1e9,
            }
        )
    zone_df = pd.DataFrame(zone_rows)

    outrigger_rows = []
    for o in outriggers:
        outrigger_rows.append(
            {
                "Story": o.story,
                "Directions": ("X " if o.enabled_x else "") + ("Y" if o.enabled_y else ""),
                "Auto-generated brace lines": len(o.generated_brace_lines),
                "CHS area (cm2)": o.brace_area_m2 * 1e4,
                "Brace length (m)": o.brace_length_m,
                "KL/r": o.brace_slenderness,
                "Axial brace stiffness (MN/m)": o.brace_axial_stiffness_N_m / 1e6,
                "Equivalent story stiffness (GN/m)": o.equivalent_story_stiffness_N_m / 1e9,
            }
        )
    outrigger_df = pd.DataFrame(outrigger_rows)

    summary = pd.DataFrame(
        {
            "Parameter": [
                "Stories",
                "Basements",
                "Total height (m)",
                "Plan (m)",
                "Core outer (m)",
                "Core opening (m)",
                "T1 modal (s)",
                "T2 modal (s)",
                "T3 modal (s)",
                "Code reference period T_code (s)",
                "Upper code period Cu*T_code (s)",
                "Roof displacement (m)",
                "Max story drift ratio",
                "Base shear coefficient used",
                "Total seismic weight (kN)",
                "Retaining wall active",
            ],
            "Value": [
                sized.n_story,
                sized.n_basement,
                total_height(sized),
                f"{sized.plan_x_m:.1f} x {sized.plan_y_m:.1f}",
                f"{sized.core_outer_x_m:.1f} x {sized.core_outer_y_m:.1f}",
                f"{sized.core_opening_x_m:.1f} x {sized.core_opening_y_m:.1f}",
                modal.periods_s[0] if len(modal.periods_s) > 0 else np.nan,
                modal.periods_s[1] if len(modal.periods_s) > 1 else np.nan,
                modal.periods_s[2] if len(modal.periods_s) > 2 else np.nan,
                T_code,
                T_upper,
                float(modal.story_displacements_m[-1]),
                float(np.max(np.abs(modal.story_drift_ratios))),
                sized.seismic_base_shear_coeff,
                total_weight_kN,
                "Yes" if sized.retaining_wall.enabled and sized.n_basement > 0 else "No",
            ],
        }
    )

    notes = [
        "Auto-sizing no longer asks the user for target first-mode period.",
        "If auto-size is active, members are resized only against code-reference period ceiling and drift limit.",
        "Brace layout is auto-generated from core-to-perimeter geometry; braces-per-side is not a user input.",
        "Modal periods are outputs from eigenvalue analysis, not user-entered targets.",
    ]

    return AnalysisResult(
        inp=sized,
        zones=zones,
        zone_results=zone_results,
        outriggers=outriggers,
        modal=modal,
        story_table=story_df,
        zone_table=zone_df,
        outrigger_table=outrigger_df,
        summary_table=summary,
        total_weight_kN=total_weight_kN,
        T_code_s=T_code,
        T_upper_s=T_upper,
        notes=notes,
    )


# ----------------------------- PLOTS -----------------------------

def plot_plan(result: AnalysisResult, show_outrigger_story: int | None = None) -> plt.Figure:
    inp = result.inp
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black", linewidth=1.5)
    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#d0d0d0", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#d0d0d0", linewidth=0.8)

    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    cy0 = (inp.plan_y_m - inp.core_outer_y_m) / 2.0
    cx1 = cx0 + inp.core_outer_x_m
    cy1 = cy0 + inp.core_outer_y_m
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core_outer_x_m, inp.core_outer_y_m, fill=False, linewidth=2.2, edgecolor="#0b6e4f"))
    ax.text(inp.plan_x_m / 2.0, cy1 + 0.6, f"Core {inp.core_outer_x_m:.1f} x {inp.core_outer_y_m:.1f} m", ha="center", color="#0b6e4f")

    ix0 = (inp.plan_x_m - inp.core_opening_x_m) / 2.0
    iy0 = (inp.plan_y_m - inp.core_opening_y_m) / 2.0
    ax.add_patch(plt.Rectangle((ix0, iy0), inp.core_opening_x_m, inp.core_opening_y_m, fill=False, linewidth=1.4, linestyle="--", edgecolor="#1f77b4"))

    zmap = {zr.zone.name: zr for zr in result.zone_results}
    lower = zmap["Lower Zone"].perimeter_wall
    t = lower.thickness_m
    sx = (inp.plan_x_m - lower.top_length_m) / 2.0
    sy = (inp.plan_y_m - lower.left_length_m) / 2.0
    color_pw = "#7a3e00"
    ax.add_patch(plt.Rectangle((sx, inp.plan_y_m - t), lower.top_length_m, t, color=color_pw, alpha=0.35))
    ax.add_patch(plt.Rectangle((sx, 0.0), lower.bottom_length_m, t, color=color_pw, alpha=0.35))
    ax.add_patch(plt.Rectangle((0.0, sy), t, lower.left_length_m, color=color_pw, alpha=0.35))
    ax.add_patch(plt.Rectangle((inp.plan_x_m - t, sy), t, lower.right_length_m, color=color_pw, alpha=0.35))

    rw = inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m)
    if rw.enabled and inp.n_basement > 0:
        ax.plot([0, inp.plan_x_m], [0, 0], color="#6a0dad", linewidth=2.0)
        ax.plot([0, 0], [0, inp.plan_y_m], color="#6a0dad", linewidth=2.0)
        ax.plot([inp.plan_x_m, inp.plan_x_m], [0, inp.plan_y_m], color="#6a0dad", linewidth=2.0)
        ax.plot([0, inp.plan_x_m], [inp.plan_y_m, inp.plan_y_m], color="#6a0dad", linewidth=2.0)

    active_story = show_outrigger_story or (result.outriggers[0].story if result.outriggers else None)
    if active_story is not None and any(o.story == active_story for o in result.outriggers):
        lines = result.outriggers[0].generated_brace_lines if result.outriggers else []
        for (x1, y1), (x2, y2) in lines:
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=2.2)
        ax.text(inp.plan_x_m / 2.0, -1.3, f"Outrigger plan line logic at story {active_story}", ha="center", color="red")

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, inp.plan_x_m + 1.5)
    ax.set_ylim(-2.0, inp.plan_y_m + 2.0)
    ax.set_title("Plan: core, perimeter walls, retaining wall, auto-generated outrigger lines")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig


def plot_elevation(result: AnalysisResult) -> plt.Figure:
    inp = result.inp
    H = total_height(inp)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot([0, 0, inp.plan_x_m, inp.plan_x_m, 0], [0, H, H, 0, 0], color="black", linewidth=1.2)
    colors = {"Lower Zone": "#d9ead3", "Middle Zone": "#fff2cc", "Upper Zone": "#cfe2f3"}
    for z in result.zones:
        y0 = (z.story_start - 1) * inp.story_height_m
        y1 = z.story_end * inp.story_height_m
        ax.add_patch(plt.Rectangle((0, y0), inp.plan_x_m, y1 - y0, color=colors[z.name], alpha=0.6))
        ax.text(inp.plan_x_m + 0.5, (y0 + y1) / 2.0, f"{z.name}\n{z.story_start}-{z.story_end}", va="center")
    for s in result.inp.outrigger_story_levels:
        if 1 <= s <= inp.n_story:
            y = s * inp.story_height_m
            ax.plot([0, inp.plan_x_m], [y, y], color="red", linewidth=2.4)
            ax.text(inp.plan_x_m / 2.0, y + 0.2, f"Outrigger story {s}", color="red", ha="center")
    if inp.n_basement > 0 and inp.retaining_wall.enabled:
        base = -inp.n_basement * inp.basement_height_m
        ax.add_patch(plt.Rectangle((0, base), inp.plan_x_m, -base, color="#ead1dc", alpha=0.6))
        ax.text(inp.plan_x_m + 0.5, base / 2.0, f"Basement\n{inp.n_basement} levels\nretaining wall active", va="center", color="#6a0dad")
    ax.set_xlim(-1.0, inp.plan_x_m + 12.0)
    ax.set_ylim(-inp.n_basement * inp.basement_height_m - 1.0, H + 1.0)
    ax.set_title("Elevation: zones, outrigger levels, basement retaining wall")
    ax.set_xlabel("Representative building width (m)")
    ax.set_ylabel("Elevation (m)")
    return fig


def plot_mode_shape(result: AnalysisResult, mode_index: int) -> plt.Figure:
    modal = result.modal
    idx = clamp(mode_index, 1, len(modal.mode_shapes)) - 1
    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(1, len(modal.mode_shapes[idx]) + 1) * result.inp.story_height_m
    ax.plot(modal.mode_shapes[idx], y, marker="o")
    ax.set_title(f"Mode {idx + 1} shape  |  T = {modal.periods_s[idx]:.3f} s")
    ax.set_xlabel("Normalized displacement")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_story_drifts(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(1, len(result.modal.story_drift_ratios) + 1)
    ax.plot(result.modal.story_drift_ratios, y, marker="o")
    ax.set_title("Story drift ratios")
    ax.set_xlabel("Drift ratio")
    ax.set_ylabel("Story")
    ax.grid(True, alpha=0.3)
    return fig


# ----------------------------- STREAMLIT UI -----------------------------

def zone_block(label: str, defaults: ZoneMemberInput) -> ZoneMemberInput:
    st.sidebar.markdown(f"**{label} zone members**")
    c1, c2 = st.sidebar.columns(2)
    wall_t = c1.number_input(f"{label} wall t (m)", 0.20, 1.20, defaults.wall_thickness_m, 0.01)
    wall_count = int(c2.number_input(f"{label} wall count", 4, 8, defaults.wall_count, 1))
    beam_b = c1.number_input(f"{label} beam b (m)", 0.25, 1.00, defaults.beam_width_m, 0.01)
    beam_h = c2.number_input(f"{label} beam h (m)", 0.40, 1.40, defaults.beam_depth_m, 0.01)
    slab_t = c1.number_input(f"{label} slab t (m)", 0.16, 0.40, defaults.slab_thickness_m, 0.01)
    ccx = c1.number_input(f"{label} corner col X (m)", 0.40, 2.00, defaults.corner_col_x_m, 0.01)
    ccy = c2.number_input(f"{label} corner col Y (m)", 0.40, 2.00, defaults.corner_col_y_m, 0.01)
    pcx = c1.number_input(f"{label} perim col X (m)", 0.35, 2.00, defaults.perimeter_col_x_m, 0.01)
    pcy = c2.number_input(f"{label} perim col Y (m)", 0.35, 2.00, defaults.perimeter_col_y_m, 0.01)
    icx = c1.number_input(f"{label} inner col X (m)", 0.30, 2.00, defaults.interior_col_x_m, 0.01)
    icy = c2.number_input(f"{label} inner col Y (m)", 0.30, 2.00, defaults.interior_col_y_m, 0.01)
    return ZoneMemberInput(wall_t, beam_b, beam_h, slab_t, ccx, ccy, pcx, pcy, icx, icy, wall_count)


def perimeter_wall_block(label: str, defaults: PerimeterWallInput) -> PerimeterWallInput:
    st.sidebar.markdown(f"**{label} perimeter walls**")
    c1, c2 = st.sidebar.columns(2)
    t = c1.number_input(f"{label} wall t (m)", 0.15, 0.80, defaults.thickness_m, 0.01)
    top = c1.number_input(f"{label} top length (m)", 0.0, 100.0, defaults.top_length_m, 0.5)
    bottom = c2.number_input(f"{label} bottom length (m)", 0.0, 100.0, defaults.bottom_length_m, 0.5)
    left = c1.number_input(f"{label} left length (m)", 0.0, 100.0, defaults.left_length_m, 0.5)
    right = c2.number_input(f"{label} right length (m)", 0.0, 100.0, defaults.right_length_m, 0.5)
    return PerimeterWallInput(t, top, bottom, left, right)


def streamlit_input_panel() -> BuildingInput:
    d = BuildingInput()
    st.sidebar.header("Input Data")

    st.sidebar.subheader("Geometry")
    n_story = int(st.sidebar.number_input("Above-grade stories", 10, 120, d.n_story, 1))
    n_basement = int(st.sidebar.number_input("Basement stories", 0, 10, d.n_basement, 1))
    story_h = st.sidebar.number_input("Story height (m)", 2.8, 5.0, d.story_height_m, 0.1)
    basement_h = st.sidebar.number_input("Basement height (m)", 2.8, 6.0, d.basement_height_m, 0.1)
    plan_x = st.sidebar.number_input("Plan X (m)", 20.0, 120.0, d.plan_x_m, 0.5)
    plan_y = st.sidebar.number_input("Plan Y (m)", 20.0, 120.0, d.plan_y_m, 0.5)
    n_bays_x = int(st.sidebar.number_input("Bays in X", 2, 20, d.n_bays_x, 1))
    n_bays_y = int(st.sidebar.number_input("Bays in Y", 2, 20, d.n_bays_y, 1))
    bay_x = st.sidebar.number_input("Bay X (m)", 4.0, 15.0, d.bay_x_m, 0.1)
    bay_y = st.sidebar.number_input("Bay Y (m)", 4.0, 15.0, d.bay_y_m, 0.1)

    st.sidebar.subheader("Core")
    core_x = st.sidebar.number_input("Core outer X (m)", 8.0, 50.0, d.core_outer_x_m, 0.5)
    core_y = st.sidebar.number_input("Core outer Y (m)", 8.0, 50.0, d.core_outer_y_m, 0.5)
    open_x = st.sidebar.number_input("Core opening X (m)", 4.0, 40.0, d.core_opening_x_m, 0.5)
    open_y = st.sidebar.number_input("Core opening Y (m)", 4.0, 40.0, d.core_opening_y_m, 0.5)

    st.sidebar.subheader("Materials / loads")
    Ec = st.sidebar.number_input("Ec (MPa)", 20000.0, 45000.0, d.Ec_mpa, 500.0)
    Es = st.sidebar.number_input("Es (MPa)", 180000.0, 220000.0, d.Es_mpa, 1000.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 235.0, 500.0, d.fy_mpa, 5.0)
    dl = st.sidebar.number_input("DL (kN/m2)", 0.0, 10.0, d.dl_kn_m2, 0.1)
    ll = st.sidebar.number_input("LL (kN/m2)", 0.0, 10.0, d.ll_kn_m2, 0.1)
    sidl = st.sidebar.number_input("Superimposed dead load (kN/m2)", 0.0, 10.0, d.superimposed_dead_kn_m2, 0.1)
    facade = st.sidebar.number_input("Facade line load (kN/m)", 0.0, 10.0, d.facade_line_load_kn_m, 0.1)
    vb = st.sidebar.number_input("Equivalent base shear coeff.", 0.01, 0.20, d.seismic_base_shear_coeff, 0.005, format="%.3f")
    drift_lim = st.sidebar.number_input("Drift limit ratio (1/x)", 200.0, 1000.0, 500.0, 10.0)

    st.sidebar.subheader("Cracked factors")
    wall_cf = st.sidebar.number_input("Wall cracked factor", 0.10, 1.00, d.wall_cracked_factor, 0.05)
    col_cf = st.sidebar.number_input("Column cracked factor", 0.10, 1.00, d.column_cracked_factor, 0.05)
    beam_cf = st.sidebar.number_input("Beam cracked factor", 0.10, 1.00, d.beam_cracked_factor, 0.05)
    slab_cf = st.sidebar.number_input("Slab cracked factor", 0.05, 1.00, d.slab_cracked_factor, 0.05)
    pw_cf = st.sidebar.number_input("Perimeter wall cracked factor", 0.10, 1.00, d.perimeter_wall_cracked_factor, 0.05)
    rw_cf = st.sidebar.number_input("Retaining wall cracked factor", 0.10, 1.00, d.retaining_wall_cracked_factor, 0.05)
    auto_size = st.sidebar.checkbox("Auto-size members to satisfy drift + code-reference period ceiling", value=d.auto_size)

    lower = zone_block("Lower", d.lower_zone)
    middle = zone_block("Middle", d.middle_zone)
    upper = zone_block("Upper", d.upper_zone)

    lower_pw = perimeter_wall_block("Lower", d.lower_perimeter_walls)
    middle_pw = perimeter_wall_block("Middle", d.middle_perimeter_walls)
    upper_pw = perimeter_wall_block("Upper", d.upper_perimeter_walls)

    st.sidebar.subheader("Basement retaining wall")
    rw_enabled = st.sidebar.checkbox("Enable retaining wall", value=d.retaining_wall.enabled)
    rw_t = st.sidebar.number_input("Retaining wall thickness (m)", 0.20, 1.00, d.retaining_wall.thickness_m, 0.01)

    st.sidebar.subheader("Outrigger steel CHS")
    default_levels = ",".join(str(v) for v in d.outrigger_story_levels)
    levels_txt = st.sidebar.text_input("Outrigger stories (comma-separated)", value=default_levels)
    depth = st.sidebar.number_input("Outrigger depth (m)", 1.0, 8.0, d.outrigger_depth_m, 0.1)
    chs_do = st.sidebar.number_input("CHS outer diameter (mm)", 168.0, 1000.0, d.chs_outer_diameter_mm, 1.0)
    chs_t = st.sidebar.number_input("CHS thickness (mm)", 6.0, 50.0, d.chs_thickness_mm, 1.0)
    kfac = st.sidebar.number_input("Brace K factor", 0.5, 2.0, d.brace_k_factor, 0.05)
    buck_red = st.sidebar.number_input("Brace buckling reduction", 0.10, 1.00, d.brace_buckling_reduction, 0.05)
    ex = st.sidebar.checkbox("Enable X-direction outrigger lines", value=d.enable_x_outrigger)
    ey = st.sidebar.checkbox("Enable Y-direction outrigger lines", value=d.enable_y_outrigger)

    levels = []
    for token in levels_txt.split(","):
        token = token.strip()
        if token.isdigit():
            v = int(token)
            if 1 <= v <= n_story:
                levels.append(v)
    levels = sorted(list(dict.fromkeys(levels)))

    return BuildingInput(
        n_story=n_story,
        n_basement=n_basement,
        story_height_m=story_h,
        basement_height_m=basement_h,
        plan_x_m=plan_x,
        plan_y_m=plan_y,
        n_bays_x=n_bays_x,
        n_bays_y=n_bays_y,
        bay_x_m=bay_x,
        bay_y_m=bay_y,
        core_outer_x_m=core_x,
        core_outer_y_m=core_y,
        core_opening_x_m=min(open_x, core_x - 1.0),
        core_opening_y_m=min(open_y, core_y - 1.0),
        Ec_mpa=Ec,
        Es_mpa=Es,
        fy_mpa=fy,
        dl_kn_m2=dl,
        ll_kn_m2=ll,
        superimposed_dead_kn_m2=sidl,
        facade_line_load_kn_m=facade,
        seismic_base_shear_coeff=vb,
        drift_limit_ratio=1.0 / drift_lim,
        wall_cracked_factor=wall_cf,
        column_cracked_factor=col_cf,
        beam_cracked_factor=beam_cf,
        slab_cracked_factor=slab_cf,
        perimeter_wall_cracked_factor=pw_cf,
        retaining_wall_cracked_factor=rw_cf,
        auto_size=auto_size,
        lower_zone=lower,
        middle_zone=middle,
        upper_zone=upper,
        lower_perimeter_walls=lower_pw,
        middle_perimeter_walls=middle_pw,
        upper_perimeter_walls=upper_pw,
        retaining_wall=RetainingWallInput(enabled=rw_enabled, thickness_m=rw_t),
        outrigger_story_levels=levels,
        outrigger_depth_m=depth,
        chs_outer_diameter_mm=chs_do,
        chs_thickness_mm=chs_t,
        brace_k_factor=kfac,
        brace_buckling_reduction=buck_red,
        enable_x_outrigger=ex,
        enable_y_outrigger=ey,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info(
        "This rebuild removes user-entered target first-mode period and user-entered braces-per-side. "
        "Outrigger lines are auto-generated from core-to-perimeter geometry, and member auto-sizing uses only drift limit and code-reference period ceiling."
    )

    inp = streamlit_input_panel()
    if st.button("Analyze"):
        try:
            result = analyze(inp)
            tabs = st.tabs(["Summary", "Zones", "Stories", "Outriggers", "Plan", "Elevation", "Modes", "Drifts"])

            with tabs[0]:
                st.dataframe(result.summary_table, use_container_width=True)
                for note in result.notes:
                    st.write(f"- {note}")

            with tabs[1]:
                st.dataframe(result.zone_table, use_container_width=True)

            with tabs[2]:
                st.dataframe(result.story_table, use_container_width=True, height=600)

            with tabs[3]:
                if len(result.outrigger_table):
                    st.dataframe(result.outrigger_table, use_container_width=True)
                else:
                    st.warning("No outrigger levels are currently active.")

            with tabs[4]:
                selected = None
                if result.outriggers:
                    selected = st.selectbox("Outrigger story shown in plan", [o.story for o in result.outriggers])
                st.pyplot(plot_plan(result, selected))

            with tabs[5]:
                st.pyplot(plot_elevation(result))

            with tabs[6]:
                mode_num = st.selectbox("Mode", list(range(1, len(result.modal.mode_shapes) + 1)))
                st.pyplot(plot_mode_shape(result, mode_num))

            with tabs[7]:
                st.pyplot(plot_story_drifts(result))

        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
