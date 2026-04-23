from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st

G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0   # kN/m³
STEEL_DENSITY        = 7850.0  # kg/m³
APP_TITLE   = "Tall Building Rational Analysis + Steel Braced Outrigger"
APP_VERSION = "v9.0-professional"

# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════

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
    n_basement: int = 2
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

    # Lateral (perimeter shear) walls at facade — X and Y direction counts
    lateral_wall_x_count: int = 2   # e.g. 2 shear walls in X direction on facade
    lateral_wall_y_count: int = 2
    lateral_wall_thickness_m: float = 0.35
    lateral_wall_length_x_m: float = 6.0   # plan length of each wall in X-face
    lateral_wall_length_y_m: float = 6.0

    # Braced bay positions (bay indices, 0-based)
    braced_bays_x: List[int] = field(default_factory=lambda: [0, 5])   # bay col indices
    braced_bays_y: List[int] = field(default_factory=lambda: [0, 5])
    braced_bay_stiffness_factor: float = 1.5  # multiplier on column stiffness for braced bay

    # Basement retaining wall
    retaining_wall_thickness_m: float = 0.50
    retaining_wall_height_m: float = 3.0      # = basement_height_m typically
    soil_unit_weight_kn_m3: float = 18.0
    soil_ka: float = 0.33                     # active earth pressure coeff
    surcharge_kn_m2: float = 10.0

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

    # Response spectrum parameters (for T1-based base shear override)
    use_response_spectrum: bool = False
    spectrum_T0_s: float = 0.20   # corner period (short)
    spectrum_Ts_s: float = 0.60   # corner period (long)
    spectrum_Sa_max_g: float = 0.40  # plateau Sa/g
    spectrum_Sa_min_g: float = 0.05  # minimum Sa/g
    response_modification_R: float = 6.0
    importance_factor_I: float = 1.0

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
    lower_zone:  ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.60, 1.20, 1.20, 1.00, 1.00, 0.90, 0.90))
    middle_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.45, 1.00, 1.00, 0.85, 0.85, 0.75, 0.75))
    upper_zone:  ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(0.35, 0.85, 0.85, 0.75, 0.75, 0.65, 0.65))

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

    # Limits
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
    lateral_wall_stiffness_n_m: float
    diaphragm_stiffness_n_m: float
    total_story_stiffness_n_m: float


@dataclass
class RetainingWallResult:
    height_m: float
    thickness_m: float
    active_pressure_top_kn_m2: float
    active_pressure_bot_kn_m2: float
    total_thrust_kn_m: float
    moment_base_kn_m_m: float
    required_d_mm: float      # minimum d from bending
    required_As_mm2_m: float  # steel area per metre


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
    base_shear_method: str        # "Coefficient" or "Spectrum"
    spectrum_Sa_g: float          # Sa/g at T1 (0 if not used)
    total_weight_kn: float
    total_mass_kg: float
    k_wall_total_n_m: float
    k_column_total_n_m: float
    k_beam_total_n_m: float
    k_lateral_wall_total_n_m: float
    k_outrigger_total_n_m: float
    k_diaphragm_median_n_m: float
    outriggers: List[OutriggerResult]
    zone_results: List[ZoneStiffnessResult]
    retaining_wall: Optional[RetainingWallResult]
    zone_table: pd.DataFrame
    story_table: pd.DataFrame
    outrigger_table: pd.DataFrame
    retaining_table: pd.DataFrame
    summary_table: pd.DataFrame


# ═══════════════════════════════════════════════════════════════════
# GEOMETRY / MATERIAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone",  1,      z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone",  z2 + 1, n_story),
    ]


def get_zone_input(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    return {"Lower Zone": inp.lower_zone, "Middle Zone": inp.middle_zone, "Upper Zone": inp.upper_zone}[zone_name]


def get_wall_count(inp: BuildingInput, zone_name: str) -> int:
    return {"Lower Zone": inp.lower_zone_wall_count, "Middle Zone": inp.middle_zone_wall_count,
            "Upper Zone": inp.upper_zone_wall_count}[zone_name]


def gross_floor_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m


def required_service_opening_area(inp: BuildingInput) -> float:
    return (inp.elevator_count * inp.elevator_area_each_m2
            + inp.stair_count * inp.stair_area_each_m2
            + inp.service_area_m2) * inp.corridor_factor


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


def circular_hollow_area(do_m, t_m):
    di = max(do_m - 2.0 * t_m, 1e-6)
    return (pi / 4.0) * (do_m**2 - di**2)


def circular_hollow_inertia(do_m, t_m):
    di = max(do_m - 2.0 * t_m, 1e-6)
    return (pi / 64.0) * (do_m**4 - di**4)


def circular_hollow_radius_of_gyration(do_m, t_m):
    area = circular_hollow_area(do_m, t_m)
    I = circular_hollow_inertia(do_m, t_m)
    return sqrt(I / max(area, 1e-12))


def concrete_shear_modulus_pa(inp: BuildingInput) -> float:
    return inp.Ec_mpa * 1e6 / (2.0 * (1.0 + inp.nu_concrete))


# ═══════════════════════════════════════════════════════════════════
# CORE WALL STIFFNESS
# ═══════════════════════════════════════════════════════════════════

def wall_lengths_for_core(inp: BuildingInput, wall_count: int) -> List[float]:
    ox, oy = inp.core_outer_x_m, inp.core_outer_y_m
    if wall_count == 4:
        return [ox, ox, oy, oy]
    if wall_count == 6:
        return [ox, ox, oy, oy, 0.45*ox, 0.45*ox]
    return [ox, ox, oy, oy, 0.45*ox, 0.45*ox, 0.45*oy, 0.45*oy]


def core_min_inertia(inp: BuildingInput, t: float, wall_count: int) -> float:
    lengths = wall_lengths_for_core(inp, wall_count)
    ox, oy = inp.core_outer_x_m, inp.core_outer_y_m
    x_side, y_side = ox/2.0, oy/2.0
    Ix = Iy = 0.0
    for L in lengths[0:2]:
        Ix += rect_inertia_x(L, t) + L*t*(y_side**2)
        Iy += rect_inertia_y(L, t)
    for L in lengths[2:4]:
        Iy += rect_inertia_y(t, L) + L*t*(x_side**2)
        Ix += rect_inertia_x(t, L)
    if wall_count >= 6:
        inner_x = 0.22*ox
        for offset, L in zip([-inner_x, inner_x], lengths[4:6]):
            Iy += rect_inertia_y(t, L) + L*t*offset**2
            Ix += rect_inertia_x(t, L)
    if wall_count >= 8:
        inner_y = 0.22*oy
        for offset, L in zip([-inner_y, inner_y], lengths[6:8]):
            Ix += rect_inertia_x(L, t) + L*t*offset**2
            Iy += rect_inertia_y(L, t)
    return min(Ix, Iy)


def wall_story_stiffness_n_m(inp: BuildingInput, wall_count: int, t: float) -> float:
    E  = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus_pa(inp)
    h  = inp.story_height_m
    Ieff = inp.wall_cracked_factor * core_min_inertia(inp, t, wall_count)
    total_wall_area = sum(L*t for L in wall_lengths_for_core(inp, wall_count))
    kb = 12.0 * E * Ieff / max(h**3, 1e-12)
    ks = 1.2  * Gc * total_wall_area / max(h, 1e-12)
    return 1.0 / max((1.0/max(kb,1e-12)) + (1.0/max(ks,1e-12)), 1e-18)


# ═══════════════════════════════════════════════════════════════════
# LATERAL (PERIMETER) SHEAR WALLS
# ═══════════════════════════════════════════════════════════════════

def lateral_wall_story_stiffness_n_m(inp: BuildingInput) -> float:
    """Cantilever stiffness of perimeter shear walls (both directions combined)."""
    E  = inp.Ec_mpa * 1e6
    Gc = concrete_shear_modulus_pa(inp)
    h  = inp.story_height_m
    t  = inp.lateral_wall_thickness_m

    def one_wall_k(Lw: float) -> float:
        Ieff = inp.wall_cracked_factor * rect_inertia_x(Lw, t)
        Aeff = Lw * t
        kb = 12.0 * E * Ieff / max(h**3, 1e-12)
        ks = 1.2 * Gc * Aeff / max(h, 1e-12)
        return 1.0 / max((1.0/max(kb,1e-12)) + (1.0/max(ks,1e-12)), 1e-18)

    kx = inp.lateral_wall_x_count * one_wall_k(inp.lateral_wall_length_x_m)
    ky = inp.lateral_wall_y_count * one_wall_k(inp.lateral_wall_length_y_m)
    return kx + ky


# ═══════════════════════════════════════════════════════════════════
# FRAME STIFFNESS (columns + beams)
# ═══════════════════════════════════════════════════════════════════

def frame_story_stiffness_from_columns(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E   = inp.Ec_mpa * 1e6
    h   = inp.story_height_m
    total_cols    = (inp.n_bays_x+1)*(inp.n_bays_y+1)
    corner_cols   = 4
    perimeter_cols= max(0, 2*(inp.n_bays_x-1) + 2*(inp.n_bays_y-1))
    interior_cols = max(0, total_cols - corner_cols - perimeter_cols)

    I_corner= min(rect_inertia_x(z.corner_col_x_m,   z.corner_col_y_m),
                  rect_inertia_y(z.corner_col_x_m,   z.corner_col_y_m))
    I_perim = min(rect_inertia_x(z.perimeter_col_x_m, z.perimeter_col_y_m),
                  rect_inertia_y(z.perimeter_col_x_m, z.perimeter_col_y_m))
    I_inter = min(rect_inertia_x(z.interior_col_x_m,  z.interior_col_y_m),
                  rect_inertia_y(z.interior_col_x_m,  z.interior_col_y_m))

    I_total = inp.column_cracked_factor * (
        corner_cols*I_corner + perimeter_cols*I_perim + interior_cols*I_inter
    )
    k_base = 12.0 * E * I_total / max(h**3, 1e-12)

    # Braced bay bonus: braced bays have diagonal, treated as shear stiffness amplifier
    n_braced_bays = len(inp.braced_bays_x) + len(inp.braced_bays_y)
    k_braced_bonus = (inp.braced_bay_stiffness_factor - 1.0) * n_braced_bays * (
        inp.column_cracked_factor * I_corner * 12.0 * E / max(h**3, 1e-12)
    )
    return k_base + k_braced_bonus


def beam_rotational_restraint_factor(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E  = inp.Ec_mpa * 1e6
    h  = inp.story_height_m
    Ib = inp.beam_cracked_factor * rect_inertia_x(inp.beam_width_m, inp.beam_depth_m)

    n_beams_x = inp.n_bays_x * (inp.n_bays_y+1)
    n_beams_y = inp.n_bays_y * (inp.n_bays_x+1)
    kb = (n_beams_x*(4.0*E*Ib/max(inp.bay_x_m, 1e-12))
         + n_beams_y*(4.0*E*Ib/max(inp.bay_y_m, 1e-12)))

    total_cols    = (inp.n_bays_x+1)*(inp.n_bays_y+1)
    corner_cols   = 4
    perimeter_cols= max(0, 2*(inp.n_bays_x-1)+2*(inp.n_bays_y-1))
    interior_cols = max(0, total_cols-corner_cols-perimeter_cols)

    kc  = corner_cols   * (4.0*E*inp.column_cracked_factor*rect_inertia_x(z.corner_col_x_m,   z.corner_col_y_m) / max(h,1e-12))
    kc += perimeter_cols* (4.0*E*inp.column_cracked_factor*rect_inertia_x(z.perimeter_col_x_m, z.perimeter_col_y_m)/max(h,1e-12))
    kc += interior_cols * (4.0*E*inp.column_cracked_factor*rect_inertia_x(z.interior_col_x_m,  z.interior_col_y_m)/max(h,1e-12))
    return kb / max(kb+kc, 1e-12)


def beam_story_stiffness_n_m(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E  = inp.Ec_mpa * 1e6
    Ib = inp.beam_cracked_factor * rect_inertia_x(inp.beam_width_m, inp.beam_depth_m)
    n_beams_x = inp.n_bays_x*(inp.n_bays_y+1)
    n_beams_y = inp.n_bays_y*(inp.n_bays_x+1)
    kx = n_beams_x * 12.0*E*Ib / max(inp.bay_x_m**3, 1e-12)
    ky = n_beams_y * 12.0*E*Ib / max(inp.bay_y_m**3, 1e-12)
    restraint = beam_rotational_restraint_factor(inp, z)
    return (kx+ky) * restraint * (inp.story_height_m / max(0.5*(inp.bay_x_m+inp.bay_y_m),1e-12))**2


def diaphragm_stiffness_n_m(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    A = inp.slab_cracked_factor * inp.slab_thickness_m * gross_floor_area(inp)
    L = max(inp.plan_x_m, inp.plan_y_m)
    return E * A / max(L, 1e-12)


def floor_weight_kn(inp: BuildingInput) -> float:
    area = gross_floor_area(inp)
    slab_self = inp.slab_thickness_m * CONCRETE_UNIT_WEIGHT
    beam_length = (inp.n_bays_x*(inp.n_bays_y+1)*inp.bay_x_m
                 + inp.n_bays_y*(inp.n_bays_x+1)*inp.bay_y_m)
    beam_self = inp.beam_width_m * inp.beam_depth_m * CONCRETE_UNIT_WEIGHT * beam_length / max(area,1e-12)
    gravity   = inp.dl_kn_m2 + inp.superimposed_dead_kn_m2 + slab_self + beam_self + inp.live_load_mass_factor*inp.ll_kn_m2
    facade    = inp.facade_line_load_kn_m * 2.0*(inp.plan_x_m+inp.plan_y_m) / max(area,1e-12)
    return area * (gravity + facade)


def floor_mass_kg(inp: BuildingInput) -> float:
    return floor_weight_kn(inp) * 1000.0 / G


# ═══════════════════════════════════════════════════════════════════
# ZONE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

def build_zone_stiffness_results(inp: BuildingInput) -> List[ZoneStiffnessResult]:
    results: List[ZoneStiffnessResult] = []
    kd  = diaphragm_stiffness_n_m(inp)
    klw = lateral_wall_story_stiffness_n_m(inp)
    for zone in define_three_zones(inp.n_story):
        z   = get_zone_input(inp, zone.name)
        wc  = get_wall_count(inp, zone.name)
        kw  = wall_story_stiffness_n_m(inp, wc, z.wall_thickness_m)
        kc  = frame_story_stiffness_from_columns(inp, z)
        kb  = beam_story_stiffness_n_m(inp, z)
        ktotal = kw + kc + kb + klw
        results.append(ZoneStiffnessResult(
            zone_name=zone.name, story_start=zone.story_start, story_end=zone.story_end,
            wall_count=wc, wall_thickness_m=z.wall_thickness_m,
            wall_story_stiffness_n_m=kw,
            column_story_stiffness_n_m=kc,
            beam_story_stiffness_n_m=kb,
            lateral_wall_stiffness_n_m=klw,
            diaphragm_stiffness_n_m=kd,
            total_story_stiffness_n_m=ktotal,
        ))
    return results


# ═══════════════════════════════════════════════════════════════════
# OUTRIGGERS
# ═══════════════════════════════════════════════════════════════════

def brace_arm_length_m(inp: BuildingInput) -> float:
    return 0.5 * max(inp.plan_x_m - inp.core_outer_x_m, inp.plan_y_m - inp.core_outer_y_m)


def outrigger_results(inp: BuildingInput) -> List[OutriggerResult]:
    if inp.outrigger_count <= 0:
        return []
    levels = [l for l in inp.outrigger_story_levels[:inp.outrigger_count] if 1 <= l <= inp.n_story]
    if not levels:
        return []

    do   = inp.brace_outer_diameter_mm / 1000.0
    t    = inp.brace_thickness_mm / 1000.0
    area = circular_hollow_area(do, t)
    rg   = circular_hollow_radius_of_gyration(do, t)
    E    = inp.Es_mpa * 1e6
    arm  = brace_arm_length_m(inp)
    depth= inp.outrigger_depth_m
    kd   = diaphragm_stiffness_n_m(inp)

    out: List[OutriggerResult] = []
    for level in levels:
        Lb      = sqrt(arm**2 + depth**2)
        axial_k = E * area / max(Lb, 1e-12)
        slender = inp.brace_effective_length_factor * Lb / max(rg, 1e-12)
        k_one   = axial_k * (depth/max(Lb,1e-12))**2
        k_story = 4.0 * inp.braces_per_side * inp.brace_buckling_reduction * k_one
        k_eff   = 1.0 / max((1.0/max(k_story,1e-12)) + (1.0/max(kd,1e-12)), 1e-18)
        steel_w = 4.0 * inp.braces_per_side * Lb * area * STEEL_DENSITY
        out.append(OutriggerResult(
            story_level=level, arm_m=arm, brace_length_m=Lb,
            brace_area_m2=area, brace_radius_gyration_m=rg,
            slenderness=slender, axial_stiffness_n=axial_k,
            lateral_stiffness_n_m=k_story,
            diaphragm_limited_stiffness_n_m=k_eff,
            steel_weight_kg=steel_w,
        ))
    return out


# ═══════════════════════════════════════════════════════════════════
# RETAINING WALL DESIGN (basement)
# ═══════════════════════════════════════════════════════════════════

def calc_retaining_wall(inp: BuildingInput) -> Optional[RetainingWallResult]:
    if inp.n_basement <= 0:
        return None
    H  = inp.retaining_wall_height_m
    γs = inp.soil_unit_weight_kn_m3
    Ka = inp.soil_ka
    q  = inp.surcharge_kn_m2
    fy = inp.fy_mpa
    fck= inp.fck_mpa

    # Active pressure diagram (triangular + uniform from surcharge)
    pa_top = Ka * q                    # kN/m²  (surcharge only at top)
    pa_bot = Ka * (q + γs * H)         # kN/m²  (surcharge + soil)
    Ptri   = 0.5 * Ka * γs * H**2      # kN/m  (triangular part)
    Prect  = Ka * q * H                # kN/m  (rectangular part from surcharge)
    thrust = Ptri + Prect              # kN per metre width

    # Moment at base (propped-cantilever simplified as cantilever)
    M_tri  = Ptri  * H / 3.0           # moment from triangular pressure
    M_rect = Prect * H / 2.0           # moment from rectangular pressure
    Mbase  = M_tri + M_rect            # kN·m / m

    # Required d from M_base (EC2 simplified — μ method)
    b    = 1000.0     # mm
    fcd  = fck / 1.5  # MPa design
    Mu   = Mbase * 1e6  # N·mm/m
    # μ = Mu / (b·d²·fcd) → solve for d with μ_lim = 0.295
    mu_lim = 0.295
    d_req = sqrt(Mu / (mu_lim * b * fcd))   # mm

    # Steel area (moment lever arm z ≈ 0.9d)
    z_lever = 0.9 * d_req
    fyd = fy / 1.15
    As  = Mu / (z_lever * fyd)   # mm²/m

    return RetainingWallResult(
        height_m=H,
        thickness_m=inp.retaining_wall_thickness_m,
        active_pressure_top_kn_m2=pa_top,
        active_pressure_bot_kn_m2=pa_bot,
        total_thrust_kn_m=thrust,
        moment_base_kn_m_m=Mbase,
        required_d_mm=d_req,
        required_As_mm2_m=As,
    )


# ═══════════════════════════════════════════════════════════════════
# RESPONSE SPECTRUM
# ═══════════════════════════════════════════════════════════════════

def spectrum_Sa_g(inp: BuildingInput, T: float) -> float:
    """Trapezoidal response spectrum — returns Sa/g."""
    T0, Ts  = inp.spectrum_T0_s, inp.spectrum_Ts_s
    Sa_max  = inp.spectrum_Sa_max_g
    Sa_min  = inp.spectrum_Sa_min_g
    if T <= T0:
        # Ascending branch
        return Sa_min + (Sa_max - Sa_min) * T / max(T0, 1e-6)
    elif T <= Ts:
        return Sa_max
    else:
        # Descending branch (1/T)
        return max(Sa_min, Sa_max * Ts / max(T, 1e-6))


def base_shear_from_spectrum(inp: BuildingInput, T1: float, total_weight_kn: float) -> Tuple[float, float]:
    Sa = spectrum_Sa_g(inp, T1)
    Cs = Sa * inp.importance_factor_I / max(inp.response_modification_R, 1e-6)
    V  = Cs * total_weight_kn   # kN
    return V, Sa


# ═══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def build_story_stiffness_vector(inp, zone_results, outriggers):
    by_story = {}
    for zr in zone_results:
        for s in range(zr.story_start, zr.story_end+1):
            by_story[s] = zr.total_story_stiffness_n_m
    for o in outriggers:
        by_story[o.story_level] = by_story.get(o.story_level, 0.0) + o.diaphragm_limited_stiffness_n_m
    return [by_story[i] for i in range(1, inp.n_story+1)]


def build_story_mass_vector(inp, outriggers):
    m = [floor_mass_kg(inp)] * inp.n_story
    for o in outriggers:
        if 1 <= o.story_level <= inp.n_story:
            m[o.story_level-1] += o.steel_weight_kg
    return m


def assemble_m_k(masses, stiffnesses):
    n = len(masses)
    M = np.diag(np.array(masses, dtype=float))
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = stiffnesses[i]
        K[i, i] += ki
        if i > 0:
            K[i, i-1] -= ki
            K[i-1, i] -= ki
            K[i-1, i-1] += ki
    return M, K


def solve_modes(M, K, n_modes=5):
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals); eigvecs = np.real(eigvecs)
    mask = eigvals > 1e-12
    eigvals = eigvals[mask]; eigvecs = eigvecs[:, mask]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    omegas = np.sqrt(eigvals[:n_modes])
    periods = [2.0*pi/max(w,1e-12) for w in omegas]
    freqs   = [w/(2.0*pi)           for w in omegas]
    shapes  = []
    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].copy()
        phi = phi / max(np.max(np.abs(phi)), 1e-12)
        if phi[-1] < 0: phi = -phi
        shapes.append(phi.tolist())
    return periods, freqs, shapes


def lateral_force_distribution_n(inp, masses, total_weight_kn, base_shear_kn):
    V = base_shear_kn * 1000.0
    heights = np.arange(1, inp.n_story+1, dtype=float) * inp.story_height_m
    w = np.array(masses) * G
    coeff = w * heights
    return V * coeff / max(np.sum(coeff), 1e-12)


def solve_static_displacements(K, F, story_height_m):
    u      = np.linalg.solve(K, F)
    drifts = np.zeros_like(u)
    drifts[0]  = u[0]
    drifts[1:] = u[1:] - u[:-1]
    return u.tolist(), drifts.tolist(), (drifts / max(story_height_m,1e-12)).tolist()


def analyze(inp: BuildingInput) -> AnalysisResult:
    valid, msg = core_geometry_is_valid(inp)
    if not valid:
        raise ValueError(msg)

    zone_results = build_zone_stiffness_results(inp)
    outrs        = outrigger_results(inp)
    k_story      = build_story_stiffness_vector(inp, zone_results, outrs)
    m_story      = build_story_mass_vector(inp, outrs)

    total_weight_kn = sum(m_story) * G / 1000.0
    total_mass_kg   = sum(m_story)

    M_mat, K_mat = assemble_m_k(m_story, k_story)
    periods, freqs, shapes = solve_modes(M_mat, K_mat, n_modes=5)

    # Base shear — coefficient or spectrum
    T1 = periods[0] if periods else 1.0
    if inp.use_response_spectrum:
        V_kn, sa_g = base_shear_from_spectrum(inp, T1, total_weight_kn)
        method = "Spectrum"
    else:
        V_kn = inp.seismic_base_shear_coeff * total_weight_kn
        sa_g = 0.0
        method = "Coefficient"

    F   = lateral_force_distribution_n(inp, m_story, total_weight_kn, V_kn)
    u, drifts, drift_ratios = solve_static_displacements(K_mat, F, inp.story_height_m)

    # Stiffness totals
    wall_total = col_total = beam_total = lat_total = 0.0
    for z in zone_results:
        n = z.story_end - z.story_start + 1
        wall_total += z.wall_story_stiffness_n_m    * n
        col_total  += z.column_story_stiffness_n_m  * n
        beam_total += z.beam_story_stiffness_n_m    * n
        lat_total  += z.lateral_wall_stiffness_n_m  * n
    k_outrigger_total = sum(o.diaphragm_limited_stiffness_n_m for o in outrs)
    kd_median = float(np.median([z.diaphragm_stiffness_n_m for z in zone_results])) if zone_results else 0.0

    # Retaining wall
    rw = calc_retaining_wall(inp)

    # Tables
    zone_table = pd.DataFrame([{
        "Zone": z.zone_name, "Stories": f"{z.story_start}-{z.story_end}",
        "Wall count": z.wall_count, "Wall thickness (m)": z.wall_thickness_m,
        "Wall K/story (N/m)": z.wall_story_stiffness_n_m,
        "Column K/story (N/m)": z.column_story_stiffness_n_m,
        "Beam K/story (N/m)": z.beam_story_stiffness_n_m,
        "Lateral wall K/story (N/m)": z.lateral_wall_stiffness_n_m,
        "Diaphragm K (N/m)": z.diaphragm_stiffness_n_m,
        "Total K/story (N/m)": z.total_story_stiffness_n_m,
    } for z in zone_results])

    story_table = pd.DataFrame({
        "Story": np.arange(1, inp.n_story+1),
        "Mass (kg)": m_story,
        "Stiffness (N/m)": k_story,
        "Lateral force (N)": F.tolist(),
        "Floor displacement (m)": u,
        "Story drift (m)": drifts,
        "Story drift ratio": drift_ratios,
    })

    outrigger_table = pd.DataFrame([{
        "Story": o.story_level, "Brace arm (m)": o.arm_m,
        "Brace length (m)": o.brace_length_m, "CHS area (m²)": o.brace_area_m2,
        "Radius of gyration (m)": o.brace_radius_gyration_m, "KL/r": o.slenderness,
        "Axial stiffness EA/L (N)": o.axial_stiffness_n,
        "Brace lateral stiffness (N/m)": o.lateral_stiffness_n_m,
        "Effective outrigger stiffness (N/m)": o.diaphragm_limited_stiffness_n_m,
        "Steel weight (kg)": o.steel_weight_kg,
    } for o in outrs]) if outrs else pd.DataFrame(columns=[
        "Story","Brace arm (m)","Brace length (m)","CHS area (m²)","Radius of gyration (m)",
        "KL/r","Axial stiffness EA/L (N)","Brace lateral stiffness (N/m)",
        "Effective outrigger stiffness (N/m)","Steel weight (kg)"])

    if rw:
        retaining_table = pd.DataFrame({
            "Parameter": [
                "Wall height (m)", "Wall thickness (m)",
                "Active pressure top (kN/m²)", "Active pressure bottom (kN/m²)",
                "Total lateral thrust (kN/m)", "Moment at base (kN·m/m)",
                "Required effective depth d (mm)", "Required steel As (mm²/m)",
            ],
            "Value": [
                rw.height_m, rw.thickness_m,
                round(rw.active_pressure_top_kn_m2, 2), round(rw.active_pressure_bot_kn_m2, 2),
                round(rw.total_thrust_kn_m, 2), round(rw.moment_base_kn_m_m, 2),
                round(rw.required_d_mm, 1), round(rw.required_As_mm2_m, 1),
            ]
        })
    else:
        retaining_table = pd.DataFrame({"Parameter": ["No basement defined"], "Value": ["-"]})

    summary_table = pd.DataFrame({
        "Parameter": [
            "Total weight (kN)", "Base shear (kN)", "Base shear method",
            "Period T1 (s)", "Frequency f1 (Hz)", "Sa/g at T1 (spectrum)",
            "Roof displacement (m)", "Max story drift (m)", "Max story drift ratio",
            "Drift limit ratio", "Drift check",
            "Wall stiffness sum (N/m)", "Column stiffness sum (N/m)",
            "Beam stiffness sum (N/m)", "Lateral wall stiffness sum (N/m)",
            "Outrigger stiffness (N/m)", "Median diaphragm stiffness (N/m)",
        ],
        "Value": [
            total_weight_kn, V_kn, method,
            periods[0] if periods else float("nan"),
            freqs[0]   if freqs   else float("nan"),
            sa_g,
            u[-1] if u else float("nan"),
            max(drifts) if drifts else float("nan"),
            max(drift_ratios) if drift_ratios else float("nan"),
            inp.drift_limit_ratio,
            "OK" if max(drift_ratios) <= inp.drift_limit_ratio else "NOT OK",
            wall_total, col_total, beam_total, lat_total,
            k_outrigger_total, kd_median,
        ]
    })

    return AnalysisResult(
        story_stiffness_n_m=k_story, story_mass_kg=m_story,
        periods_s=periods, frequencies_hz=freqs, mode_shapes=shapes,
        lateral_forces_n=F.tolist(), floor_displacements_m=u,
        story_drifts_m=drifts, story_drift_ratios=drift_ratios,
        base_shear_kn=V_kn, base_shear_method=method, spectrum_Sa_g=sa_g,
        total_weight_kn=total_weight_kn, total_mass_kg=total_mass_kg,
        k_wall_total_n_m=wall_total, k_column_total_n_m=col_total,
        k_beam_total_n_m=beam_total, k_lateral_wall_total_n_m=lat_total,
        k_outrigger_total_n_m=k_outrigger_total, k_diaphragm_median_n_m=kd_median,
        outriggers=outrs, zone_results=zone_results, retaining_wall=rw,
        zone_table=zone_table, story_table=story_table,
        outrigger_table=outrigger_table, retaining_table=retaining_table,
        summary_table=summary_table,
    )


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_mode_shapes(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.story_mass_kg)+1)
    for i, (phi, T) in enumerate(zip(result.mode_shapes[:5], result.periods_s[:5])):
        ax.plot(phi, y, marker="o", ms=3, label=f"Mode {i+1}  T={T:.3f}s")
    ax.axvline(0, color="k", linewidth=0.7)
    ax.set_xlabel("Normalised mode shape"); ax.set_ylabel("Story")
    ax.set_title("Mode shapes"); ax.grid(True, alpha=0.3); ax.legend()
    return fig


def plot_story_response(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.floor_displacements_m)+1)
    ax.plot(result.floor_displacements_m, y, marker="o", ms=3)
    ax.set_xlabel("Displacement (m)"); ax.set_ylabel("Story")
    ax.set_title("Static floor displacements"); ax.grid(True, alpha=0.3)
    return fig


def plot_story_drifts(result: AnalysisResult, drift_limit: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, len(result.story_drifts_m)+1)
    ax.plot(result.story_drift_ratios, y, marker="o", ms=3, label="Drift ratio")
    ax.axvline(drift_limit, color="red", linestyle="--", label=f"Limit {drift_limit:.4f}")
    ax.set_xlabel("Drift ratio"); ax.set_ylabel("Story")
    ax.set_title("Story drift ratios"); ax.grid(True, alpha=0.3); ax.legend()
    return fig


def plot_spectrum(inp: BuildingInput, T1: float) -> plt.Figure:
    Ts_arr = np.linspace(0, 4.0, 400)
    Sa_arr = [spectrum_Sa_g(inp, t) for t in Ts_arr]
    Sa_T1  = spectrum_Sa_g(inp, T1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ts_arr, Sa_arr, "b-", linewidth=2, label="Design spectrum")
    ax.plot(T1, Sa_T1, "ro", ms=10, label=f"T₁ = {T1:.3f}s  Sa={Sa_T1:.3f}g")
    ax.set_xlabel("Period T (s)"); ax.set_ylabel("Sa/g")
    ax.set_title("Response spectrum"); ax.grid(True, alpha=0.3); ax.legend()
    return fig


def plot_plan(inp: BuildingInput, result: AnalysisResult, zone_name: str) -> plt.Figure:
    """Plan view with beams, braced bays, lateral shear walls, and outrigger zones."""
    zone = next(z for z in result.zone_results if z.zone_name == zone_name)
    z_in = get_zone_input(inp, zone_name)
    fig, ax = plt.subplots(figsize=(12, 10))

    # ── grid lines ──────────────────────────────────────────────
    for i in range(inp.n_bays_x+1):
        ax.plot([i*inp.bay_x_m]*2, [0, inp.plan_y_m], color="#cccccc", lw=0.7)
    for j in range(inp.n_bays_y+1):
        ax.plot([0, inp.plan_x_m], [j*inp.bay_y_m]*2, color="#cccccc", lw=0.7)

    # ── building outline ─────────────────────────────────────────
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0],
            [0, 0, inp.plan_y_m, inp.plan_y_m, 0], "k-", lw=2)

    # ── beams ────────────────────────────────────────────────────
    bw = inp.beam_width_m
    bd = inp.beam_depth_m
    beam_color = "#a0c4ff"
    # beams in X direction
    for j in range(inp.n_bays_y+1):
        y = j * inp.bay_y_m
        for i in range(inp.n_bays_x):
            x0 = i * inp.bay_x_m
            x1 = (i+1) * inp.bay_x_m
            ax.add_patch(patches.FancyArrow(0, 0, 0, 0))  # placeholder flush
            # Draw beam as a narrow rectangle along its axis
            rect = patches.Rectangle((x0 + z_in.corner_col_x_m/2,
                                       y - bw/2),
                                      x1 - x0 - z_in.corner_col_x_m,
                                      bw,
                                      facecolor=beam_color, edgecolor="#5599cc",
                                      linewidth=0.8, alpha=0.75, zorder=2)
            ax.add_patch(rect)
    # beams in Y direction
    for i in range(inp.n_bays_x+1):
        x = i * inp.bay_x_m
        for j in range(inp.n_bays_y):
            y0 = j * inp.bay_y_m
            y1 = (j+1) * inp.bay_y_m
            rect = patches.Rectangle((x - bw/2,
                                       y0 + z_in.corner_col_y_m/2),
                                      bw,
                                      y1 - y0 - z_in.corner_col_y_m,
                                      facecolor=beam_color, edgecolor="#5599cc",
                                      linewidth=0.8, alpha=0.75, zorder=2)
            ax.add_patch(rect)

    # ── braced bays (diagonal hatch) ─────────────────────────────
    brace_color = "#ff9933"
    for bi in inp.braced_bays_x:
        if 0 <= bi < inp.n_bays_x:
            x0 = bi * inp.bay_x_m
            # shade full bay column strip
            rect = patches.Rectangle((x0, 0), inp.bay_x_m, inp.plan_y_m,
                                       facecolor=brace_color, alpha=0.18,
                                       edgecolor=brace_color, lw=1.5, zorder=1,
                                       hatch="//")
            ax.add_patch(rect)
            ax.text(x0 + inp.bay_x_m/2, inp.plan_y_m + 0.5, f"BX{bi+1}",
                    ha="center", va="bottom", fontsize=7, color=brace_color)
    for bj in inp.braced_bays_y:
        if 0 <= bj < inp.n_bays_y:
            y0 = bj * inp.bay_y_m
            rect = patches.Rectangle((0, y0), inp.plan_x_m, inp.bay_y_m,
                                       facecolor=brace_color, alpha=0.18,
                                       edgecolor=brace_color, lw=1.5, zorder=1,
                                       hatch="\\\\")
            ax.add_patch(rect)
            ax.text(-1.0, y0 + inp.bay_y_m/2, f"BY{bj+1}",
                    ha="right", va="center", fontsize=7, color=brace_color)

    # ── lateral (perimeter) shear walls ─────────────────────────
    lw_color = "#22aa55"
    lw_t  = inp.lateral_wall_thickness_m
    lw_Lx = inp.lateral_wall_length_x_m
    lw_Ly = inp.lateral_wall_length_y_m
    # top and bottom facades — walls in X direction (resist Y loads)
    mid_x = inp.plan_x_m / 2.0
    for sign, y_face in [(-1, 0), (1, inp.plan_y_m)]:
        for k in range(inp.lateral_wall_x_count):
            offset = (k - (inp.lateral_wall_x_count-1)/2.0) * (inp.plan_x_m / (inp.lateral_wall_x_count+1))
            x_start = mid_x + offset - lw_Lx/2.0
            y_start = y_face - (lw_t if sign == 1 else 0)
            ax.add_patch(patches.Rectangle((x_start, y_start), lw_Lx, lw_t,
                          facecolor=lw_color, edgecolor="#006622", lw=1.5,
                          alpha=0.75, zorder=4))
    # left and right facades — walls in Y direction (resist X loads)
    mid_y = inp.plan_y_m / 2.0
    for x_face in [0, inp.plan_x_m]:
        for k in range(inp.lateral_wall_y_count):
            offset = (k - (inp.lateral_wall_y_count-1)/2.0) * (inp.plan_y_m / (inp.lateral_wall_y_count+1))
            y_start = mid_y + offset - lw_Ly/2.0
            x_start = x_face - (lw_t if x_face > 0 else 0)
            ax.add_patch(patches.Rectangle((x_start, y_start), lw_t, lw_Ly,
                          facecolor=lw_color, edgecolor="#006622", lw=1.5,
                          alpha=0.75, zorder=4))

    # ── core ─────────────────────────────────────────────────────
    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    cy0 = (inp.plan_y_m - inp.core_outer_y_m) / 2.0
    ix0 = (inp.plan_x_m - inp.core_opening_x_m) / 2.0
    iy0 = (inp.plan_y_m - inp.core_opening_y_m) / 2.0
    t   = z_in.wall_thickness_m
    for x, y, w, h in [
        (cx0, cy0, inp.core_outer_x_m, t),
        (cx0, cy0+inp.core_outer_y_m-t, inp.core_outer_x_m, t),
        (cx0, cy0, t, inp.core_outer_y_m),
        (cx0+inp.core_outer_x_m-t, cy0, t, inp.core_outer_y_m),
    ]:
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#336633",
                     edgecolor="green", alpha=0.50, zorder=5))
    ax.add_patch(patches.Rectangle((cx0, cy0), inp.core_outer_x_m, inp.core_outer_y_m,
                  fill=False, edgecolor="green", lw=2.0, zorder=5))
    ax.add_patch(patches.Rectangle((ix0, iy0), inp.core_opening_x_m, inp.core_opening_y_m,
                  fill=False, edgecolor="gray", linestyle="--", zorder=5))

    # ── columns ─────────────────────────────────────────────────
    for i in range(inp.n_bays_x+1):
        for j in range(inp.n_bays_y+1):
            x = i*inp.bay_x_m; y = j*inp.bay_y_m
            at_x = i in (0, inp.n_bays_x)
            at_y = j in (0, inp.n_bays_y)
            is_braced_x = any(i in (bb, bb+1) for bb in inp.braced_bays_x)
            is_braced_y = any(j in (bb, bb+1) for bb in inp.braced_bays_y)
            if at_x and at_y:
                dx, dy, c = z_in.corner_col_x_m, z_in.corner_col_y_m, "#8b0000"
            elif at_x or at_y:
                dx, dy, c = z_in.perimeter_col_x_m, z_in.perimeter_col_y_m, "#cc5500"
            else:
                dx, dy, c = z_in.interior_col_x_m, z_in.interior_col_y_m, "#3366aa"
            edge_c = "#ff6600" if (is_braced_x or is_braced_y) else c
            ax.add_patch(patches.Rectangle((x-dx/2, y-dy/2), dx, dy,
                          facecolor=c, edgecolor=edge_c,
                          linewidth=2.0 if (is_braced_x or is_braced_y) else 0.8,
                          alpha=0.90, zorder=6))

    # ── outrigger arm zones ──────────────────────────────────────
    if inp.outrigger_count > 0:
        arm   = brace_arm_length_m(inp)
        depth = inp.outrigger_depth_m
        cy_mid= inp.plan_y_m / 2.0
        ax.add_patch(patches.Rectangle((cx0-arm, cy_mid-depth/2), arm, depth,
                      facecolor="#ff8c00", alpha=0.55, edgecolor="#cc6600", lw=1.5, zorder=3))
        ax.add_patch(patches.Rectangle((cx0+inp.core_outer_x_m, cy_mid-depth/2), arm, depth,
                      facecolor="#ff8c00", alpha=0.55, edgecolor="#cc6600", lw=1.5, zorder=3))
        ax.text(cx0 - arm/2, cy_mid, "Outrigger\nzone", ha="center", va="center",
                fontsize=7, color="#aa4400")

    # ── legend patches ───────────────────────────────────────────
    legend_elements = [
        patches.Patch(facecolor=beam_color, edgecolor="#5599cc", label="Beams"),
        patches.Patch(facecolor="#336633", edgecolor="green", alpha=0.5, label="Core walls"),
        patches.Patch(facecolor=lw_color, edgecolor="#006622", label="Lateral shear walls"),
        patches.Patch(facecolor=brace_color, alpha=0.35, hatch="//", label="Braced bays"),
        patches.Patch(facecolor="#8b0000", label="Corner columns"),
        patches.Patch(facecolor="#cc5500", label="Perimeter columns"),
        patches.Patch(facecolor="#3366aa", label="Interior columns"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlim(-3, inp.plan_x_m+3); ax.set_ylim(-3, inp.plan_y_m+3)
    ax.set_title(f"Plan view — {zone_name}  (beam bw={inp.beam_width_m:.2f}m  bd={inp.beam_depth_m:.2f}m)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    return fig


def plot_elevation(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 10))
    H_above = inp.n_story * inp.story_height_m
    H_bsmt  = inp.n_basement * inp.basement_height_m
    total_H = H_above + H_bsmt

    # Grade line reference (y=0 → ground level, negative → basement)
    ax.axhline(0, color="saddlebrown", lw=2, linestyle="--", label="Grade level")

    # Basement below grade
    if inp.n_basement > 0:
        ax.add_patch(patches.Rectangle((0, -H_bsmt), inp.plan_x_m, H_bsmt,
                      facecolor="#d4b896", edgecolor="saddlebrown", alpha=0.40,
                      linewidth=1.5, label="Basement / soil"))
        # Retaining wall hatching on sides
        rw_t = inp.retaining_wall_thickness_m
        ax.add_patch(patches.Rectangle((0, -H_bsmt), rw_t, H_bsmt,
                      facecolor="gray", edgecolor="black", alpha=0.55,
                      hatch="||", label="Retaining wall"))
        ax.add_patch(patches.Rectangle((inp.plan_x_m-rw_t, -H_bsmt), rw_t, H_bsmt,
                      facecolor="gray", edgecolor="black", alpha=0.55, hatch="||"))
        for lvl in range(1, inp.n_basement+1):
            y = -lvl * inp.basement_height_m
            ax.plot([0, inp.plan_x_m], [y, y], color="#888888", lw=0.7)

    # Above-grade floors
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0],
            [0, 0, H_above, H_above, 0], "k-", lw=2)
    for i in range(inp.n_story+1):
        y = i * inp.story_height_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#dddddd", lw=0.5)

    # Core band
    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    ax.add_patch(patches.Rectangle((cx0, 0), inp.core_outer_x_m, H_above,
                  edgecolor="green", fill=False, linewidth=2.0, label="Core"))

    # Zone colour bands
    zone_colours = ["#eaf5ea", "#d0ead0", "#b6dfb6"]
    for zr, clr in zip(result.zone_results, zone_colours):
        y0 = (zr.story_start-1) * inp.story_height_m
        yh = zr.n_stories * inp.story_height_m
        ax.add_patch(patches.Rectangle((0, y0), inp.plan_x_m, yh,
                      facecolor=clr, alpha=0.25, edgecolor="none"))
        ax.text(inp.plan_x_m+0.5, y0+yh/2, zr.zone_name,
                va="center", fontsize=7, color="darkgreen")

    # Outriggers
    for o in result.outriggers:
        y = o.story_level * inp.story_height_m
        ax.plot([cx0, 0], [y, y], color="#ff8c00", lw=4, solid_capstyle="round")
        ax.plot([cx0+inp.core_outer_x_m, inp.plan_x_m], [y, y],
                color="#ff8c00", lw=4, solid_capstyle="round",
                label=f"Outrigger @ story {o.story_level}")

    # Lateral shear wall zones (schematic vertical bands)
    lw_t = inp.lateral_wall_thickness_m
    ax.add_patch(patches.Rectangle((0, 0), lw_t, H_above,
                  facecolor="#22aa55", alpha=0.45, edgecolor="#006622",
                  lw=1.5, label="Lateral shear walls"))
    ax.add_patch(patches.Rectangle((inp.plan_x_m-lw_t, 0), lw_t, H_above,
                  facecolor="#22aa55", alpha=0.45, edgecolor="#006622", lw=1.5))

    ax.set_xlim(-1, inp.plan_x_m+8)
    ax.set_ylim(-H_bsmt - inp.basement_height_m, H_above + inp.story_height_m)
    ax.set_xlabel("Plan width (schematic, m)"); ax.set_ylabel("Height (m)")
    ax.set_title("Elevation — schematic with basement")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_retaining_wall(inp: BuildingInput, rw: RetainingWallResult) -> plt.Figure:
    """Pressure diagram for the retaining wall."""
    fig, ax = plt.subplots(figsize=(6, 5))
    H = rw.height_m
    pressures = [rw.active_pressure_top_kn_m2, rw.active_pressure_bot_kn_m2]
    ax.plot([0, 0], [0, H], "k-", lw=3, label="Wall")
    ax.fill_betweenx([0, H], [rw.active_pressure_top_kn_m2, rw.active_pressure_bot_kn_m2],
                     [0, 0], alpha=0.4, color="#8b4513", label="Active earth pressure")
    ax.set_xlabel("Active pressure (kN/m²)")
    ax.set_ylabel("Height from base (m)")
    ax.set_title(f"Retaining wall pressure diagram\nH={H:.1f}m  M_base={rw.moment_base_kn_m_m:.1f} kN·m/m")
    ax.invert_yaxis()
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


# ═══════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════

def build_text_report(inp: BuildingInput, result: AnalysisResult) -> str:
    lines = ["="*78,
             f"{APP_TITLE} — {APP_VERSION}",
             "="*78, "",
             "MODEL BASIS", "-"*78,
             "1) Story stiffness assembled from core walls, frame columns, beams,",
             "   perimeter shear walls, and steel CHS outriggers.",
             "2) Cracked section factors applied per user inputs.",
             "3) Base shear = max(coefficient method, response spectrum) if spectrum enabled.",
             "4) Retaining wall designed by EC2 simplified approach (cantilever).",
             "", "GLOBAL RESPONSE", "-"*78,
             f"  Base shear method            : {result.base_shear_method}",
             f"  Total weight                 : {result.total_weight_kn:,.1f} kN",
             f"  Base shear                   : {result.base_shear_kn:,.1f} kN",
             f"  Sa/g at T1 (spectrum)        : {result.spectrum_Sa_g:.4f}",
             f"  Fundamental period T1        : {result.periods_s[0]:.3f} s",
             f"  Fundamental frequency f1     : {result.frequencies_hz[0]:.3f} Hz",
             f"  Roof displacement            : {result.floor_displacements_m[-1]:.4f} m",
             f"  Max story drift ratio        : {max(result.story_drift_ratios):.6f}",
             f"  Drift limit                  : {inp.drift_limit_ratio:.6f}",
             f"  Drift check                  : {'OK' if max(result.story_drift_ratios) <= inp.drift_limit_ratio else 'NOT OK'}",
             "", "STIFFNESS CONTRIBUTIONS", "-"*78,
             f"  Core wall sum                : {result.k_wall_total_n_m:,.3e} N/m",
             f"  Column sum                   : {result.k_column_total_n_m:,.3e} N/m",
             f"  Beam sum                     : {result.k_beam_total_n_m:,.3e} N/m",
             f"  Lateral shear wall sum       : {result.k_lateral_wall_total_n_m:,.3e} N/m",
             f"  Outrigger effective sum      : {result.k_outrigger_total_n_m:,.3e} N/m",
             f"  Median diaphragm stiffness   : {result.k_diaphragm_median_n_m:,.3e} N/m",
             "", "ZONE SUMMARY", "-"*78,
    ]
    for z in result.zone_results:
        lines.append(
            f"  {z.zone_name:12s} | stories {z.story_start:>3}-{z.story_end:<3}"
            f" | t={z.wall_thickness_m:.3f}m"
            f" | K_wall={z.wall_story_stiffness_n_m:.3e}"
            f" | K_col={z.column_story_stiffness_n_m:.3e}"
            f" | K_beam={z.beam_story_stiffness_n_m:.3e}"
            f" | K_latwall={z.lateral_wall_stiffness_n_m:.3e}"
        )
    lines += ["", "OUTRIGGER SUMMARY", "-"*78]
    if result.outriggers:
        for o in result.outriggers:
            lines.append(
                f"  Story {o.story_level:>3} | Lb={o.brace_length_m:.3f}m"
                f" | KL/r={o.slenderness:.1f}"
                f" | K_eff={o.diaphragm_limited_stiffness_n_m:.3e} N/m"
            )
    else:
        lines.append("  No outriggers defined.")

    if result.retaining_wall:
        rw = result.retaining_wall
        lines += ["", "RETAINING WALL", "-"*78,
                  f"  Height                       : {rw.height_m:.2f} m",
                  f"  Thickness                    : {rw.thickness_m:.2f} m",
                  f"  Total lateral thrust         : {rw.total_thrust_kn_m:.2f} kN/m",
                  f"  Moment at base               : {rw.moment_base_kn_m_m:.2f} kN·m/m",
                  f"  Required effective depth d   : {rw.required_d_mm:.1f} mm",
                  f"  Required steel As            : {rw.required_As_mm2_m:.1f} mm²/m",
        ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

def zone_block(label: str, defaults: ZoneMemberInput) -> ZoneMemberInput:
    st.sidebar.markdown(f"**{label} zone members**")
    wall_t = st.sidebar.number_input(f"{label} wall thickness (m)", 0.15, 2.00, float(defaults.wall_thickness_m), 0.01, format="%.2f")
    cxx = st.sidebar.number_input(f"{label} corner col X (m)", 0.20, 3.00, float(defaults.corner_col_x_m), 0.05, format="%.2f")
    cxy = st.sidebar.number_input(f"{label} corner col Y (m)", 0.20, 3.00, float(defaults.corner_col_y_m), 0.05, format="%.2f")
    pxx = st.sidebar.number_input(f"{label} perimeter col X (m)", 0.20, 3.00, float(defaults.perimeter_col_x_m), 0.05, format="%.2f")
    pxy = st.sidebar.number_input(f"{label} perimeter col Y (m)", 0.20, 3.00, float(defaults.perimeter_col_y_m), 0.05, format="%.2f")
    ixx = st.sidebar.number_input(f"{label} interior col X (m)", 0.20, 3.00, float(defaults.interior_col_x_m), 0.05, format="%.2f")
    ixy = st.sidebar.number_input(f"{label} interior col Y (m)", 0.20, 3.00, float(defaults.interior_col_y_m), 0.05, format="%.2f")
    return ZoneMemberInput(wall_t, cxx, cxy, pxx, pxy, ixx, ixy)


def streamlit_input_panel() -> BuildingInput:
    st.sidebar.header("Input Data")
    d = BuildingInput()

    st.sidebar.subheader("Geometry")
    n_story         = int(st.sidebar.number_input("Above-grade stories",  1,  120, d.n_story, 1))
    n_basement      = int(st.sidebar.number_input("Basement stories",      0,   20, d.n_basement, 1))
    story_height_m  = float(st.sidebar.number_input("Story height (m)",   2.5,  6.0, d.story_height_m, 0.1))
    basement_height_m = float(st.sidebar.number_input("Basement height (m)", 2.5, 6.0, d.basement_height_m, 0.1))
    plan_x_m        = float(st.sidebar.number_input("Plan X (m)",        10.0, 300.0, d.plan_x_m, 0.5))
    plan_y_m        = float(st.sidebar.number_input("Plan Y (m)",        10.0, 300.0, d.plan_y_m, 0.5))
    n_bays_x        = int(st.sidebar.number_input("Bays in X",           1,   40, d.n_bays_x, 1))
    n_bays_y        = int(st.sidebar.number_input("Bays in Y",           1,   40, d.n_bays_y, 1))
    bay_x_m         = float(st.sidebar.number_input("Bay X (m)",          2.0,  20.0, d.bay_x_m, 0.1))
    bay_y_m         = float(st.sidebar.number_input("Bay Y (m)",          2.0,  20.0, d.bay_y_m, 0.1))

    st.sidebar.subheader("Core geometry")
    core_outer_x_m   = float(st.sidebar.number_input("Core outer X (m)",   2.0, 100.0, d.core_outer_x_m,   0.1))
    core_outer_y_m   = float(st.sidebar.number_input("Core outer Y (m)",   2.0, 100.0, d.core_outer_y_m,   0.1))
    core_opening_x_m = float(st.sidebar.number_input("Core opening X (m)", 1.0, 100.0, d.core_opening_x_m, 0.1))
    core_opening_y_m = float(st.sidebar.number_input("Core opening Y (m)", 1.0, 100.0, d.core_opening_y_m, 0.1))
    lower_zone_wall_count  = int(st.sidebar.selectbox("Lower zone wall count",  [4,6,8], index=2))
    middle_zone_wall_count = int(st.sidebar.selectbox("Middle zone wall count", [4,6,8], index=1))
    upper_zone_wall_count  = int(st.sidebar.selectbox("Upper zone wall count",  [4,6,8], index=0))

    st.sidebar.subheader("Lateral (perimeter) shear walls")
    lateral_wall_x_count   = int(st.sidebar.number_input("Shear walls in X (top/bot facade)", 0, 10, d.lateral_wall_x_count, 1))
    lateral_wall_y_count   = int(st.sidebar.number_input("Shear walls in Y (left/right facade)", 0, 10, d.lateral_wall_y_count, 1))
    lateral_wall_thickness_m = float(st.sidebar.number_input("Lateral wall thickness (m)", 0.15, 1.50, d.lateral_wall_thickness_m, 0.05, format="%.2f"))
    lateral_wall_length_x_m  = float(st.sidebar.number_input("Lateral wall length — X face (m)", 1.0, 30.0, d.lateral_wall_length_x_m, 0.5))
    lateral_wall_length_y_m  = float(st.sidebar.number_input("Lateral wall length — Y face (m)", 1.0, 30.0, d.lateral_wall_length_y_m, 0.5))

    st.sidebar.subheader("Braced bays (plan indices, 0-based)")
    braced_bays_x_str = st.sidebar.text_input("Braced bay indices in X (comma-sep, e.g. 0,5)", "0,5")
    braced_bays_y_str = st.sidebar.text_input("Braced bay indices in Y (comma-sep, e.g. 0,5)", "0,5")
    def parse_indices(s):
        try: return [int(x.strip()) for x in s.split(",") if x.strip()]
        except: return []
    braced_bays_x = parse_indices(braced_bays_x_str)
    braced_bays_y = parse_indices(braced_bays_y_str)
    braced_bay_stiffness_factor = float(st.sidebar.number_input("Braced bay stiffness factor", 1.0, 5.0, d.braced_bay_stiffness_factor, 0.1, format="%.1f"))

    st.sidebar.subheader("Basement retaining wall")
    retaining_wall_thickness_m = float(st.sidebar.number_input("Retaining wall thickness (m)", 0.20, 2.00, d.retaining_wall_thickness_m, 0.05, format="%.2f"))
    retaining_wall_height_m    = float(st.sidebar.number_input("Retaining wall height (m)",    1.0, 15.0, d.retaining_wall_height_m, 0.1))
    soil_unit_weight_kn_m3     = float(st.sidebar.number_input("Soil unit weight (kN/m³)",     14.0, 22.0, d.soil_unit_weight_kn_m3, 0.5))
    soil_ka                    = float(st.sidebar.number_input("Active earth pressure Ka",      0.15,  0.50, d.soil_ka, 0.01, format="%.2f"))
    surcharge_kn_m2            = float(st.sidebar.number_input("Surcharge (kN/m²)",             0.0,  50.0, d.surcharge_kn_m2, 1.0))

    st.sidebar.subheader("Materials")
    fck_mpa = float(st.sidebar.number_input("fck (MPa)", 20.0, 100.0, d.fck_mpa, 1.0))
    Ec_mpa  = float(st.sidebar.number_input("Ec (MPa)", 15000.0, 50000.0, d.Ec_mpa, 500.0))
    fy_mpa  = float(st.sidebar.number_input("fy steel (MPa)", 200.0, 700.0, d.fy_mpa, 5.0))
    Es_mpa  = float(st.sidebar.number_input("Es (MPa)", 150000.0, 250000.0, d.Es_mpa, 1000.0))

    st.sidebar.subheader("Loads")
    dl_kn_m2               = float(st.sidebar.number_input("DL (kN/m²)", 0.0, 20.0, d.dl_kn_m2, 0.1))
    ll_kn_m2               = float(st.sidebar.number_input("LL (kN/m²)", 0.0, 20.0, d.ll_kn_m2, 0.1))
    superimposed_dead_kn_m2= float(st.sidebar.number_input("SDL (kN/m²)", 0.0, 20.0, d.superimposed_dead_kn_m2, 0.1))
    facade_line_load_kn_m  = float(st.sidebar.number_input("Facade line load (kN/m)", 0.0, 20.0, d.facade_line_load_kn_m, 0.1))
    live_load_mass_factor  = float(st.sidebar.number_input("LL mass factor", 0.0, 1.0, d.live_load_mass_factor, 0.05, format="%.2f"))
    seismic_base_shear_coeff = float(st.sidebar.number_input("Base shear coeff (if not spectrum)", 0.01, 0.50, d.seismic_base_shear_coeff, 0.005, format="%.3f"))
    drift_limit_ratio        = float(st.sidebar.number_input("Drift limit ratio", 0.0005, 0.0200, d.drift_limit_ratio, 0.0005, format="%.4f"))

    st.sidebar.subheader("Response spectrum (optional)")
    use_response_spectrum    = st.sidebar.checkbox("Use response spectrum for base shear", value=False)
    spectrum_T0_s            = float(st.sidebar.number_input("Spectrum T0 (s)", 0.05, 1.0, d.spectrum_T0_s, 0.05, format="%.2f"))
    spectrum_Ts_s            = float(st.sidebar.number_input("Spectrum Ts (s)", 0.10, 3.0, d.spectrum_Ts_s, 0.05, format="%.2f"))
    spectrum_Sa_max_g        = float(st.sidebar.number_input("Spectrum Sa_max/g", 0.05, 2.0, d.spectrum_Sa_max_g, 0.01, format="%.2f"))
    spectrum_Sa_min_g        = float(st.sidebar.number_input("Spectrum Sa_min/g", 0.01, 0.50, d.spectrum_Sa_min_g, 0.01, format="%.2f"))
    response_modification_R  = float(st.sidebar.number_input("Response modification R", 1.0, 12.0, d.response_modification_R, 0.5, format="%.1f"))
    importance_factor_I      = float(st.sidebar.number_input("Importance factor I", 0.8, 2.0, d.importance_factor_I, 0.05, format="%.2f"))

    st.sidebar.subheader("Cracked section factors")
    wall_cracked_factor   = float(st.sidebar.number_input("Wall cracked factor",   0.10, 1.00, d.wall_cracked_factor,   0.05, format="%.2f"))
    column_cracked_factor = float(st.sidebar.number_input("Column cracked factor", 0.10, 1.00, d.column_cracked_factor, 0.05, format="%.2f"))
    beam_cracked_factor   = float(st.sidebar.number_input("Beam cracked factor",   0.10, 1.00, d.beam_cracked_factor,   0.05, format="%.2f"))
    slab_cracked_factor   = float(st.sidebar.number_input("Slab cracked factor",   0.10, 1.00, d.slab_cracked_factor,   0.05, format="%.2f"))

    st.sidebar.subheader("Floor system")
    slab_thickness_m = float(st.sidebar.number_input("Slab thickness (m)",   0.10, 0.80, d.slab_thickness_m, 0.01, format="%.2f"))
    beam_width_m     = float(st.sidebar.number_input("Beam width (m)",        0.20, 2.00, d.beam_width_m,     0.05, format="%.2f"))
    beam_depth_m     = float(st.sidebar.number_input("Beam depth (m)",        0.30, 3.00, d.beam_depth_m,     0.05, format="%.2f"))

    st.sidebar.subheader("Circulation / services")
    stair_count            = int(st.sidebar.number_input("Stairs",       0, 20, d.stair_count, 1))
    elevator_count         = int(st.sidebar.number_input("Elevators",    0, 40, d.elevator_count, 1))
    elevator_area_each_m2  = float(st.sidebar.number_input("Elevator area each (m²)", 0.0, 20.0, d.elevator_area_each_m2, 0.1))
    stair_area_each_m2     = float(st.sidebar.number_input("Stair area each (m²)",    0.0, 50.0, d.stair_area_each_m2,    0.5))
    service_area_m2        = float(st.sidebar.number_input("Service area (m²)",        0.0, 100.0, d.service_area_m2,      0.5))
    corridor_factor        = float(st.sidebar.number_input("Corridor factor",           1.0, 2.0, d.corridor_factor,       0.05, format="%.2f"))

    st.sidebar.subheader("Zone member sizes")
    lower_zone  = zone_block("Lower",  d.lower_zone)
    middle_zone = zone_block("Middle", d.middle_zone)
    upper_zone  = zone_block("Upper",  d.upper_zone)

    st.sidebar.subheader("Steel braced outriggers (CHS)")
    outrigger_count = int(st.sidebar.selectbox("Number of outriggers", [0,1,2,3], index=0))
    suggested_levels= []
    if outrigger_count >= 1: suggested_levels.append(max(1, int(round(n_story*0.45))))
    if outrigger_count >= 2: suggested_levels.append(max(1, int(round(n_story*0.65))))
    if outrigger_count >= 3: suggested_levels.append(max(1, int(round(n_story*0.82))))
    outrigger_story_levels: List[int] = []
    for i in range(outrigger_count):
        lvl = int(st.sidebar.number_input(
            f"Outrigger level {i+1}", 1, max(1,n_story),
            min(suggested_levels[i], n_story), 1))
        outrigger_story_levels.append(lvl)
    brace_outer_diameter_mm    = float(st.sidebar.number_input("CHS outer diameter (mm)", 100.0, 2000.0, d.brace_outer_diameter_mm, 10.0))
    brace_thickness_mm         = float(st.sidebar.number_input("CHS thickness (mm)",        4.0,   80.0, d.brace_thickness_mm,       1.0))
    braces_per_side            = int(st.sidebar.number_input("Braces per side", 1, 8, d.braces_per_side, 1))
    outrigger_depth_m          = float(st.sidebar.number_input("Outrigger depth (m)", 0.5, 10.0, d.outrigger_depth_m, 0.1))
    brace_effective_length_factor = float(st.sidebar.number_input("Brace K factor", 0.5, 2.0, d.brace_effective_length_factor, 0.05, format="%.2f"))
    brace_buckling_reduction   = float(st.sidebar.number_input("Brace buckling reduction", 0.10, 1.00, d.brace_buckling_reduction, 0.05, format="%.2f"))

    return BuildingInput(
        n_story=n_story, n_basement=n_basement,
        story_height_m=story_height_m, basement_height_m=basement_height_m,
        plan_x_m=plan_x_m, plan_y_m=plan_y_m,
        n_bays_x=n_bays_x, n_bays_y=n_bays_y,
        bay_x_m=bay_x_m, bay_y_m=bay_y_m,
        core_outer_x_m=core_outer_x_m, core_outer_y_m=core_outer_y_m,
        core_opening_x_m=core_opening_x_m, core_opening_y_m=core_opening_y_m,
        lower_zone_wall_count=lower_zone_wall_count,
        middle_zone_wall_count=middle_zone_wall_count,
        upper_zone_wall_count=upper_zone_wall_count,
        lateral_wall_x_count=lateral_wall_x_count,
        lateral_wall_y_count=lateral_wall_y_count,
        lateral_wall_thickness_m=lateral_wall_thickness_m,
        lateral_wall_length_x_m=lateral_wall_length_x_m,
        lateral_wall_length_y_m=lateral_wall_length_y_m,
        braced_bays_x=braced_bays_x, braced_bays_y=braced_bays_y,
        braced_bay_stiffness_factor=braced_bay_stiffness_factor,
        retaining_wall_thickness_m=retaining_wall_thickness_m,
        retaining_wall_height_m=retaining_wall_height_m,
        soil_unit_weight_kn_m3=soil_unit_weight_kn_m3,
        soil_ka=soil_ka, surcharge_kn_m2=surcharge_kn_m2,
        fck_mpa=fck_mpa, Ec_mpa=Ec_mpa, fy_mpa=fy_mpa, Es_mpa=Es_mpa,
        dl_kn_m2=dl_kn_m2, ll_kn_m2=ll_kn_m2,
        superimposed_dead_kn_m2=superimposed_dead_kn_m2,
        facade_line_load_kn_m=facade_line_load_kn_m,
        live_load_mass_factor=live_load_mass_factor,
        seismic_base_shear_coeff=seismic_base_shear_coeff,
        use_response_spectrum=use_response_spectrum,
        spectrum_T0_s=spectrum_T0_s, spectrum_Ts_s=spectrum_Ts_s,
        spectrum_Sa_max_g=spectrum_Sa_max_g, spectrum_Sa_min_g=spectrum_Sa_min_g,
        response_modification_R=response_modification_R,
        importance_factor_I=importance_factor_I,
        drift_limit_ratio=drift_limit_ratio,
        wall_cracked_factor=wall_cracked_factor,
        column_cracked_factor=column_cracked_factor,
        beam_cracked_factor=beam_cracked_factor,
        slab_cracked_factor=slab_cracked_factor,
        slab_thickness_m=slab_thickness_m,
        beam_width_m=beam_width_m, beam_depth_m=beam_depth_m,
        lower_zone=lower_zone, middle_zone=middle_zone, upper_zone=upper_zone,
        stair_count=stair_count, elevator_count=elevator_count,
        elevator_area_each_m2=elevator_area_each_m2,
        stair_area_each_m2=stair_area_each_m2,
        service_area_m2=service_area_m2, corridor_factor=corridor_factor,
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
        "**v9.0 additions:** Beams shown in plan | Braced bays highlighted | "
        "Perimeter shear walls in plan & elevation | "
        "Basement retaining wall design (EC2) | "
        "Response spectrum base shear with T₁-dependent Sa."
    )

    for key in ("analysis_result", "input_data", "report_text"):
        if key not in st.session_state:
            st.session_state[key] = None

    inp = streamlit_input_panel()

    c1, c2 = st.columns(2)
    analyze_btn = c1.button("Analyze", use_container_width=True)
    clear_btn   = c2.button("Clear results", use_container_width=True)

    if clear_btn:
        st.session_state.analysis_result = None
        st.session_state.input_data      = None
        st.session_state.report_text     = None
        st.rerun()

    if analyze_btn:
        try:
            with st.spinner("Running rational analysis…"):
                result = analyze(inp)
                st.session_state.analysis_result = result
                st.session_state.input_data      = inp
                st.session_state.report_text     = build_text_report(inp, result)
            st.success("Analysis completed.")
        except Exception as exc:
            st.exception(exc)

    result    = st.session_state.analysis_result
    saved_inp = st.session_state.input_data
    if result is None or saved_inp is None:
        st.warning("Enter inputs and click **Analyze**.")
        return

    tabs = st.tabs(["Summary", "Story Tables", "Mode Shapes",
                    "Spectrum", "Plan / Elevation", "Outriggers",
                    "Retaining Wall", "Report"])

    with tabs[0]:
        st.subheader("Summary")
        st.dataframe(result.summary_table, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("T₁ (s)",          f"{result.periods_s[0]:.3f}" if result.periods_s else "-")
        m2.metric("Roof disp. (m)",   f"{result.floor_displacements_m[-1]:.4f}")
        m3.metric("Max drift ratio",  f"{max(result.story_drift_ratios):.6f}")
        m4.metric("Base shear (kN)",  f"{result.base_shear_kn:,.1f}")
        st.subheader("Zone stiffness table")
        st.dataframe(result.zone_table, use_container_width=True)

    with tabs[1]:
        st.subheader("Story response table")
        st.dataframe(result.story_table, use_container_width=True)
        st.pyplot(plot_story_response(result))
        st.pyplot(plot_story_drifts(result, saved_inp.drift_limit_ratio))
        st.download_button("Download story table CSV",
                           data=result.story_table.to_csv(index=False).encode(),
                           file_name="story_response.csv", mime="text/csv")

    with tabs[2]:
        st.subheader("Mode shapes")
        st.pyplot(plot_mode_shapes(result))

    with tabs[3]:
        st.subheader("Response spectrum")
        if saved_inp.use_response_spectrum:
            st.pyplot(plot_spectrum(saved_inp, result.periods_s[0] if result.periods_s else 1.0))
            st.info(f"Sa/g at T₁ = {result.spectrum_Sa_g:.4f}   |   "
                    f"Cs = Sa·I/R = {result.spectrum_Sa_g*saved_inp.importance_factor_I/saved_inp.response_modification_R:.4f}   |   "
                    f"Base shear = {result.base_shear_kn:,.1f} kN")
        else:
            st.info("Response spectrum not enabled. Activate it in the sidebar under 'Response spectrum'.")

    with tabs[4]:
        st.subheader("Plan and elevation")
        zone_names  = [z.zone_name for z in result.zone_results]
        sel_zone    = st.selectbox("Plan zone", zone_names, index=0)
        st.pyplot(plot_plan(saved_inp, result, sel_zone))
        st.pyplot(plot_elevation(saved_inp, result))

    with tabs[5]:
        st.subheader("Steel CHS outriggers")
        if result.outriggers:
            st.dataframe(result.outrigger_table, use_container_width=True)
        else:
            st.info("No outriggers defined.")

    with tabs[6]:
        st.subheader("Basement retaining wall")
        st.dataframe(result.retaining_table, use_container_width=True)
        if result.retaining_wall:
            st.pyplot(plot_retaining_wall(saved_inp, result.retaining_wall))
        else:
            st.info("No basement defined — set 'Basement stories' > 0.")

    with tabs[7]:
        st.subheader("Text report")
        st.text_area("Report", st.session_state.report_text, height=550)
        st.download_button("Download report TXT",
                           data=st.session_state.report_text,
                           file_name="tall_building_v9_report.txt",
                           mime="text/plain")


if __name__ == "__main__":
    main()
