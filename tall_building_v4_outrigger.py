from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception:  # allows offline validation without streamlit installed
    st = None

APP_TITLE = "Tall Building Rational Analysis + Auto Outrigger Layout"
APP_VERSION = "v9.1-rebuilt"
G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3
STEEL_DENSITY = 7850.0  # kg/m3


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

    def normalized(self, plan_x: float, plan_y: float) -> "PerimeterWallInput":
        return PerimeterWallInput(
            thickness_m=self.thickness_m,
            top_length_m=min(max(self.top_length_m, 0.0), plan_x),
            bottom_length_m=min(max(self.bottom_length_m, 0.0), plan_x),
            left_length_m=min(max(self.left_length_m, 0.0), plan_y),
            right_length_m=min(max(self.right_length_m, 0.0), plan_y),
        )


@dataclass
class RetainingWallInput:
    enabled: bool = True
    thickness_m: float = 0.40
    top_length_m: float = 0.0
    bottom_length_m: float = 0.0
    left_length_m: float = 0.0
    right_length_m: float = 0.0
    stiffness_reduction: float = 0.15

    def normalized(self, plan_x: float, plan_y: float) -> "RetainingWallInput":
        return RetainingWallInput(
            enabled=self.enabled,
            thickness_m=self.thickness_m,
            top_length_m=plan_x if self.top_length_m <= 0 else min(self.top_length_m, plan_x),
            bottom_length_m=plan_x if self.bottom_length_m <= 0 else min(self.bottom_length_m, plan_x),
            left_length_m=plan_y if self.left_length_m <= 0 else min(self.left_length_m, plan_y),
            right_length_m=plan_y if self.right_length_m <= 0 else min(self.right_length_m, plan_y),
            stiffness_reduction=clamp(self.stiffness_reduction, 0.02, 0.30),
        )


@dataclass
class BraceInput:
    outer_diameter_mm: float = 355.6
    thickness_mm: float = 16.0
    k_factor: float = 1.0
    buckling_reduction: float = 0.80


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
    core_opening_x_m: float = 11.5
    core_opening_y_m: float = 9.0

    Ec_mpa: float = 34000.0
    Es_mpa: float = 200000.0

    wall_cracked_factor: float = 0.50
    perimeter_wall_cracked_factor: float = 0.45
    column_cracked_factor: float = 0.70
    beam_cracked_factor: float = 0.35

    super_dead_load_kpa: float = 4.5
    live_load_kpa: float = 2.0
    facade_line_load_kn_m: float = 12.0
    seismic_mass_factor: float = 1.0

    lower_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(
        wall_thickness_m=0.70,
        beam_width_m=0.45,
        beam_depth_m=0.90,
        slab_thickness_m=0.24,
        corner_col_x_m=1.20,
        corner_col_y_m=1.20,
        perimeter_col_x_m=1.00,
        perimeter_col_y_m=1.00,
        interior_col_x_m=0.90,
        interior_col_y_m=0.90,
        wall_count=8,
    ))
    middle_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(
        wall_thickness_m=0.55,
        beam_width_m=0.40,
        beam_depth_m=0.80,
        slab_thickness_m=0.22,
        corner_col_x_m=1.00,
        corner_col_y_m=1.00,
        perimeter_col_x_m=0.85,
        perimeter_col_y_m=0.85,
        interior_col_x_m=0.75,
        interior_col_y_m=0.75,
        wall_count=6,
    ))
    upper_zone: ZoneMemberInput = field(default_factory=lambda: ZoneMemberInput(
        wall_thickness_m=0.40,
        beam_width_m=0.35,
        beam_depth_m=0.70,
        slab_thickness_m=0.20,
        corner_col_x_m=0.80,
        corner_col_y_m=0.80,
        perimeter_col_x_m=0.70,
        perimeter_col_y_m=0.70,
        interior_col_x_m=0.60,
        interior_col_y_m=0.60,
        wall_count=4,
    ))

    lower_perimeter_wall: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.35, 18.0, 18.0, 14.0, 14.0))
    middle_perimeter_wall: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.30, 14.0, 14.0, 10.0, 10.0))
    upper_perimeter_wall: PerimeterWallInput = field(default_factory=lambda: PerimeterWallInput(0.25, 10.0, 10.0, 8.0, 8.0))

    retaining_wall: RetainingWallInput = field(default_factory=RetainingWallInput)

    outrigger_story_levels: List[int] = field(default_factory=lambda: [24, 42])
    brace_input: BraceInput = field(default_factory=BraceInput)

    drift_limit_ratio: float = 1 / 500
    Ct: float = 0.0488
    x_exp: float = 0.75
    Cu: float = 1.4
    auto_size: bool = True


@dataclass
class StoryRow:
    story: int
    zone: str
    brace_level: str
    wall_t_m: float
    perimeter_wall_t_m: float
    beam_b_m: float
    beam_h_m: float
    slab_t_m: float
    corner_col_m: str
    perimeter_col_m: str
    interior_col_m: str
    story_stiffness_gn_m: float
    story_mass_ton: float


@dataclass
class ZoneSummary:
    zone: str
    stories: str
    core_wall_t_m: float
    perimeter_wall_t_m: float
    beam_b_m: float
    beam_h_m: float
    slab_t_m: float
    corner_col: str
    perimeter_col: str
    interior_col: str
    core_wall_count: int


@dataclass
class BraceLine:
    x0: float
    y0: float
    x1: float
    y1: float
    side: str
    target_label: str


@dataclass
class OutriggerLevelResult:
    story: int
    elevation_m: float
    brace_count: int
    od_mm: float
    t_mm: float
    area_cm2: float
    slenderness: float
    k_eff_gn_m: float


@dataclass
class ModalResult:
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[np.ndarray]
    effective_mass_ratios: List[float]


@dataclass
class AnalysisResult:
    inp: BuildingInput
    zones: List[ZoneDefinition]
    zone_summaries: List[ZoneSummary]
    story_rows: List[StoryRow]
    modal: ModalResult
    roof_disp_m: float
    max_story_drift_m: float
    max_story_drift_ratio: float
    story_displacements_m: np.ndarray
    story_drifts_m: np.ndarray
    story_stiffness_n_m: np.ndarray
    story_masses_kg: np.ndarray
    outrigger_results: List[OutriggerLevelResult]
    brace_lines_by_story: Dict[int, List[BraceLine]]
    t_code_s: float
    t_upper_s: float
    total_weight_kn: float


# ----------------------------- CORE LOGIC -----------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def define_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, int(round(0.30 * n_story)))
    z2 = max(z1 + 1, int(round(0.70 * n_story)))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def zone_for_story(zones: List[ZoneDefinition], story: int) -> ZoneDefinition:
    for z in zones:
        if z.story_start <= story <= z.story_end:
            return z
    raise ValueError(f"Story {story} out of range")


def zone_member(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    return {
        "Lower Zone": inp.lower_zone,
        "Middle Zone": inp.middle_zone,
        "Upper Zone": inp.upper_zone,
    }[zone_name]


def zone_perimeter_wall(inp: BuildingInput, zone_name: str) -> PerimeterWallInput:
    return {
        "Lower Zone": inp.lower_perimeter_wall,
        "Middle Zone": inp.middle_perimeter_wall,
        "Upper Zone": inp.upper_perimeter_wall,
    }[zone_name].normalized(inp.plan_x_m, inp.plan_y_m)


def normalize_input(inp: BuildingInput) -> BuildingInput:
    min_story_beam_max = 0.28 * inp.story_height_m

    def norm_zone(z: ZoneMemberInput, band: str) -> ZoneMemberInput:
        if band == "lower":
            wall_min, col_min = 0.55, 0.95
        elif band == "middle":
            wall_min, col_min = 0.40, 0.75
        else:
            wall_min, col_min = 0.30, 0.55
        beam_depth_min = max(0.55, 0.10 * max(inp.bay_x_m, inp.bay_y_m))
        return replace(
            z,
            wall_thickness_m=clamp(z.wall_thickness_m, wall_min, 1.20),
            beam_width_m=clamp(z.beam_width_m, 0.30, 0.70),
            beam_depth_m=clamp(z.beam_depth_m, max(0.60, beam_depth_min), min(1.40, min_story_beam_max)),
            slab_thickness_m=clamp(z.slab_thickness_m, 0.18, 0.35),
            corner_col_x_m=clamp(z.corner_col_x_m, col_min, 1.80),
            corner_col_y_m=clamp(z.corner_col_y_m, col_min, 1.80),
            perimeter_col_x_m=clamp(z.perimeter_col_x_m, col_min - 0.10, 1.60),
            perimeter_col_y_m=clamp(z.perimeter_col_y_m, col_min - 0.10, 1.60),
            interior_col_x_m=clamp(z.interior_col_x_m, col_min - 0.20, 1.40),
            interior_col_y_m=clamp(z.interior_col_y_m, col_min - 0.20, 1.40),
            wall_count=4 if z.wall_count <= 4 else (6 if z.wall_count <= 6 else 8),
        )

    return replace(
        inp,
        lower_zone=norm_zone(inp.lower_zone, "lower"),
        middle_zone=norm_zone(inp.middle_zone, "middle"),
        upper_zone=norm_zone(inp.upper_zone, "upper"),
        lower_perimeter_wall=replace(inp.lower_perimeter_wall.normalized(inp.plan_x_m, inp.plan_y_m), thickness_m=clamp(inp.lower_perimeter_wall.thickness_m, 0.25, 0.60)),
        middle_perimeter_wall=replace(inp.middle_perimeter_wall.normalized(inp.plan_x_m, inp.plan_y_m), thickness_m=clamp(inp.middle_perimeter_wall.thickness_m, 0.22, 0.50)),
        upper_perimeter_wall=replace(inp.upper_perimeter_wall.normalized(inp.plan_x_m, inp.plan_y_m), thickness_m=clamp(inp.upper_perimeter_wall.thickness_m, 0.20, 0.45)),
        retaining_wall=inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m),
        outrigger_story_levels=sorted({s for s in inp.outrigger_story_levels if 1 <= s <= inp.n_story}),
    )


def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height_m


def plan_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m


def core_wall_lengths(inp: BuildingInput, wall_count: int) -> List[float]:
    ox, oy = inp.core_outer_x_m, inp.core_outer_y_m
    base = [ox, ox, oy, oy]
    if wall_count >= 6:
        base += [0.45 * ox, 0.45 * ox]
    if wall_count >= 8:
        base += [0.45 * oy, 0.45 * oy]
    return base


def wall_local_I(length: float, thickness: float) -> float:
    return thickness * length ** 3 / 12.0


def zone_column_counts(inp: BuildingInput) -> Tuple[int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return corner, perimeter, interior


def core_story_stiffness(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    lengths = core_wall_lengths(inp, z.wall_count)
    # local wall action only; no huge plan-width parallel-axis inflation
    k = 0.0
    for L in lengths:
        I = wall_local_I(L, z.wall_thickness_m)
        k += 12.0 * E * inp.wall_cracked_factor * I / (h ** 3)
    # coupling penalty because walls form perforated core, not independent perfect cantilevers
    return 0.0022 * k


def perimeter_wall_story_stiffness(inp: BuildingInput, p: PerimeterWallInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    lengths = [p.top_length_m, p.bottom_length_m, p.left_length_m, p.right_length_m]
    k = 0.0
    for L in lengths:
        if L <= 0.05:
            continue
        I = wall_local_I(L, p.thickness_m)
        k += 12.0 * E * inp.perimeter_wall_cracked_factor * I / (h ** 3)
    # strong reduction because not all façade walls participate fully in one translational direction
    return 0.0008 * k


def column_story_stiffness(inp: BuildingInput, z: ZoneMemberInput) -> float:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height_m
    corner_n, perim_n, inter_n = zone_column_counts(inp)

    def I_rect(b: float, d: float) -> float:
        return max(b * d ** 3 / 12.0, d * b ** 3 / 12.0)

    k = 0.0
    k += corner_n * 12 * E * inp.column_cracked_factor * I_rect(z.corner_col_x_m, z.corner_col_y_m) / (h ** 3)
    k += perim_n * 12 * E * inp.column_cracked_factor * I_rect(z.perimeter_col_x_m, z.perimeter_col_y_m) / (h ** 3)
    k += inter_n * 12 * E * inp.column_cracked_factor * I_rect(z.interior_col_x_m, z.interior_col_y_m) / (h ** 3)
    return 0.012 * k


def beam_coupling_factor(inp: BuildingInput, z: ZoneMemberInput) -> float:
    span = 0.5 * (inp.bay_x_m + inp.bay_y_m)
    E = inp.Ec_mpa * 1e6
    Ib = z.beam_width_m * z.beam_depth_m ** 3 / 12.0
    kb = 12 * E * inp.beam_cracked_factor * Ib / (span ** 3)
    # transform to moderate continuity factor for frame action
    return 1.0 + clamp(kb / 1.2e8, 0.05, 0.30)


def floor_mass_kg(inp: BuildingInput, z: ZoneMemberInput, p: PerimeterWallInput) -> float:
    area = plan_area(inp)
    q = (inp.super_dead_load_kpa + 0.25 * inp.live_load_kpa)  # seismic mass load in kN/m²
    slab_w = z.slab_thickness_m * area * CONCRETE_UNIT_WEIGHT
    façade = inp.facade_line_load_kn_m * (2 * (inp.plan_x_m + inp.plan_y_m))
    lengths = sum(core_wall_lengths(inp, z.wall_count))
    core_w = lengths * z.wall_thickness_m * inp.story_height_m * CONCRETE_UNIT_WEIGHT
    perim_w = (p.top_length_m + p.bottom_length_m + p.left_length_m + p.right_length_m) * p.thickness_m * inp.story_height_m * CONCRETE_UNIT_WEIGHT
    corner_n, perim_n, inter_n = zone_column_counts(inp)
    col_area = (
        corner_n * z.corner_col_x_m * z.corner_col_y_m
        + perim_n * z.perimeter_col_x_m * z.perimeter_col_y_m
        + inter_n * z.interior_col_x_m * z.interior_col_y_m
    )
    col_w = col_area * inp.story_height_m * CONCRETE_UNIT_WEIGHT
    beam_len = inp.n_bays_y * (inp.n_bays_x + 1) * inp.bay_x_m + inp.n_bays_x * (inp.n_bays_y + 1) * inp.bay_y_m
    beam_w = beam_len * z.beam_width_m * z.beam_depth_m * CONCRETE_UNIT_WEIGHT
    total_kn = (q * area + slab_w + façade + core_w + perim_w + col_w + beam_w) * inp.seismic_mass_factor
    return total_kn * 1000.0 / G


def chs_properties(od_mm: float, t_mm: float) -> Tuple[float, float]:
    D = od_mm / 1000.0
    t = min(t_mm / 1000.0, 0.49 * D)
    A = pi / 4.0 * (D ** 2 - (D - 2 * t) ** 2)
    I = pi / 64.0 * (D ** 4 - (D - 2 * t) ** 4)
    r = sqrt(I / max(A, 1e-12))
    return A, r


def auto_generated_brace_lines(inp: BuildingInput) -> Dict[int, List[BraceLine]]:
    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    cx1 = cx0 + inp.core_outer_x_m
    cy0 = (inp.plan_y_m - inp.core_outer_y_m) / 2.0
    cy1 = cy0 + inp.core_outer_y_m

    x_cols = [i * inp.bay_x_m for i in range(inp.n_bays_x + 1)]
    y_cols = [j * inp.bay_y_m for j in range(inp.n_bays_y + 1)]

    center_x = inp.plan_x_m / 2.0
    center_y = inp.plan_y_m / 2.0

    def nearest_two(values: List[float], center: float, lo: float, hi: float) -> List[float]:
        cand = [v for v in values if v < lo or v > hi]
        cand.sort(key=lambda v: abs(v - center))
        chosen = cand[:2] if len(cand) >= 2 else cand[:1]
        return sorted(chosen)

    top_targets = nearest_two(x_cols, center_x, cx0, cx1)
    bot_targets = top_targets[:]
    left_targets = nearest_two(y_cols, center_y, cy0, cy1)
    right_targets = left_targets[:]

    lines: List[BraceLine] = []
    for x in top_targets:
        lines.append(BraceLine((cx0 + cx1) / 2, cy1, x, inp.plan_y_m, "top", f"X={x:.1f}"))
    for x in bot_targets:
        lines.append(BraceLine((cx0 + cx1) / 2, cy0, x, 0.0, "bottom", f"X={x:.1f}"))
    for y in left_targets:
        lines.append(BraceLine(cx0, (cy0 + cy1) / 2, 0.0, y, "left", f"Y={y:.1f}"))
    for y in right_targets:
        lines.append(BraceLine(cx1, (cy0 + cy1) / 2, inp.plan_x_m, y, "right", f"Y={y:.1f}"))

    return {story: lines[:] for story in inp.outrigger_story_levels}


def brace_story_stiffness(inp: BuildingInput, brace_lines: List[BraceLine]) -> Tuple[float, OutriggerLevelResult]:
    E = inp.Es_mpa * 1e6
    A, r = chs_properties(inp.brace_input.outer_diameter_mm, inp.brace_input.thickness_mm)
    h = inp.story_height_m
    k_total = 0.0
    lam_max = 0.0
    for line in brace_lines:
        dx = line.x1 - line.x0
        dy = line.y1 - line.y0
        horizontal = sqrt(dx * dx + dy * dy)
        L = sqrt(horizontal ** 2 + h ** 2)
        cos2 = (horizontal / max(L, 1e-9)) ** 2
        lam = inp.brace_input.k_factor * L / max(r, 1e-9)
        lam_max = max(lam_max, lam)
        k_axial = E * A / max(L, 1e-9)
        k_total += inp.brace_input.buckling_reduction * k_axial * cos2
    k_total *= 15.00  # equivalent outrigger system efficiency
    story = 0
    res = OutriggerLevelResult(
        story=story,
        elevation_m=0.0,
        brace_count=len(brace_lines),
        od_mm=inp.brace_input.outer_diameter_mm,
        t_mm=inp.brace_input.thickness_mm,
        area_cm2=A * 1e4,
        slenderness=lam_max,
        k_eff_gn_m=k_total / 1e9,
    )
    return k_total, res


def retaining_base_spring(inp: BuildingInput) -> float:
    rw = inp.retaining_wall.normalized(inp.plan_x_m, inp.plan_y_m)
    if not rw.enabled or inp.n_basement <= 0:
        return 0.0
    E = inp.Ec_mpa * 1e6
    h = max(inp.basement_height_m * max(inp.n_basement, 1), 1.0)
    lengths = [rw.top_length_m, rw.bottom_length_m, rw.left_length_m, rw.right_length_m]
    k = 0.0
    for L in lengths:
        I = wall_local_I(L, rw.thickness_m)
        k += 3.0 * E * I / (h ** 3)
    return 0.03 * rw.stiffness_reduction * k


def build_story_arrays(inp: BuildingInput) -> Tuple[np.ndarray, np.ndarray, List[StoryRow], List[ZoneSummary], List[OutriggerLevelResult], Dict[int, List[BraceLine]]]:
    zones = define_zones(inp.n_story)
    brace_lines_by_story = auto_generated_brace_lines(inp)
    k_story: List[float] = []
    m_story: List[float] = []
    rows: List[StoryRow] = []
    outrigger_rows: List[OutriggerLevelResult] = []

    base_spring = retaining_base_spring(inp)

    for story in range(1, inp.n_story + 1):
        zone = zone_for_story(zones, story)
        zm = zone_member(inp, zone.name)
        pw = zone_perimeter_wall(inp, zone.name)

        k_core = core_story_stiffness(inp, zm)
        k_perim = perimeter_wall_story_stiffness(inp, pw)
        k_cols = column_story_stiffness(inp, zm)
        frame_factor = beam_coupling_factor(inp, zm)
        k_frame = k_cols * frame_factor
        k = k_core + k_perim + k_frame

        brace_level = "No"
        if story in brace_lines_by_story:
            k_br, br_res = brace_story_stiffness(inp, brace_lines_by_story[story])
            brace_level = "Yes"
            br_res.story = story
            br_res.elevation_m = story * inp.story_height_m
            outrigger_rows.append(br_res)
            k += k_br

        if story == 1 and base_spring > 0.0:
            k += base_spring

        m = floor_mass_kg(inp, zm, pw)
        k_story.append(k)
        m_story.append(m)
        rows.append(StoryRow(
            story=story,
            zone=zone.name,
            brace_level=brace_level,
            wall_t_m=zm.wall_thickness_m,
            perimeter_wall_t_m=pw.thickness_m,
            beam_b_m=zm.beam_width_m,
            beam_h_m=zm.beam_depth_m,
            slab_t_m=zm.slab_thickness_m,
            corner_col_m=f"{zm.corner_col_x_m:.2f}x{zm.corner_col_y_m:.2f}",
            perimeter_col_m=f"{zm.perimeter_col_x_m:.2f}x{zm.perimeter_col_y_m:.2f}",
            interior_col_m=f"{zm.interior_col_x_m:.2f}x{zm.interior_col_y_m:.2f}",
            story_stiffness_gn_m=k / 1e9,
            story_mass_ton=m / 1000.0,
        ))

    zone_summaries = []
    for z in zones:
        zm = zone_member(inp, z.name)
        pw = zone_perimeter_wall(inp, z.name)
        zone_summaries.append(ZoneSummary(
            zone=z.name,
            stories=f"{z.story_start}-{z.story_end}",
            core_wall_t_m=zm.wall_thickness_m,
            perimeter_wall_t_m=pw.thickness_m,
            beam_b_m=zm.beam_width_m,
            beam_h_m=zm.beam_depth_m,
            slab_t_m=zm.slab_thickness_m,
            corner_col=f"{zm.corner_col_x_m:.2f}x{zm.corner_col_y_m:.2f}",
            perimeter_col=f"{zm.perimeter_col_x_m:.2f}x{zm.perimeter_col_y_m:.2f}",
            interior_col=f"{zm.interior_col_x_m:.2f}x{zm.interior_col_y_m:.2f}",
            core_wall_count=zm.wall_count,
        ))

    return np.array(k_story, dtype=float), np.array(m_story, dtype=float), rows, zone_summaries, outrigger_rows, brace_lines_by_story


def build_m_k(masses: np.ndarray, stiffness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(masses)
    M = np.diag(masses)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = stiffness[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K


def modal_analysis(masses: np.ndarray, stiffness: np.ndarray, modes: int = 5) -> ModalResult:
    M, K = build_m_k(masses, stiffness)
    A = np.linalg.solve(M, K)
    vals, vecs = np.linalg.eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    mask = vals > 1e-9
    vals = vals[mask]
    vecs = vecs[:, mask]
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    omegas = np.sqrt(vals[:modes])
    periods = [2 * pi / w for w in omegas]
    freqs = [w / (2 * pi) for w in omegas]

    ones = np.ones((len(masses), 1))
    Mmat = np.diag(masses)
    total_m = masses.sum()
    mode_shapes = []
    ratios = []
    for i in range(len(periods)):
        phi = vecs[:, i].reshape(-1, 1)
        denom = (phi.T @ Mmat @ phi).item()
        gamma = (phi.T @ Mmat @ ones).item() / denom
        meff = gamma ** 2 * denom
        ratios.append(meff / total_m)
        shp = phi.flatten()
        shp = shp / max(abs(shp[-1]), 1e-12)
        if shp[-1] < 0:
            shp = -shp
        mode_shapes.append(shp)
    return ModalResult(periods_s=periods, frequencies_hz=freqs, mode_shapes=mode_shapes, effective_mass_ratios=ratios)


def static_response(masses: np.ndarray, stiffness: np.ndarray, base_shear_ratio: float = 0.015) -> Tuple[np.ndarray, np.ndarray, float, float]:
    M, K = build_m_k(masses, stiffness)
    W = masses.sum() * G
    V = base_shear_ratio * W
    n = len(masses)
    heights = np.arange(1, n + 1, dtype=float)
    F = V * heights / heights.sum()
    u = np.linalg.solve(K, F)
    drifts = np.empty_like(u)
    drifts[0] = u[0]
    drifts[1:] = u[1:] - u[:-1]
    return u, drifts, float(u[-1]), float(np.max(np.abs(drifts)))


def code_period(inp: BuildingInput) -> float:
    return inp.Ct * total_height(inp) ** inp.x_exp


def scale_zone_member(inp: BuildingInput, z: ZoneMemberInput, factor: float, band: str) -> ZoneMemberInput:
    story_beam_max = 0.28 * inp.story_height_m
    if band == "lower":
        wall_min, col_min = 0.55, 0.95
    elif band == "middle":
        wall_min, col_min = 0.40, 0.75
    else:
        wall_min, col_min = 0.30, 0.55
    return replace(
        z,
        wall_thickness_m=clamp(z.wall_thickness_m * factor, wall_min, 1.20),
        beam_width_m=clamp(z.beam_width_m * (0.90 + 0.10 * factor), 0.30, 0.75),
        beam_depth_m=clamp(z.beam_depth_m * factor, 0.60, min(1.40, story_beam_max)),
        slab_thickness_m=clamp(z.slab_thickness_m * (0.95 + 0.05 * factor), 0.18, 0.35),
        corner_col_x_m=clamp(z.corner_col_x_m * factor, col_min, 1.80),
        corner_col_y_m=clamp(z.corner_col_y_m * factor, col_min, 1.80),
        perimeter_col_x_m=clamp(z.perimeter_col_x_m * factor, col_min - 0.10, 1.60),
        perimeter_col_y_m=clamp(z.perimeter_col_y_m * factor, col_min - 0.10, 1.60),
        interior_col_x_m=clamp(z.interior_col_x_m * factor, col_min - 0.20, 1.40),
        interior_col_y_m=clamp(z.interior_col_y_m * factor, col_min - 0.20, 1.40),
    )


def scale_perimeter_wall(p: PerimeterWallInput, factor: float, band: str) -> PerimeterWallInput:
    if band == "lower":
        tmin = 0.25
    elif band == "middle":
        tmin = 0.22
    else:
        tmin = 0.20
    return replace(p, thickness_m=clamp(p.thickness_m * factor, tmin, 0.60))


def auto_size(inp: BuildingInput, iterations: int = 16) -> BuildingInput:
    if not inp.auto_size:
        return inp
    work = normalize_input(inp)
    T_code = code_period(work)
    T_low = 0.65 * T_code
    T_high = work.Cu * T_code

    for _ in range(iterations):
        k, m, *_ = build_story_arrays(work)
        modal = modal_analysis(m, k, modes=1)
        T1 = modal.periods_s[0]
        _, dr, _, dmax = static_response(m, k)
        drift_ratio = dmax / work.story_height_m

        need_stiffer = T1 > T_high or drift_ratio > work.drift_limit_ratio
        need_softer = T1 < T_low and drift_ratio < 0.65 * work.drift_limit_ratio
        if not need_stiffer and not need_softer:
            break
        fac = 1.08 if need_stiffer else 0.94
        work = replace(
            work,
            lower_zone=scale_zone_member(work, work.lower_zone, fac, "lower"),
            middle_zone=scale_zone_member(work, work.middle_zone, fac, "middle"),
            upper_zone=scale_zone_member(work, work.upper_zone, fac, "upper"),
            lower_perimeter_wall=scale_perimeter_wall(work.lower_perimeter_wall, fac, "lower"),
            middle_perimeter_wall=scale_perimeter_wall(work.middle_perimeter_wall, fac, "middle"),
            upper_perimeter_wall=scale_perimeter_wall(work.upper_perimeter_wall, fac, "upper"),
        )
    return work


def analyze(inp: BuildingInput) -> AnalysisResult:
    inp = normalize_input(inp)
    inp = auto_size(inp)
    zones = define_zones(inp.n_story)
    k, m, story_rows, zone_summaries, outrigger_rows, brace_lines_by_story = build_story_arrays(inp)
    modal = modal_analysis(m, k, modes=5)
    u, drifts, roof_disp, max_drift = static_response(m, k)
    total_weight_kn = float(m.sum() * G / 1000.0)
    return AnalysisResult(
        inp=inp,
        zones=zones,
        zone_summaries=zone_summaries,
        story_rows=story_rows,
        modal=modal,
        roof_disp_m=roof_disp,
        max_story_drift_m=max_drift,
        max_story_drift_ratio=max_drift / inp.story_height_m,
        story_displacements_m=u,
        story_drifts_m=drifts,
        story_stiffness_n_m=k,
        story_masses_kg=m,
        outrigger_results=outrigger_rows,
        brace_lines_by_story=brace_lines_by_story,
        t_code_s=code_period(inp),
        t_upper_s=inp.Cu * code_period(inp),
        total_weight_kn=total_weight_kn,
    )


# ----------------------------- TABLES -----------------------------

def summary_df(res: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame({
        "Parameter": [
            "Height (m)", "Stories", "Basements", "Plan X (m)", "Plan Y (m)",
            "Core outer X (m)", "Core outer Y (m)", "Core opening X (m)", "Core opening Y (m)",
            "T1 (s)", "T2 (s)", "T3 (s)", "Code reference period T_code (s)", "Upper code period Cu*T_code (s)",
            "Roof displacement (m)", "Max story drift (m)", "Max story drift ratio", "Total weight (kN)",
            "Retaining wall active", "Outrigger levels",
        ],
        "Value": [
            total_height(res.inp), res.inp.n_story, res.inp.n_basement, res.inp.plan_x_m, res.inp.plan_y_m,
            res.inp.core_outer_x_m, res.inp.core_outer_y_m, res.inp.core_opening_x_m, res.inp.core_opening_y_m,
            res.modal.periods_s[0], res.modal.periods_s[1], res.modal.periods_s[2], res.t_code_s, res.t_upper_s,
            res.roof_disp_m, res.max_story_drift_m, res.max_story_drift_ratio, res.total_weight_kn,
            "Yes" if res.inp.retaining_wall.enabled and res.inp.n_basement > 0 else "No",
            ", ".join(str(s) for s in res.inp.outrigger_story_levels) if res.inp.outrigger_story_levels else "None",
        ]
    })


def zones_df(res: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame([z.__dict__ for z in res.zone_summaries])


def stories_df(res: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in res.story_rows])


def outriggers_df(res: AnalysisResult) -> pd.DataFrame:
    if not res.outrigger_results:
        return pd.DataFrame(columns=["story", "elevation_m", "brace_count", "od_mm", "t_mm", "area_cm2", "slenderness", "k_eff_gn_m"])
    return pd.DataFrame([o.__dict__ for o in res.outrigger_results])


def drifts_df(res: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame({
        "story": np.arange(1, res.inp.n_story + 1),
        "displacement_m": res.story_displacements_m,
        "story_drift_m": res.story_drifts_m,
        "story_drift_ratio": res.story_drifts_m / res.inp.story_height_m,
    })


# ----------------------------- PLOTS -----------------------------

def plot_plan(res: AnalysisResult, story_for_braces: Optional[int] = None) -> plt.Figure:
    inp = res.inp
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black", lw=1.5)

    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#d0d0d0", lw=0.7)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#d0d0d0", lw=0.7)

    cx0 = (inp.plan_x_m - inp.core_outer_x_m) / 2.0
    cy0 = (inp.plan_y_m - inp.core_outer_y_m) / 2.0
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core_outer_x_m, inp.core_outer_y_m, fill=False, edgecolor="#007f5f", lw=2.2))
    ox0 = (inp.plan_x_m - inp.core_opening_x_m) / 2.0
    oy0 = (inp.plan_y_m - inp.core_opening_y_m) / 2.0
    ax.add_patch(plt.Rectangle((ox0, oy0), inp.core_opening_x_m, inp.core_opening_y_m, fill=False, edgecolor="#555", lw=1.2, ls="--"))

    # perimeter walls by zone lengths shown using lower zone for plan footprint reference
    pw = zone_perimeter_wall(inp, "Lower Zone")
    t = pw.thickness_m
    sx = (inp.plan_x_m - pw.top_length_m) / 2.0
    sy = (inp.plan_y_m - pw.left_length_m) / 2.0
    if pw.top_length_m > 0:
        ax.add_patch(plt.Rectangle((sx, inp.plan_y_m - t), pw.top_length_m, t, color="#9ecae1", alpha=0.8))
    if pw.bottom_length_m > 0:
        ax.add_patch(plt.Rectangle((sx, 0), pw.bottom_length_m, t, color="#9ecae1", alpha=0.8))
    if pw.left_length_m > 0:
        ax.add_patch(plt.Rectangle((0, sy), t, pw.left_length_m, color="#9ecae1", alpha=0.8))
    if pw.right_length_m > 0:
        ax.add_patch(plt.Rectangle((inp.plan_x_m - t, sy), t, pw.right_length_m, color="#9ecae1", alpha=0.8))

    story = story_for_braces or (res.inp.outrigger_story_levels[0] if res.inp.outrigger_story_levels else None)
    if story in res.brace_lines_by_story:
        for line in res.brace_lines_by_story[story]:
            ax.plot([line.x0, line.x1], [line.y0, line.y1], color="red", lw=2.0)

    ax.set_title(f"Plan view{' - brace story ' + str(story) if story else ''}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(False)
    return fig


def plot_elevation(res: AnalysisResult) -> plt.Figure:
    inp = res.inp
    H = total_height(inp)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot([0, 0], [0, H], color="black", lw=2)
    ax.plot([inp.plan_x_m, inp.plan_x_m], [0, H], color="black", lw=2)
    for i in range(inp.n_story + 1):
        y = i * inp.story_height_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#d9d9d9", lw=0.5)

    zone_colors = {"Lower Zone": "#cfe8cf", "Middle Zone": "#fff2ae", "Upper Zone": "#fdd0cf"}
    for z in res.zones:
        y0 = (z.story_start - 1) * inp.story_height_m
        y1 = z.story_end * inp.story_height_m
        ax.axhspan(y0, y1, color=zone_colors[z.name], alpha=0.35)
        ax.text(inp.plan_x_m + 0.8, 0.5 * (y0 + y1), f"{z.name}\n{z.story_start}-{z.story_end}", va="center")

    for story in inp.outrigger_story_levels:
        y = story * inp.story_height_m
        ax.plot([0, inp.plan_x_m], [y, y], color="red", lw=2.8)
        ax.text(inp.plan_x_m / 2, y + 0.5, f"Outrigger @ story {story}", color="red", ha="center")

    if inp.n_basement > 0 and inp.retaining_wall.enabled:
        bh = inp.n_basement * inp.basement_height_m
        ax.axhspan(-bh, 0, color="#b3cde3", alpha=0.45)
        ax.text(inp.plan_x_m + 0.8, -bh / 2, "Basement\nretaining wall active", va="center")
        ax.plot([0, 0], [-bh, 0], color="#1f78b4", lw=5)
        ax.plot([inp.plan_x_m, inp.plan_x_m], [-bh, 0], color="#1f78b4", lw=5)

    ax.set_xlim(-1, inp.plan_x_m + 12)
    ax.set_ylim(-inp.n_basement * inp.basement_height_m - 1, H + 2)
    ax.set_xlabel("Building width reference (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Elevation with zones, basement, and outrigger levels")
    return fig


def plot_mode(res: AnalysisResult, mode_index: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, res.inp.n_story + 1)
    shp = res.modal.mode_shapes[mode_index]
    ax.plot(shp, y, marker="o")
    ax.set_title(f"Mode {mode_index + 1} - T = {res.modal.periods_s[mode_index]:.3f} s")
    ax.set_xlabel("Normalized shape")
    ax.set_ylabel("Story")
    ax.grid(True, alpha=0.3)
    return fig


def plot_drifts(res: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, res.inp.n_story + 1)
    ax.plot(res.story_drifts_m / res.inp.story_height_m, y, marker="o")
    ax.axvline(res.inp.drift_limit_ratio, color="red", ls="--", label="limit")
    ax.set_xlabel("Story drift ratio")
    ax.set_ylabel("Story")
    ax.set_title("Story drift ratios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


# ----------------------------- STREAMLIT UI -----------------------------

def numeric(label: str, value: float, min_value: float, max_value: float, step: float, fmt: Optional[str] = None) -> float:
    if st is None:
        return value
    kwargs = dict(label=label, min_value=min_value, max_value=max_value, value=value, step=step)
    if fmt is not None:
        kwargs["format"] = fmt
    return float(st.sidebar.number_input(**kwargs))


def integer(label: str, value: int, min_value: int, max_value: int, step: int = 1) -> int:
    if st is None:
        return value
    return int(st.sidebar.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step))


def zone_block(title: str, z: ZoneMemberInput) -> ZoneMemberInput:
    if st is None:
        return z
    with st.sidebar.expander(title, expanded=(title == "Lower Zone")):
        wall_t = float(st.number_input(f"{title} core wall thickness (m)", 0.20, 1.20, z.wall_thickness_m, 0.01))
        wall_count = int(st.selectbox(f"{title} core wall count", [4, 6, 8], index=[4, 6, 8].index(z.wall_count)))
        beam_b = float(st.number_input(f"{title} beam width (m)", 0.25, 0.80, z.beam_width_m, 0.01))
        beam_h = float(st.number_input(f"{title} beam depth (m)", 0.50, 1.40, z.beam_depth_m, 0.01))
        slab_t = float(st.number_input(f"{title} slab thickness (m)", 0.15, 0.35, z.slab_thickness_m, 0.01))
        c1, c2 = st.columns(2)
        corner_x = float(c1.number_input(f"{title} corner col bx", 0.40, 1.80, z.corner_col_x_m, 0.01))
        corner_y = float(c2.number_input(f"{title} corner col by", 0.40, 1.80, z.corner_col_y_m, 0.01))
        c1, c2 = st.columns(2)
        per_x = float(c1.number_input(f"{title} perimeter col bx", 0.40, 1.60, z.perimeter_col_x_m, 0.01))
        per_y = float(c2.number_input(f"{title} perimeter col by", 0.40, 1.60, z.perimeter_col_y_m, 0.01))
        c1, c2 = st.columns(2)
        in_x = float(c1.number_input(f"{title} interior col bx", 0.40, 1.40, z.interior_col_x_m, 0.01))
        in_y = float(c2.number_input(f"{title} interior col by", 0.40, 1.40, z.interior_col_y_m, 0.01))
    return ZoneMemberInput(wall_t, beam_b, beam_h, slab_t, corner_x, corner_y, per_x, per_y, in_x, in_y, wall_count)


def perimeter_wall_block(title: str, p: PerimeterWallInput, plan_x: float, plan_y: float) -> PerimeterWallInput:
    if st is None:
        return p
    with st.sidebar.expander(title, expanded=False):
        t = float(st.number_input(f"{title} thickness (m)", 0.15, 0.60, p.thickness_m, 0.01))
        top = float(st.number_input(f"{title} top length (m)", 0.0, plan_x, p.top_length_m, 0.5))
        bottom = float(st.number_input(f"{title} bottom length (m)", 0.0, plan_x, p.bottom_length_m, 0.5))
        left = float(st.number_input(f"{title} left length (m)", 0.0, plan_y, p.left_length_m, 0.5))
        right = float(st.number_input(f"{title} right length (m)", 0.0, plan_y, p.right_length_m, 0.5))
    return PerimeterWallInput(t, top, bottom, left, right)


def streamlit_input_panel() -> BuildingInput:
    default = BuildingInput()
    if st is None:
        return default

    st.sidebar.header("Input Data")
    n_story = integer("Above-grade stories", default.n_story, 10, 120)
    n_basement = integer("Basement stories", default.n_basement, 0, 10)
    story_h = numeric("Story height (m)", default.story_height_m, 2.8, 5.0, 0.1)
    base_h = numeric("Basement height (m)", default.basement_height_m, 2.8, 5.0, 0.1)
    plan_x = numeric("Plan X (m)", default.plan_x_m, 20.0, 120.0, 1.0)
    plan_y = numeric("Plan Y (m)", default.plan_y_m, 20.0, 120.0, 1.0)
    bays_x = integer("Bays in X", default.n_bays_x, 3, 12)
    bays_y = integer("Bays in Y", default.n_bays_y, 3, 12)
    bay_x = numeric("Bay X (m)", default.bay_x_m, 5.0, 12.0, 0.5)
    bay_y = numeric("Bay Y (m)", default.bay_y_m, 5.0, 12.0, 0.5)

    st.sidebar.subheader("Core")
    core_outer_x = numeric("Core outer X (m)", default.core_outer_x_m, 8.0, plan_x - 6.0, 0.5)
    core_outer_y = numeric("Core outer Y (m)", default.core_outer_y_m, 8.0, plan_y - 6.0, 0.5)
    core_open_x = numeric("Core opening X (m)", default.core_opening_x_m, 4.0, core_outer_x - 2.0, 0.5)
    core_open_y = numeric("Core opening Y (m)", default.core_opening_y_m, 4.0, core_outer_y - 2.0, 0.5)

    st.sidebar.subheader("Materials / cracked factors")
    Ec = numeric("Ec (MPa)", default.Ec_mpa, 25000.0, 45000.0, 500.0)
    wall_cf = numeric("Core wall cracked factor", default.wall_cracked_factor, 0.20, 1.0, 0.05)
    pwall_cf = numeric("Perimeter wall cracked factor", default.perimeter_wall_cracked_factor, 0.20, 1.0, 0.05)
    col_cf = numeric("Column cracked factor", default.column_cracked_factor, 0.30, 1.0, 0.05)
    beam_cf = numeric("Beam cracked factor", default.beam_cracked_factor, 0.20, 1.0, 0.05)

    st.sidebar.subheader("Loads")
    sdl = numeric("Super dead load (kPa)", default.super_dead_load_kpa, 1.0, 10.0, 0.1)
    ll = numeric("Live load (kPa)", default.live_load_kpa, 0.5, 6.0, 0.1)
    facade = numeric("Facade line load (kN/m)", default.facade_line_load_kn_m, 2.0, 25.0, 0.5)

    lower_zone = zone_block("Lower Zone", default.lower_zone)
    middle_zone = zone_block("Middle Zone", default.middle_zone)
    upper_zone = zone_block("Upper Zone", default.upper_zone)

    lower_pw = perimeter_wall_block("Lower perimeter wall", default.lower_perimeter_wall, plan_x, plan_y)
    middle_pw = perimeter_wall_block("Middle perimeter wall", default.middle_perimeter_wall, plan_x, plan_y)
    upper_pw = perimeter_wall_block("Upper perimeter wall", default.upper_perimeter_wall, plan_x, plan_y)

    st.sidebar.subheader("Basement retaining wall")
    rw_enabled = st.sidebar.checkbox("Retaining wall active", value=default.retaining_wall.enabled)
    rw_t = numeric("Retaining wall thickness (m)", default.retaining_wall.thickness_m, 0.25, 0.80, 0.01)
    rw_red = numeric("Retaining wall stiffness reduction", default.retaining_wall.stiffness_reduction, 0.02, 0.40, 0.01)

    st.sidebar.subheader("Outrigger")
    og_count = integer("Number of outrigger levels", len(default.outrigger_story_levels), 0, 4)
    og_levels = []
    suggested = [max(5, int(round(n_story * 0.4))), max(8, int(round(n_story * 0.7))), max(10, int(round(n_story * 0.85))), max(12, int(round(n_story * 0.93)))]
    for i in range(og_count):
        og_levels.append(integer(f"Outrigger level {i+1}", min(suggested[i], n_story), 1, n_story))
    od = numeric("Brace CHS OD (mm)", default.brace_input.outer_diameter_mm, 168.0, 610.0, 1.0)
    tt = numeric("Brace CHS thickness (mm)", default.brace_input.thickness_mm, 8.0, 40.0, 1.0)
    kf = numeric("Brace K factor", default.brace_input.k_factor, 0.7, 1.5, 0.05)
    br_red = numeric("Brace buckling reduction", default.brace_input.buckling_reduction, 0.4, 1.0, 0.05)

    st.sidebar.subheader("Code checks")
    drift_lim = numeric("Story drift limit ratio", default.drift_limit_ratio, 1/1000, 1/250, 1/1000, fmt="%.5f")
    Ct = numeric("Ct", default.Ct, 0.02, 0.10, 0.001, fmt="%.4f")
    xexp = numeric("x exponent", default.x_exp, 0.50, 1.00, 0.01, fmt="%.2f")
    Cu = numeric("Cu", default.Cu, 1.0, 2.0, 0.05, fmt="%.2f")
    auto_size_flag = st.sidebar.checkbox("Auto-size using drift + code ceiling", value=True)

    return BuildingInput(
        n_story=n_story,
        n_basement=n_basement,
        story_height_m=story_h,
        basement_height_m=base_h,
        plan_x_m=plan_x,
        plan_y_m=plan_y,
        n_bays_x=bays_x,
        n_bays_y=bays_y,
        bay_x_m=bay_x,
        bay_y_m=bay_y,
        core_outer_x_m=core_outer_x,
        core_outer_y_m=core_outer_y,
        core_opening_x_m=core_open_x,
        core_opening_y_m=core_open_y,
        Ec_mpa=Ec,
        wall_cracked_factor=wall_cf,
        perimeter_wall_cracked_factor=pwall_cf,
        column_cracked_factor=col_cf,
        beam_cracked_factor=beam_cf,
        super_dead_load_kpa=sdl,
        live_load_kpa=ll,
        facade_line_load_kn_m=facade,
        lower_zone=lower_zone,
        middle_zone=middle_zone,
        upper_zone=upper_zone,
        lower_perimeter_wall=lower_pw,
        middle_perimeter_wall=middle_pw,
        upper_perimeter_wall=upper_pw,
        retaining_wall=RetainingWallInput(enabled=rw_enabled, thickness_m=rw_t, stiffness_reduction=rw_red),
        outrigger_story_levels=og_levels,
        brace_input=BraceInput(outer_diameter_mm=od, thickness_mm=tt, k_factor=kf, buckling_reduction=br_red),
        drift_limit_ratio=drift_lim,
        Ct=Ct,
        x_exp=xexp,
        Cu=Cu,
        auto_size=auto_size_flag,
    )


def main() -> None:
    if st is None:
        raise SystemExit("Streamlit is required to run the app UI.")
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info("No manual target period and no manual brace count input. Outrigger layout is generated automatically from core-to-perimeter geometry. Auto-sizing uses only drift limit and code period ceiling.")

    inp = streamlit_input_panel()
    if st.button("Analyze") or "analysis_result" not in st.session_state:
        st.session_state.analysis_result = analyze(inp)
    res: AnalysisResult = st.session_state.analysis_result

    tabs = st.tabs(["Summary", "Zones", "Stories", "Outriggers", "Plan", "Elevation", "Modes", "Drifts"])

    with tabs[0]:
        st.dataframe(summary_df(res), use_container_width=True)
    with tabs[1]:
        st.dataframe(zones_df(res), use_container_width=True)
    with tabs[2]:
        st.dataframe(stories_df(res), use_container_width=True, height=520)
    with tabs[3]:
        st.dataframe(outriggers_df(res), use_container_width=True)
    with tabs[4]:
        story_choice = st.selectbox("Show brace layout for story", options=res.inp.outrigger_story_levels if res.inp.outrigger_story_levels else [0])
        st.pyplot(plot_plan(res, story_choice if story_choice != 0 else None))
    with tabs[5]:
        st.pyplot(plot_elevation(res))
    with tabs[6]:
        mode_id = st.selectbox("Mode", options=list(range(1, min(5, len(res.modal.periods_s)) + 1)))
        st.pyplot(plot_mode(res, mode_id - 1))
    with tabs[7]:
        st.dataframe(drifts_df(res), use_container_width=True, height=520)
        st.pyplot(plot_drifts(res))


if __name__ == "__main__":
    main()
