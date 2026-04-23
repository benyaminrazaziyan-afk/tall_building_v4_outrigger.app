
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


G = 9.81
CONCRETE_DENSITY = 25.0  # kN/m3
STEEL_DENSITY = 7850.0   # kg/m3
APP_TITLE = "Tall Building Rational Analysis + Steel Braced Outrigger"
APP_VERSION = "v7.0-rational"


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
    n_story: int = 60
    n_basement: int = 0
    story_height: float = 3.2
    basement_height: float = 3.0
    plan_x: float = 48.0
    plan_y: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6
    bay_x: float = 8.0
    bay_y: float = 7.0

    fck_mpa: float = 50.0
    Ec_mpa: float = 34000.0
    fy_mpa: float = 420.0
    Es_mpa: float = 200000.0

    dl_kn_m2: float = 3.0
    ll_kn_m2: float = 2.0
    superimposed_dead_kn_m2: float = 1.5
    facade_line_load_kn_m: float = 1.0
    live_load_mass_factor: float = 0.30

    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.70
    beam_cracked_factor: float = 0.35
    slab_cracked_factor: float = 0.25

    slab_thickness_m: float = 0.22
    beam_width_m: float = 0.45
    beam_depth_m: float = 0.80

    lower_zone: ZoneMemberInput = ZoneMemberInput(0.60, 1.20, 1.20, 1.00, 1.00, 0.90, 0.90)
    middle_zone: ZoneMemberInput = ZoneMemberInput(0.45, 1.00, 1.00, 0.85, 0.85, 0.75, 0.75)
    upper_zone: ZoneMemberInput = ZoneMemberInput(0.35, 0.85, 0.85, 0.75, 0.75, 0.65, 0.65)

    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each_m2: float = 3.5
    stair_area_each_m2: float = 20.0
    service_area_m2: float = 35.0
    corridor_factor: float = 1.35

    seismic_base_shear_coeff: float = 0.08

    outrigger_count: int = 0
    outrigger_story_levels: Optional[List[int]] = None
    brace_outer_diameter_mm: float = 355.6
    brace_thickness_mm: float = 16.0
    braces_per_side: int = 2
    outrigger_depth_m: float = 3.0
    brace_effective_length_factor: float = 1.0
    brace_buckling_reduction: float = 0.85


@dataclass
class OutriggerResult:
    story_level: int
    arm_m: float
    brace_length_m: float
    brace_area_m2: float
    brace_radius_gyration_m: float
    axial_stiffness_kn: float
    lateral_stiffness_n_m: float
    steel_weight_kg: float


@dataclass
class AnalysisResult:
    story_stiffness_n_m: List[float]
    story_mass_kg: List[float]
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    roof_displacements_m: List[float]
    story_drifts_m: List[float]
    story_drift_ratios: List[float]
    base_shear_kn: float
    total_weight_kn: float
    total_mass_kg: float
    k_wall_total_n_m: float
    k_column_total_n_m: float
    k_beam_total_n_m: float
    k_outrigger_total_n_m: float
    outriggers: List[OutriggerResult]
    diaphragm_inplane_stiffness_n_m: float
    zone_table: pd.DataFrame


def define_three_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def get_zone_input(inp: BuildingInput, zone_name: str) -> ZoneMemberInput:
    if zone_name == "Lower Zone":
        return inp.lower_zone
    if zone_name == "Middle Zone":
        return inp.middle_zone
    return inp.upper_zone


def floor_area(inp: BuildingInput) -> float:
    return inp.plan_x * inp.plan_y


def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height


def required_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each_m2
        + inp.stair_count * inp.stair_area_each_m2
        + inp.service_area_m2
    ) * inp.corridor_factor


def opening_dimensions(inp: BuildingInput) -> tuple[float, float]:
    area = required_opening_area(inp)
    aspect = 1.6
    oy = sqrt(area / aspect)
    ox = aspect * oy
    return ox, oy


def core_outer_dimensions(inp: BuildingInput) -> tuple[float, float]:
    ox, oy = opening_dimensions(inp)
    outer_x = max(ox + 2.5, 0.22 * inp.plan_x)
    outer_y = max(oy + 2.5, 0.20 * inp.plan_y)
    return outer_x, outer_y


def wall_layout_lengths(inp: BuildingInput) -> dict[str, list[float]]:
    core_x, core_y = core_outer_dimensions(inp)
    return {
        "Lower Zone": [core_x, core_x, core_y, core_y, 0.35 * core_x, 0.35 * core_x, 0.35 * core_y, 0.35 * core_y],
        "Middle Zone": [core_x, core_x, core_y, core_y, 0.30 * core_x, 0.30 * core_x],
        "Upper Zone": [core_x, core_x, core_y, core_y],
    }


def core_equivalent_inertia(inp: BuildingInput, zone_name: str, wall_t: float) -> float:
    lengths = wall_layout_lengths(inp)[zone_name]
    core_x, core_y = core_outer_dimensions(inp)
    x_side = core_x / 2.0
    y_side = core_y / 2.0
    I_x = 0.0
    I_y = 0.0

    top_len, bot_len, left_len, right_len = lengths[0], lengths[1], lengths[2], lengths[3]
    I_x += top_len * wall_t**3 / 12.0 + (top_len * wall_t) * y_side**2
    I_x += bot_len * wall_t**3 / 12.0 + (bot_len * wall_t) * y_side**2
    I_y += wall_t * top_len**3 / 12.0
    I_y += wall_t * bot_len**3 / 12.0

    I_y += left_len * wall_t**3 / 12.0 + (left_len * wall_t) * x_side**2
    I_y += right_len * wall_t**3 / 12.0 + (right_len * wall_t) * x_side**2
    I_x += wall_t * left_len**3 / 12.0
    I_x += wall_t * right_len**3 / 12.0

    if len(lengths) >= 6:
        inner_x = 0.18 * core_x
        l1, l2 = lengths[4], lengths[5]
        I_y += l1 * wall_t**3 / 12.0 + (l1 * wall_t) * inner_x**2
        I_y += l2 * wall_t**3 / 12.0 + (l2 * wall_t) * inner_x**2
        I_x += wall_t * l1**3 / 12.0 + wall_t * l2**3 / 12.0

    if len(lengths) >= 8:
        inner_y = 0.18 * core_y
        l3, l4 = lengths[6], lengths[7]
        I_x += l3 * wall_t**3 / 12.0 + (l3 * wall_t) * inner_y**2
        I_x += l4 * wall_t**3 / 12.0 + (l4 * wall_t) * inner_y**2
        I_y += wall_t * l3**3 / 12.0 + wall_t * l4**3 / 12.0

    return min(I_x, I_y)


def column_counts(inp: BuildingInput) -> tuple[int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return corner, perimeter, interior


def rectangle_i_for_sway(b: float, h: float) -> float:
    return max(b * h**3 / 12.0, h * b**3 / 12.0)


def beam_line_counts(inp: BuildingInput) -> tuple[int, int]:
    n_x_dir = (inp.n_bays_y + 1) * inp.n_bays_x
    n_y_dir = (inp.n_bays_x + 1) * inp.n_bays_y
    return n_x_dir, n_y_dir


def beam_story_stiffness(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    I_beam = rectangle_i_for_sway(inp.beam_width_m, inp.beam_depth_m)
    n_x, n_y = beam_line_counts(inp)
    kx = n_x * (12.0 * E * inp.beam_cracked_factor * I_beam / max(inp.bay_x**3, 1e-9))
    ky = n_y * (12.0 * E * inp.beam_cracked_factor * I_beam / max(inp.bay_y**3, 1e-9))
    return min(kx + ky, 0.65 * (kx + ky))


def diaphragm_inplane_stiffness(inp: BuildingInput) -> float:
    E = inp.Ec_mpa * 1e6
    nu = 0.20
    Gc = E / (2.0 * (1.0 + nu))
    A = floor_area(inp)
    t = inp.slab_thickness_m
    L = max(inp.plan_x, inp.plan_y)
    return inp.slab_cracked_factor * Gc * A * t / max(L, 1e-9)


def chs_area_and_r(outer_d_mm: float, t_mm: float) -> tuple[float, float]:
    D = outer_d_mm / 1000.0
    t = t_mm / 1000.0
    d = max(D - 2.0 * t, 1e-6)
    area = pi / 4.0 * (D**2 - d**2)
    I = pi / 64.0 * (D**4 - d**4)
    r = sqrt(I / max(area, 1e-12))
    return area, r


def calculate_outriggers(inp: BuildingInput) -> List[OutriggerResult]:
    if inp.outrigger_count <= 0 or not inp.outrigger_story_levels:
        return []

    E = inp.Es_mpa * 1e6
    core_x, core_y = core_outer_dimensions(inp)
    arm_x = max((inp.plan_x - core_x) / 2.0, 0.5)
    arm_y = max((inp.plan_y - core_y) / 2.0, 0.5)
    arm = min(arm_x, arm_y)
    area, r = chs_area_and_r(inp.brace_outer_diameter_mm, inp.brace_thickness_mm)

    results = []
    for lvl in inp.outrigger_story_levels[:inp.outrigger_count]:
        brace_len = sqrt(arm**2 + inp.outrigger_depth_m**2)
        k_ax = inp.braces_per_side * 4.0 * E * area / max(inp.brace_effective_length_factor * brace_len, 1e-9)
        slenderness = (inp.brace_effective_length_factor * brace_len) / max(r, 1e-9)
        reduction = min(1.0, inp.brace_buckling_reduction * (180.0 / max(slenderness, 1.0)))
        reduction = max(0.20, min(1.0, reduction))
        k_eff = k_ax * (arm / max(brace_len, 1e-9))**2 * reduction
        steel_weight = inp.braces_per_side * 4.0 * brace_len * area * STEEL_DENSITY
        results.append(OutriggerResult(
            story_level=lvl,
            arm_m=arm,
            brace_length_m=brace_len,
            brace_area_m2=area,
            brace_radius_gyration_m=r,
            axial_stiffness_kn=k_ax / 1000.0,
            lateral_stiffness_n_m=k_eff,
            steel_weight_kg=steel_weight,
        ))
    return results


def story_lateral_force_distribution(inp: BuildingInput, base_shear_kn: float) -> np.ndarray:
    n = inp.n_story
    z = np.arange(1, n + 1, dtype=float) * inp.story_height
    w = z / z.sum()
    return base_shear_kn * 1000.0 * w


def zone_label_for_story(inp: BuildingInput, story: int) -> str:
    for z in define_three_zones(inp.n_story):
        if z.story_start <= story <= z.story_end:
            return z.name
    return "Upper Zone"


def story_stiffness_breakdown(inp: BuildingInput) -> tuple[list[float], list[float], list[float], list[float], pd.DataFrame]:
    E = inp.Ec_mpa * 1e6
    h = inp.story_height
    corner_n, perim_n, interior_n = column_counts(inp)
    beam_k = beam_story_stiffness(inp)

    k_walls = []
    k_cols = []
    k_beams = []
    rows = []

    for story in range(1, inp.n_story + 1):
        zone_name = zone_label_for_story(inp, story)
        z = get_zone_input(inp, zone_name)

        I_core = inp.wall_cracked_factor * core_equivalent_inertia(inp, zone_name, z.wall_thickness_m)
        k_wall = 12.0 * E * I_core / max(h**3, 1e-9)

        I_corner = rectangle_i_for_sway(z.corner_col_x_m, z.corner_col_y_m)
        I_perim = rectangle_i_for_sway(z.perimeter_col_x_m, z.perimeter_col_y_m)
        I_interior = rectangle_i_for_sway(z.interior_col_x_m, z.interior_col_y_m)
        I_col_total = inp.column_cracked_factor * (
            corner_n * I_corner + perim_n * I_perim + interior_n * I_interior
        )
        k_col = 12.0 * E * I_col_total / max(h**3, 1e-9)
        k_beam = beam_k

        k_walls.append(k_wall)
        k_cols.append(k_col)
        k_beams.append(k_beam)
        rows.append({
            "Story": story,
            "Zone": zone_name,
            "Wall thickness (m)": z.wall_thickness_m,
            "Corner col (m)": f"{z.corner_col_x_m:.2f}x{z.corner_col_y_m:.2f}",
            "Perimeter col (m)": f"{z.perimeter_col_x_m:.2f}x{z.perimeter_col_y_m:.2f}",
            "Interior col (m)": f"{z.interior_col_x_m:.2f}x{z.interior_col_y_m:.2f}",
            "K_wall (N/m)": k_wall,
            "K_cols (N/m)": k_col,
            "K_beams (N/m)": k_beam,
            "K_story_no_outrigger (N/m)": k_wall + k_col + k_beam,
        })
    return k_walls, k_cols, k_beams, pd.DataFrame(rows)


def story_masses(inp: BuildingInput, outriggers: List[OutriggerResult]) -> tuple[list[float], float]:
    A = floor_area(inp)
    super_dead = inp.dl_kn_m2 + inp.superimposed_dead_kn_m2
    slab_self = inp.slab_thickness_m * CONCRETE_DENSITY
    mass_load = super_dead + slab_self + inp.live_load_mass_factor * inp.ll_kn_m2
    floor_weight_kn = A * mass_load + inp.facade_line_load_kn_m * 2.0 * (inp.plan_x + inp.plan_y)

    masses = []
    levels = {o.story_level: o.steel_weight_kg for o in outriggers}
    for story in range(1, inp.n_story + 1):
        extra_kn = levels.get(story, 0.0) * G / 1000.0
        masses.append((floor_weight_kn + extra_kn) * 1000.0 / G)
    total_weight = sum(m * G / 1000.0 for m in masses)
    return masses, total_weight


def assemble_m_k(masses: list[float], story_k: list[float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(masses)
    M = np.diag(masses)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = story_k[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return M, K


def solve_modes(masses: list[float], story_k: list[float], n_modes: int = 5):
    M, K = assemble_m_k(masses, story_k)
    vals, vecs = np.linalg.eig(np.linalg.solve(M, K))
    vals = np.real(vals)
    vecs = np.real(vecs)
    keep = vals > 1e-12
    vals = vals[keep]
    vecs = vecs[:, keep]
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    omegas = np.sqrt(vals)
    periods = [2.0 * pi / w for w in omegas[:n_modes]]
    freqs = [w / (2.0 * pi) for w in omegas[:n_modes]]

    shapes = []
    for i in range(min(n_modes, vecs.shape[1])):
        phi = vecs[:, i].copy()
        phi = phi / max(np.max(np.abs(phi)), 1e-12)
        if phi[-1] < 0:
            phi = -phi
        shapes.append(phi.tolist())
    return periods, freqs, shapes


def solve_drifts(inp: BuildingInput, masses: list[float], story_k: list[float], total_weight_kn: float):
    _, K = assemble_m_k(masses, story_k)
    base_shear_kn = inp.seismic_base_shear_coeff * total_weight_kn
    F = story_lateral_force_distribution(inp, base_shear_kn)
    u = np.linalg.solve(K, F)
    roof = u.tolist()
    drifts = [u[0]] + [u[i] - u[i - 1] for i in range(1, len(u))]
    ratios = [d / inp.story_height for d in drifts]
    return roof, drifts, ratios, base_shear_kn


def run_analysis(inp: BuildingInput) -> AnalysisResult:
    outriggers = calculate_outriggers(inp)
    k_walls, k_cols, k_beams, zone_df = story_stiffness_breakdown(inp)
    story_k = [a + b + c for a, b, c in zip(k_walls, k_cols, k_beams)]

    for o in outriggers:
        idx = o.story_level - 1
        if 0 <= idx < len(story_k):
            story_k[idx] += o.lateral_stiffness_n_m

    masses, total_weight_kn = story_masses(inp, outriggers)
    total_mass_kg = sum(masses)
    periods, freqs, shapes = solve_modes(masses, story_k)
    roof, drifts, ratios, base_shear_kn = solve_drifts(inp, masses, story_k, total_weight_kn)
    dia_k = diaphragm_inplane_stiffness(inp)

    zone_df["K_story_with_outrigger (N/m)"] = story_k
    return AnalysisResult(
        story_stiffness_n_m=story_k,
        story_mass_kg=masses,
        periods_s=periods,
        frequencies_hz=freqs,
        mode_shapes=shapes,
        roof_displacements_m=roof,
        story_drifts_m=drifts,
        story_drift_ratios=ratios,
        base_shear_kn=base_shear_kn,
        total_weight_kn=total_weight_kn,
        total_mass_kg=total_mass_kg,
        k_wall_total_n_m=sum(k_walls),
        k_column_total_n_m=sum(k_cols),
        k_beam_total_n_m=sum(k_beams),
        k_outrigger_total_n_m=sum(o.lateral_stiffness_n_m for o in outriggers),
        outriggers=outriggers,
        diaphragm_inplane_stiffness_n_m=dia_k,
        zone_table=zone_df,
    )


def summary_table(inp: BuildingInput, res: AnalysisResult) -> pd.DataFrame:
    rows = [
        ["Total height (m)", total_height(inp)],
        ["Floor area (m2)", floor_area(inp)],
        ["Total weight (kN)", res.total_weight_kn],
        ["Base shear (kN)", res.base_shear_kn],
        ["Mode 1 period (s)", res.periods_s[0] if res.periods_s else np.nan],
        ["Mode 2 period (s)", res.periods_s[1] if len(res.periods_s) > 1 else np.nan],
        ["Roof displacement (m)", res.roof_displacements_m[-1] if res.roof_displacements_m else np.nan],
        ["Max story drift (m)", max(abs(v) for v in res.story_drifts_m)],
        ["Max story drift ratio", max(abs(v) for v in res.story_drift_ratios)],
        ["Wall stiffness sum (N/m)", res.k_wall_total_n_m],
        ["Column stiffness sum (N/m)", res.k_column_total_n_m],
        ["Beam stiffness sum (N/m)", res.k_beam_total_n_m],
        ["Outrigger stiffness sum (N/m)", res.k_outrigger_total_n_m],
        ["Diaphragm in-plane stiffness (N/m)", res.diaphragm_inplane_stiffness_n_m],
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def outrigger_table(res: AnalysisResult) -> pd.DataFrame:
    if not res.outriggers:
        return pd.DataFrame(columns=[
            "Story", "Brace length (m)", "Area (m2)", "r (m)", "Axial stiffness (kN)", "Lateral stiffness (N/m)", "Steel weight (kg)"
        ])
    return pd.DataFrame([{
        "Story": o.story_level,
        "Brace length (m)": o.brace_length_m,
        "Area (m2)": o.brace_area_m2,
        "r (m)": o.brace_radius_gyration_m,
        "Axial stiffness (kN)": o.axial_stiffness_kn,
        "Lateral stiffness (N/m)": o.lateral_stiffness_n_m,
        "Steel weight (kg)": o.steel_weight_kg,
    } for o in res.outriggers])


def drift_table(inp: BuildingInput, res: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame({
        "Story": list(range(1, inp.n_story + 1)),
        "Displacement (m)": res.roof_displacements_m,
        "Story drift (m)": res.story_drifts_m,
        "Story drift ratio": res.story_drift_ratios,
        "Story stiffness (N/m)": res.story_stiffness_n_m,
        "Story mass (kg)": res.story_mass_kg,
    })


def build_report(inp: BuildingInput, res: AnalysisResult) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"{APP_TITLE} - {APP_VERSION}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("GLOBAL RESPONSE")
    lines.append("-" * 78)
    lines.append(f"Mode 1 period                  = {res.periods_s[0]:.3f} s" if res.periods_s else "Mode 1 period                  = N/A")
    lines.append(f"Base shear                     = {res.base_shear_kn:,.0f} kN")
    lines.append(f"Roof displacement              = {res.roof_displacements_m[-1]:.4f} m")
    lines.append(f"Max story drift                = {max(abs(v) for v in res.story_drifts_m):.5f} m")
    lines.append(f"Max story drift ratio          = {max(abs(v) for v in res.story_drift_ratios):.6f}")
    lines.append(f"Wall stiffness sum             = {res.k_wall_total_n_m:,.3e} N/m")
    lines.append(f"Column stiffness sum           = {res.k_column_total_n_m:,.3e} N/m")
    lines.append(f"Beam stiffness sum             = {res.k_beam_total_n_m:,.3e} N/m")
    lines.append(f"Outrigger stiffness sum        = {res.k_outrigger_total_n_m:,.3e} N/m")
    lines.append(f"Diaphragm in-plane stiffness   = {res.diaphragm_inplane_stiffness_n_m:,.3e} N/m")
    lines.append("")
    lines.append("MODELING NOTES")
    lines.append("-" * 78)
    lines.append("1. Empirical target-period sizing was removed.")
    lines.append("2. Beam stiffness is included explicitly in story stiffness.")
    lines.append("3. Slab is included through self-weight and reported diaphragm in-plane stiffness.")
    lines.append("4. Outrigger is modeled as steel CHS braces, not a belt-truss tuning factor.")
    lines.append("5. Cracked section factors for walls, columns, beams, and slabs are user inputs.")
    return "\n".join(lines)


def plot_mode_shapes(inp: BuildingInput, res: AnalysisResult, n_modes: int = 3) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, inp.n_story + 1)
    for i in range(min(n_modes, len(res.mode_shapes))):
        ax.plot(res.mode_shapes[i], y, marker="o", label=f"Mode {i+1} | T={res.periods_s[i]:.3f} s")
    ax.set_xlabel("Normalized mode shape")
    ax.set_ylabel("Story")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_story_drifts(inp: BuildingInput, res: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(1, inp.n_story + 1)
    ax.plot(res.story_drifts_m, y, marker="o")
    ax.set_xlabel("Story drift (m)")
    ax.set_ylabel("Story")
    ax.set_title("Story drift profile")
    ax.grid(True, alpha=0.3)
    return fig


def streamlit_input_panel() -> BuildingInput:
    st.sidebar.header("Input Data")

    st.sidebar.subheader("Geometry")
    n_story = int(st.sidebar.number_input("Above-grade stories", 1, 120, 60, 1))
    n_basement = int(st.sidebar.number_input("Basement stories", 0, 20, 0, 1))
    story_height = float(st.sidebar.number_input("Story height (m)", 2.5, 6.0, 3.2, 0.1))
    basement_height = float(st.sidebar.number_input("Basement height (m)", 2.5, 6.0, 3.0, 0.1))
    plan_x = float(st.sidebar.number_input("Plan X (m)", 10.0, 200.0, 48.0, 1.0))
    plan_y = float(st.sidebar.number_input("Plan Y (m)", 10.0, 200.0, 42.0, 1.0))
    n_bays_x = int(st.sidebar.number_input("Bays in X", 1, 30, 6, 1))
    n_bays_y = int(st.sidebar.number_input("Bays in Y", 1, 30, 6, 1))
    bay_x = float(st.sidebar.number_input("Bay X (m)", 2.0, 20.0, 8.0, 0.1))
    bay_y = float(st.sidebar.number_input("Bay Y (m)", 2.0, 20.0, 7.0, 0.1))

    st.sidebar.subheader("Materials / Loads")
    fck_mpa = float(st.sidebar.number_input("fck (MPa)", 20.0, 100.0, 50.0, 1.0))
    Ec_mpa = float(st.sidebar.number_input("Ec (MPa)", 15000.0, 50000.0, 34000.0, 500.0))
    fy_mpa = float(st.sidebar.number_input("Rebar fy (MPa)", 200.0, 700.0, 420.0, 10.0))
    Es_mpa = float(st.sidebar.number_input("Steel brace Es (MPa)", 150000.0, 220000.0, 200000.0, 1000.0))
    dl_kn_m2 = float(st.sidebar.number_input("Dead load DL (kN/m2)", 0.0, 20.0, 3.0, 0.1))
    ll_kn_m2 = float(st.sidebar.number_input("Live load LL (kN/m2)", 0.0, 20.0, 2.0, 0.1))
    super_dead = float(st.sidebar.number_input("Superimposed dead load (kN/m2)", 0.0, 10.0, 1.5, 0.1))
    facade_line = float(st.sidebar.number_input("Facade line load (kN/m)", 0.0, 10.0, 1.0, 0.1))
    live_mass_factor = float(st.sidebar.number_input("Live load mass factor", 0.0, 1.0, 0.30, 0.05))
    seismic_coeff = float(st.sidebar.number_input("Seismic base shear coefficient", 0.005, 0.50, 0.08, 0.005, format="%.3f"))

    st.sidebar.subheader("Cracked section factors")
    wall_cf = float(st.sidebar.number_input("Wall cracked factor", 0.10, 1.00, 0.40, 0.05))
    col_cf = float(st.sidebar.number_input("Column cracked factor", 0.10, 1.00, 0.70, 0.05))
    beam_cf = float(st.sidebar.number_input("Beam cracked factor", 0.10, 1.00, 0.35, 0.05))
    slab_cf = float(st.sidebar.number_input("Slab cracked factor", 0.05, 1.00, 0.25, 0.05))

    st.sidebar.subheader("Beams / Slab")
    slab_t = float(st.sidebar.number_input("Slab thickness (m)", 0.10, 0.50, 0.22, 0.01))
    beam_b = float(st.sidebar.number_input("Beam width (m)", 0.20, 1.50, 0.45, 0.01))
    beam_h = float(st.sidebar.number_input("Beam depth (m)", 0.30, 2.00, 0.80, 0.01))

    def zone_block(name: str, defaults: ZoneMemberInput) -> ZoneMemberInput:
        st.sidebar.markdown(f"**{name} members**")
        wt = float(st.sidebar.number_input(f"{name} wall thickness (m)", 0.20, 1.50, defaults.wall_thickness_m, 0.01))
        cx = float(st.sidebar.number_input(f"{name} corner col X (m)", 0.30, 2.50, defaults.corner_col_x_m, 0.01))
        cy = float(st.sidebar.number_input(f"{name} corner col Y (m)", 0.30, 2.50, defaults.corner_col_y_m, 0.01))
        px = float(st.sidebar.number_input(f"{name} perimeter col X (m)", 0.30, 2.50, defaults.perimeter_col_x_m, 0.01))
        py = float(st.sidebar.number_input(f"{name} perimeter col Y (m)", 0.30, 2.50, defaults.perimeter_col_y_m, 0.01))
        ix = float(st.sidebar.number_input(f"{name} interior col X (m)", 0.30, 2.50, defaults.interior_col_x_m, 0.01))
        iy = float(st.sidebar.number_input(f"{name} interior col Y (m)", 0.30, 2.50, defaults.interior_col_y_m, 0.01))
        return ZoneMemberInput(wt, cx, cy, px, py, ix, iy)

    st.sidebar.subheader("Zone member sizes")
    lower_zone = zone_block("Lower", BuildingInput.lower_zone)
    middle_zone = zone_block("Middle", BuildingInput.middle_zone)
    upper_zone = zone_block("Upper", BuildingInput.upper_zone)

    st.sidebar.subheader("Core opening")
    stair_count = int(st.sidebar.number_input("Stairs", 0, 20, 2, 1))
    elevator_count = int(st.sidebar.number_input("Elevators", 0, 30, 4, 1))
    elevator_area_each = float(st.sidebar.number_input("Elevator area each (m2)", 0.0, 20.0, 3.5, 0.1))
    stair_area_each = float(st.sidebar.number_input("Stair area each (m2)", 0.0, 50.0, 20.0, 0.5))
    service_area = float(st.sidebar.number_input("Service area (m2)", 0.0, 200.0, 35.0, 1.0))
    corridor_factor = float(st.sidebar.number_input("Corridor factor", 1.0, 2.0, 1.35, 0.05))

    st.sidebar.subheader("Steel braced outriggers")
    outrigger_count = int(st.sidebar.selectbox("Number of outriggers", [0, 1, 2, 3], index=0))
    suggested = [max(1, int(round(n_story * r))) for r in [0.45, 0.70, 0.85]]
    levels = []
    for i in range(outrigger_count):
        levels.append(int(st.sidebar.number_input(f"Outrigger story level {i+1}", 1, max(1, n_story), suggested[i], 1)))
    brace_D = float(st.sidebar.number_input("Brace outer diameter (mm)", 60.0, 1200.0, 355.6, 1.0))
    brace_t = float(st.sidebar.number_input("Brace thickness (mm)", 4.0, 80.0, 16.0, 1.0))
    braces_per_side = int(st.sidebar.number_input("Braces per side", 1, 8, 2, 1))
    out_depth = float(st.sidebar.number_input("Outrigger depth (m)", 0.5, 10.0, 3.0, 0.1))
    k_fac = float(st.sidebar.number_input("Brace effective length factor K", 0.5, 2.0, 1.0, 0.05))
    buck_red = float(st.sidebar.number_input("Brace buckling reduction", 0.20, 1.00, 0.85, 0.05))

    return BuildingInput(
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
        fck_mpa=fck_mpa,
        Ec_mpa=Ec_mpa,
        fy_mpa=fy_mpa,
        Es_mpa=Es_mpa,
        dl_kn_m2=dl_kn_m2,
        ll_kn_m2=ll_kn_m2,
        superimposed_dead_kn_m2=super_dead,
        facade_line_load_kn_m=facade_line,
        live_load_mass_factor=live_mass_factor,
        wall_cracked_factor=wall_cf,
        column_cracked_factor=col_cf,
        beam_cracked_factor=beam_cf,
        slab_cracked_factor=slab_cf,
        slab_thickness_m=slab_t,
        beam_width_m=beam_b,
        beam_depth_m=beam_h,
        lower_zone=lower_zone,
        middle_zone=middle_zone,
        upper_zone=upper_zone,
        stair_count=stair_count,
        elevator_count=elevator_count,
        elevator_area_each_m2=elevator_area_each,
        stair_area_each_m2=stair_area_each,
        service_area_m2=service_area,
        corridor_factor=corridor_factor,
        seismic_base_shear_coeff=seismic_coeff,
        outrigger_count=outrigger_count,
        outrigger_story_levels=levels,
        brace_outer_diameter_mm=brace_D,
        brace_thickness_mm=brace_t,
        braces_per_side=braces_per_side,
        outrigger_depth_m=out_depth,
        brace_effective_length_factor=k_fac,
        brace_buckling_reduction=buck_red,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_VERSION)
    st.info("Empirical target-period sizing and outrigger relief/boost tuning were removed. The app now uses direct member stiffness input, explicit beam stiffness, slab diaphragm stiffness reporting, and steel CHS-braced outriggers.")

    inp = streamlit_input_panel()

    if "result_v7" not in st.session_state:
        st.session_state.result_v7 = None
        st.session_state.report_v7 = ""

    c1, c2 = st.columns(2)
    analyze = c1.button("Analyze")
    clear_btn = c2.button("Clear Results")

    if clear_btn:
        st.session_state.result_v7 = None
        st.session_state.report_v7 = ""
        st.rerun()

    if analyze:
        try:
            res = run_analysis(inp)
            st.session_state.result_v7 = res
            st.session_state.report_v7 = build_report(inp, res)
            st.success("Analysis completed.")
        except Exception as exc:
            st.exception(exc)

    res = st.session_state.result_v7
    if res is None:
        st.warning("Set the member sizes and click Analyze.")
        return

    tabs = st.tabs(["Summary", "Stiffness by Story", "Outriggers", "Drifts", "Modes", "Report"])

    with tabs[0]:
        st.subheader("Summary")
        st.dataframe(summary_table(inp, res), use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mode 1 period (s)", f"{res.periods_s[0]:.3f}")
        m2.metric("Roof disp. (m)", f"{res.roof_displacements_m[-1]:.4f}")
        m3.metric("Max drift ratio", f"{max(abs(v) for v in res.story_drift_ratios):.5f}")
        m4.metric("Base shear (kN)", f"{res.base_shear_kn:,.0f}")

    with tabs[1]:
        st.subheader("Story stiffness breakdown")
        st.dataframe(res.zone_table, use_container_width=True)

    with tabs[2]:
        st.subheader("Steel braced outriggers")
        st.dataframe(outrigger_table(res), use_container_width=True)

    with tabs[3]:
        st.subheader("Story drifts")
        df_drift = drift_table(inp, res)
        st.dataframe(df_drift, use_container_width=True)
        st.pyplot(plot_story_drifts(inp, res))

    with tabs[4]:
        st.subheader("Mode shapes")
        st.pyplot(plot_mode_shapes(inp, res, n_modes=3))

    with tabs[5]:
        st.subheader("Report")
        st.text_area("Text report", st.session_state.report_v7, height=420)

if __name__ == "__main__":
    main()
