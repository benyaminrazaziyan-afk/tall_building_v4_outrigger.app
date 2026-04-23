from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

G = 9.81
CONCRETE_UNIT_WEIGHT = 25.0  # kN/m3


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int

    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1


@dataclass
class ZoneMembers:
    wall_t: float
    perimeter_wall_t: float
    beam_b: float
    beam_h: float
    slab_t: float
    corner_col_x: float
    corner_col_y: float
    perimeter_col_x: float
    perimeter_col_y: float
    interior_col_x: float
    interior_col_y: float


@dataclass
class BraceSystem:
    n_outrigger_levels: int = 2
    outer_diameter_mm: float = 406.4
    thickness_mm: float = 18.0
    k_factor: float = 1.0
    buckling_reduction: float = 0.85


@dataclass
class BuildingInput:
    n_story: int = 60
    n_basement: int = 2
    story_h: float = 3.2
    basement_h: float = 3.2
    plan_x: float = 48.0
    plan_y: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6
    bay_x: float = 8.0
    bay_y: float = 7.0

    core_outer_x: float = 18.0
    core_outer_y: float = 15.0
    core_open_x: float = 11.5
    core_open_y: float = 9.0

    Ec_mpa: float = 34000.0
    Es_mpa: float = 200000.0

    wall_cr: float = 0.50
    perimeter_wall_cr: float = 0.40
    beam_cr: float = 0.35
    col_cr: float = 0.70
    slab_cr: float = 0.25

    drift_limit: float = 1.0 / 500.0
    Ct: float = 0.0488
    x_exp: float = 0.75
    Cu: float = 1.4
    seismic_coeff: float = 0.015

    retaining_enabled: bool = True
    retaining_t: float = 0.50
    retaining_reduction: float = 0.08

    brace: BraceSystem = field(default_factory=BraceSystem)

    lower: ZoneMembers = field(default_factory=lambda: ZoneMembers(
        wall_t=0.80, perimeter_wall_t=0.35, beam_b=0.60, beam_h=0.95, slab_t=0.24,
        corner_col_x=1.20, corner_col_y=1.20,
        perimeter_col_x=1.05, perimeter_col_y=1.05,
        interior_col_x=0.90, interior_col_y=0.90,
    ))
    middle: ZoneMembers = field(default_factory=lambda: ZoneMembers(
        wall_t=0.65, perimeter_wall_t=0.28, beam_b=0.55, beam_h=0.82, slab_t=0.22,
        corner_col_x=1.00, corner_col_y=1.00,
        perimeter_col_x=0.90, perimeter_col_y=0.90,
        interior_col_x=0.78, interior_col_y=0.78,
    ))
    upper: ZoneMembers = field(default_factory=lambda: ZoneMembers(
        wall_t=0.50, perimeter_wall_t=0.22, beam_b=0.45, beam_h=0.68, slab_t=0.20,
        corner_col_x=0.82, corner_col_y=0.82,
        perimeter_col_x=0.74, perimeter_col_y=0.74,
        interior_col_x=0.65, interior_col_y=0.65,
    ))


@dataclass
class StoryState:
    story: int
    zone: str
    brace_level: bool
    wall_t: float
    perimeter_wall_t: float
    beam_b: float
    beam_h: float
    slab_t: float
    corner_col: str
    perimeter_col: str
    interior_col: str
    k_core: float
    k_perimeter_walls: float
    k_frame: float
    k_outrigger: float
    k_total: float
    mass_t: float


@dataclass
class AnalysisResult:
    periods: List[float]
    roof_disp_m: float
    max_story_drift_m: float
    max_story_drift_ratio: float
    code_period: float
    period_ceiling: float
    brace_levels: List[int]
    brace_zone_count: Dict[str, int]
    story_table: pd.DataFrame
    zone_table: pd.DataFrame
    contribution_table: pd.DataFrame
    iteration_log: pd.DataFrame
    system_force_table: pd.DataFrame
    brace_force_table: pd.DataFrame
    resized_input: BuildingInput


def define_zones(n_story: int) -> List[ZoneDefinition]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        ZoneDefinition("Lower Zone", 1, z1),
        ZoneDefinition("Middle Zone", z1 + 1, z2),
        ZoneDefinition("Upper Zone", z2 + 1, n_story),
    ]


def zone_for_story(inp: BuildingInput, story: int) -> Tuple[ZoneDefinition, ZoneMembers]:
    for z in define_zones(inp.n_story):
        if z.story_start <= story <= z.story_end:
            members = inp.lower if z.name == "Lower Zone" else inp.middle if z.name == "Middle Zone" else inp.upper
            return z, members
    raise ValueError("Story out of range")


def auto_brace_levels(inp: BuildingInput) -> List[int]:
    n = inp.brace.n_outrigger_levels
    if n <= 0:
        return []
    ratios = [0.34, 0.62, 0.82][:n]
    return sorted({max(2, min(inp.n_story - 1, round(inp.n_story * r))) for r in ratios})


def brace_zone_count(inp: BuildingInput, levels: List[int]) -> Dict[str, int]:
    counts = {z.name: 0 for z in define_zones(inp.n_story)}
    for lv in levels:
        z, _ = zone_for_story(inp, lv)
        counts[z.name] += 1
    return counts


def chs_area_mm2(d_mm: float, t_mm: float) -> float:
    di = max(d_mm - 2.0 * t_mm, 0.0)
    return 0.25 * pi * (d_mm**2 - di**2)


def core_wall_group_I(inp: BuildingInput, t: float) -> float:
    x = inp.core_outer_x / 2.0
    y = inp.core_outer_y / 2.0
    top = t * inp.core_outer_x**3 / 12.0 + inp.core_outer_x * t * y**2
    left = t * inp.core_outer_y**3 / 12.0 + inp.core_outer_y * t * x**2
    return min(2.0 * top, 2.0 * left)


def perimeter_wall_group_I(inp: BuildingInput, t: float, reduction_len: float = 0.16) -> float:
    lx = inp.plan_x * reduction_len
    ly = inp.plan_y * reduction_len
    x = 0.30 * inp.plan_x
    y = 0.30 * inp.plan_y
    top = t * lx**3 / 12.0 + lx * t * y**2
    left = t * ly**3 / 12.0 + ly * t * x**2
    return min(2.0 * top, 2.0 * left)


def _rect_I_min(x: float, y: float) -> float:
    return min(x * y**3 / 12.0, y * x**3 / 12.0)


def frame_components(inp: BuildingInput, m: ZoneMembers) -> Tuple[float, float, float, float]:
    E = inp.Ec_mpa * 1e6
    h = inp.story_h
    n_corner = 4
    n_per = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    n_total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    n_int = max(0, n_total - n_corner - n_per)
    n_beam = inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1)

    Icols = inp.col_cr * (
        n_corner * _rect_I_min(m.corner_col_x, m.corner_col_y) +
        n_per * _rect_I_min(m.perimeter_col_x, m.perimeter_col_y) +
        n_int * _rect_I_min(m.interior_col_x, m.interior_col_y)
    )
    k_col = 12.0 * E * Icols / max(h**3, 1e-9)

    Ibeam = m.beam_b * m.beam_h**3 / 12.0
    span = 0.5 * (inp.bay_x + inp.bay_y)
    k_beam = inp.beam_cr * n_beam * 12.0 * E * Ibeam / max(span**3, 1e-9)

    area = inp.plan_x * inp.plan_y
    k_slab = inp.slab_cr * 0.0008 * E * area * m.slab_t / max(0.5 * (inp.plan_x + inp.plan_y), 1e-9)

    beam_col_coupled = 1.0 / (1.0 / max(k_col, 1e-9) + 1.0 / max(0.90 * k_beam, 1e-9))
    frame_total = beam_col_coupled + 0.25 * k_slab
    return k_col, k_beam, k_slab, frame_total


def outrigger_story_stiffness(inp: BuildingInput) -> float:
    E = inp.Es_mpa * 1e6
    d = inp.brace.outer_diameter_mm
    t = inp.brace.thickness_mm
    A = chs_area_mm2(d, t) * 1e-6
    arm = 0.5 * max(inp.plan_x - inp.core_outer_x, inp.plan_y - inp.core_outer_y)
    L = sqrt(arm**2 + inp.story_h**2)
    k_axial = E * A / max(inp.brace.k_factor * L, 1e-9)
    return inp.brace.buckling_reduction * 4.0 * k_axial * (arm / max(inp.story_h, 1e-9))**2 * 0.08


def retaining_base_spring(inp: BuildingInput) -> float:
    if not inp.retaining_enabled or inp.n_basement <= 0:
        return 0.0
    E = inp.Ec_mpa * 1e6
    h = max(inp.n_basement * inp.basement_h, 1e-9)
    t = inp.retaining_t
    l_eff_x = 0.16 * inp.plan_x
    l_eff_y = 0.16 * inp.plan_y
    I = min(2.0 * (t * l_eff_x**3 / 12.0), 2.0 * (t * l_eff_y**3 / 12.0))
    return inp.retaining_reduction * 3.0 * E * I / max(h**3, 1e-9)


def story_mass(inp: BuildingInput, m: ZoneMembers) -> float:
    area = inp.plan_x * inp.plan_y
    q = 6.0 + CONCRETE_UNIT_WEIGHT * m.slab_t
    W = area * q
    return (W * 1000.0) / G


def build_story_arrays(inp: BuildingInput) -> Tuple[List[StoryState], np.ndarray, np.ndarray, List[int], Dict[str, int]]:
    brace_levels = auto_brace_levels(inp)
    brace_counts = brace_zone_count(inp, brace_levels)
    E = inp.Ec_mpa * 1e6
    base_spring = retaining_base_spring(inp)

    states: List[StoryState] = []
    masses: List[float] = []
    k_story: List[float] = []

    for s in range(1, inp.n_story + 1):
        z, m = zone_for_story(inp, s)
        brace_here = s in brace_levels
        z_braces = brace_counts[z.name]
        zone_h = z.n_stories * inp.story_h

        I_core = inp.wall_cr * core_wall_group_I(inp, m.wall_t)
        k_core_raw = 3.0 * E * I_core / max((zone_h**3) / z.n_stories, 1e-9)

        I_per = inp.perimeter_wall_cr * perimeter_wall_group_I(inp, m.perimeter_wall_t)
        k_per_raw = 3.0 * E * I_per / max((zone_h**3) / z.n_stories, 1e-9)

        k_col, k_beam, k_slab, k_frame_raw = frame_components(inp, m)

        beam_col_ratio = k_beam / max(k_col, 1e-9)
        frame_core_ratio = k_frame_raw / max(k_core_raw, 1e-9)

        frame_compat = clamp(0.82 + 0.16 * min(beam_col_ratio, 1.20), 0.72, 0.96)
        core_compat = clamp(0.82 + 0.16 * min(frame_core_ratio, 1.10), 0.72, 0.96)
        perim_compat = clamp(0.68 + 0.10 * min(k_frame_raw / max(k_per_raw, 1e-9), 1.00), 0.56, 0.82)

        k_core = k_core_raw * core_compat
        k_per = k_per_raw * perim_compat
        k_frame = k_frame_raw * frame_compat

        zone_force_amplifier = 1.0 + 0.06 * z_braces
        if brace_here:
            zone_force_amplifier += 0.04

        k_core *= zone_force_amplifier
        k_frame *= (1.0 + 0.04 * z_braces)
        k_per *= (1.0 + 0.02 * z_braces)

        k_out = outrigger_story_stiffness(inp) if brace_here else 0.0
        k_out_eff = k_out * (0.55 + 0.20 * min(k_core / max(k_frame, 1e-9), 1.2))

        k_total = k_core + k_per + k_frame + k_out_eff
        if s == 1:
            k_total += base_spring

        masses.append(story_mass(inp, m))
        k_story.append(k_total)
        states.append(StoryState(
            story=s,
            zone=z.name,
            brace_level=brace_here,
            wall_t=m.wall_t,
            perimeter_wall_t=m.perimeter_wall_t,
            beam_b=m.beam_b,
            beam_h=m.beam_h,
            slab_t=m.slab_t,
            corner_col=f"{m.corner_col_x:.2f}x{m.corner_col_y:.2f}",
            perimeter_col=f"{m.perimeter_col_x:.2f}x{m.perimeter_col_y:.2f}",
            interior_col=f"{m.interior_col_x:.2f}x{m.interior_col_y:.2f}",
            k_core=k_core,
            k_perimeter_walls=k_per,
            k_frame=k_frame,
            k_outrigger=k_out_eff,
            k_total=k_total,
            mass_t=masses[-1] / 1000.0,
        ))

    M = np.diag(masses)
    n = len(k_story)
    K = np.zeros((n, n), dtype=float)
    for i, ki in enumerate(k_story):
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
            K[i - 1, i - 1] += ki
    return states, M, K, brace_levels, brace_counts


def modal_analysis(M: np.ndarray, K: np.ndarray, modes: int = 3) -> Tuple[List[float], np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eig(np.linalg.solve(M, K))
    evals = np.real(evals)
    evecs = np.real(evecs)
    pos = evals > 1e-10
    evals = evals[pos]
    evecs = evecs[:, pos]
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    omegas = np.sqrt(evals[:modes])
    periods = [2.0 * pi / w for w in omegas]
    return periods, omegas, evecs[:, :modes]


def static_response(inp: BuildingInput, K: np.ndarray, total_weight_kN: float) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    n = K.shape[0]
    V = inp.seismic_coeff * total_weight_kN * 1000.0
    heights = np.arange(1, n + 1, dtype=float) * inp.story_h
    F = V * heights / heights.sum()
    u = np.linalg.solve(K, F.reshape(-1, 1)).flatten()
    roof = float(u[-1])
    drifts = np.diff(np.r_[0.0, u])
    max_drift = float(np.max(np.abs(drifts)))
    return roof, max_drift, max_drift / inp.story_h, u, drifts


def total_weight_kN(inp: BuildingInput) -> float:
    W = 0.0
    for s in range(1, inp.n_story + 1):
        _, m = zone_for_story(inp, s)
        area = inp.plan_x * inp.plan_y
        q = 6.0 + CONCRETE_UNIT_WEIGHT * m.slab_t
        W += area * q
    return W


def code_periods(inp: BuildingInput) -> Tuple[float, float]:
    H = inp.n_story * inp.story_h
    t_code = inp.Ct * H**inp.x_exp
    return t_code, inp.Cu * t_code


def stiffness_contributions(states: List[StoryState]) -> Dict[str, float]:
    return {
        "Core": sum(s.k_core for s in states),
        "Perimeter walls": sum(s.k_perimeter_walls for s in states),
        "Frame (beam+column+slab)": sum(s.k_frame for s in states),
        "Outriggers": sum(s.k_outrigger for s in states),
    }


def scale_zone(m: ZoneMembers, story_h: float, beam_col_factor: float, wall_factor: float, perim_factor: float, slab_factor: float, brace_bonus: float = 1.0) -> ZoneMembers:
    beam_h = clamp(m.beam_h * beam_col_factor * brace_bonus, 0.60, 0.32 * story_h)
    beam_b = clamp(m.beam_b * (0.94 + 0.10 * beam_col_factor) * (0.98 + 0.02 * brace_bonus), 0.40, 1.00)
    return ZoneMembers(
        wall_t=clamp(m.wall_t * wall_factor * brace_bonus, 0.45, 1.40),
        perimeter_wall_t=clamp(m.perimeter_wall_t * perim_factor * (0.98 + 0.02 * brace_bonus), 0.20, 0.70),
        beam_b=beam_b,
        beam_h=beam_h,
        slab_t=clamp(m.slab_t * slab_factor, 0.18, 0.34),
        corner_col_x=clamp(m.corner_col_x * beam_col_factor * brace_bonus, 0.80, 1.80),
        corner_col_y=clamp(m.corner_col_y * beam_col_factor * brace_bonus, 0.80, 1.80),
        perimeter_col_x=clamp(m.perimeter_col_x * beam_col_factor * brace_bonus, 0.75, 1.60),
        perimeter_col_y=clamp(m.perimeter_col_y * beam_col_factor * brace_bonus, 0.75, 1.60),
        interior_col_x=clamp(m.interior_col_x * beam_col_factor, 0.70, 1.50),
        interior_col_y=clamp(m.interior_col_y * beam_col_factor, 0.70, 1.50),
    )


def auto_size(inp: BuildingInput, max_iter: int = 20) -> Tuple[BuildingInput, pd.DataFrame]:
    log = []
    current = replace(inp)
    t_code, t_ceiling = code_periods(current)
    target_floor = 0.72 * t_code

    for it in range(1, max_iter + 1):
        states, M, K, brace_levels, brace_counts = build_story_arrays(current)
        periods, _, _ = modal_analysis(M, K, modes=3)
        W = total_weight_kN(current)
        roof, _, drift_ratio, _, _ = static_response(current, K, W)
        contrib = stiffness_contributions(states)
        total_k = sum(contrib.values())
        shares = {k: v / max(total_k, 1e-9) for k, v in contrib.items()}

        log.append({
            "iter": it,
            "T1": periods[0],
            "roof_disp_m": roof,
            "max_story_drift_ratio": drift_ratio,
            "core_share": shares["Core"],
            "frame_share": shares["Frame (beam+column+slab)"],
            "perim_share": shares["Perimeter walls"],
            "outrigger_share": shares["Outriggers"],
        })

        ok = (periods[0] <= t_ceiling) and (drift_ratio <= current.drift_limit) and (periods[0] >= target_floor)
        if ok:
            break

        stiff_need = max((periods[0] / t_ceiling) ** 2 if periods[0] > t_ceiling else 1.0,
                         (drift_ratio / current.drift_limit) ** 1.10 if drift_ratio > current.drift_limit else 1.0)
        soften_need = (target_floor / max(periods[0], 1e-9)) ** 0.32 if periods[0] < target_floor else 1.0

        desired = {
            "Core": 0.40,
            "Frame (beam+column+slab)": 0.32,
            "Perimeter walls": 0.16,
            "Outriggers": 0.12,
        }

        if stiff_need > 1.0:
            beam_col = stiff_need ** 0.15
            wall = stiff_need ** 0.16
            perim = stiff_need ** 0.09
            slab = stiff_need ** 0.04
        else:
            beam_col = 1.0 / (soften_need ** 0.12)
            wall = 1.0 / (soften_need ** 0.14)
            perim = 1.0 / (soften_need ** 0.07)
            slab = 1.0 / (soften_need ** 0.03)

        frame_gap = desired["Frame (beam+column+slab)"] - shares["Frame (beam+column+slab)"]
        core_gap = desired["Core"] - shares["Core"]
        perim_gap = desired["Perimeter walls"] - shares["Perimeter walls"]

        beam_col *= 1.0 + 0.22 * frame_gap + 0.10 * max(core_gap, 0.0)
        wall *= 1.0 + 0.24 * core_gap + 0.10 * max(frame_gap, 0.0)
        perim *= 1.0 + 0.16 * perim_gap

        lower_bonus = 1.0 + 0.05 * brace_counts["Lower Zone"]
        middle_bonus = 1.0 + 0.05 * brace_counts["Middle Zone"]
        upper_bonus = 1.0 + 0.05 * brace_counts["Upper Zone"]

        current = replace(
            current,
            lower=scale_zone(current.lower, current.story_h, beam_col * 1.05, wall * 1.10, perim * 1.03, slab, lower_bonus),
            middle=scale_zone(current.middle, current.story_h, beam_col, wall, perim, slab, middle_bonus),
            upper=scale_zone(current.upper, current.story_h, beam_col * 0.96, wall * 0.94, perim * 0.97, slab, upper_bonus),
        )

    return current, pd.DataFrame(log)


def system_force_table(states: List[StoryState], base_shear_kN: float) -> pd.DataFrame:
    contrib = stiffness_contributions(states)
    total = sum(contrib.values())
    rows = []
    for name, k in contrib.items():
        share = k / max(total, 1e-9)
        rows.append({
            "System": name,
            "Stiffness share": share,
            "Approx. shear (kN)": share * base_shear_kN,
        })
    return pd.DataFrame(rows)


def brace_force_table(inp: BuildingInput, states: List[StoryState], displacements: np.ndarray) -> pd.DataFrame:
    levels = [s.story for s in states if s.brace_level]
    k_out = outrigger_story_stiffness(inp)
    rows = []
    for lv in levels:
        idx = lv - 1
        u = abs(displacements[idx])
        F_level = k_out * u / 1000.0
        rows.append({
            "Brace level": lv,
            "Story disp. (m)": u,
            "Approx. total outrigger force (kN)": F_level,
            "Approx. force per line (kN)": F_level / 4.0,
        })
    return pd.DataFrame(rows)


def analyze(inp: BuildingInput) -> AnalysisResult:
    sized, log = auto_size(inp)
    states, M, K, brace_levels, brace_counts = build_story_arrays(sized)
    periods, _, _ = modal_analysis(M, K, modes=3)
    W = total_weight_kN(sized)
    roof, max_drift, drift_ratio, u, _ = static_response(sized, K, W)
    t_code, t_ceiling = code_periods(sized)

    story_df = pd.DataFrame([{
        "Story": s.story,
        "Zone": s.zone,
        "Brace level": "Yes" if s.brace_level else "No",
        "Wall t (m)": round(s.wall_t, 3),
        "Perimeter wall t (m)": round(s.perimeter_wall_t, 3),
        "Beam b (m)": round(s.beam_b, 3),
        "Beam h (m)": round(s.beam_h, 3),
        "Slab t (m)": round(s.slab_t, 3),
        "Corner col": s.corner_col,
        "Perimeter col": s.perimeter_col,
        "Interior col": s.interior_col,
        "K_core (GN/m)": s.k_core / 1e9,
        "K_perim (GN/m)": s.k_perimeter_walls / 1e9,
        "K_frame (GN/m)": s.k_frame / 1e9,
        "K_outrigger (GN/m)": s.k_outrigger / 1e9,
        "K_total (GN/m)": s.k_total / 1e9,
        "Mass (t)": s.mass_t,
        "Disp. (m)": u[s.story - 1],
    } for s in states])

    zone_rows = []
    for z in define_zones(sized.n_story):
        m = sized.lower if z.name == "Lower Zone" else sized.middle if z.name == "Middle Zone" else sized.upper
        zone_rows.append({
            "Zone": z.name,
            "Stories": f"{z.story_start}-{z.story_end}",
            "Brace levels in zone": brace_counts[z.name],
            "Core wall t (m)": m.wall_t,
            "Perimeter wall t (m)": m.perimeter_wall_t,
            "Beam b x h (m)": f"{m.beam_b:.2f} x {m.beam_h:.2f}",
            "Slab t (m)": m.slab_t,
            "Corner col (m)": f"{m.corner_col_x:.2f} x {m.corner_col_y:.2f}",
            "Perimeter col (m)": f"{m.perimeter_col_x:.2f} x {m.perimeter_col_y:.2f}",
            "Interior col (m)": f"{m.interior_col_x:.2f} x {m.interior_col_y:.2f}",
        })
    zone_df = pd.DataFrame(zone_rows)

    contrib = stiffness_contributions(states)
    contrib_df = pd.DataFrame({
        "System": list(contrib.keys()),
        "Total stiffness (GN/m)": [v / 1e9 for v in contrib.values()],
        "Share": [v / sum(contrib.values()) for v in contrib.values()],
    })
    V_kN = sized.seismic_coeff * W

    return AnalysisResult(
        periods=periods,
        roof_disp_m=roof,
        max_story_drift_m=max_drift,
        max_story_drift_ratio=drift_ratio,
        code_period=t_code,
        period_ceiling=t_ceiling,
        brace_levels=brace_levels,
        brace_zone_count=brace_counts,
        story_table=story_df,
        zone_table=zone_df,
        contribution_table=contrib_df,
        iteration_log=log,
        system_force_table=system_force_table(states, V_kN),
        brace_force_table=brace_force_table(sized, states, u),
        resized_input=sized,
    )


def plot_mode_shape(result: AnalysisResult, mode_index: int) -> plt.Figure:
    # simple surrogate from displacement envelope for display purposes
    fig, ax = plt.subplots(figsize=(5, 7))
    y = result.story_table["Story"].to_numpy()
    if mode_index == 1:
        x = y / y.max()
    elif mode_index == 2:
        x = np.sin(np.pi * y / y.max())
    else:
        x = np.sin(2 * np.pi * y / y.max())
    ax.plot(x, y)
    ax.set_xlabel(f"Mode {mode_index} shape (normalized)")
    ax.set_ylabel("Story")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Mode {mode_index}")
    return fig


def plot_drift(result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 7))
    dr = result.story_table["Disp. (m)"].diff().fillna(result.story_table["Disp. (m)"]).abs() / result.resized_input.story_h
    ax.plot(dr.to_numpy(), result.story_table["Story"].to_numpy())
    ax.axvline(result.resized_input.drift_limit, linestyle="--")
    ax.set_xlabel("Story drift ratio")
    ax.set_ylabel("Story")
    ax.grid(True, alpha=0.3)
    ax.set_title("Story drift ratios")
    return fig


def plot_plan(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black")
    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x
        ax.plot([x, x], [0, inp.plan_y], color="lightgray", linewidth=0.7)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y
        ax.plot([0, inp.plan_x], [y, y], color="lightgray", linewidth=0.7)
    cx0 = (inp.plan_x - inp.core_outer_x) / 2.0
    cy0 = (inp.plan_y - inp.core_outer_y) / 2.0
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core_outer_x, inp.core_outer_y, fill=False, linewidth=2))
    ix0 = (inp.plan_x - inp.core_open_x) / 2.0
    iy0 = (inp.plan_y - inp.core_open_y) / 2.0
    ax.add_patch(plt.Rectangle((ix0, iy0), inp.core_open_x, inp.core_open_y, fill=False, linestyle="--"))
    if inp.retaining_enabled:
        ax.add_patch(plt.Rectangle((0, 0), inp.plan_x, inp.plan_y, fill=False, linewidth=1.0, linestyle=':'))
    # core to perimeter lines, shown once as layout concept
    center_x = inp.plan_x / 2.0
    center_y = inp.plan_y / 2.0
    ax.plot([cx0, 0], [center_y, center_y], linewidth=2)
    ax.plot([cx0 + inp.core_outer_x, inp.plan_x], [center_y, center_y], linewidth=2)
    ax.plot([center_x, center_x], [cy0, 0], linewidth=2)
    ax.plot([center_x, center_x], [cy0 + inp.core_outer_y, inp.plan_y], linewidth=2)
    ax.set_aspect("equal")
    ax.set_title("Plan: core, grids, perimeter, and outrigger layout concept")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig


def plot_elevation(inp: BuildingInput, result: AnalysisResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    H = inp.n_story * inp.story_h
    ax.plot([0, 0], [0, H], color="black")
    ax.plot([inp.plan_x * 0.2, inp.plan_x * 0.2], [0, H], color="black")
    for s in range(inp.n_story + 1):
        y = s * inp.story_h
        ax.plot([0, inp.plan_x * 0.2], [y, y], color="lightgray", linewidth=0.6)
    for lv in result.brace_levels:
        y = lv * inp.story_h
        ax.plot([0, inp.plan_x * 0.2], [y, y], color="red", linewidth=2)
    for z in define_zones(inp.n_story):
        y = z.story_end * inp.story_h
        ax.text(inp.plan_x * 0.22, y, f"{z.name}: {z.story_start}-{z.story_end}")
    ax.set_title("Elevation: outrigger levels and zones")
    ax.set_xlabel("Structural width (schematic)")
    ax.set_ylabel("Height (m)")
    return fig


def input_sidebar() -> BuildingInput:
    st.sidebar.header("Inputs")
    default = BuildingInput()
    n_story = st.sidebar.number_input("Stories", 10, 100, default.n_story)
    n_basement = st.sidebar.number_input("Basements", 0, 10, default.n_basement)
    story_h = st.sidebar.number_input("Story height (m)", 2.8, 5.0, default.story_h)
    plan_x = st.sidebar.number_input("Plan X (m)", 20.0, 100.0, default.plan_x)
    plan_y = st.sidebar.number_input("Plan Y (m)", 20.0, 100.0, default.plan_y)
    n_bays_x = st.sidebar.number_input("Bays X", 3, 12, default.n_bays_x)
    n_bays_y = st.sidebar.number_input("Bays Y", 3, 12, default.n_bays_y)
    bay_x = st.sidebar.number_input("Bay X (m)", 5.0, 12.0, default.bay_x)
    bay_y = st.sidebar.number_input("Bay Y (m)", 5.0, 12.0, default.bay_y)

    st.sidebar.subheader("Core")
    core_outer_x = st.sidebar.number_input("Core outer X (m)", 8.0, 30.0, default.core_outer_x)
    core_outer_y = st.sidebar.number_input("Core outer Y (m)", 8.0, 30.0, default.core_outer_y)
    core_open_x = st.sidebar.number_input("Core opening X (m)", 4.0, 24.0, default.core_open_x)
    core_open_y = st.sidebar.number_input("Core opening Y (m)", 4.0, 24.0, default.core_open_y)

    st.sidebar.subheader("Cracked stiffness")
    wall_cr = st.sidebar.number_input("Core wall factor", 0.2, 1.0, default.wall_cr, 0.05)
    perimeter_wall_cr = st.sidebar.number_input("Perimeter wall factor", 0.2, 1.0, default.perimeter_wall_cr, 0.05)
    beam_cr = st.sidebar.number_input("Beam factor", 0.2, 1.0, default.beam_cr, 0.05)
    col_cr = st.sidebar.number_input("Column factor", 0.2, 1.0, default.col_cr, 0.05)
    slab_cr = st.sidebar.number_input("Slab factor", 0.1, 1.0, default.slab_cr, 0.05)

    st.sidebar.subheader("Seismic / code")
    drift_limit = st.sidebar.number_input("Drift limit", 0.001, 0.01, default.drift_limit, 0.0005, format="%.4f")
    seismic_coeff = st.sidebar.number_input("Seismic coeff.", 0.005, 0.08, default.seismic_coeff, 0.0025)
    Ct = st.sidebar.number_input("Ct", 0.01, 0.12, default.Ct, 0.001)
    x_exp = st.sidebar.number_input("x exponent", 0.5, 1.0, default.x_exp, 0.01)
    Cu = st.sidebar.number_input("Cu", 1.0, 2.0, default.Cu, 0.05)

    st.sidebar.subheader("Outrigger CHS")
    n_outrigger_levels = st.sidebar.number_input("Outrigger levels", 0, 3, default.brace.n_outrigger_levels)
    outer_diameter_mm = st.sidebar.number_input("CHS OD (mm)", 200.0, 1000.0, default.brace.outer_diameter_mm)
    thickness_mm = st.sidebar.number_input("CHS thickness (mm)", 8.0, 50.0, default.brace.thickness_mm)
    k_factor = st.sidebar.number_input("K factor", 0.5, 2.0, default.brace.k_factor, 0.05)
    buckling_reduction = st.sidebar.number_input("Buckling reduction", 0.3, 1.0, default.brace.buckling_reduction, 0.05)

    st.sidebar.subheader("Retaining wall")
    retaining_enabled = st.sidebar.checkbox("Retaining enabled", value=default.retaining_enabled)
    retaining_t = st.sidebar.number_input("Retaining wall t (m)", 0.2, 1.0, default.retaining_t)
    retaining_reduction = st.sidebar.number_input("Retaining reduction", 0.0, 0.20, default.retaining_reduction, 0.01)

    return BuildingInput(
        n_story=int(n_story),
        n_basement=int(n_basement),
        story_h=float(story_h),
        basement_h=default.basement_h,
        plan_x=float(plan_x),
        plan_y=float(plan_y),
        n_bays_x=int(n_bays_x),
        n_bays_y=int(n_bays_y),
        bay_x=float(bay_x),
        bay_y=float(bay_y),
        core_outer_x=float(core_outer_x),
        core_outer_y=float(core_outer_y),
        core_open_x=float(core_open_x),
        core_open_y=float(core_open_y),
        wall_cr=float(wall_cr),
        perimeter_wall_cr=float(perimeter_wall_cr),
        beam_cr=float(beam_cr),
        col_cr=float(col_cr),
        slab_cr=float(slab_cr),
        drift_limit=float(drift_limit),
        seismic_coeff=float(seismic_coeff),
        Ct=float(Ct),
        x_exp=float(x_exp),
        Cu=float(Cu),
        retaining_enabled=bool(retaining_enabled),
        retaining_t=float(retaining_t),
        retaining_reduction=float(retaining_reduction),
        brace=BraceSystem(
            n_outrigger_levels=int(n_outrigger_levels),
            outer_diameter_mm=float(outer_diameter_mm),
            thickness_mm=float(thickness_mm),
            k_factor=float(k_factor),
            buckling_reduction=float(buckling_reduction),
        ),
    )


def compare_summary(no_brace: AnalysisResult, with_brace: AnalysisResult) -> pd.DataFrame:
    rows = []
    items = [
        ("T1 (s)", no_brace.periods[0], with_brace.periods[0]),
        ("Roof disp. (m)", no_brace.roof_disp_m, with_brace.roof_disp_m),
        ("Max drift ratio", no_brace.max_story_drift_ratio, with_brace.max_story_drift_ratio),
        ("Core wall lower (m)", no_brace.resized_input.lower.wall_t, with_brace.resized_input.lower.wall_t),
        ("Beam h lower (m)", no_brace.resized_input.lower.beam_h, with_brace.resized_input.lower.beam_h),
        ("Corner col lower x (m)", no_brace.resized_input.lower.corner_col_x, with_brace.resized_input.lower.corner_col_x),
    ]
    for label, a, b in items:
        rows.append({"Metric": label, "No outriggers": a, "With outriggers": b, "Delta": b - a, "Delta %": 100*(b-a)/a if abs(a)>1e-9 else np.nan})
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Outrigger Stage 4", layout="wide")
    st.title("Stage 4 — Comparison + system forces")
    st.caption("Approximate MDOF preliminary design tool. Use for calibration, not as a substitute for ETABS.")

    inp = input_sidebar()
    if st.button("Analyze"):
        with st.spinner("Running analyses..."):
            no_brace_inp = replace(inp, brace=replace(inp.brace, n_outrigger_levels=0))
            res_no = analyze(no_brace_inp)
            res_yes = analyze(inp)
            st.session_state["res_no"] = res_no
            st.session_state["res_yes"] = res_yes

    if "res_yes" not in st.session_state:
        st.info("Set inputs and click Analyze.")
        return

    res_no: AnalysisResult = st.session_state["res_no"]
    res_yes: AnalysisResult = st.session_state["res_yes"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T1 no-outrigger (s)", f"{res_no.periods[0]:.3f}")
    c2.metric("T1 with-outrigger (s)", f"{res_yes.periods[0]:.3f}")
    c3.metric("Roof disp. with-outrigger (m)", f"{res_yes.roof_disp_m:.3f}")
    c4.metric("Max drift ratio with-outrigger", f"{res_yes.max_story_drift_ratio:.5f}")

    tabs = st.tabs(["Comparison", "With outriggers", "Without outriggers", "Plan / Elevation", "Modes / Drift"])

    with tabs[0]:
        st.subheader("Global comparison")
        st.dataframe(compare_summary(res_no, res_yes), use_container_width=True)
        st.subheader("System shear split — with outriggers")
        st.dataframe(res_yes.system_force_table, use_container_width=True)
        st.subheader("Approximate outrigger forces")
        st.dataframe(res_yes.brace_force_table, use_container_width=True)

    with tabs[1]:
        st.subheader("Zone table — with outriggers")
        st.dataframe(res_yes.zone_table, use_container_width=True)
        st.subheader("Stiffness contributions — with outriggers")
        st.dataframe(res_yes.contribution_table, use_container_width=True)
        st.subheader("Story table — with outriggers")
        st.dataframe(res_yes.story_table, use_container_width=True, height=500)
        st.subheader("Sizing iterations")
        st.dataframe(res_yes.iteration_log, use_container_width=True)

    with tabs[2]:
        st.subheader("Zone table — without outriggers")
        st.dataframe(res_no.zone_table, use_container_width=True)
        st.subheader("Stiffness contributions — without outriggers")
        st.dataframe(res_no.contribution_table, use_container_width=True)
        st.subheader("Story table — without outriggers")
        st.dataframe(res_no.story_table, use_container_width=True, height=500)

    with tabs[3]:
        st.pyplot(plot_plan(inp, res_yes))
        st.pyplot(plot_elevation(inp, res_yes))

    with tabs[4]:
        mode = st.selectbox("Mode", [1, 2, 3], index=0)
        st.pyplot(plot_mode_shape(res_yes, mode))
        st.pyplot(plot_drift(res_yes))


if __name__ == "__main__":
    main()
