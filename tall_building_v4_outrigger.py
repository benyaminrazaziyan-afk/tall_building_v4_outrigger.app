
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# PRELIMINARY TOWER DESIGNER
# Revised for:
# 1) story-by-story MDOF stiffness assembly
# 2) period targeting based on empirical standard period
# 3) section-stiffness-driven iteration
# 4) outrigger modeled at actual story levels, not as a single global spring
# 5) comparison of belt-truss outrigger vs pipe-braced outrigger
# ============================================================

st.set_page_config(
    page_title="Tower Pre-Design MDOF",
    layout="wide",
    initial_sidebar_state="expanded",
)

G = 9.81
RHO_STEEL = 7850.0
RHO_RC = 2500.0

CORE_COLOR = "#2E8B57"
COL_COLOR = "#B22222"
OUTRIGGER_COLOR = "#FF8C00"
PIPE_COLOR = "#4169E1"

AUTHOR = "Benyamin"
VERSION = "v5.0-MDOF-SectionStiffness"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class OutriggerInput:
    story: int
    system_type: str  # "belt_truss" or "pipe_brace"
    truss_depth_m: float
    main_area_m2: float
    diagonal_area_m2: float
    active: bool = True


@dataclass
class BuildingInput:
    n_story: int
    n_basement: int
    story_height_m: float
    basement_height_m: float
    plan_x_m: float
    plan_y_m: float
    bay_x_m: float
    bay_y_m: float
    n_bays_x: int
    n_bays_y: int

    fck_mpa: float
    Ec_mpa: float
    fy_mpa: float

    DL_kN_m2: float
    LL_kN_m2: float
    finishes_kN_m2: float
    facade_line_load_kN_m: float

    min_wall_thk_m: float
    max_wall_thk_m: float
    min_col_dim_m: float
    max_col_dim_m: float
    min_beam_width_m: float
    min_beam_depth_m: float
    min_slab_thk_m: float
    max_slab_thk_m: float

    wall_cracked: float
    col_cracked: float

    Ct: float
    x_exp: float
    upper_period_factor: float
    target_position_factor: float

    drift_limit_ratio: float
    seismic_mass_factor: float

    lower_zone_wall_count: int
    middle_zone_wall_count: int
    upper_zone_wall_count: int

    perimeter_column_factor: float
    corner_column_factor: float

    core_opening_factor_x: float = 0.18
    core_opening_factor_y: float = 0.16

    outriggers: List[OutriggerInput] = field(default_factory=list)


@dataclass
class StoryState:
    story: int
    elevation_m: float
    zone_name: str

    core_outer_x_m: float
    core_outer_y_m: float
    core_opening_x_m: float
    core_opening_y_m: float
    wall_thk_m: float
    wall_count: int

    interior_col_x_m: float
    interior_col_y_m: float
    perimeter_col_x_m: float
    perimeter_col_y_m: float
    corner_col_x_m: float
    corner_col_y_m: float

    slab_thk_m: float
    beam_b_m: float
    beam_h_m: float

    mass_kg: float
    k_story_N_m: float
    k_core_N_m: float
    k_col_N_m: float
    k_outrigger_N_m: float

    steel_kg: float
    concrete_m3: float


@dataclass
class ModalResult:
    periods_s: List[float]
    freqs_hz: List[float]
    mode_shapes: List[np.ndarray]
    effective_mass_ratios: List[float]
    cumulative_mass_ratios: List[float]


@dataclass
class AnalysisResult:
    building: BuildingInput
    stories: List[StoryState]
    modal: ModalResult
    target_period_s: float
    reference_period_s: float
    upper_period_s: float
    estimated_period_s: float
    period_error_ratio: float
    total_weight_kN: float
    total_mass_kg: float
    total_steel_kg: float
    total_concrete_m3: float
    top_drift_m: float
    drift_ratio: float
    core_scale: float
    col_scale: float
    outr_scale: float
    convergence_table: pd.DataFrame
    sustainability_table: pd.DataFrame


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def total_height(inp: BuildingInput) -> float:
    return inp.n_story * inp.story_height_m

def floor_area(inp: BuildingInput) -> float:
    return inp.plan_x_m * inp.plan_y_m

def code_period(inp: BuildingInput) -> float:
    return inp.Ct * (total_height(inp) ** inp.x_exp)

def target_period(inp: BuildingInput) -> Tuple[float, float, float]:
    tref = code_period(inp)
    tupper = inp.upper_period_factor * tref
    ttarg = tref + inp.target_position_factor * (tupper - tref)
    return tref, ttarg, tupper

def story_zone_name(n_story: int, story: int) -> str:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    if story <= z1:
        return "Lower Zone"
    elif story <= z2:
        return "Middle Zone"
    return "Upper Zone"

def zone_wall_count(inp: BuildingInput, story: int) -> int:
    z = story_zone_name(inp.n_story, story)
    if z == "Lower Zone":
        return inp.lower_zone_wall_count
    if z == "Middle Zone":
        return inp.middle_zone_wall_count
    return inp.upper_zone_wall_count

def core_dimensions(inp: BuildingInput) -> Tuple[float, float, float, float]:
    outer_x = max(0.24 * inp.plan_x_m, 10.0)
    outer_y = max(0.22 * inp.plan_y_m, 10.0)
    open_x = inp.core_opening_factor_x * inp.plan_x_m
    open_y = inp.core_opening_factor_y * inp.plan_y_m
    open_x = min(open_x, outer_x - 2.5)
    open_y = min(open_y, outer_y - 2.5)
    return outer_x, outer_y, open_x, open_y

def slab_thickness_prelim(inp: BuildingInput, col_scale: float) -> float:
    span = max(inp.bay_x_m, inp.bay_y_m)
    t = (span / 28.0) * (0.95 + 0.10 * col_scale)
    return float(np.clip(t, inp.min_slab_thk_m, inp.max_slab_thk_m))

def beam_prelim(inp: BuildingInput, col_scale: float) -> Tuple[float, float]:
    span = max(inp.bay_x_m, inp.bay_y_m)
    h = (span / 12.0) * (0.95 + 0.12 * col_scale)
    h = float(np.clip(h, inp.min_beam_depth_m, 1.60))
    b = max(inp.min_beam_width_m, 0.45 * h)
    return b, h

def wall_thickness_by_story(inp: BuildingInput, story: int, core_scale: float) -> float:
    H = total_height(inp)
    base = H / 180.0
    z = story_zone_name(inp.n_story, story)
    zfac = {"Lower Zone": 1.00, "Middle Zone": 0.82, "Upper Zone": 0.64}[z]
    t = base * zfac * core_scale
    return float(np.clip(t, inp.min_wall_thk_m, inp.max_wall_thk_m))

def column_dimensions_by_story(inp: BuildingInput, story: int, col_scale: float) -> Dict[str, Tuple[float, float]]:
    z = story_zone_name(inp.n_story, story)
    zfac = {"Lower Zone": 1.00, "Middle Zone": 0.85, "Upper Zone": 0.72}[z]
    base = 0.85 * zfac * col_scale
    base = float(np.clip(base, inp.min_col_dim_m, inp.max_col_dim_m))

    inter = (base, base)
    per = (
        float(np.clip(base * inp.perimeter_column_factor, inp.min_col_dim_m, inp.max_col_dim_m)),
        float(np.clip(base * inp.perimeter_column_factor, inp.min_col_dim_m, inp.max_col_dim_m)),
    )
    cor = (
        float(np.clip(base * inp.corner_column_factor, inp.min_col_dim_m, inp.max_col_dim_m)),
        float(np.clip(base * inp.corner_column_factor, inp.min_col_dim_m, inp.max_col_dim_m)),
    )
    return {"interior": inter, "perimeter": per, "corner": cor}

def n_columns(inp: BuildingInput) -> Tuple[int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return corner, perimeter, interior

def rect_I_major(b: float, h: float) -> float:
    return max(b * h**3 / 12.0, h * b**3 / 12.0)

def cantilever_story_stiffness(EI: float, h: float) -> float:
    # effective story spring for lateral shear-building idealization
    return 12.0 * EI / (h**3)

def frame_story_stiffness_from_columns(EI_cols: float, h: float) -> float:
    # approximate sway stiffness contribution of columns in a storey
    return 12.0 * EI_cols / (h**3)

def outrigger_efficiency_factor(system_type: str) -> float:
    # belt truss generally offers better global coupling efficiency for towers
    if system_type == "belt_truss":
        return 1.00
    elif system_type == "pipe_brace":
        return 0.72
    return 0.85

def outrigger_embodied_factor(system_type: str) -> float:
    # proxy sustainability factor (lower is better)
    if system_type == "belt_truss":
        return 1.00
    elif system_type == "pipe_brace":
        return 1.15
    return 1.00


# ============================================================
# SECTION STIFFNESS MODEL
# ============================================================

def core_equivalent_I(inp: BuildingInput, story: int, wall_t: float) -> float:
    outer_x, outer_y, _, _ = core_dimensions(inp)
    wc = zone_wall_count(inp, story)

    x_side = outer_x / 2.0
    y_side = outer_y / 2.0

    # gross approximation of a coupled wall/core tube in each direction
    I_x = 2 * (outer_x * wall_t**3 / 12.0 + outer_x * wall_t * y_side**2)
    I_y = 2 * (outer_y * wall_t**3 / 12.0 + outer_y * wall_t * x_side**2)

    if wc >= 6:
        inner_x = 0.22 * outer_x
        add = 2 * (0.45 * outer_x * wall_t**3 / 12.0 + (0.45 * outer_x * wall_t) * inner_x**2)
        I_y += add
    if wc >= 8:
        inner_y = 0.22 * outer_y
        add = 2 * (0.45 * outer_y * wall_t**3 / 12.0 + (0.45 * outer_y * wall_t) * inner_y**2)
        I_x += add

    Ieq = min(I_x, I_y)
    return inp.wall_cracked * Ieq

def column_group_I(inp: BuildingInput, story: int, col_scale: float) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    dims = column_dimensions_by_story(inp, story, col_scale)
    nc, npf, ni = n_columns(inp)

    I_corner = rect_I_major(*dims["corner"])
    I_perim = rect_I_major(*dims["perimeter"])
    I_inter = rect_I_major(*dims["interior"])

    # direct column bending contribution only
    Ig = nc * I_corner + npf * I_perim + ni * I_inter
    return inp.col_cracked * Ig, dims

def beam_slab_mass(inp: BuildingInput, slab_t: float, beam_b: float, beam_h: float) -> Tuple[float, float]:
    A = floor_area(inp)

    slab_vol = A * slab_t

    beam_lines = inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1)
    avg_span = 0.5 * (inp.bay_x_m + inp.bay_y_m)
    beam_len = beam_lines * avg_span
    beam_vol = beam_b * beam_h * beam_len

    return slab_vol, beam_vol

def storey_mass(inp: BuildingInput, slab_t: float, beam_b: float, beam_h: float) -> Tuple[float, float, float]:
    A = floor_area(inp)
    slab_vol, beam_vol = beam_slab_mass(inp, slab_t, beam_b, beam_h)

    gravity_load_kN = (inp.DL_kN_m2 + inp.finishes_kN_m2 + 0.25 * inp.LL_kN_m2) * A
    slab_beam_selfweight_kN = 25.0 * (slab_vol + beam_vol)
    facade_kN = inp.facade_line_load_kN_m * 2 * (inp.plan_x_m + inp.plan_y_m)

    total_kN = (gravity_load_kN + slab_beam_selfweight_kN + facade_kN) * inp.seismic_mass_factor
    mass_kg = total_kN * 1000.0 / G
    return mass_kg, slab_vol, beam_vol

def outrigger_story_stiffness(inp: BuildingInput, story: int, outr_scale: float) -> Tuple[float, float]:
    if not inp.outriggers:
        return 0.0, 0.0

    outer_x, outer_y, _, _ = core_dimensions(inp)
    arm_x = 0.5 * (inp.plan_x_m - outer_x)
    arm_y = 0.5 * (inp.plan_y_m - outer_y)
    arm = max(arm_x, arm_y)

    E = inp.Ec_mpa * 1e6
    total_k = 0.0
    steel_kg = 0.0

    for o in inp.outriggers:
        if (not o.active) or (o.story != story):
            continue

        eta = outrigger_efficiency_factor(o.system_type)

        A_main = o.main_area_m2 * outr_scale
        A_diag = o.diagonal_area_m2 * outr_scale

        L_main = max(arm, 1.0)
        L_diag = sqrt(arm**2 + o.truss_depth_m**2)

        k_ax_main = 4.0 * E * A_main / L_main
        k_ax_diag = 4.0 * E * A_diag / L_diag

        # translate outrigger axial stiffness to equivalent lateral coupling stiffness
        # story-level contribution through overturning restraint:
        k_eq = eta * (k_ax_main + 0.60 * k_ax_diag)

        total_k += k_eq

        volume = 4.0 * A_main * L_main + 4.0 * A_diag * L_diag
        steel_kg += volume * RHO_STEEL * outrigger_embodied_factor(o.system_type)

    return total_k, steel_kg

def story_stiffness(inp: BuildingInput, story: int, core_scale: float, col_scale: float, outr_scale: float) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[float, float]]]:
    h = inp.story_height_m
    wall_t = wall_thickness_by_story(inp, story, core_scale)
    I_core = core_equivalent_I(inp, story, wall_t)

    I_cols, dims = column_group_I(inp, story, col_scale)

    E = inp.Ec_mpa * 1e6
    k_core = cantilever_story_stiffness(E * I_core, h)
    k_col = frame_story_stiffness_from_columns(E * I_cols, h)
    k_outr, outr_steel_kg = outrigger_story_stiffness(inp, story, outr_scale)

    # mild coupling bonus when outrigger is active and both core + perimeter columns exist
    coupling_bonus = 0.0
    if k_outr > 0.0:
        coupling_bonus = 0.18 * min(k_core, k_col)

    k_total = k_core + k_col + k_outr + coupling_bonus

    return (
        {
            "k_story": k_total,
            "k_core": k_core,
            "k_col": k_col,
            "k_outrigger": k_outr + coupling_bonus,
            "wall_t": wall_t,
            "outrigger_steel_kg": outr_steel_kg,
        },
        {
            "I_core": I_core,
            "I_cols": I_cols,
        },
        dims,
    )


# ============================================================
# MDOF ASSEMBLY
# ============================================================

def assemble_M_K(stories: List[StoryState]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(stories)
    M = np.diag([s.mass_kg for s in stories])

    K = np.zeros((n, n), dtype=float)
    for i, s in enumerate(stories):
        k = s.k_story_N_m
        if i == 0:
            K[i, i] += k
        else:
            K[i, i] += k
            K[i, i - 1] -= k
            K[i - 1, i] -= k
            K[i - 1, i - 1] += k
    return M, K

def solve_modal(stories: List[StoryState], n_modes: int = 5) -> ModalResult:
    M, K = assemble_M_K(stories)

    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    keep = eigvals > 1e-10
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    omegas = np.sqrt(eigvals[:n_modes])
    periods = [2.0 * pi / w for w in omegas]
    freqs = [w / (2.0 * pi) for w in omegas]

    ones = np.ones((len(stories), 1))
    total_mass = sum(s.mass_kg for s in stories)

    mass_ratios = []
    cumulative = []
    shapes = []
    cum = 0.0

    for i in range(min(n_modes, eigvecs.shape[1])):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = float(phi.T @ M @ phi)
        gamma = float((phi.T @ M @ ones) / denom)
        meff = gamma**2 * denom
        ratio = meff / total_mass
        cum += ratio

        ph = phi.flatten()
        if abs(ph[-1]) > 1e-9:
            ph = ph / ph[-1]
        if ph[-1] < 0:
            ph = -ph

        shapes.append(ph)
        mass_ratios.append(ratio)
        cumulative.append(cum)

    return ModalResult(
        periods_s=periods,
        freqs_hz=freqs,
        mode_shapes=shapes,
        effective_mass_ratios=mass_ratios,
        cumulative_mass_ratios=cumulative,
    )


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def build_story_states(inp: BuildingInput, core_scale: float, col_scale: float, outr_scale: float) -> List[StoryState]:
    stories: List[StoryState] = []
    outer_x, outer_y, open_x, open_y = core_dimensions(inp)

    slab_t = slab_thickness_prelim(inp, col_scale)
    beam_b, beam_h = beam_prelim(inp, col_scale)

    for story in range(1, inp.n_story + 1):
        zname = story_zone_name(inp.n_story, story)
        wc = zone_wall_count(inp, story)

        stiff, sec, dims = story_stiffness(inp, story, core_scale, col_scale, outr_scale)
        mass_kg, slab_vol, beam_vol = storey_mass(inp, slab_t, beam_b, beam_h)

        nc, npf, ni = n_columns(inp)
        col_vol = (
            nc * dims["corner"][0] * dims["corner"][1] * inp.story_height_m
            + npf * dims["perimeter"][0] * dims["perimeter"][1] * inp.story_height_m
            + ni * dims["interior"][0] * dims["interior"][1] * inp.story_height_m
        )

        wall_len = 2 * outer_x + 2 * outer_y
        if wc >= 6:
            wall_len += 0.90 * outer_x
        if wc >= 8:
            wall_len += 0.90 * outer_y
        wall_vol = wall_len * stiff["wall_t"] * inp.story_height_m

        concrete_m3 = slab_vol + beam_vol + col_vol + wall_vol
        steel_kg = stiff["outrigger_steel_kg"]

        stories.append(
            StoryState(
                story=story,
                elevation_m=story * inp.story_height_m,
                zone_name=zname,
                core_outer_x_m=outer_x,
                core_outer_y_m=outer_y,
                core_opening_x_m=open_x,
                core_opening_y_m=open_y,
                wall_thk_m=stiff["wall_t"],
                wall_count=wc,
                interior_col_x_m=dims["interior"][0],
                interior_col_y_m=dims["interior"][1],
                perimeter_col_x_m=dims["perimeter"][0],
                perimeter_col_y_m=dims["perimeter"][1],
                corner_col_x_m=dims["corner"][0],
                corner_col_y_m=dims["corner"][1],
                slab_thk_m=slab_t,
                beam_b_m=beam_b,
                beam_h_m=beam_h,
                mass_kg=mass_kg,
                k_story_N_m=stiff["k_story"],
                k_core_N_m=stiff["k_core"],
                k_col_N_m=stiff["k_col"],
                k_outrigger_N_m=stiff["k_outrigger"],
                steel_kg=steel_kg,
                concrete_m3=concrete_m3,
            )
        )

    return stories

def equivalent_lateral_force(inp: BuildingInput, total_weight_kN: float) -> float:
    # preliminary global lateral demand proxy for predesign drift screening
    return 0.015 * total_weight_kN * 1000.0

def estimate_drift(stories: List[StoryState], lateral_force_N: float) -> float:
    # distribute triangular load and estimate shear-building drifts
    n = len(stories)
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()
    Fi = lateral_force_N * weights

    V = np.zeros(n)
    for i in range(n - 1, -1, -1):
        V[i] = Fi[i] + (V[i + 1] if i < n - 1 else 0.0)

    drifts = np.array([V[i] / max(stories[i].k_story_N_m, 1e-9) for i in range(n)])
    return float(drifts.sum())

def summarize_sustainability(stories: List[StoryState], inp: BuildingInput) -> pd.DataFrame:
    total_steel = sum(s.steel_kg for s in stories)
    total_concrete = sum(s.concrete_m3 for s in stories)

    floor_area_total = floor_area(inp) * inp.n_story
    steel_intensity = total_steel / max(floor_area_total, 1e-9)
    concrete_intensity = total_concrete / max(floor_area_total, 1e-9)

    return pd.DataFrame(
        {
            "Metric": [
                "Total steel (kg)",
                "Total concrete (m³)",
                "Steel intensity (kg/m²)",
                "Concrete intensity (m³/m²)",
            ],
            "Value": [
                total_steel,
                total_concrete,
                steel_intensity,
                concrete_intensity,
            ],
        }
    )

def evaluate(inp: BuildingInput, core_scale: float, col_scale: float, outr_scale: float):
    tref, ttarg, tupper = target_period(inp)
    stories = build_story_states(inp, core_scale, col_scale, outr_scale)
    modal = solve_modal(stories, n_modes=5)
    t1 = modal.periods_s[0]

    total_mass = sum(s.mass_kg for s in stories)
    total_weight_kN = total_mass * G / 1000.0
    lateral_force = equivalent_lateral_force(inp, total_weight_kN)
    top_drift = estimate_drift(stories, lateral_force)
    drift_ratio = top_drift / max(total_height(inp), 1e-9)

    total_steel = sum(s.steel_kg for s in stories)
    total_concrete = sum(s.concrete_m3 for s in stories)

    sust = summarize_sustainability(stories, inp)

    return {
        "stories": stories,
        "modal": modal,
        "tref": tref,
        "ttarg": ttarg,
        "tupper": tupper,
        "t1": t1,
        "period_error": abs(t1 - ttarg) / max(ttarg, 1e-9),
        "weight_kN": total_weight_kN,
        "mass_kg": total_mass,
        "steel_kg": total_steel,
        "concrete_m3": total_concrete,
        "top_drift": top_drift,
        "drift_ratio": drift_ratio,
        "sustainability": sust,
    }

def iterate_to_target(inp: BuildingInput, max_iter: int = 24, tol: float = 0.02) -> AnalysisResult:
    core_scale = 1.00
    col_scale = 1.00
    outr_scale = 1.00

    logs = []

    best = None
    best_score = 1e99

    for it in range(1, max_iter + 1):
        ev = evaluate(inp, core_scale, col_scale, outr_scale)
        t1 = ev["t1"]
        ttarg = ev["ttarg"]
        tupper = ev["tupper"]
        err = ev["period_error"]

        # score: period accuracy first, drift second, material moderation third
        score = (
            1000.0 * err**2
            + 600.0 * max(ev["drift_ratio"] / inp.drift_limit_ratio - 1.0, 0.0) ** 2
            + 0.10 * (ev["steel_kg"] / 1000.0)
            + 0.02 * ev["concrete_m3"]
        )

        if score < best_score:
            best_score = score
            best = (core_scale, col_scale, outr_scale, ev)

        logs.append(
            {
                "Iter": it,
                "Core Scale": core_scale,
                "Column Scale": col_scale,
                "Outrigger Scale": outr_scale,
                "T1 (s)": t1,
                "Target (s)": ttarg,
                "Upper (s)": tupper,
                "Error %": 100.0 * err,
                "Drift Ratio": ev["drift_ratio"],
                "Steel (t)": ev["steel_kg"] / 1000.0,
            }
        )

        ok = (err <= tol) and (ev["drift_ratio"] <= inp.drift_limit_ratio) and (t1 <= tupper)
        if ok:
            best = (core_scale, col_scale, outr_scale, ev)
            break

        # stiffness correction
        # T ~ sqrt(M/K)  => K_req/K_cur ~ (Tcur/Ttar)^2
        k_ratio = (t1 / ttarg) ** 2

        # if t1 > target => too soft => increase sizes
        # if t1 < target => too stiff => decrease sizes
        # map by sensitivity:
        # wall thickness ~ core_scale -> I ~ t^3 approx
        # column dim ~ col_scale -> I ~ b^4 approx
        core_factor = k_ratio ** (0.55 / 3.0)
        col_factor = k_ratio ** (0.30 / 4.0)
        outr_factor = k_ratio ** (0.15 / 1.0)

        # drift correction adds stiffness if drift is high
        if ev["drift_ratio"] > inp.drift_limit_ratio:
            drift_mult = min(1.15, (ev["drift_ratio"] / inp.drift_limit_ratio) ** 0.18)
            core_factor *= drift_mult
            col_factor *= drift_mult
            outr_factor *= drift_mult

        # damping
        damp = 0.70
        core_scale = float(np.clip(core_scale * (1.0 + damp * (core_factor - 1.0)), 0.35, 2.50))
        col_scale = float(np.clip(col_scale * (1.0 + damp * (col_factor - 1.0)), 0.35, 2.50))
        outr_scale = float(np.clip(outr_scale * (1.0 + damp * (outr_factor - 1.0)), 0.35, 3.00))

    assert best is not None
    core_scale, col_scale, outr_scale, ev = best
    conv_df = pd.DataFrame(logs)

    return AnalysisResult(
        building=inp,
        stories=ev["stories"],
        modal=ev["modal"],
        target_period_s=ev["ttarg"],
        reference_period_s=ev["tref"],
        upper_period_s=ev["tupper"],
        estimated_period_s=ev["t1"],
        period_error_ratio=ev["period_error"],
        total_weight_kN=ev["weight_kN"],
        total_mass_kg=ev["mass_kg"],
        total_steel_kg=ev["steel_kg"],
        total_concrete_m3=ev["concrete_m3"],
        top_drift_m=ev["top_drift"],
        drift_ratio=ev["drift_ratio"],
        core_scale=core_scale,
        col_scale=col_scale,
        outr_scale=outr_scale,
        convergence_table=conv_df,
        sustainability_table=ev["sustainability"],
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_story_stiffness(res: AnalysisResult):
    df = pd.DataFrame(
        {
            "Story": [s.story for s in res.stories],
            "Core": [s.k_core_N_m / 1e6 for s in res.stories],
            "Columns": [s.k_col_N_m / 1e6 for s in res.stories],
            "Outrigger": [s.k_outrigger_N_m / 1e6 for s in res.stories],
            "Total": [s.k_story_N_m / 1e6 for s in res.stories],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(df["Core"], df["Story"], label="Core")
    ax.plot(df["Columns"], df["Story"], label="Columns")
    ax.plot(df["Outrigger"], df["Story"], label="Outrigger")
    ax.plot(df["Total"], df["Story"], linewidth=2.5, label="Total")
    ax.set_xlabel("Story stiffness (MN/m)")
    ax.set_ylabel("Story")
    ax.set_title("Story-by-story lateral stiffness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def plot_mode_shapes(res: AnalysisResult):
    H = total_height(res.building)
    y = np.array([s.elevation_m for s in res.stories])
    n_modes = min(5, len(res.modal.mode_shapes))
    fig, axes = plt.subplots(1, n_modes, figsize=(17, 6))
    if n_modes == 1:
        axes = [axes]

    outrigger_levels = {o.story for o in res.building.outriggers if o.active}

    for i in range(n_modes):
        ax = axes[i]
        phi = res.modal.mode_shapes[i]
        phi = phi / max(np.max(np.abs(phi)), 1e-9)
        ax.plot(phi, y, linewidth=2.0)
        ax.scatter(phi, y, s=14)
        ax.axvline(0.0, linestyle="--", linewidth=1.0)

        for s in res.stories:
            if s.story in outrigger_levels:
                ax.axhline(s.elevation_m, color=OUTRIGGER_COLOR, linestyle=":", alpha=0.6)

        ax.set_title(f"Mode {i+1}\nT={res.modal.periods_s[i]:.3f}s")
        ax.set_xlabel("Normalized shape")
        if i == 0:
            ax.set_ylabel("Elevation (m)")
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig.suptitle("Mode shapes")
    fig.tight_layout()
    return fig

def plot_convergence(res: AnalysisResult):
    df = res.convergence_table
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(df["Iter"], df["T1 (s)"], marker="o", label="T1")
    ax.plot(df["Iter"], df["Target (s)"], linestyle="--", label="Target")
    ax.plot(df["Iter"], df["Upper (s)"], linestyle=":", label="Upper")
    ax.set_title("Period convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Period (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["Iter"], df["Error %"], marker="s")
    ax.axhline(2.0, linestyle="--")
    ax.set_title("Period error")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["Iter"], df["Core Scale"], marker="o", label="Core")
    ax.plot(df["Iter"], df["Column Scale"], marker="s", label="Columns")
    ax.plot(df["Iter"], df["Outrigger Scale"], marker="^", label="Outrigger")
    ax.set_title("Scale factors")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Scale")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(df["Iter"], df["Steel (t)"], marker="d")
    ax.set_title("Steel trend")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Steel (t)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def plot_plan(res: AnalysisResult, story_to_show: int):
    inp = res.building
    s = next(ss for ss in res.stories if ss.story == story_to_show)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, inp.plan_x_m, inp.plan_x_m, 0, 0], [0, 0, inp.plan_y_m, inp.plan_y_m, 0], color="black")

    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x_m
        ax.plot([x, x], [0, inp.plan_y_m], color="#d9d9d9", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y_m
        ax.plot([0, inp.plan_x_m], [y, y], color="#d9d9d9", linewidth=0.8)

    # columns
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x_m
            y = j * inp.bay_y_m

            is_corner = (i in [0, inp.n_bays_x]) and (j in [0, inp.n_bays_y])
            is_perim = ((i in [0, inp.n_bays_x]) or (j in [0, inp.n_bays_y])) and not is_corner

            if is_corner:
                w, h = s.corner_col_x_m, s.corner_col_y_m
            elif is_perim:
                w, h = s.perimeter_col_x_m, s.perimeter_col_y_m
            else:
                w, h = s.interior_col_x_m, s.interior_col_y_m

            rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=COL_COLOR, edgecolor="black", linewidth=0.4)
            ax.add_patch(rect)

    # core
    cx0 = 0.5 * (inp.plan_x_m - s.core_outer_x_m)
    cy0 = 0.5 * (inp.plan_y_m - s.core_outer_y_m)
    cxo = 0.5 * (inp.plan_x_m - s.core_opening_x_m)
    cyo = 0.5 * (inp.plan_y_m - s.core_opening_y_m)

    core_rect = plt.Rectangle((cx0, cy0), s.core_outer_x_m, s.core_outer_y_m, fill=False, edgecolor=CORE_COLOR, linewidth=2.2)
    open_rect = plt.Rectangle((cxo, cyo), s.core_opening_x_m, s.core_opening_y_m, fill=False, edgecolor=CORE_COLOR, linestyle="--", linewidth=1.2)
    ax.add_patch(core_rect)
    ax.add_patch(open_rect)

    # highlight outrigger stories
    active_here = [o for o in inp.outriggers if o.active and o.story == story_to_show]
    if active_here:
        armx = 0.5 * (inp.plan_x_m - s.core_outer_x_m)
        army = 0.5 * (inp.plan_y_m - s.core_outer_y_m)
        ox = inp.plan_x_m / 2.0
        oy = inp.plan_y_m / 2.0

        ax.plot([cx0, 0], [oy, oy], color=OUTRIGGER_COLOR, linewidth=5)
        ax.plot([cx0 + s.core_outer_x_m, inp.plan_x_m], [oy, oy], color=OUTRIGGER_COLOR, linewidth=5)
        ax.plot([ox, ox], [cy0, 0], color=OUTRIGGER_COLOR, linewidth=5)
        ax.plot([ox, ox], [cy0 + s.core_outer_y_m, inp.plan_y_m], color=OUTRIGGER_COLOR, linewidth=5)

    ax.set_title(
        f"Plan view - Story {s.story} - {s.zone_name}\n"
        f"Wall t={s.wall_thk_m:.2f}m | K_story={s.k_story_N_m/1e6:.1f} MN/m"
    )
    ax.set_aspect("equal")
    ax.set_xlim(-2, inp.plan_x_m + 2)
    ax.set_ylim(-2, inp.plan_y_m + 2)
    ax.grid(False)
    return fig


# ============================================================
# STREAMLIT INPUTS
# ============================================================

def read_inputs() -> BuildingInput:
    st.sidebar.header("Geometry")
    n_story = st.sidebar.number_input("Above-grade stories", 10, 120, 60)
    n_basement = st.sidebar.number_input("Basement stories", 0, 20, 6)
    story_height_m = st.sidebar.number_input("Story height (m)", 2.8, 5.0, 3.2)
    basement_height_m = st.sidebar.number_input("Basement height (m)", 2.8, 5.0, 3.2)
    plan_x_m = st.sidebar.number_input("Plan X (m)", 20.0, 200.0, 48.0)
    plan_y_m = st.sidebar.number_input("Plan Y (m)", 20.0, 200.0, 42.0)
    n_bays_x = st.sidebar.number_input("Bays X", 2, 20, 6)
    n_bays_y = st.sidebar.number_input("Bays Y", 2, 20, 6)
    bay_x_m = st.sidebar.number_input("Bay X (m)", 4.0, 15.0, 8.0)
    bay_y_m = st.sidebar.number_input("Bay Y (m)", 4.0, 15.0, 7.0)

    st.sidebar.header("Materials / Loads")
    fck_mpa = st.sidebar.number_input("fck (MPa)", 25.0, 100.0, 60.0)
    Ec_mpa = st.sidebar.number_input("Ec (MPa)", 22000.0, 50000.0, 34000.0)
    fy_mpa = st.sidebar.number_input("fy (MPa)", 300.0, 700.0, 420.0)

    DL_kN_m2 = st.sidebar.number_input("DL (kN/m²)", 0.5, 12.0, 3.5)
    LL_kN_m2 = st.sidebar.number_input("LL (kN/m²)", 0.5, 10.0, 2.5)
    finishes_kN_m2 = st.sidebar.number_input("Finishes (kN/m²)", 0.0, 8.0, 1.5)
    facade_line_load_kN_m = st.sidebar.number_input("Facade line load (kN/m)", 0.0, 15.0, 1.2)

    st.sidebar.header("Section bounds")
    min_wall_thk_m = st.sidebar.number_input("Min wall thickness (m)", 0.20, 1.20, 0.30)
    max_wall_thk_m = st.sidebar.number_input("Max wall thickness (m)", 0.30, 2.00, 1.00)
    min_col_dim_m = st.sidebar.number_input("Min column dimension (m)", 0.40, 2.00, 0.70)
    max_col_dim_m = st.sidebar.number_input("Max column dimension (m)", 0.60, 3.00, 1.60)
    min_beam_width_m = st.sidebar.number_input("Min beam width (m)", 0.25, 1.20, 0.40)
    min_beam_depth_m = st.sidebar.number_input("Min beam depth (m)", 0.40, 2.00, 0.80)
    min_slab_thk_m = st.sidebar.number_input("Min slab thickness (m)", 0.12, 0.50, 0.22)
    max_slab_thk_m = st.sidebar.number_input("Max slab thickness (m)", 0.18, 0.60, 0.35)

    st.sidebar.header("Behavior / targeting")
    wall_cracked = st.sidebar.number_input("Wall cracked factor", 0.10, 1.00, 0.35)
    col_cracked = st.sidebar.number_input("Column cracked factor", 0.10, 1.00, 0.70)
    Ct = st.sidebar.number_input("Ct", 0.001, 0.200, 0.0488, format="%.4f")
    x_exp = st.sidebar.number_input("x exponent", 0.10, 1.50, 0.75)
    upper_period_factor = st.sidebar.number_input("Upper period factor", 1.00, 2.00, 1.20)
    target_position_factor = st.sidebar.number_input("Target position factor", 0.10, 0.95, 0.80)
    drift_den = st.sidebar.number_input("Drift denominator", 250.0, 2000.0, 500.0)
    seismic_mass_factor = st.sidebar.number_input("Seismic mass factor", 0.50, 1.50, 1.00)

    st.sidebar.header("Layout factors")
    lower_zone_wall_count = st.sidebar.number_input("Lower zone wall count", 4, 8, 8)
    middle_zone_wall_count = st.sidebar.number_input("Middle zone wall count", 4, 8, 6)
    upper_zone_wall_count = st.sidebar.number_input("Upper zone wall count", 4, 8, 4)
    perimeter_column_factor = st.sidebar.number_input("Perimeter column factor", 1.00, 2.00, 1.10)
    corner_column_factor = st.sidebar.number_input("Corner column factor", 1.00, 2.00, 1.30)

    st.sidebar.header("Outriggers")
    n_or = st.sidebar.number_input("Number of outriggers", 0, 4, 2)
    outriggers = []
    for i in range(int(n_or)):
        st.sidebar.markdown(f"**Outrigger {i+1}**")
        story = st.sidebar.number_input(f"Story #{i+1}", 1, int(n_story), min(int((i+1) * n_story / 3), int(n_story)), key=f"story_{i}")
        system_type = st.sidebar.selectbox(f"System type #{i+1}", ["belt_truss", "pipe_brace"], index=0, key=f"type_{i}")
        truss_depth_m = st.sidebar.number_input(f"Depth #{i+1} (m)", 1.0, 8.0, 3.0, key=f"depth_{i}")
        main_area_m2 = st.sidebar.number_input(f"Main member area #{i+1} (m²)", 0.01, 0.50, 0.08, key=f"amain_{i}")
        diagonal_area_m2 = st.sidebar.number_input(f"Diagonal area #{i+1} (m²)", 0.01, 0.50, 0.04, key=f"adiag_{i}")
        outriggers.append(
            OutriggerInput(
                story=int(story),
                system_type=system_type,
                truss_depth_m=float(truss_depth_m),
                main_area_m2=float(main_area_m2),
                diagonal_area_m2=float(diagonal_area_m2),
                active=True,
            )
        )

    return BuildingInput(
        n_story=int(n_story),
        n_basement=int(n_basement),
        story_height_m=float(story_height_m),
        basement_height_m=float(basement_height_m),
        plan_x_m=float(plan_x_m),
        plan_y_m=float(plan_y_m),
        bay_x_m=float(bay_x_m),
        bay_y_m=float(bay_y_m),
        n_bays_x=int(n_bays_x),
        n_bays_y=int(n_bays_y),
        fck_mpa=float(fck_mpa),
        Ec_mpa=float(Ec_mpa),
        fy_mpa=float(fy_mpa),
        DL_kN_m2=float(DL_kN_m2),
        LL_kN_m2=float(LL_kN_m2),
        finishes_kN_m2=float(finishes_kN_m2),
        facade_line_load_kN_m=float(facade_line_load_kN_m),
        min_wall_thk_m=float(min_wall_thk_m),
        max_wall_thk_m=float(max_wall_thk_m),
        min_col_dim_m=float(min_col_dim_m),
        max_col_dim_m=float(max_col_dim_m),
        min_beam_width_m=float(min_beam_width_m),
        min_beam_depth_m=float(min_beam_depth_m),
        min_slab_thk_m=float(min_slab_thk_m),
        max_slab_thk_m=float(max_slab_thk_m),
        wall_cracked=float(wall_cracked),
        col_cracked=float(col_cracked),
        Ct=float(Ct),
        x_exp=float(x_exp),
        upper_period_factor=float(upper_period_factor),
        target_position_factor=float(target_position_factor),
        drift_limit_ratio=1.0 / float(drift_den),
        seismic_mass_factor=float(seismic_mass_factor),
        lower_zone_wall_count=int(lower_zone_wall_count),
        middle_zone_wall_count=int(middle_zone_wall_count),
        upper_zone_wall_count=int(upper_zone_wall_count),
        perimeter_column_factor=float(perimeter_column_factor),
        corner_column_factor=float(corner_column_factor),
        outriggers=outriggers,
    )


# ============================================================
# REPORTING
# ============================================================

def build_story_dataframe(res: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Story": [s.story for s in res.stories],
            "Elevation (m)": [s.elevation_m for s in res.stories],
            "Zone": [s.zone_name for s in res.stories],
            "Wall t (m)": [s.wall_thk_m for s in res.stories],
            "Interior Col (m)": [f"{s.interior_col_x_m:.2f}x{s.interior_col_y_m:.2f}" for s in res.stories],
            "Perimeter Col (m)": [f"{s.perimeter_col_x_m:.2f}x{s.perimeter_col_y_m:.2f}" for s in res.stories],
            "Corner Col (m)": [f"{s.corner_col_x_m:.2f}x{s.corner_col_y_m:.2f}" for s in res.stories],
            "K_core (MN/m)": [s.k_core_N_m / 1e6 for s in res.stories],
            "K_col (MN/m)": [s.k_col_N_m / 1e6 for s in res.stories],
            "K_outrigger (MN/m)": [s.k_outrigger_N_m / 1e6 for s in res.stories],
            "K_total (MN/m)": [s.k_story_N_m / 1e6 for s in res.stories],
        }
    )

def build_text_report(res: AnalysisResult) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("TOWER PRE-DESIGN REPORT - REVISED MDOF")
    lines.append("=" * 78)
    lines.append(f"Author: {AUTHOR}")
    lines.append(f"Version: {VERSION}")
    lines.append("")
    lines.append("GLOBAL RESULTS")
    lines.append("-" * 78)
    lines.append(f"Reference period            = {res.reference_period_s:.3f} s")
    lines.append(f"Target period               = {res.target_period_s:.3f} s")
    lines.append(f"Upper period                = {res.upper_period_s:.3f} s")
    lines.append(f"Estimated first period      = {res.estimated_period_s:.3f} s")
    lines.append(f"Period error                = {100.0 * res.period_error_ratio:.2f} %")
    lines.append(f"Top drift                   = {res.top_drift_m:.3f} m")
    lines.append(f"Drift ratio                 = {res.drift_ratio:.5f}")
    lines.append(f"Total weight                = {res.total_weight_kN:,.0f} kN")
    lines.append(f"Total mass                  = {res.total_mass_kg:,.0f} kg")
    lines.append(f"Total steel                 = {res.total_steel_kg:,.0f} kg")
    lines.append(f"Total concrete              = {res.total_concrete_m3:,.0f} m³")
    lines.append(f"Core scale                  = {res.core_scale:.3f}")
    lines.append(f"Column scale                = {res.col_scale:.3f}")
    lines.append(f"Outrigger scale             = {res.outr_scale:.3f}")
    lines.append("")
    lines.append("MODAL RESULTS")
    lines.append("-" * 78)
    for i, (T, f, mr, cmr) in enumerate(zip(
        res.modal.periods_s,
        res.modal.freqs_hz,
        res.modal.effective_mass_ratios,
        res.modal.cumulative_mass_ratios,
    ), start=1):
        lines.append(
            f"Mode {i}: T={T:.4f}s | f={f:.4f}Hz | "
            f"eff.mass={100*mr:.2f}% | cum.mass={100*cmr:.2f}%"
        )
    lines.append("")
    lines.append("OUTRIGGER SYSTEMS")
    lines.append("-" * 78)
    if res.building.outriggers:
        for o in res.building.outriggers:
            lines.append(
                f"Story {o.story}: {o.system_type}, depth={o.truss_depth_m:.2f}m, "
                f"A_main={o.main_area_m2:.4f}m², A_diag={o.diagonal_area_m2:.4f}m²"
            )
    else:
        lines.append("No outriggers defined.")
    lines.append("")
    lines.append("DESIGN NOTE")
    lines.append("-" * 78)
    lines.append(
        "This model is intended for preliminary sizing and period targeting. "
        "It is not a replacement for full 3D analysis, cracked-section calibration, "
        "P-Delta analysis, wind tunnel assessment, or code-based final design."
    )
    return "\n".join(lines)


# ============================================================
# APP
# ============================================================

st.title("Tower Preliminary Design - MDOF + Section Stiffness")
st.caption(f"{AUTHOR} | {VERSION}")

st.markdown(
    """
    This revised version improves the original logic by:
    - assembling **story-by-story stiffness**
    - computing **modal periods from the MDOF mass-stiffness system**
    - iterating on **wall, column, and outrigger section stiffness**
    - placing outriggers at their **actual floor level**
    - comparing **belt-truss outrigger** and **pipe-braced outrigger**
    """
)

inp = read_inputs()

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "text_report" not in st.session_state:
    st.session_state.text_report = ""

c1, c2 = st.columns([1.1, 2.2])

with c1:
    run = st.button("Run MDOF pre-design", type="primary")
    if run:
        with st.spinner("Running iterative section-stiffness design..."):
            res = iterate_to_target(inp)
            st.session_state.analysis_result = res
            st.session_state.text_report = build_text_report(res)

with c2:
    res = st.session_state.analysis_result
    if res is None:
        st.info("Run the analysis to see the revised MDOF pre-design results.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("T_ref (s)", f"{res.reference_period_s:.3f}")
        m2.metric("T_target (s)", f"{res.target_period_s:.3f}")
        m3.metric("T1 (s)", f"{res.estimated_period_s:.3f}")
        m4.metric("Drift ratio", f"{res.drift_ratio:.5f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Period error (%)", f"{100*res.period_error_ratio:.2f}")
        m6.metric("Steel (t)", f"{res.total_steel_kg/1000:.1f}")
        m7.metric("Concrete (m³)", f"{res.total_concrete_m3:.0f}")
        m8.metric("Weight (MN)", f"{res.total_weight_kN/1000:.1f}")

        tabs = st.tabs([
            "Story stiffness",
            "Modes",
            "Convergence",
            "Plan",
            "Tables",
            "Sustainability",
            "Report",
        ])

        with tabs[0]:
            st.pyplot(plot_story_stiffness(res), use_container_width=True)

        with tabs[1]:
            st.pyplot(plot_mode_shapes(res), use_container_width=True)
            df_modal = pd.DataFrame(
                {
                    "Mode": list(range(1, len(res.modal.periods_s) + 1)),
                    "Period (s)": res.modal.periods_s,
                    "Frequency (Hz)": res.modal.freqs_hz,
                    "Eff. Mass (%)": [100*x for x in res.modal.effective_mass_ratios],
                    "Cum. Mass (%)": [100*x for x in res.modal.cumulative_mass_ratios],
                }
            )
            st.dataframe(df_modal, use_container_width=True, hide_index=True)

        with tabs[2]:
            st.pyplot(plot_convergence(res), use_container_width=True)
            st.dataframe(res.convergence_table, use_container_width=True, hide_index=True)

        with tabs[3]:
            story_to_show = st.slider("Story to show", 1, inp.n_story, min(inp.n_story, max(1, inp.n_story // 2)))
            st.pyplot(plot_plan(res, story_to_show), use_container_width=True)

        with tabs[4]:
            st.dataframe(build_story_dataframe(res), use_container_width=True, hide_index=True)

        with tabs[5]:
            st.dataframe(res.sustainability_table, use_container_width=True, hide_index=True)
            st.markdown(
                """
                **Interpretation**
                - For a tall tower, **belt-truss outrigger** is generally the better starting option for global stiffness efficiency.
                - **Pipe-brace outrigger** can still be useful, but in pre-design it usually gives lower global coupling efficiency per unit material.
                - The sustainable solution is not simply the lightest system; it is the system that meets target period and drift with the lowest reliable material demand.
                """
            )

        with tabs[6]:
            st.text_area("Text report", st.session_state.text_report, height=460)
            st.download_button(
                "Download report",
                data=st.session_state.text_report.encode("utf-8"),
                file_name="tower_pre_design_report.txt",
                mime="text/plain",
            )
