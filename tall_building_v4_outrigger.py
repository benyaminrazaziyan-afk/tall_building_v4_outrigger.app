"""
tower_mdof_professional_v6.py
Professional preliminary tower pre-design tool using MDOF modal analysis.

Scope:
- Preliminary concept/pre-design only.
- Story-by-story stiffness assembly in X and Y directions.
- Empirical target period and upper-period screening.
- Section-stiffness iteration for core walls, columns, and outriggers.
- Outrigger comparison: belt truss vs pipe bracing.
- Streamlit interface with plots, tables, and downloadable report.

Author: Benyamin
Version: v6.0-professional-mdof
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi, sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

G = 9.81
GAMMA_RC = 25.0  # kN/m3
RHO_STEEL = 7850.0  # kg/m3
APP_VERSION = "v6.0-professional-mdof"


class Direction(str, Enum):
    X = "X"
    Y = "Y"


class OutriggerType(str, Enum):
    BELT_TRUSS = "Belt Truss"
    PIPE_BRACE = "Pipe Bracing"


@dataclass
class Material:
    fck_mpa: float = 60.0
    Ec_mpa: float = 34000.0
    fy_mpa: float = 420.0

    @property
    def Ec_pa(self) -> float:
        return self.Ec_mpa * 1e6


@dataclass
class Geometry:
    n_story: int = 60
    n_basement: int = 6
    story_height_m: float = 3.2
    basement_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6
    core_ratio_x: float = 0.24
    core_ratio_y: float = 0.22
    opening_ratio_x: float = 0.16
    opening_ratio_y: float = 0.14

    @property
    def height_m(self) -> float:
        return self.n_story * self.story_height_m

    @property
    def floor_area_m2(self) -> float:
        return self.plan_x_m * self.plan_y_m

    @property
    def bay_x_m(self) -> float:
        return self.plan_x_m / max(self.n_bays_x, 1)

    @property
    def bay_y_m(self) -> float:
        return self.plan_y_m / max(self.n_bays_y, 1)


@dataclass
class Loads:
    dead_kN_m2: float = 3.5
    live_kN_m2: float = 2.5
    finish_kN_m2: float = 1.5
    facade_kN_m: float = 1.2
    live_mass_factor: float = 0.25
    seismic_mass_factor: float = 1.0
    base_shear_coeff: float = 0.015


@dataclass
class Limits:
    min_wall_t_m: float = 0.30
    max_wall_t_m: float = 1.00
    min_col_m: float = 0.70
    max_col_m: float = 1.60
    min_slab_t_m: float = 0.22
    max_slab_t_m: float = 0.35
    min_beam_b_m: float = 0.40
    max_beam_b_m: float = 0.90
    min_beam_h_m: float = 0.80
    max_beam_h_m: float = 1.80
    drift_limit_ratio: float = 1.0 / 500.0


@dataclass
class PeriodRule:
    Ct: float = 0.0488
    x_exp: float = 0.75
    upper_factor: float = 1.20
    target_position: float = 0.80

    def periods(self, H: float) -> Tuple[float, float, float]:
        t_ref = self.Ct * H ** self.x_exp
        t_upper = self.upper_factor * t_ref
        t_target = t_ref + self.target_position * (t_upper - t_ref)
        return t_ref, t_target, t_upper


@dataclass
class CrackedFactors:
    wall: float = 0.35
    column: float = 0.70
    outrigger_connection: float = 0.80


@dataclass
class Outrigger:
    story: int
    system: OutriggerType = OutriggerType.BELT_TRUSS
    depth_m: float = 3.0
    chord_area_m2: float = 0.08
    diagonal_area_m2: float = 0.04
    active: bool = True


@dataclass
class BuildingModel:
    material: Material = field(default_factory=Material)
    geometry: Geometry = field(default_factory=Geometry)
    loads: Loads = field(default_factory=Loads)
    limits: Limits = field(default_factory=Limits)
    period: PeriodRule = field(default_factory=PeriodRule)
    cracked: CrackedFactors = field(default_factory=CrackedFactors)
    lower_wall_count: int = 8
    middle_wall_count: int = 6
    upper_wall_count: int = 4
    perimeter_col_factor: float = 1.10
    corner_col_factor: float = 1.30
    wall_rebar_ratio: float = 0.004
    column_rebar_ratio: float = 0.012
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.004
    outriggers: List[Outrigger] = field(default_factory=list)


@dataclass
class StoryResult:
    story: int
    elevation_m: float
    zone: str
    wall_t_m: float
    wall_count: int
    core_x_m: float
    core_y_m: float
    open_x_m: float
    open_y_m: float
    interior_col_m: float
    perimeter_col_m: float
    corner_col_m: float
    slab_t_m: float
    beam_b_m: float
    beam_h_m: float
    mass_kg: float
    weight_kN: float
    kx_core: float
    ky_core: float
    kx_col: float
    ky_col: float
    kx_outrigger: float
    ky_outrigger: float
    concrete_m3: float
    steel_kg: float

    @property
    def kx_total(self) -> float:
        return self.kx_core + self.kx_col + self.kx_outrigger

    @property
    def ky_total(self) -> float:
        return self.ky_core + self.ky_col + self.ky_outrigger


@dataclass
class ModalResult:
    direction: Direction
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[np.ndarray]
    effective_mass_ratios: List[float]
    cumulative_mass_ratios: List[float]


@dataclass
class DesignResult:
    model: BuildingModel
    stories: List[StoryResult]
    modal_x: ModalResult
    modal_y: ModalResult
    t_ref: float
    t_target: float
    t_upper: float
    governing_direction: Direction
    governing_period: float
    period_error: float
    total_weight_kN: float
    total_mass_kg: float
    total_concrete_m3: float
    total_steel_kg: float
    drift_x_m: float
    drift_y_m: float
    drift_x_ratio: float
    drift_y_ratio: float
    core_scale: float
    column_scale: float
    outrigger_scale: float
    iteration_table: pd.DataFrame
    messages: List[str]


# ----------------------------- Validation -----------------------------

def validate_model(model: BuildingModel) -> None:
    g = model.geometry
    if g.n_story < 1:
        raise ValueError("Number of stories must be positive.")
    if min(g.plan_x_m, g.plan_y_m, g.story_height_m) <= 0:
        raise ValueError("Geometry dimensions must be positive.")
    if g.n_bays_x < 1 or g.n_bays_y < 1:
        raise ValueError("Number of bays must be positive.")
    if model.material.Ec_mpa <= 0:
        raise ValueError("Ec must be positive.")
    if not (0.05 <= model.cracked.wall <= 1.0):
        raise ValueError("Wall cracked factor must be between 0.05 and 1.0.")
    if not (0.05 <= model.cracked.column <= 1.0):
        raise ValueError("Column cracked factor must be between 0.05 and 1.0.")
    for o in model.outriggers:
        if not (1 <= o.story <= g.n_story):
            raise ValueError(f"Outrigger story {o.story} is outside the tower height.")


# ----------------------------- Geometry helpers -----------------------------

def zone_name(model: BuildingModel, story: int) -> str:
    n = model.geometry.n_story
    z1 = round(0.30 * n)
    z2 = round(0.70 * n)
    if story <= z1:
        return "Lower"
    if story <= z2:
        return "Middle"
    return "Upper"


def wall_count(model: BuildingModel, story: int) -> int:
    z = zone_name(model, story)
    if z == "Lower":
        return model.lower_wall_count
    if z == "Middle":
        return model.middle_wall_count
    return model.upper_wall_count


def core_dimensions(model: BuildingModel) -> Tuple[float, float, float, float]:
    g = model.geometry
    cx = max(8.0, g.core_ratio_x * g.plan_x_m)
    cy = max(8.0, g.core_ratio_y * g.plan_y_m)
    ox = min(g.opening_ratio_x * g.plan_x_m, cx - 2.0)
    oy = min(g.opening_ratio_y * g.plan_y_m, cy - 2.0)
    return cx, cy, ox, oy


def wall_thickness(model: BuildingModel, story: int, scale: float) -> float:
    H = model.geometry.height_m
    factor = {"Lower": 1.00, "Middle": 0.82, "Upper": 0.65}[zone_name(model, story)]
    t = (H / 180.0) * factor * scale
    return float(np.clip(t, model.limits.min_wall_t_m, model.limits.max_wall_t_m))


def column_dims(model: BuildingModel, story: int, scale: float) -> Tuple[float, float, float]:
    factor = {"Lower": 1.00, "Middle": 0.86, "Upper": 0.72}[zone_name(model, story)]
    base = float(np.clip(0.85 * factor * scale, model.limits.min_col_m, model.limits.max_col_m))
    interior = base
    perimeter = float(np.clip(base * model.perimeter_col_factor, model.limits.min_col_m, model.limits.max_col_m))
    corner = float(np.clip(base * model.corner_col_factor, model.limits.min_col_m, model.limits.max_col_m))
    return interior, perimeter, corner


def slab_thickness(model: BuildingModel, col_scale: float) -> float:
    span = max(model.geometry.bay_x_m, model.geometry.bay_y_m)
    t = span / 30.0 * (0.95 + 0.10 * col_scale)
    return float(np.clip(t, model.limits.min_slab_t_m, model.limits.max_slab_t_m))


def beam_size(model: BuildingModel, col_scale: float) -> Tuple[float, float]:
    span = max(model.geometry.bay_x_m, model.geometry.bay_y_m)
    h = span / 12.0 * (0.95 + 0.10 * col_scale)
    h = float(np.clip(h, model.limits.min_beam_h_m, model.limits.max_beam_h_m))
    b = float(np.clip(0.45 * h, model.limits.min_beam_b_m, model.limits.max_beam_b_m))
    return b, h


def column_counts(model: BuildingModel) -> Tuple[int, int, int]:
    g = model.geometry
    total = (g.n_bays_x + 1) * (g.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (g.n_bays_x - 1) + 2 * (g.n_bays_y - 1))
    interior = max(0, total - corner - perimeter)
    return interior, perimeter, corner


# ----------------------------- Section stiffness -----------------------------

def rect_I(b: float, h: float) -> float:
    return b * h**3 / 12.0


def core_Ix_Iy(model: BuildingModel, story: int, t: float) -> Tuple[float, float]:
    cx, cy, _, _ = core_dimensions(model)
    wc = wall_count(model, story)

    Ix = 2 * (cx * t**3 / 12.0 + cx * t * (cy / 2) ** 2) + 2 * rect_I(t, cy)
    Iy = 2 * (cy * t**3 / 12.0 + cy * t * (cx / 2) ** 2) + 2 * rect_I(t, cx)

    if wc >= 6:
        Ix += 2 * rect_I(t, 0.45 * cy)
        Iy += 2 * (0.45 * cx * t**3 / 12.0 + 0.45 * cx * t * (0.22 * cx) ** 2)
    if wc >= 8:
        Ix += 2 * (0.45 * cy * t**3 / 12.0 + 0.45 * cy * t * (0.22 * cy) ** 2)
        Iy += 2 * rect_I(t, 0.45 * cx)

    return model.cracked.wall * Ix, model.cracked.wall * Iy


def column_I(model: BuildingModel, interior: float, perimeter: float, corner: float) -> Tuple[float, float]:
    ni, np_, nc = column_counts(model)
    I_total = ni * rect_I(interior, interior) + np_ * rect_I(perimeter, perimeter) + nc * rect_I(corner, corner)
    I_eff = model.cracked.column * I_total
    return I_eff, I_eff


def story_spring(EI: float, h: float) -> float:
    return 12.0 * EI / max(h**3, 1e-9)


def outrigger_efficiency(system: OutriggerType) -> float:
    return 1.00 if system == OutriggerType.BELT_TRUSS else 0.70


def outrigger_material_factor(system: OutriggerType) -> float:
    return 1.00 if system == OutriggerType.BELT_TRUSS else 1.15


def outrigger_k_and_steel(model: BuildingModel, story: int, scale: float) -> Tuple[float, float, float]:
    g = model.geometry
    E = model.material.Ec_pa
    cx, cy, _, _ = core_dimensions(model)
    arm_x = max(0.5 * (g.plan_x_m - cx), 1.0)
    arm_y = max(0.5 * (g.plan_y_m - cy), 1.0)

    kx = ky = steel = 0.0
    for o in model.outriggers:
        if not o.active or o.story != story:
            continue
        eta = outrigger_efficiency(o.system) * model.cracked.outrigger_connection
        mat_fac = outrigger_material_factor(o.system)
        A_ch = o.chord_area_m2 * scale
        A_d = o.diagonal_area_m2 * scale
        Lx = sqrt(arm_x**2 + o.depth_m**2)
        Ly = sqrt(arm_y**2 + o.depth_m**2)
        kx += eta * (2 * E * A_ch / arm_x + 2 * E * A_d / Lx)
        ky += eta * (2 * E * A_ch / arm_y + 2 * E * A_d / Ly)
        steel_vol = 2 * A_ch * (arm_x + arm_y) + 2 * A_d * (Lx + Ly)
        steel += steel_vol * RHO_STEEL * mat_fac
    return kx, ky, steel


# ----------------------------- Quantities -----------------------------

def floor_quantities(model: BuildingModel, t: float, interior: float, perimeter: float, corner: float,
                     slab_t: float, beam_b: float, beam_h: float, out_steel: float) -> Tuple[float, float, float, float]:
    g = model.geometry
    loads = model.loads
    A = g.floor_area_m2

    slab_vol = A * slab_t
    beam_lines = g.n_bays_y * (g.n_bays_x + 1) + g.n_bays_x * (g.n_bays_y + 1)
    avg_span = 0.5 * (g.bay_x_m + g.bay_y_m)
    beam_vol = beam_lines * avg_span * beam_b * beam_h

    ni, np_, nc = column_counts(model)
    col_vol = (ni * interior**2 + np_ * perimeter**2 + nc * corner**2) * g.story_height_m

    cx, cy, _, _ = core_dimensions(model)
    wall_vol = 2 * (cx + cy) * t * g.story_height_m
    concrete = slab_vol + beam_vol + col_vol + wall_vol

    steel = (
        model.slab_rebar_ratio * slab_vol * RHO_STEEL
        + model.beam_rebar_ratio * beam_vol * RHO_STEEL
        + model.column_rebar_ratio * col_vol * RHO_STEEL
        + model.wall_rebar_ratio * wall_vol * RHO_STEEL
        + out_steel
    )

    area_load = (loads.dead_kN_m2 + loads.finish_kN_m2 + loads.live_mass_factor * loads.live_kN_m2) * A
    facade = loads.facade_kN_m * 2 * (g.plan_x_m + g.plan_y_m)
    weight = (area_load + concrete * GAMMA_RC + steel * G / 1000.0 + facade) * loads.seismic_mass_factor
    mass = weight * 1000.0 / G
    return mass, weight, concrete, steel


# ----------------------------- Analysis engine -----------------------------

def build_stories(model: BuildingModel, core_scale: float, column_scale: float, outr_scale: float) -> List[StoryResult]:
    g = model.geometry
    E = model.material.Ec_pa
    cx, cy, ox, oy = core_dimensions(model)
    rows = []
    for story in range(1, g.n_story + 1):
        z = zone_name(model, story)
        wc = wall_count(model, story)
        t = wall_thickness(model, story, core_scale)
        interior, perimeter, corner = column_dims(model, story, column_scale)
        slab_t = slab_thickness(model, column_scale)
        beam_b, beam_h = beam_size(model, column_scale)

        Ix_core, Iy_core = core_Ix_Iy(model, story, t)
        Ix_col, Iy_col = column_I(model, interior, perimeter, corner)
        kx_out, ky_out, out_steel = outrigger_k_and_steel(model, story, outr_scale)

        kx_core = story_spring(E * Iy_core, g.story_height_m)
        ky_core = story_spring(E * Ix_core, g.story_height_m)
        kx_col = story_spring(E * Iy_col, g.story_height_m)
        ky_col = story_spring(E * Ix_col, g.story_height_m)

        if kx_out > 0:
            kx_out += 0.15 * min(kx_core, kx_col)
        if ky_out > 0:
            ky_out += 0.15 * min(ky_core, ky_col)

        mass, weight, concrete, steel = floor_quantities(model, t, interior, perimeter, corner, slab_t, beam_b, beam_h, out_steel)

        rows.append(StoryResult(story, story * g.story_height_m, z, t, wc, cx, cy, ox, oy,
                                interior, perimeter, corner, slab_t, beam_b, beam_h,
                                mass, weight, kx_core, ky_core, kx_col, ky_col,
                                kx_out, ky_out, concrete, steel))
    return rows


def assemble_mk(stories: List[StoryResult], direction: Direction) -> Tuple[np.ndarray, np.ndarray]:
    n = len(stories)
    M = np.diag([s.mass_kg for s in stories])
    k = np.array([s.kx_total if direction == Direction.X else s.ky_total for s in stories])
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] += k[i]
        else:
            K[i, i] += k[i]
            K[i, i - 1] -= k[i]
            K[i - 1, i] -= k[i]
            K[i - 1, i - 1] += k[i]
    return M, K


def solve_modal(stories: List[StoryResult], direction: Direction, n_modes: int = 5) -> ModalResult:
    M, K = assemble_mk(stories, direction)
    A = np.linalg.solve(M, K)
    vals, vecs = np.linalg.eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    keep = vals > 1e-8
    vals = vals[keep]
    vecs = vecs[:, keep]
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    m = min(n_modes, len(vals))
    omega = np.sqrt(vals[:m])
    periods = [float(2 * pi / w) for w in omega]
    freqs = [float(w / (2 * pi)) for w in omega]

    ones = np.ones((len(stories), 1))
    total_mass = np.sum(np.diag(M))
    shapes, ratios, cum_ratios = [], [], []
    cum = 0.0
    for i in range(m):
        phi = vecs[:, i].reshape(-1, 1)
        denom = float(phi.T @ M @ phi)
        gamma = float((phi.T @ M @ ones) / denom)
        meff = gamma**2 * denom
        ratio = float(meff / total_mass)
        cum += ratio
        ph = phi.flatten()
        if abs(ph[-1]) > 1e-12:
            ph = ph / ph[-1]
        if ph[-1] < 0:
            ph = -ph
        shapes.append(ph)
        ratios.append(ratio)
        cum_ratios.append(cum)
    return ModalResult(direction, periods, freqs, shapes, ratios, cum_ratios)


def top_drift(model: BuildingModel, stories: List[StoryResult], direction: Direction) -> float:
    total_weight = sum(s.weight_kN for s in stories)
    Vb = model.loads.base_shear_coeff * total_weight * 1000.0
    heights = np.array([s.elevation_m for s in stories])
    Fi = Vb * heights / max(np.sum(heights), 1e-9)
    V = np.zeros(len(stories))
    for i in range(len(stories) - 1, -1, -1):
        V[i] = Fi[i] + (V[i + 1] if i < len(stories) - 1 else 0.0)
    k = np.array([s.kx_total if direction == Direction.X else s.ky_total for s in stories])
    return float(np.sum(V / np.maximum(k, 1e-9)))


def evaluate(model: BuildingModel, core_scale: float, col_scale: float, out_scale: float):
    stories = build_stories(model, core_scale, col_scale, out_scale)
    mx = solve_modal(stories, Direction.X)
    my = solve_modal(stories, Direction.Y)
    t_ref, t_target, t_upper = model.period.periods(model.geometry.height_m)
    tx, ty = mx.periods_s[0], my.periods_s[0]
    gov_dir = Direction.X if tx >= ty else Direction.Y
    gov_t = max(tx, ty)
    total_w = sum(s.weight_kN for s in stories)
    total_m = sum(s.mass_kg for s in stories)
    total_c = sum(s.concrete_m3 for s in stories)
    total_s = sum(s.steel_kg for s in stories)
    dx = top_drift(model, stories, Direction.X)
    dy = top_drift(model, stories, Direction.Y)
    return dict(stories=stories, modal_x=mx, modal_y=my, t_ref=t_ref, t_target=t_target, t_upper=t_upper,
                gov_dir=gov_dir, gov_t=gov_t, error=abs(gov_t - t_target) / max(t_target, 1e-9),
                total_w=total_w, total_m=total_m, total_c=total_c, total_s=total_s,
                dx=dx, dy=dy, dx_ratio=dx / model.geometry.height_m, dy_ratio=dy / model.geometry.height_m)


def run_design(model: BuildingModel, max_iter: int = 30, tolerance: float = 0.025) -> DesignResult:
    validate_model(model)
    core_scale = col_scale = out_scale = 1.0
    logs = []
    best = None
    best_score = 1e99
    for it in range(1, max_iter + 1):
        ev = evaluate(model, core_scale, col_scale, out_scale)
        drift_max = max(ev['dx_ratio'], ev['dy_ratio'])
        over_drift = max(drift_max / model.limits.drift_limit_ratio - 1.0, 0.0)
        over_period = max(ev['gov_t'] / ev['t_upper'] - 1.0, 0.0)
        score = 1200 * ev['error']**2 + 1500 * over_drift**2 + 2000 * over_period**2 + 0.000002 * ev['total_c'] + 0.0000005 * ev['total_s']
        logs.append(dict(Iteration=it, Core_scale=core_scale, Column_scale=col_scale, Outrigger_scale=out_scale,
                         T_governing_s=ev['gov_t'], T_target_s=ev['t_target'], T_upper_s=ev['t_upper'],
                         Period_error_percent=100 * ev['error'], Drift_X=ev['dx_ratio'], Drift_Y=ev['dy_ratio'],
                         Weight_MN=ev['total_w'] / 1000, Concrete_m3=ev['total_c'], Steel_t=ev['total_s'] / 1000,
                         Governing_direction=ev['gov_dir'].value))
        if score < best_score:
            best_score = score
            best = (core_scale, col_scale, out_scale, ev)
        if ev['error'] <= tolerance and drift_max <= model.limits.drift_limit_ratio and ev['gov_t'] <= ev['t_upper']:
            best = (core_scale, col_scale, out_scale, ev)
            break
        ratio = (ev['gov_t'] / max(ev['t_target'], 1e-9))**2
        core_factor = ratio ** (0.55 / 3.0)
        col_factor = ratio ** (0.30 / 4.0)
        out_factor = ratio ** 0.15
        if drift_max > model.limits.drift_limit_ratio:
            dfac = min(1.20, (drift_max / model.limits.drift_limit_ratio)**0.20)
            core_factor *= dfac
            col_factor *= dfac
            out_factor *= dfac
        damp = 0.65
        core_scale = float(np.clip(core_scale * (1 + damp * (core_factor - 1)), 0.40, 2.50))
        col_scale = float(np.clip(col_scale * (1 + damp * (col_factor - 1)), 0.40, 2.50))
        out_scale = float(np.clip(out_scale * (1 + damp * (out_factor - 1)), 0.40, 3.00))
    assert best is not None
    core_scale, col_scale, out_scale, ev = best
    messages = []
    messages.append('Upper period check: OK' if ev['gov_t'] <= ev['t_upper'] else 'Upper period check: NOT OK')
    messages.append('Drift check: OK' if max(ev['dx_ratio'], ev['dy_ratio']) <= model.limits.drift_limit_ratio else 'Drift check: NOT OK')
    messages.append('This is a preliminary MDOF shear-building model. Final design requires calibrated 3D analysis, torsion, P-Delta, wind/seismic cases, and member design.')
    return DesignResult(model, ev['stories'], ev['modal_x'], ev['modal_y'], ev['t_ref'], ev['t_target'], ev['t_upper'], ev['gov_dir'], ev['gov_t'], ev['error'], ev['total_w'], ev['total_m'], ev['total_c'], ev['total_s'], ev['dx'], ev['dy'], ev['dx_ratio'], ev['dy_ratio'], core_scale, col_scale, out_scale, pd.DataFrame(logs), messages)


# ----------------------------- Tables and reports -----------------------------

def story_table(res: DesignResult) -> pd.DataFrame:
    return pd.DataFrame([dict(Story=s.story, Elevation_m=s.elevation_m, Zone=s.zone, Wall_t_m=s.wall_t_m, Wall_count=s.wall_count,
                              Interior_col_m=s.interior_col_m, Perimeter_col_m=s.perimeter_col_m, Corner_col_m=s.corner_col_m,
                              Kx_core_MNm=s.kx_core / 1e6, Kx_col_MNm=s.kx_col / 1e6, Kx_outrigger_MNm=s.kx_outrigger / 1e6, Kx_total_MNm=s.kx_total / 1e6,
                              Ky_core_MNm=s.ky_core / 1e6, Ky_col_MNm=s.ky_col / 1e6, Ky_outrigger_MNm=s.ky_outrigger / 1e6, Ky_total_MNm=s.ky_total / 1e6,
                              Mass_t=s.mass_kg / 1000) for s in res.stories])


def modal_table(modal: ModalResult) -> pd.DataFrame:
    return pd.DataFrame(dict(Mode=list(range(1, len(modal.periods_s) + 1)), Direction=modal.direction.value, Period_s=modal.periods_s,
                             Frequency_Hz=modal.frequencies_hz, Effective_mass_percent=[100*x for x in modal.effective_mass_ratios],
                             Cumulative_mass_percent=[100*x for x in modal.cumulative_mass_ratios]))


def sustainability_table(res: DesignResult) -> pd.DataFrame:
    area = res.model.geometry.floor_area_m2 * res.model.geometry.n_story
    return pd.DataFrame(dict(Indicator=['Total concrete', 'Total steel', 'Concrete intensity', 'Steel intensity', 'Seismic weight'],
                             Value=[res.total_concrete_m3, res.total_steel_kg, res.total_concrete_m3 / area, res.total_steel_kg / area, res.total_weight_kN],
                             Unit=['m3', 'kg', 'm3/m2', 'kg/m2', 'kN']))


def build_report(res: DesignResult) -> str:
    lines = []
    lines.append('=' * 88)
    lines.append('PROFESSIONAL TOWER PRE-DESIGN REPORT')
    lines.append('=' * 88)
    lines.append(f'Version: {APP_VERSION}')
    lines.append('')
    lines.append('1. PERIOD TARGETING')
    lines.append('-' * 88)
    lines.append(f'Reference empirical period : {res.t_ref:.3f} s')
    lines.append(f'Target period              : {res.t_target:.3f} s')
    lines.append(f'Upper period limit         : {res.t_upper:.3f} s')
    lines.append(f'Governing direction        : {res.governing_direction.value}')
    lines.append(f'Governing MDOF period      : {res.governing_period:.3f} s')
    lines.append(f'Period error               : {100*res.period_error:.2f} %')
    lines.append('')
    lines.append('2. DRIFT SCREENING')
    lines.append('-' * 88)
    lines.append(f'Top drift X                : {res.drift_x_m:.4f} m')
    lines.append(f'Top drift Y                : {res.drift_y_m:.4f} m')
    lines.append(f'Drift ratio X              : {res.drift_x_ratio:.6f}')
    lines.append(f'Drift ratio Y              : {res.drift_y_ratio:.6f}')
    lines.append(f'Allowable drift ratio      : {res.model.limits.drift_limit_ratio:.6f}')
    lines.append('')
    lines.append('3. MATERIAL QUANTITIES')
    lines.append('-' * 88)
    lines.append(f'Total weight               : {res.total_weight_kN:,.0f} kN')
    lines.append(f'Total concrete             : {res.total_concrete_m3:,.0f} m3')
    lines.append(f'Total steel                : {res.total_steel_kg:,.0f} kg')
    lines.append('')
    lines.append('4. FINAL SCALE FACTORS')
    lines.append('-' * 88)
    lines.append(f'Core scale                 : {res.core_scale:.3f}')
    lines.append(f'Column scale               : {res.column_scale:.3f}')
    lines.append(f'Outrigger scale            : {res.outrigger_scale:.3f}')
    lines.append('')
    lines.append('5. OUTRIGGERS')
    lines.append('-' * 88)
    if res.model.outriggers:
        for o in res.model.outriggers:
            lines.append(f'Story {o.story:3d} | {o.system.value:12s} | depth={o.depth_m:.2f} m | A_chord={o.chord_area_m2:.4f} m2 | A_diag={o.diagonal_area_m2:.4f} m2')
    else:
        lines.append('No outrigger used.')
    lines.append('')
    lines.append('6. STATUS')
    lines.append('-' * 88)
    for m in res.messages:
        lines.append(f'- {m}')
    lines.append('')
    lines.append('ENGINEERING CONCLUSION')
    lines.append('-' * 88)
    lines.append('For preliminary tower design, belt-truss outriggers are generally more efficient than pipe bracing for global period and drift control because they couple the core and perimeter columns more directly. Pipe bracing may still be useful for constructability or architectural constraints, but its lower global coupling efficiency should be verified through MDOF period, drift, and story-stiffness profiles.')
    return '\n'.join(lines)


# ----------------------------- Plots -----------------------------

def plot_stiffness(res: DesignResult, direction: Direction):
    df = story_table(res)
    fig, ax = plt.subplots(figsize=(8, 8))
    if direction == Direction.X:
        ax.plot(df.Kx_core_MNm, df.Story, label='Core')
        ax.plot(df.Kx_col_MNm, df.Story, label='Columns')
        ax.plot(df.Kx_outrigger_MNm, df.Story, label='Outrigger')
        ax.plot(df.Kx_total_MNm, df.Story, label='Total', linewidth=2.5)
    else:
        ax.plot(df.Ky_core_MNm, df.Story, label='Core')
        ax.plot(df.Ky_col_MNm, df.Story, label='Columns')
        ax.plot(df.Ky_outrigger_MNm, df.Story, label='Outrigger')
        ax.plot(df.Ky_total_MNm, df.Story, label='Total', linewidth=2.5)
    ax.set_xlabel('Story stiffness (MN/m)')
    ax.set_ylabel('Story')
    ax.set_title(f'Story stiffness profile - {direction.value}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_modes(res: DesignResult, modal: ModalResult):
    y = np.array([s.elevation_m for s in res.stories])
    n = min(5, len(modal.mode_shapes))
    fig, axes = plt.subplots(1, n, figsize=(16, 6))
    if n == 1:
        axes = [axes]
    out_stories = [o.story for o in res.model.outriggers if o.active]
    for i in range(n):
        ax = axes[i]
        phi = modal.mode_shapes[i]
        phi = phi / max(np.max(np.abs(phi)), 1e-9)
        ax.plot(phi, y, linewidth=2)
        ax.scatter(phi, y, s=12)
        ax.axvline(0, linestyle='--', linewidth=0.8)
        for st_ in out_stories:
            ax.axhline(st_ * res.model.geometry.story_height_m, linestyle=':', linewidth=1, alpha=0.6)
        ax.set_title(f'Mode {i+1}\nT={modal.periods_s[i]:.3f}s')
        ax.set_xlabel('Normalized shape')
        if i == 0:
            ax.set_ylabel('Elevation (m)')
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)
    fig.suptitle(f'Mode shapes - {modal.direction.value}')
    fig.tight_layout()
    return fig


def plot_iteration(res: DesignResult):
    df = res.iteration_table
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(df.Iteration, df.T_governing_s, marker='o', label='T governing')
    axes[0, 0].plot(df.Iteration, df.T_target_s, linestyle='--', label='Target')
    axes[0, 0].plot(df.Iteration, df.T_upper_s, linestyle=':', label='Upper')
    axes[0, 0].set_title('Period convergence')
    axes[0, 0].set_ylabel('Period (s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].plot(df.Iteration, df.Period_error_percent, marker='s')
    axes[0, 1].axhline(2.5, linestyle='--')
    axes[0, 1].set_title('Period error')
    axes[0, 1].set_ylabel('Error (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(df.Iteration, df.Core_scale, marker='o', label='Core')
    axes[1, 0].plot(df.Iteration, df.Column_scale, marker='s', label='Column')
    axes[1, 0].plot(df.Iteration, df.Outrigger_scale, marker='^', label='Outrigger')
    axes[1, 0].set_title('Scale factors')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 1].plot(df.Iteration, df.Steel_t, marker='d')
    axes[1, 1].set_title('Steel trend')
    axes[1, 1].set_ylabel('Steel (t)')
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_plan(res: DesignResult, story: int):
    g = res.model.geometry
    s = next(x for x in res.stories if x.story == story)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, g.plan_x_m, g.plan_x_m, 0, 0], [0, 0, g.plan_y_m, g.plan_y_m, 0], color='black')
    for i in range(g.n_bays_x + 1):
        x = i * g.bay_x_m
        ax.plot([x, x], [0, g.plan_y_m], color='#d0d0d0', linewidth=0.7)
    for j in range(g.n_bays_y + 1):
        y = j * g.bay_y_m
        ax.plot([0, g.plan_x_m], [y, y], color='#d0d0d0', linewidth=0.7)
    for i in range(g.n_bays_x + 1):
        for j in range(g.n_bays_y + 1):
            x = i * g.bay_x_m
            y = j * g.bay_y_m
            is_corner = (i in [0, g.n_bays_x]) and (j in [0, g.n_bays_y])
            is_perim = (i in [0, g.n_bays_x] or j in [0, g.n_bays_y]) and not is_corner
            dim = s.corner_col_m if is_corner else s.perimeter_col_m if is_perim else s.interior_col_m
            ax.add_patch(plt.Rectangle((x - dim/2, y - dim/2), dim, dim, facecolor='#8B0000', edgecolor='black', alpha=0.85))
    cx0 = 0.5 * (g.plan_x_m - s.core_x_m)
    cy0 = 0.5 * (g.plan_y_m - s.core_y_m)
    ox0 = 0.5 * (g.plan_x_m - s.open_x_m)
    oy0 = 0.5 * (g.plan_y_m - s.open_y_m)
    ax.add_patch(plt.Rectangle((cx0, cy0), s.core_x_m, s.core_y_m, fill=False, edgecolor='#2E8B57', linewidth=2.4))
    ax.add_patch(plt.Rectangle((ox0, oy0), s.open_x_m, s.open_y_m, fill=False, edgecolor='#2E8B57', linewidth=1.2, linestyle='--'))
    active = [o for o in res.model.outriggers if o.active and o.story == story]
    if active:
        mx, my = g.plan_x_m / 2, g.plan_y_m / 2
        ax.plot([0, cx0], [my, my], linewidth=4.5, color='#FF8C00')
        ax.plot([cx0 + s.core_x_m, g.plan_x_m], [my, my], linewidth=4.5, color='#FF8C00')
        ax.plot([mx, mx], [0, cy0], linewidth=4.5, color='#FF8C00')
        ax.plot([mx, mx], [cy0 + s.core_y_m, g.plan_y_m], linewidth=4.5, color='#FF8C00')
        ax.text(mx, my, 'Outrigger', ha='center', va='center')
    ax.set_title(f'Plan - Story {story} | {s.zone} zone | wall t={s.wall_t_m:.2f} m')
    ax.set_aspect('equal')
    ax.set_xlim(-2, g.plan_x_m + 2)
    ax.set_ylim(-2, g.plan_y_m + 2)
    return fig


# ----------------------------- Streamlit UI -----------------------------

def read_sidebar() -> BuildingModel:
    st.sidebar.header('Geometry')
    g = Geometry(
        n_story=st.sidebar.number_input('Stories', 10, 120, 60),
        n_basement=st.sidebar.number_input('Basements', 0, 20, 6),
        story_height_m=st.sidebar.number_input('Story height (m)', 2.8, 5.0, 3.2),
        basement_height_m=st.sidebar.number_input('Basement height (m)', 2.8, 5.0, 3.2),
        plan_x_m=st.sidebar.number_input('Plan X (m)', 20.0, 200.0, 48.0),
        plan_y_m=st.sidebar.number_input('Plan Y (m)', 20.0, 200.0, 42.0),
        n_bays_x=st.sidebar.number_input('Bays X', 2, 20, 6),
        n_bays_y=st.sidebar.number_input('Bays Y', 2, 20, 6),
        core_ratio_x=st.sidebar.number_input('Core ratio X', 0.10, 0.45, 0.24),
        core_ratio_y=st.sidebar.number_input('Core ratio Y', 0.10, 0.45, 0.22),
    )
    st.sidebar.header('Materials')
    mat = Material(st.sidebar.number_input('fck (MPa)', 25.0, 100.0, 60.0), st.sidebar.number_input('Ec (MPa)', 22000.0, 60000.0, 34000.0), st.sidebar.number_input('fy (MPa)', 300.0, 700.0, 420.0))
    st.sidebar.header('Loads')
    loads = Loads(st.sidebar.number_input('DL (kN/m²)', 0.0, 15.0, 3.5), st.sidebar.number_input('LL (kN/m²)', 0.0, 10.0, 2.5), st.sidebar.number_input('Finishes (kN/m²)', 0.0, 8.0, 1.5), st.sidebar.number_input('Facade (kN/m)', 0.0, 20.0, 1.2), st.sidebar.number_input('Live load mass factor', 0.0, 1.0, 0.25), st.sidebar.number_input('Seismic mass factor', 0.5, 1.5, 1.0), st.sidebar.number_input('Base shear coeff.', 0.001, 0.100, 0.015))
    st.sidebar.header('Limits and stiffness factors')
    drift_den = st.sidebar.number_input('Drift denominator', 250.0, 2000.0, 500.0)
    limits = Limits(drift_limit_ratio=1.0 / drift_den)
    cracked = CrackedFactors(st.sidebar.number_input('Wall cracked factor', 0.05, 1.0, 0.35), st.sidebar.number_input('Column cracked factor', 0.05, 1.0, 0.70), st.sidebar.number_input('Outrigger connection efficiency', 0.30, 1.0, 0.80))
    st.sidebar.header('Period target')
    period = PeriodRule(st.sidebar.number_input('Ct', 0.001, 0.2, 0.0488, format='%.4f'), st.sidebar.number_input('x exponent', 0.1, 1.5, 0.75), st.sidebar.number_input('Upper period factor', 1.0, 2.0, 1.2), st.sidebar.number_input('Target position factor', 0.1, 0.95, 0.8))
    st.sidebar.header('Outriggers')
    n_out = st.sidebar.number_input('Number of outriggers', 0, 5, 2)
    outs = []
    for i in range(int(n_out)):
        default_story = int(round([0.50, 0.70, 0.33, 0.85, 0.20][i] * g.n_story))
        st.sidebar.markdown(f'Outrigger {i+1}')
        story = st.sidebar.number_input(f'Story {i+1}', 1, int(g.n_story), max(1, min(default_story, int(g.n_story))), key=f'out_story_{i}')
        sys_label = st.sidebar.selectbox(f'Type {i+1}', [OutriggerType.BELT_TRUSS.value, OutriggerType.PIPE_BRACE.value], key=f'out_type_{i}')
        depth = st.sidebar.number_input(f'Depth {i+1} (m)', 1.0, 8.0, 3.0, key=f'out_depth_{i}')
        ach = st.sidebar.number_input(f'Chord area {i+1} (m²)', 0.005, 0.5, 0.08, key=f'out_ach_{i}')
        adi = st.sidebar.number_input(f'Diagonal area {i+1} (m²)', 0.005, 0.5, 0.04, key=f'out_adi_{i}')
        outs.append(Outrigger(int(story), OutriggerType(sys_label), float(depth), float(ach), float(adi)))
    return BuildingModel(material=mat, geometry=g, loads=loads, limits=limits, period=period, cracked=cracked, outriggers=outs)


def main():
    st.set_page_config(page_title='Professional Tower MDOF', layout='wide')
    st.title('Professional Tower Preliminary Design - MDOF + Outrigger')
    st.caption(APP_VERSION)
    st.info('Professional pre-design model: X/Y MDOF modal analysis, story stiffness, section-scale iteration, and outrigger comparison.')
    model = read_sidebar()
    if 'res_v6' not in st.session_state:
        st.session_state.res_v6 = None
        st.session_state.report_v6 = ''
    if st.button('Run professional MDOF pre-design', type='primary'):
        try:
            with st.spinner('Running iterative MDOF analysis...'):
                res = run_design(model)
                st.session_state.res_v6 = res
                st.session_state.report_v6 = build_report(res)
            st.success('Analysis completed.')
        except Exception as e:
            st.error(f'Analysis failed: {e}')
    res = st.session_state.res_v6
    if res is None:
        st.warning('Run the analysis to see results.')
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('T_ref (s)', f'{res.t_ref:.3f}')
    c2.metric('T_target (s)', f'{res.t_target:.3f}')
    c3.metric('T_gov (s)', f'{res.governing_period:.3f}')
    c4.metric('Error (%)', f'{100*res.period_error:.2f}')
    d1, d2, d3, d4 = st.columns(4)
    d1.metric('Direction', res.governing_direction.value)
    d2.metric('Drift X', f'{res.drift_x_ratio:.5f}')
    d3.metric('Drift Y', f'{res.drift_y_ratio:.5f}')
    d4.metric('Weight (MN)', f'{res.total_weight_kN/1000:.1f}')
    tabs = st.tabs(['Stiffness X', 'Stiffness Y', 'Modes X', 'Modes Y', 'Convergence', 'Plan', 'Story Table', 'Modal Tables', 'Sustainability', 'Report'])
    with tabs[0]: st.pyplot(plot_stiffness(res, Direction.X), use_container_width=True)
    with tabs[1]: st.pyplot(plot_stiffness(res, Direction.Y), use_container_width=True)
    with tabs[2]: st.pyplot(plot_modes(res, res.modal_x), use_container_width=True)
    with tabs[3]: st.pyplot(plot_modes(res, res.modal_y), use_container_width=True)
    with tabs[4]:
        st.pyplot(plot_iteration(res), use_container_width=True)
        st.dataframe(res.iteration_table, use_container_width=True, hide_index=True)
    with tabs[5]:
        story = st.slider('Story', 1, res.model.geometry.n_story, max(1, res.model.geometry.n_story // 2))
        st.pyplot(plot_plan(res, story), use_container_width=True)
    with tabs[6]: st.dataframe(story_table(res), use_container_width=True, hide_index=True)
    with tabs[7]:
        st.markdown('X Direction')
        st.dataframe(modal_table(res.modal_x), use_container_width=True, hide_index=True)
        st.markdown('Y Direction')
        st.dataframe(modal_table(res.modal_y), use_container_width=True, hide_index=True)
    with tabs[8]: st.dataframe(sustainability_table(res), use_container_width=True, hide_index=True)
    with tabs[9]:
        st.text_area('Report', st.session_state.report_v6, height=520)
        st.download_button('Download report', st.session_state.report_v6.encode('utf-8'), 'professional_tower_mdof_report.txt', 'text/plain')


if __name__ == '__main__':
    main()
