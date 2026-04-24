
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import pi, sqrt, ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 9.81
RHO_STEEL_KG_M3 = 7850.0
APP_VERSION = "v12.0-auto-sizing-ASCE7-modal"


class Direction(str, Enum):
    X = "X"
    Y = "Y"


class ModalCombination(str, Enum):
    CQC = "CQC"
    SRSS = "SRSS"


class OutriggerType(str, Enum):
    TUBULAR_BRACE = "Tubular Bracing"
    BELT_TRUSS = "Belt Truss"
    NONE = "None"


@dataclass(frozen=True)
class ASCE7Input:
    SDS: float = 0.70
    SD1: float = 0.35
    S1: float = 0.30
    TL: float = 8.0
    R: float = 5.0
    Ie: float = 1.0
    Cd: float = 5.0
    damping_ratio: float = 0.05
    Ct: float = 0.016
    x: float = 0.90
    Cu: float = 1.40
    rsa_to_elf_min_ratio: float = 0.85
    use_CuTa_period_cap: bool = True
    scale_drift_with_base_shear_factor: bool = False


@dataclass(frozen=True)
class TowerGeometry:
    n_story: int = 60
    story_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0
    n_bays_x: int = 6
    n_bays_y: int = 6

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


@dataclass(frozen=True)
class MaterialMass:
    Ec_MPa: float = 34000.0
    fck_MPa: float = 60.0
    fy_MPa: float = 420.0
    floor_mass_ton: float = 2500.0

    @property
    def E_N_m2(self) -> float:
        return self.Ec_MPa * 1e6

    @property
    def floor_mass_kg(self) -> float:
        return self.floor_mass_ton * 1000.0

    @property
    def fy_N_m2(self) -> float:
        return self.fy_MPa * 1e6


@dataclass(frozen=True)
class SizingRules:
    core_ratio_x_min: float = 0.18
    core_ratio_x_max: float = 0.32
    core_ratio_y_min: float = 0.16
    core_ratio_y_max: float = 0.30
    wall_t_min_m: float = 0.25
    wall_t_max_m: float = 2.50
    side_wall_t_min_m: float = 0.25
    side_wall_t_max_m: float = 1.50
    side_wall_length_ratio: float = 0.18
    col_min_m: float = 0.50
    col_max_m: float = 4.00
    beam_b_min_m: float = 0.35
    beam_b_max_m: float = 1.20
    beam_h_min_m: float = 0.60
    beam_h_max_m: float = 2.50
    slab_t_min_m: float = 0.18
    slab_t_max_m: float = 0.45
    wall_effective_I_factor: float = 0.35
    column_effective_I_factor: float = 0.70
    side_wall_effective_I_factor: float = 0.35
    max_design_drift_ratio: float = 0.015
    min_modal_mass_ratio: float = 0.90
    max_iterations: int = 25
    convergence_tolerance: float = 0.03
    too_small_drift_ratio_fraction: float = 0.35
    max_member_growth_per_iteration: float = 1.15
    max_member_reduction_per_iteration: float = 0.95


@dataclass(frozen=True)
class SectionState:
    core_x_m: float
    core_y_m: float
    core_wall_t_base_m: float
    core_wall_t_top_m: float
    side_wall_length_x_m: float
    side_wall_length_y_m: float
    side_wall_t_base_m: float
    side_wall_t_top_m: float
    column_dim_base_m: float
    column_dim_top_m: float
    n_perimeter_columns: int
    beam_b_m: float
    beam_h_m: float
    slab_t_m: float
    outrigger_type: OutriggerType
    outrigger_story_1: int
    outrigger_story_2: int
    tubular_brace_diameter_m: float
    tubular_brace_thickness_m: float
    tubular_brace_area_m2: float
    tubular_braced_spans_x: int
    tubular_braced_spans_y: int
    outrigger_depth_m: float
    outrigger_connection_efficiency: float


@dataclass(frozen=True)
class ModelInput:
    asce: ASCE7Input
    geometry: TowerGeometry
    material: MaterialMass
    rules: SizingRules
    section: SectionState
    combination: ModalCombination = ModalCombination.CQC
    n_modes: int = 12


def clamp(value: float, vmin: float, vmax: float) -> float:
    return float(max(vmin, min(vmax, value)))


def interpolate(base: float, top: float, story: int, n_story: int) -> float:
    r = (story - 1) / max(n_story - 1, 1)
    return base + r * (top - base)


def propose_initial_section(geom: TowerGeometry, mat: MaterialMass, rules: SizingRules, use_tubular_outrigger: bool = True) -> SectionState:
    H = geom.height_m
    core_x = clamp(0.24 * geom.plan_x_m, rules.core_ratio_x_min * geom.plan_x_m, rules.core_ratio_x_max * geom.plan_x_m)
    core_y = clamp(0.22 * geom.plan_y_m, rules.core_ratio_y_min * geom.plan_y_m, rules.core_ratio_y_max * geom.plan_y_m)
    wall_base = clamp(H / 220.0, rules.wall_t_min_m, rules.wall_t_max_m)
    wall_top = clamp(0.45 * wall_base, rules.wall_t_min_m, rules.wall_t_max_m)
    side_len_x = rules.side_wall_length_ratio * geom.plan_x_m
    side_len_y = rules.side_wall_length_ratio * geom.plan_y_m
    side_t_base = clamp(0.70 * wall_base, rules.side_wall_t_min_m, rules.side_wall_t_max_m)
    side_t_top = clamp(0.50 * side_t_base, rules.side_wall_t_min_m, rules.side_wall_t_max_m)

    n_cols = max(2 * (geom.n_bays_x + geom.n_bays_y), 12)
    total_weight_kN = mat.floor_mass_kg * geom.n_story * G / 1000.0
    P_col_kN = 0.35 * total_weight_kN / n_cols
    sigma_allow_kN_m2 = 0.25 * mat.fck_MPa * 1000.0
    col_base = clamp(sqrt(max(P_col_kN / max(sigma_allow_kN_m2, 1e-9), 1e-9)) * 1.35, rules.col_min_m, rules.col_max_m)
    col_top = clamp(0.65 * col_base, rules.col_min_m, rules.col_max_m)

    span = max(geom.bay_x_m, geom.bay_y_m)
    slab_t = clamp(span / 32.0, rules.slab_t_min_m, rules.slab_t_max_m)
    beam_h = clamp(span / 10.0, rules.beam_h_min_m, rules.beam_h_max_m)
    beam_b = clamp(0.45 * beam_h, rules.beam_b_min_m, rules.beam_b_max_m)

    story1 = max(1, min(geom.n_story, round(0.50 * geom.n_story)))
    story2 = max(1, min(geom.n_story, round(0.70 * geom.n_story)))

    D = clamp(0.12 * span, 0.30, 1.20)
    t = clamp(D / 30.0, 0.012, 0.060)
    A_tube = np.pi / 4.0 * (D**2 - max(D - 2 * t, 0.001)**2)
    braced_x = max(1, min(geom.n_bays_x, ceil(geom.n_bays_x / 3)))
    braced_y = max(1, min(geom.n_bays_y, ceil(geom.n_bays_y / 3)))
    out_type = OutriggerType.TUBULAR_BRACE if use_tubular_outrigger else OutriggerType.BELT_TRUSS

    return SectionState(
        core_x_m=core_x, core_y_m=core_y,
        core_wall_t_base_m=wall_base, core_wall_t_top_m=wall_top,
        side_wall_length_x_m=side_len_x, side_wall_length_y_m=side_len_y,
        side_wall_t_base_m=side_t_base, side_wall_t_top_m=side_t_top,
        column_dim_base_m=col_base, column_dim_top_m=col_top,
        n_perimeter_columns=n_cols,
        beam_b_m=beam_b, beam_h_m=beam_h, slab_t_m=slab_t,
        outrigger_type=out_type,
        outrigger_story_1=story1, outrigger_story_2=story2,
        tubular_brace_diameter_m=D, tubular_brace_thickness_m=t,
        tubular_brace_area_m2=float(A_tube),
        tubular_braced_spans_x=braced_x, tubular_braced_spans_y=braced_y,
        outrigger_depth_m=max(geom.story_height_m, 2.5),
        outrigger_connection_efficiency=0.75
    )


def asce_corner_periods(asce: ASCE7Input) -> Tuple[float, float]:
    Ts = asce.SD1 / max(asce.SDS, 1e-9)
    return 0.2 * Ts, Ts


def asce_spectrum_sa_g(T: float, asce: ASCE7Input) -> float:
    T = max(float(T), 1e-9)
    T0, Ts = asce_corner_periods(asce)
    if T < T0 and T0 > 0:
        return asce.SDS * (0.4 + 0.6 * T / T0)
    if T <= Ts:
        return asce.SDS
    if T <= asce.TL:
        return asce.SD1 / T
    return asce.SD1 * asce.TL / T**2


def rsa_force_spectrum_sa_g(T: float, asce: ASCE7Input) -> float:
    return asce_spectrum_sa_g(T, asce) * asce.Ie / asce.R


def approximate_period_Ta_s(geom: TowerGeometry, asce: ASCE7Input) -> float:
    h_ft = geom.height_m * 3.28084
    return asce.Ct * h_ft**asce.x


def asce_Cs(T: float, asce: ASCE7Input) -> float:
    T = max(float(T), 1e-6)
    R_over_Ie = asce.R / asce.Ie
    Cs_short = asce.SDS / R_over_Ie
    Cs_long = asce.SD1 / (T * R_over_Ie) if T <= asce.TL else asce.SD1 * asce.TL / (T**2 * R_over_Ie)
    Cs = min(Cs_short, Cs_long)
    Cs_min = max(0.044 * asce.SDS * asce.Ie, 0.01)
    if asce.S1 >= 0.6:
        Cs_min = max(Cs_min, 0.5 * asce.S1 / R_over_Ie)
    return max(Cs, Cs_min)


def elf_base_shear(total_mass_kg: float, T_modal: float, model: ModelInput) -> Dict[str, float]:
    Ta = approximate_period_Ta_s(model.geometry, model.asce)
    T_used = min(T_modal, model.asce.Cu * Ta) if model.asce.use_CuTa_period_cap else T_modal
    Cs = asce_Cs(T_used, model.asce)
    return {"V_N": Cs * total_mass_kg * G, "Cs": Cs, "Ta_s": Ta, "T_used_s": T_used, "CuTa_s": model.asce.Cu * Ta}


def rectangular_tube_inertia(outer_x: float, outer_y: float, t: float) -> Tuple[float, float]:
    ix_o = outer_x * outer_y**3 / 12.0
    iy_o = outer_y * outer_x**3 / 12.0
    inner_x = max(outer_x - 2 * t, 0.10)
    inner_y = max(outer_y - 2 * t, 0.10)
    return max(ix_o - inner_x * inner_y**3 / 12.0, 1e-9), max(iy_o - inner_y * inner_x**3 / 12.0, 1e-9)


def side_wall_inertia(model: ModelInput, story: int) -> Tuple[float, float]:
    g, s, r = model.geometry, model.section, model.rules
    t = interpolate(s.side_wall_t_base_m, s.side_wall_t_top_m, story, g.n_story)
    lx, ly = s.side_wall_length_x_m, s.side_wall_length_y_m
    Ix = 2.0 * (lx * t**3 / 12.0 + lx * t * (g.plan_y_m / 2.0)**2) + 2.0 * (t * ly**3 / 12.0)
    Iy = 2.0 * (t * lx**3 / 12.0) + 2.0 * (ly * t**3 / 12.0 + ly * t * (g.plan_x_m / 2.0)**2)
    return r.side_wall_effective_I_factor * Ix, r.side_wall_effective_I_factor * Iy


def perimeter_column_coordinates(g: TowerGeometry, n_cols: int) -> List[Tuple[float, float]]:
    n = max(n_cols, 4)
    n_side = max(n // 4, 1)
    pts = []
    for i in range(n_side):
        x = -g.plan_x_m / 2 + i * g.plan_x_m / max(n_side - 1, 1)
        pts += [(x, -g.plan_y_m / 2), (x, g.plan_y_m / 2)]
    for i in range(n_side):
        y = -g.plan_y_m / 2 + i * g.plan_y_m / max(n_side - 1, 1)
        pts += [(-g.plan_x_m / 2, y), (g.plan_x_m / 2, y)]
    return pts[:n]


def perimeter_column_global_inertia(model: ModelInput, story: int) -> Tuple[float, float]:
    g, s, r = model.geometry, model.section, model.rules
    col = interpolate(s.column_dim_base_m, s.column_dim_top_m, story, g.n_story)
    A = col**2
    I_local = col**4 / 12.0
    Ix = Iy = 0.0
    for x, y in perimeter_column_coordinates(g, s.n_perimeter_columns):
        Ix += I_local + A * y**2
        Iy += I_local + A * x**2
    return r.column_effective_I_factor * Ix, r.column_effective_I_factor * Iy


def effective_EI_profile(model: ModelInput, direction: Direction) -> np.ndarray:
    g, s, r = model.geometry, model.section, model.rules
    E = model.material.E_N_m2
    EI = []
    for story in range(1, g.n_story + 1):
        t = interpolate(s.core_wall_t_base_m, s.core_wall_t_top_m, story, g.n_story)
        Ix_core, Iy_core = rectangular_tube_inertia(s.core_x_m, s.core_y_m, t)
        Ix_core *= r.wall_effective_I_factor
        Iy_core *= r.wall_effective_I_factor
        Ix_side, Iy_side = side_wall_inertia(model, story)
        Ix_col, Iy_col = perimeter_column_global_inertia(model, story)
        I_eff = Iy_core + Iy_side + Iy_col if direction == Direction.X else Ix_core + Ix_side + Ix_col
        EI.append(E * I_eff)
    return np.array(EI, dtype=float)


def outrigger_efficiency(system: OutriggerType) -> float:
    if system == OutriggerType.TUBULAR_BRACE:
        return 0.85
    if system == OutriggerType.BELT_TRUSS:
        return 1.00
    return 0.0


def outrigger_rotational_stiffness(model: ModelInput, story: int, direction: Direction) -> float:
    s, g = model.section, model.geometry
    if s.outrigger_type == OutriggerType.NONE or story not in [s.outrigger_story_1, s.outrigger_story_2]:
        return 0.0
    E = model.material.E_N_m2
    eta = outrigger_efficiency(s.outrigger_type) * s.outrigger_connection_efficiency
    if direction == Direction.X:
        arm = max((g.plan_x_m - s.core_x_m) / 2.0, 1.0)
        n_spans = max(s.tubular_braced_spans_x, 1)
    else:
        arm = max((g.plan_y_m - s.core_y_m) / 2.0, 1.0)
        n_spans = max(s.tubular_braced_spans_y, 1)
    L_diag = sqrt(arm**2 + s.outrigger_depth_m**2)
    return eta * n_spans * 2.0 * E * s.tubular_brace_area_m2 / L_diag * arm**2


def estimate_quantities(model: ModelInput) -> Dict[str, float]:
    g, s = model.geometry, model.section
    core_wall_vol = side_wall_vol = col_vol = 0.0
    for story in range(1, g.n_story + 1):
        core_t = interpolate(s.core_wall_t_base_m, s.core_wall_t_top_m, story, g.n_story)
        side_t = interpolate(s.side_wall_t_base_m, s.side_wall_t_top_m, story, g.n_story)
        col = interpolate(s.column_dim_base_m, s.column_dim_top_m, story, g.n_story)
        core_wall_vol += 2 * (s.core_x_m + s.core_y_m) * core_t * g.story_height_m
        side_wall_vol += 2 * (s.side_wall_length_x_m + s.side_wall_length_y_m) * side_t * g.story_height_m
        col_vol += s.n_perimeter_columns * col**2 * g.story_height_m
    beam_len_per_floor = g.n_bays_x * (g.n_bays_y + 1) * g.bay_x_m + g.n_bays_y * (g.n_bays_x + 1) * g.bay_y_m
    beam_vol = g.n_story * beam_len_per_floor * s.beam_b_m * s.beam_h_m
    slab_vol = g.n_story * g.floor_area_m2 * s.slab_t_m
    out_steel = 0.0
    if s.outrigger_type != OutriggerType.NONE:
        for direction in [Direction.X, Direction.Y]:
            if direction == Direction.X:
                arm = max((g.plan_x_m - s.core_x_m) / 2.0, 1.0)
                n_spans = max(s.tubular_braced_spans_x, 1)
            else:
                arm = max((g.plan_y_m - s.core_y_m) / 2.0, 1.0)
                n_spans = max(s.tubular_braced_spans_y, 1)
            L = sqrt(arm**2 + s.outrigger_depth_m**2)
            out_steel += 2 * n_spans * s.tubular_brace_area_m2 * L * RHO_STEEL_KG_M3
        out_steel *= 2
    return {
        "core_wall_concrete_m3": core_wall_vol,
        "side_wall_concrete_m3": side_wall_vol,
        "column_concrete_m3": col_vol,
        "beam_concrete_m3": beam_vol,
        "slab_concrete_m3": slab_vol,
        "total_concrete_m3": core_wall_vol + side_wall_vol + col_vol + beam_vol + slab_vol,
        "outrigger_steel_kg": out_steel,
        "seismic_weight_kN": model.material.floor_mass_kg * g.n_story * G / 1000.0,
    }


def beam_element_stiffness(EI: float, L: float) -> np.ndarray:
    return EI / L**3 * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2], [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]], dtype=float)


def assemble_mk(model: ModelInput, direction: Direction) -> Tuple[np.ndarray, np.ndarray]:
    g, n, L = model.geometry, model.geometry.n_story, model.geometry.story_height_m
    ndof = 2 * (n + 1)
    K = np.zeros((ndof, ndof), dtype=float)
    M = np.zeros((ndof, ndof), dtype=float)
    EI = effective_EI_profile(model, direction)
    for e in range(n):
        ke = beam_element_stiffness(EI[e], L)
        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]
    for node in range(1, n + 1):
        u = 2 * node
        theta = 2 * node + 1
        M[u, u] += model.material.floor_mass_kg
        M[theta, theta] += model.material.floor_mass_kg * L**2 * 1e-5
    for story in range(1, n + 1):
        ktheta = outrigger_rotational_stiffness(model, story, direction)
        if ktheta > 0:
            K[2*story+1, 2*story+1] += ktheta
    free = list(range(2, ndof))
    return M[np.ix_(free, free)], K[np.ix_(free, free)]


def modal_analysis(model: ModelInput, direction: Direction) -> Dict:
    M, K = assemble_mk(model, direction)
    vals, vecs = np.linalg.eig(np.linalg.solve(M, K))
    vals, vecs = np.real(vals), np.real(vecs)
    keep = vals > 1e-8
    vals, vecs = vals[keep], vecs[:, keep]
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    n_modes = min(model.n_modes, len(vals))
    vals, vecs = vals[:n_modes], vecs[:, :n_modes]
    omegas = np.sqrt(vals)
    periods = 2*pi/omegas
    freqs = omegas/(2*pi)
    n = model.geometry.n_story
    r = np.zeros((2*n, 1)); r[0::2, 0] = 1.0
    total_mass = model.material.floor_mass_kg * n
    gammas, meff, cumulative, shapes = [], [], [], []
    cum = 0.0
    for i in range(n_modes):
        phi = vecs[:, i].reshape(-1, 1)
        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ r) / denom).item()
        ratio = gamma**2 * denom / total_mass
        cum += ratio
        shape = phi.flatten()[0::2]
        if abs(shape[-1]) > 1e-12:
            shape = shape / shape[-1]
        if shape[-1] < 0:
            shape = -shape
        gammas.append(gamma); meff.append(ratio); cumulative.append(cum); shapes.append(shape)
    return {"direction": direction, "M": M, "K": K, "eigvecs": vecs, "omegas": omegas, "periods": periods, "frequencies": freqs, "gammas": np.array(gammas), "effective_mass_ratios": np.array(meff), "cumulative_mass_ratios": np.array(cumulative), "floor_shapes": shapes}


def cqc_rho(omega_i: float, omega_j: float, zeta: float) -> float:
    if abs(omega_i - omega_j) < 1e-12:
        return 1.0
    beta = omega_j / omega_i
    return (8*zeta**2*beta**1.5) / max((1-beta**2)**2 + 4*zeta**2*beta*(1+beta)**2, 1e-12)


def combine_modal(values: np.ndarray, omegas: np.ndarray, method: ModalCombination, zeta: float) -> np.ndarray:
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if method == ModalCombination.SRSS:
        return np.sqrt(np.sum(values**2, axis=0))
    n_modes, n_resp = values.shape
    result = np.zeros(n_resp)
    for k in range(n_resp):
        s = 0.0
        for i in range(n_modes):
            for j in range(n_modes):
                s += cqc_rho(omegas[i], omegas[j], zeta) * values[i, k] * values[j, k]
        result[k] = sqrt(max(s, 0.0))
    return result


def response_spectrum_analysis(model: ModelInput, direction: Direction) -> Dict:
    modal = modal_analysis(model, direction)
    n, h, m = model.geometry.n_story, model.geometry.story_height_m, model.material.floor_mass_kg
    heights = np.arange(1, n+1) * h
    modal_floor_force, modal_story_shear, modal_overturning, modal_disp, modal_drift, modal_base = [], [], [], [], [], []
    for i, T in enumerate(modal["periods"]):
        omega, gamma = float(modal["omegas"][i]), float(modal["gammas"][i])
        phi_u = modal["eigvecs"][:, i].reshape(-1, 1).flatten()[0::2]
        Sa = rsa_force_spectrum_sa_g(float(T), model.asce) * G
        u = phi_u * gamma * Sa / omega**2
        f = m * phi_u * gamma * Sa
        V = np.zeros(n)
        for j in range(n-1, -1, -1):
            V[j] = f[j] + (V[j+1] if j < n-1 else 0.0)
        M_ot = np.zeros(n)
        for j in range(n):
            M_ot[j] = np.sum(f[j:] * (heights[j:] - (heights[j] - h)))
        drift = np.zeros(n); drift[0] = u[0]; drift[1:] = np.diff(u)
        modal_floor_force.append(f); modal_story_shear.append(V); modal_overturning.append(M_ot); modal_disp.append(u); modal_drift.append(drift); modal_base.append(np.sum(f))
    omegas = modal["omegas"]; zeta = model.asce.damping_ratio
    floor_force = combine_modal(np.array(modal_floor_force), omegas, model.combination, zeta)
    story_shear = combine_modal(np.array(modal_story_shear), omegas, model.combination, zeta)
    overturning = combine_modal(np.array(modal_overturning), omegas, model.combination, zeta)
    disp_elastic = combine_modal(np.array(modal_disp), omegas, model.combination, zeta)
    drift_elastic = combine_modal(np.array(modal_drift), omegas, model.combination, zeta)
    base_rsa = combine_modal(np.array(modal_base), omegas, model.combination, zeta).item()
    elf = elf_base_shear(model.material.floor_mass_kg * n, float(modal["periods"][0]), model)
    required = model.asce.rsa_to_elf_min_ratio * elf["V_N"]
    scale = required / base_rsa if base_rsa < required and base_rsa > 1e-9 else 1.0
    drift_design = drift_elastic * model.asce.Cd / model.asce.Ie
    disp_design = disp_elastic * model.asce.Cd / model.asce.Ie
    if model.asce.scale_drift_with_base_shear_factor:
        drift_design *= scale; disp_design *= scale
    return {"modal": modal, "floor_force_N": floor_force*scale, "story_shear_N": story_shear*scale, "overturning_Nm": overturning*scale, "disp_design_m": disp_design, "drift_design_m": drift_design, "drift_ratio": drift_design/h, "base_shear_rsa_unscaled_N": base_rsa, "base_shear_scaled_N": base_rsa*scale, "elf": elf, "rsa_scale_factor": scale}


def resize_section(model: ModelInput, drift_over: float, too_stiff: bool = False) -> ModelInput:
    s, r, g = model.section, model.rules, model.geometry
    if too_stiff:
        wall_scale = column_scale = brace_scale = r.max_member_reduction_per_iteration
    else:
        wall_scale = min(r.max_member_growth_per_iteration, drift_over**0.20)
        column_scale = min(1.10, drift_over**0.12)
        brace_scale = min(r.max_member_growth_per_iteration, drift_over**0.18)
    D = clamp(s.tubular_brace_diameter_m * brace_scale, 0.20, 2.00)
    t = clamp(s.tubular_brace_thickness_m * brace_scale, 0.008, 0.120)
    A = np.pi / 4.0 * (D**2 - max(D - 2*t, 0.001)**2)
    brx, bry = s.tubular_braced_spans_x, s.tubular_braced_spans_y
    if (not too_stiff) and drift_over > 1.25:
        brx = min(g.n_bays_x, brx + 1)
        bry = min(g.n_bays_y, bry + 1)
    new_section = replace(
        s,
        core_wall_t_base_m=clamp(s.core_wall_t_base_m*wall_scale, r.wall_t_min_m, r.wall_t_max_m),
        core_wall_t_top_m=clamp(s.core_wall_t_top_m*wall_scale, r.wall_t_min_m, r.wall_t_max_m),
        side_wall_t_base_m=clamp(s.side_wall_t_base_m*wall_scale, r.side_wall_t_min_m, r.side_wall_t_max_m),
        side_wall_t_top_m=clamp(s.side_wall_t_top_m*wall_scale, r.side_wall_t_min_m, r.side_wall_t_max_m),
        column_dim_base_m=clamp(s.column_dim_base_m*column_scale, r.col_min_m, r.col_max_m),
        column_dim_top_m=clamp(s.column_dim_top_m*column_scale, r.col_min_m, r.col_max_m),
        tubular_brace_diameter_m=D,
        tubular_brace_thickness_m=t,
        tubular_brace_area_m2=float(A),
        tubular_braced_spans_x=brx,
        tubular_braced_spans_y=bry,
    )
    return replace(model, section=new_section)


def run_predesign(model: ModelInput) -> Dict:
    current = model
    logs = []
    best, best_score = None, 1e99
    for it in range(1, model.rules.max_iterations + 1):
        rsa_x = response_spectrum_analysis(current, Direction.X)
        rsa_y = response_spectrum_analysis(current, Direction.Y)
        max_drift_x, max_drift_y = float(np.max(rsa_x["drift_ratio"])), float(np.max(rsa_y["drift_ratio"]))
        max_drift = max(max_drift_x, max_drift_y)
        mass_x, mass_y = float(rsa_x["modal"]["cumulative_mass_ratios"][-1]), float(rsa_y["modal"]["cumulative_mass_ratios"][-1])
        q = estimate_quantities(current)
        drift_over = max(max_drift/current.rules.max_design_drift_ratio, 1.0)
        too_stiff = max_drift < current.rules.max_design_drift_ratio * current.rules.too_small_drift_ratio_fraction
        score = 1000*max(drift_over-1.0,0.0)**2 + 300*max(current.rules.min_modal_mass_ratio-min(mass_x,mass_y),0.0)**2 + 0.0000015*q["total_concrete_m3"] + 0.0000005*q["outrigger_steel_kg"]
        logs.append({"Iteration": it, "T1 X (s)": rsa_x["modal"]["periods"][0], "T1 Y (s)": rsa_y["modal"]["periods"][0], "Max drift X": max_drift_x, "Max drift Y": max_drift_y, "Allowable drift": current.rules.max_design_drift_ratio, "Modal mass X (%)": 100*mass_x, "Modal mass Y (%)": 100*mass_y, "Core X (m)": current.section.core_x_m, "Core Y (m)": current.section.core_y_m, "Core wall base (m)": current.section.core_wall_t_base_m, "Core wall top (m)": current.section.core_wall_t_top_m, "Side wall base (m)": current.section.side_wall_t_base_m, "Column base (m)": current.section.column_dim_base_m, "Column top (m)": current.section.column_dim_top_m, "Beam b (m)": current.section.beam_b_m, "Beam h (m)": current.section.beam_h_m, "Slab t (m)": current.section.slab_t_m, "Tube D (m)": current.section.tubular_brace_diameter_m, "Tube t (m)": current.section.tubular_brace_thickness_m, "Tube area (m²)": current.section.tubular_brace_area_m2, "Braced spans X": current.section.tubular_braced_spans_x, "Braced spans Y": current.section.tubular_braced_spans_y, "Concrete (m³)": q["total_concrete_m3"], "Outrigger steel (kg)": q["outrigger_steel_kg"], "Base shear X (kN)": rsa_x["base_shear_scaled_N"]/1000, "Base shear Y (kN)": rsa_y["base_shear_scaled_N"]/1000})
        if score < best_score:
            best_score, best = score, (current, rsa_x, rsa_y)
        ok_drift = max_drift <= current.rules.max_design_drift_ratio*(1+current.rules.convergence_tolerance)
        ok_mass = min(mass_x, mass_y) >= current.rules.min_modal_mass_ratio
        if ok_drift and ok_mass and not too_stiff:
            best = (current, rsa_x, rsa_y)
            break
        current = resize_section(current, drift_over, too_stiff)
    if best is None:
        best = (current, response_spectrum_analysis(current, Direction.X), response_spectrum_analysis(current, Direction.Y))
    final_model, final_x, final_y = best
    return {"model": final_model, "rsa_x": final_x, "rsa_y": final_y, "iteration_table": pd.DataFrame(logs)}


def story_response_table(model: ModelInput, rsa: Dict) -> pd.DataFrame:
    n, h = model.geometry.n_story, model.geometry.story_height_m
    return pd.DataFrame({"Story": np.arange(1,n+1), "Elevation (m)": np.arange(1,n+1)*h, "Floor force (kN)": rsa["floor_force_N"]/1000, "Story shear (kN)": rsa["story_shear_N"]/1000, "Overturning (kN.m)": rsa["overturning_Nm"]/1000, "Design displacement (m)": rsa["disp_design_m"], "Design drift (m)": rsa["drift_design_m"], "Drift ratio": rsa["drift_ratio"]})


def modal_table(model: ModelInput, rsa: Dict) -> pd.DataFrame:
    modal = rsa["modal"]
    rows = []
    for i, T in enumerate(modal["periods"]):
        rows.append({"Mode": i+1, "Direction": modal["direction"].value, "Period (s)": T, "Frequency (Hz)": modal["frequencies"][i], "Gamma": modal["gammas"][i], "Effective mass (%)": 100*modal["effective_mass_ratios"][i], "Cumulative mass (%)": 100*modal["cumulative_mass_ratios"][i], "ASCE Sa (g)": asce_spectrum_sa_g(T, model.asce), "RSA Sa*Ie/R (g)": rsa_force_spectrum_sa_g(T, model.asce)})
    return pd.DataFrame(rows)


def section_table(model: ModelInput) -> pd.DataFrame:
    g = model.geometry
    EI_x, EI_y = effective_EI_profile(model, Direction.X), effective_EI_profile(model, Direction.Y)
    rows = []
    for story in range(1, g.n_story+1):
        rows.append({"Story": story, "Core wall t (m)": interpolate(model.section.core_wall_t_base_m, model.section.core_wall_t_top_m, story, g.n_story), "Side wall t (m)": interpolate(model.section.side_wall_t_base_m, model.section.side_wall_t_top_m, story, g.n_story), "Column dim (m)": interpolate(model.section.column_dim_base_m, model.section.column_dim_top_m, story, g.n_story), "EI X translation (GN.m²)": EI_x[story-1]/1e9, "EI Y translation (GN.m²)": EI_y[story-1]/1e9, "Outrigger Ktheta X (GN.m/rad)": outrigger_rotational_stiffness(model, story, Direction.X)/1e9, "Outrigger Ktheta Y (GN.m/rad)": outrigger_rotational_stiffness(model, story, Direction.Y)/1e9})
    return pd.DataFrame(rows)


def final_dimensions_table(model: ModelInput) -> pd.DataFrame:
    s = model.section
    components = ["Central core X","Central core Y","Core wall thickness base","Core wall thickness top","Side wall length X direction","Side wall length Y direction","Side wall thickness base","Side wall thickness top","Column dimension base","Column dimension top","Beam width","Beam depth","Slab thickness","Tubular brace diameter","Tubular brace thickness","Tubular brace area","Braced spans in X","Braced spans in Y","Outrigger story 1","Outrigger story 2"]
    values = [s.core_x_m,s.core_y_m,s.core_wall_t_base_m,s.core_wall_t_top_m,s.side_wall_length_x_m,s.side_wall_length_y_m,s.side_wall_t_base_m,s.side_wall_t_top_m,s.column_dim_base_m,s.column_dim_top_m,s.beam_b_m,s.beam_h_m,s.slab_t_m,s.tubular_brace_diameter_m,s.tubular_brace_thickness_m,s.tubular_brace_area_m2,s.tubular_braced_spans_x,s.tubular_braced_spans_y,s.outrigger_story_1,s.outrigger_story_2]
    units = ["m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m²","-","-","story","story"]
    return pd.DataFrame({"Component": components, "Value": values, "Unit": units})


def summary_table(result: Dict) -> pd.DataFrame:
    model, rsa_x, rsa_y = result["model"], result["rsa_x"], result["rsa_y"]
    sx, sy = story_response_table(model, rsa_x), story_response_table(model, rsa_y)
    q = estimate_quantities(model)
    return pd.DataFrame({"Item": ["T1 X","T1 Y","Modal mass X","Modal mass Y","Base shear X","Base shear Y","Max drift X","Max drift Y","Concrete volume","Outrigger steel","Seismic weight"], "Value": [rsa_x["modal"]["periods"][0], rsa_y["modal"]["periods"][0], 100*rsa_x["modal"]["cumulative_mass_ratios"][-1], 100*rsa_y["modal"]["cumulative_mass_ratios"][-1], rsa_x["base_shear_scaled_N"]/1000, rsa_y["base_shear_scaled_N"]/1000, sx["Drift ratio"].max(), sy["Drift ratio"].max(), q["total_concrete_m3"], q["outrigger_steel_kg"], q["seismic_weight_kN"]], "Unit": ["s","s","%","%","kN","kN","-","-","m³","kg","kN"]})


def spectrum_table(model: ModelInput) -> pd.DataFrame:
    T = np.linspace(0, max(10.0, model.asce.TL*1.25), 160)
    return pd.DataFrame({"T (s)": T, "ASCE Sa (g)": [asce_spectrum_sa_g(x, model.asce) for x in T], "RSA Sa*Ie/R (g)": [rsa_force_spectrum_sa_g(x, model.asce) for x in T]})


def plot_spectrum(model: ModelInput):
    df = spectrum_table(model)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df["T (s)"], df["ASCE Sa (g)"], label="ASCE spectrum")
    ax.plot(df["T (s)"], df["RSA Sa*Ie/R (g)"], label="RSA force spectrum")
    ax.set_xlabel("Period (s)"); ax.set_ylabel("Sa (g)"); ax.set_title("ASCE 7 design spectrum"); ax.grid(True, alpha=0.3); ax.legend()
    return fig


def plot_modes(model: ModelInput, rsa: Dict):
    modal = rsa["modal"]; y = np.arange(1, model.geometry.n_story+1)*model.geometry.story_height_m
    n_modes = min(5, len(modal["floor_shapes"]))
    fig, axes = plt.subplots(1, n_modes, figsize=(16,6))
    if n_modes == 1: axes = [axes]
    for i in range(n_modes):
        ax=axes[i]; ax.plot(modal["floor_shapes"][i], y, linewidth=2); ax.scatter(modal["floor_shapes"][i], y, s=12); ax.axvline(0, linestyle="--", linewidth=0.8)
        for st in [model.section.outrigger_story_1, model.section.outrigger_story_2]:
            ax.axhline(st*model.geometry.story_height_m, linestyle=":", alpha=0.6)
        ax.set_title(f"Mode {i+1}\nT={modal['periods'][i]:.2f}s"); ax.set_xlabel("Normalized displacement")
        if i==0: ax.set_ylabel("Height (m)")
        else: ax.set_yticks([])
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Mode shapes - {modal['direction'].value}"); fig.tight_layout()
    return fig


def plot_story_response(model: ModelInput, rsa: Dict, response: str):
    df = story_response_table(model, rsa)
    fig, ax = plt.subplots(figsize=(7,8))
    if response == "Story shear":
        ax.plot(df["Story shear (kN)"], df["Story"]); ax.set_xlabel("Story shear (kN)")
    elif response == "Drift ratio":
        ax.plot(df["Drift ratio"], df["Story"]); ax.set_xlabel("Drift ratio")
    elif response == "Displacement":
        ax.plot(df["Design displacement (m)"], df["Story"]); ax.set_xlabel("Design displacement (m)")
    else:
        ax.plot(df["Overturning (kN.m)"], df["Story"]); ax.set_xlabel("Overturning (kN.m)")
    ax.set_ylabel("Story"); ax.set_title(response); ax.grid(True, alpha=0.3)
    return fig


def plot_iteration(iter_df: pd.DataFrame):
    fig, axes = plt.subplots(2,2,figsize=(12,9))
    axes[0,0].plot(iter_df["Iteration"], iter_df["Max drift X"], marker="o", label="X"); axes[0,0].plot(iter_df["Iteration"], iter_df["Max drift Y"], marker="s", label="Y"); axes[0,0].plot(iter_df["Iteration"], iter_df["Allowable drift"], linestyle="--", label="Allowable"); axes[0,0].set_title("Drift convergence"); axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()
    axes[0,1].plot(iter_df["Iteration"], iter_df["T1 X (s)"], marker="o", label="T1 X"); axes[0,1].plot(iter_df["Iteration"], iter_df["T1 Y (s)"], marker="s", label="T1 Y"); axes[0,1].set_title("Period evolution"); axes[0,1].grid(True, alpha=0.3); axes[0,1].legend()
    axes[1,0].plot(iter_df["Iteration"], iter_df["Core wall base (m)"], marker="o", label="Core wall"); axes[1,0].plot(iter_df["Iteration"], iter_df["Column base (m)"], marker="s", label="Column"); axes[1,0].plot(iter_df["Iteration"], iter_df["Tube D (m)"], marker="^", label="Tube D"); axes[1,0].set_title("Member size evolution"); axes[1,0].grid(True, alpha=0.3); axes[1,0].legend()
    axes[1,1].plot(iter_df["Iteration"], iter_df["Concrete (m³)"], marker="d"); axes[1,1].set_title("Concrete volume proxy"); axes[1,1].grid(True, alpha=0.3)
    fig.tight_layout(); return fig


def plot_plan(model: ModelInput):
    g, s = model.geometry, model.section
    fig, ax = plt.subplots(figsize=(8,7))
    ax.plot([-g.plan_x_m/2,g.plan_x_m/2,g.plan_x_m/2,-g.plan_x_m/2,-g.plan_x_m/2],[-g.plan_y_m/2,-g.plan_y_m/2,g.plan_y_m/2,g.plan_y_m/2,-g.plan_y_m/2],color="black")
    ax.add_patch(plt.Rectangle((-s.core_x_m/2,-s.core_y_m/2),s.core_x_m,s.core_y_m,fill=False,edgecolor="green",linewidth=2.5))
    ax.plot([-s.side_wall_length_x_m/2,s.side_wall_length_x_m/2],[g.plan_y_m/2,g.plan_y_m/2],color="green",linewidth=4)
    ax.plot([-s.side_wall_length_x_m/2,s.side_wall_length_x_m/2],[-g.plan_y_m/2,-g.plan_y_m/2],color="green",linewidth=4)
    ax.plot([g.plan_x_m/2,g.plan_x_m/2],[-s.side_wall_length_y_m/2,s.side_wall_length_y_m/2],color="green",linewidth=4)
    ax.plot([-g.plan_x_m/2,-g.plan_x_m/2],[-s.side_wall_length_y_m/2,s.side_wall_length_y_m/2],color="green",linewidth=4)
    for x,y in perimeter_column_coordinates(g, s.n_perimeter_columns):
        col=s.column_dim_base_m; ax.add_patch(plt.Rectangle((x-col/2,y-col/2),col,col,facecolor="darkred",alpha=0.85))
    if s.outrigger_type != OutriggerType.NONE:
        ax.plot([-g.plan_x_m/2,-s.core_x_m/2],[0,0],color="orange",linewidth=4); ax.plot([s.core_x_m/2,g.plan_x_m/2],[0,0],color="orange",linewidth=4); ax.plot([0,0],[-g.plan_y_m/2,-s.core_y_m/2],color="orange",linewidth=4); ax.plot([0,0],[s.core_y_m/2,g.plan_y_m/2],color="orange",linewidth=4)
    ax.set_aspect("equal"); ax.set_title("Auto-sized structural plan"); ax.grid(True, alpha=0.2)
    return fig


def build_report(result: Dict) -> str:
    model, rsa_x, rsa_y = result["model"], result["rsa_x"], result["rsa_y"]
    sx, sy = story_response_table(model, rsa_x), story_response_table(model, rsa_y)
    q = estimate_quantities(model)
    lines = []
    lines.append("="*96)
    lines.append("ASCE 7 MODAL RESPONSE SPECTRUM AUTO-SIZING TOWER PREDESIGN REPORT")
    lines.append("="*96)
    lines.append(f"Version: {APP_VERSION}")
    lines.append("")
    lines.append("1. Final preliminary dimensions")
    lines.append("-"*96)
    for _, row in final_dimensions_table(model).iterrows():
        val = row["Value"]
        try:
            val_txt = f"{float(val):.4g}"
        except Exception:
            val_txt = str(val)
        lines.append(f"{row['Component']:<35}: {val_txt} {row['Unit']}")
    lines.append("")
    lines.append("2. Modal response")
    lines.append("-"*96)
    lines.append(f"T1 X = {rsa_x['modal']['periods'][0]:.4f} s")
    lines.append(f"T1 Y = {rsa_y['modal']['periods'][0]:.4f} s")
    lines.append(f"Modal mass X = {100*rsa_x['modal']['cumulative_mass_ratios'][-1]:.2f} %")
    lines.append(f"Modal mass Y = {100*rsa_y['modal']['cumulative_mass_ratios'][-1]:.2f} %")
    lines.append("")
    lines.append("3. ASCE 7 base shear and drift")
    lines.append("-"*96)
    lines.append(f"Base shear X = {rsa_x['base_shear_scaled_N']/1000:.2f} kN")
    lines.append(f"Base shear Y = {rsa_y['base_shear_scaled_N']/1000:.2f} kN")
    lines.append(f"Max drift ratio X = {sx['Drift ratio'].max():.6f}")
    lines.append(f"Max drift ratio Y = {sy['Drift ratio'].max():.6f}")
    lines.append(f"Allowable drift ratio = {model.rules.max_design_drift_ratio:.6f}")
    lines.append("")
    lines.append("4. Quantity proxy")
    lines.append("-"*96)
    lines.append(f"Total concrete proxy = {q['total_concrete_m3']:.2f} m³")
    lines.append(f"Outrigger steel proxy = {q['outrigger_steel_kg']:.2f} kg")
    lines.append("")
    lines.append("5. Engineering conclusion")
    lines.append("-"*96)
    lines.append("The program automatically proposes core dimensions, side wall dimensions, column sizes, beam and slab sizes, tubular outrigger brace sizes, and braced span counts. It then performs ASCE 7 modal response spectrum analysis and redesigns the structure when the drift/stiffness response is not acceptable.")
    lines.append("This is appropriate for preliminary PhD-level system comparison. It is not a full replacement for ETABS because torsional diaphragm DOFs, accidental eccentricity, P-Delta, foundation flexibility, coupling beam design, load combinations, and member strength checks must still be added.")
    return "\n".join(lines)


def make_model_from_sidebar():
    import streamlit as st
    st.sidebar.header("1. ASCE 7")
    asce = ASCE7Input(SDS=st.sidebar.number_input("SDS (g)",0.01,3.0,0.70), SD1=st.sidebar.number_input("SD1 (g)",0.01,3.0,0.35), S1=st.sidebar.number_input("S1 (g)",0.0,3.0,0.30), TL=st.sidebar.number_input("TL (s)",2.0,20.0,8.0), R=st.sidebar.number_input("R",1.0,10.0,5.0), Ie=st.sidebar.number_input("Ie",0.5,2.0,1.0), Cd=st.sidebar.number_input("Cd",1.0,10.0,5.0), damping_ratio=st.sidebar.number_input("Damping ratio",0.01,0.20,0.05), rsa_to_elf_min_ratio=st.sidebar.number_input("RSA / ELF minimum",0.5,1.0,0.85))
    st.sidebar.header("2. Geometry")
    geom = TowerGeometry(n_story=st.sidebar.number_input("Stories",10,120,60), story_height_m=st.sidebar.number_input("Story height (m)",2.8,5.5,3.2), plan_x_m=st.sidebar.number_input("Plan X (m)",20.0,200.0,48.0), plan_y_m=st.sidebar.number_input("Plan Y (m)",20.0,200.0,42.0), n_bays_x=st.sidebar.number_input("Bays X",2,20,6), n_bays_y=st.sidebar.number_input("Bays Y",2,20,6))
    st.sidebar.header("3. Material and mass")
    mat = MaterialMass(Ec_MPa=st.sidebar.number_input("Ec (MPa)",20000.0,60000.0,34000.0), fck_MPa=st.sidebar.number_input("fck (MPa)",25.0,100.0,60.0), fy_MPa=st.sidebar.number_input("fy (MPa)",300.0,700.0,420.0), floor_mass_ton=st.sidebar.number_input("Floor mass (ton)",100.0,30000.0,2500.0))
    st.sidebar.header("4. Predesign criteria")
    rules = SizingRules(max_design_drift_ratio=st.sidebar.number_input("Max design drift ratio",0.001,0.05,0.015,format="%.4f"), min_modal_mass_ratio=st.sidebar.number_input("Minimum modal mass ratio",0.50,0.99,0.90), max_iterations=st.sidebar.number_input("Max redesign iterations",1,40,20), wall_effective_I_factor=st.sidebar.number_input("Wall effective I factor",0.05,1.00,0.35), column_effective_I_factor=st.sidebar.number_input("Column effective I factor",0.05,1.00,0.70), side_wall_effective_I_factor=st.sidebar.number_input("Side wall effective I factor",0.05,1.00,0.35))
    use_tube = st.sidebar.checkbox("Prefer tubular brace outrigger", True)
    section = propose_initial_section(geom, mat, rules, use_tubular_outrigger=use_tube)
    st.sidebar.header("5. Modal settings")
    comb = st.sidebar.selectbox("Combination", [ModalCombination.CQC.value, ModalCombination.SRSS.value], index=0)
    n_modes = st.sidebar.number_input("Number of modes", 1, 60, 12)
    return ModelInput(asce=asce, geometry=geom, material=mat, rules=rules, section=section, combination=ModalCombination(comb), n_modes=int(n_modes))


def main():
    import streamlit as st
    st.set_page_config(page_title="ASCE 7 Tower Auto Predesign v12", layout="wide")
    st.title("Revision 12: ASCE 7 Modal Tower Auto-Predesign")
    st.caption(APP_VERSION)
    st.markdown("This version automatically proposes dimensions for the core, side walls, columns, beams, slabs, and tubular outrigger braces; then it performs ASCE 7 modal response spectrum analysis and redesigns the system.")
    model = make_model_from_sidebar()
    if "v12_result" not in st.session_state:
        st.session_state.v12_result = None
    if st.button("Run auto-predesign", type="primary"):
        with st.spinner("Running auto-sizing and ASCE 7 modal analysis..."):
            st.session_state.v12_result = run_predesign(model)
    result = st.session_state.v12_result
    if result is None:
        st.info("Set inputs and run auto-predesign.")
        return
    final_model, rsa_x, rsa_y = result["model"], result["rsa_x"], result["rsa_y"]
    sx, sy = story_response_table(final_model, rsa_x), story_response_table(final_model, rsa_y)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("T1 X (s)", f"{rsa_x['modal']['periods'][0]:.2f}")
    c2.metric("T1 Y (s)", f"{rsa_y['modal']['periods'][0]:.2f}")
    c3.metric("Max drift X", f"{sx['Drift ratio'].max():.5f}")
    c4.metric("Max drift Y", f"{sy['Drift ratio'].max():.5f}")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("Core", f"{final_model.section.core_x_m:.1f} x {final_model.section.core_y_m:.1f} m")
    c6.metric("Column base", f"{final_model.section.column_dim_base_m:.2f} m")
    c7.metric("Tube brace", f"D={final_model.section.tubular_brace_diameter_m:.2f} m")
    c8.metric("Braced spans", f"X={final_model.section.tubular_braced_spans_x}, Y={final_model.section.tubular_braced_spans_y}")
    tabs = st.tabs(["Final dimensions","Summary","Spectrum","Modes X","Modes Y","Story X","Story Y","Section profile","Iteration","Plan","Report"])
    with tabs[0]: st.dataframe(final_dimensions_table(final_model), use_container_width=True, hide_index=True)
    with tabs[1]: st.dataframe(summary_table(result), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.pyplot(plot_spectrum(final_model), use_container_width=True); st.dataframe(spectrum_table(final_model), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.pyplot(plot_modes(final_model, rsa_x), use_container_width=True); st.dataframe(modal_table(final_model, rsa_x), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.pyplot(plot_modes(final_model, rsa_y), use_container_width=True); st.dataframe(modal_table(final_model, rsa_y), use_container_width=True, hide_index=True)
    with tabs[5]:
        p1,p2=st.columns(2)
        with p1: st.pyplot(plot_story_response(final_model, rsa_x, "Story shear"), use_container_width=True)
        with p2: st.pyplot(plot_story_response(final_model, rsa_x, "Drift ratio"), use_container_width=True)
        st.dataframe(story_response_table(final_model, rsa_x), use_container_width=True, hide_index=True)
    with tabs[6]:
        p1,p2=st.columns(2)
        with p1: st.pyplot(plot_story_response(final_model, rsa_y, "Story shear"), use_container_width=True)
        with p2: st.pyplot(plot_story_response(final_model, rsa_y, "Drift ratio"), use_container_width=True)
        st.dataframe(story_response_table(final_model, rsa_y), use_container_width=True, hide_index=True)
    with tabs[7]: st.dataframe(section_table(final_model), use_container_width=True, hide_index=True)
    with tabs[8]:
        st.pyplot(plot_iteration(result["iteration_table"]), use_container_width=True); st.dataframe(result["iteration_table"], use_container_width=True, hide_index=True)
    with tabs[9]: st.pyplot(plot_plan(final_model), use_container_width=True)
    with tabs[10]:
        report = build_report(result)
        st.text_area("Report", report, height=560)
        st.download_button("Download report", data=report.encode("utf-8"), file_name="asce7_tower_v12_report.txt", mime="text/plain")


if __name__ == "__main__":
    main()
