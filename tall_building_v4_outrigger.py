
"""
asce7_modal_tower_predesign_v10.py

ASCE 7 Modal Response Spectrum Preliminary Tower Design Framework
=================================================================

This application is written for preliminary PhD-level tower pre-design.
It is NOT an ETABS calibration tool. It follows the logic of modal response
spectrum analysis:

    1. Build mass matrix [M]
    2. Build stiffness matrix [K] from structural mechanics
    3. Solve [K]{phi} = omega^2[M]{phi}
    4. Build ASCE 7 design spectrum from SDS, SD1, TL
    5. Calculate modal responses
    6. Combine modes using SRSS or CQC
    7. Check modal mass participation
    8. Scale RSA base shear to ASCE 7 ELF minimum
    9. Report story shear, overturning, displacement, drift, and section data

Important:
----------
This is a preliminary design/research framework. It is not a final design
program and does not replace ETABS/SAP2000/Abaqus or code-compliant final
member design.

Author: Benyamin
Version: v10.0-ASCE7-RSA
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import pi, sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# CONSTANTS
# ============================================================

G = 9.81
RHO_STEEL = 7850.0
APP_VERSION = "v10.0-ASCE7-RSA"


# ============================================================
# ENUMS
# ============================================================

class Direction(str, Enum):
    X = "X"
    Y = "Y"


class CombineMethod(str, Enum):
    SRSS = "SRSS"
    CQC = "CQC"


class OutriggerType(str, Enum):
    NONE = "None"
    BELT_TRUSS = "Belt Truss"
    PIPE_BRACE = "Pipe Bracing"


# ============================================================
# INPUT DATA
# ============================================================

@dataclass
class ASCE7SpectrumInput:
    """
    ASCE 7 design spectrum input.

    T0 = 0.2 SD1 / SDS
    Ts = SD1 / SDS

    Sa(T):
      0 <= T < T0       : SDS(0.4 + 0.6T/T0)
      T0 <= T <= Ts     : SDS
      Ts < T <= TL      : SD1/T
      T > TL            : SD1*TL/T^2
    """
    SDS: float = 0.70
    SD1: float = 0.35
    TL: float = 8.0
    S1: float = 0.30

    R: float = 5.0
    Ie: float = 1.0
    Cd: float = 5.0

    damping_ratio: float = 0.05

    # ASCE 7 approximate period parameters for ELF period cap
    Ct: float = 0.016
    x: float = 0.90
    Cu: float = 1.40

    # Common ASCE 7 modal scaling requirement is often taken as 85% for older editions.
    # Some ASCE 7-16/22 implementations use 100% depending on provision/version.
    # Therefore this is user-selectable.
    rsa_scale_ratio_to_elf: float = 0.85

    use_period_cap_for_elf: bool = True
    scale_drifts_with_base_shear: bool = False


@dataclass
class TowerGeometry:
    n_story: int = 60
    story_height_m: float = 3.2
    plan_x_m: float = 48.0
    plan_y_m: float = 42.0

    core_x_m: float = 12.0
    core_y_m: float = 10.0

    wall_t_base_m: float = 0.80
    wall_t_top_m: float = 0.35

    n_perimeter_columns: int = 28
    column_dim_base_m: float = 1.40
    column_dim_top_m: float = 0.80

    @property
    def height_m(self) -> float:
        return self.n_story * self.story_height_m

    @property
    def floor_area_m2(self) -> float:
        return self.plan_x_m * self.plan_y_m


@dataclass
class MaterialAndMass:
    Ec_MPa: float = 34000.0

    # These are not "calibration factors"; they are preliminary effective cracked stiffness inputs.
    # They must be justified by ACI/engineering assumptions in a thesis.
    wall_effective_I_factor: float = 0.35
    column_effective_I_factor: float = 0.70

    floor_mass_ton: float = 2500.0

    @property
    def E_N_m2(self) -> float:
        return self.Ec_MPa * 1e6

    @property
    def floor_mass_kg(self) -> float:
        return self.floor_mass_ton * 1000.0


@dataclass
class OutriggerInput:
    use_outrigger: bool = True
    outrigger_type: OutriggerType = OutriggerType.BELT_TRUSS
    story_1: int = 30
    story_2: int = 42

    depth_m: float = 3.0
    chord_area_m2: float = 0.08
    diagonal_area_m2: float = 0.04
    connection_efficiency: float = 0.75


@dataclass
class ModelInput:
    spectrum: ASCE7SpectrumInput
    geometry: TowerGeometry
    material: MaterialAndMass
    outrigger: OutriggerInput
    combine_method: CombineMethod = CombineMethod.CQC
    n_modes: int = 12


# ============================================================
# ASCE 7 SPECTRUM AND ELF
# ============================================================

def asce7_corner_periods(sp: ASCE7SpectrumInput) -> Tuple[float, float]:
    if sp.SDS <= 0:
        return 0.0, 0.0
    Ts = sp.SD1 / sp.SDS
    T0 = 0.2 * Ts
    return T0, Ts


def asce7_design_spectrum_sa_g(T: float, sp: ASCE7SpectrumInput) -> float:
    """
    ASCE 7 design response spectrum ordinate in g.
    """
    T = max(float(T), 1e-9)
    T0, Ts = asce7_corner_periods(sp)

    if T < T0 and T0 > 0:
        return sp.SDS * (0.4 + 0.6 * T / T0)
    if T <= Ts:
        return sp.SDS
    if T <= sp.TL:
        return sp.SD1 / T
    return sp.SD1 * sp.TL / T**2


def design_spectrum_for_rsa_g(T: float, sp: ASCE7SpectrumInput) -> float:
    """
    Design-level acceleration used for force calculation.

    A common preliminary implementation applies the response modification factor
    by using Sa_design = Sa * Ie/R.
    """
    return asce7_design_spectrum_sa_g(T, sp) * sp.Ie / sp.R


def approximate_period_Ta(geom: TowerGeometry, sp: ASCE7SpectrumInput) -> float:
    # ASCE height in feet if using US units. Here the app uses SI.
    # To remain transparent, h is converted to ft for Ct values commonly tabulated in ft-based ASCE equations.
    h_ft = geom.height_m * 3.28084
    return sp.Ct * h_ft**sp.x


def elf_period_for_base_shear(T_modal: float, geom: TowerGeometry, sp: ASCE7SpectrumInput) -> float:
    if not sp.use_period_cap_for_elf:
        return T_modal
    Ta = approximate_period_Ta(geom, sp)
    return min(T_modal, sp.Cu * Ta)


def asce7_Cs(T: float, sp: ASCE7SpectrumInput) -> float:
    """
    ASCE 7 ELF seismic response coefficient, simplified.

    Cs = SDS / (R/Ie)
    need not exceed:
      SD1 / (T(R/Ie)) for T <= TL
      SD1*TL / (T^2(R/Ie)) for T > TL
    minimum:
      max(0.044*SDS*Ie, 0.01)
    if S1 >= 0.6:
      also not less than 0.5*S1/(R/Ie)
    """
    T = max(T, 1e-6)
    R_over_Ie = sp.R / sp.Ie

    Cs_upper_short = sp.SDS / R_over_Ie

    if T <= sp.TL:
        Cs_limit = sp.SD1 / (T * R_over_Ie)
    else:
        Cs_limit = sp.SD1 * sp.TL / (T**2 * R_over_Ie)

    Cs = min(Cs_upper_short, Cs_limit)

    Cs_min = max(0.044 * sp.SDS * sp.Ie, 0.01)
    if sp.S1 >= 0.6:
        Cs_min = max(Cs_min, 0.5 * sp.S1 / R_over_Ie)

    return max(Cs, Cs_min)


def elf_base_shear_N(total_mass_kg: float, T_modal: float, inp: ModelInput) -> Tuple[float, float, float]:
    T_used = elf_period_for_base_shear(T_modal, inp.geometry, inp.spectrum)
    Cs = asce7_Cs(T_used, inp.spectrum)
    W_N = total_mass_kg * G
    return Cs * W_N, Cs, T_used


# ============================================================
# STRUCTURAL MECHANICS MODEL
# ============================================================

def interpolate(base: float, top: float, story: int, n_story: int) -> float:
    r = (story - 1) / max(n_story - 1, 1)
    return base + r * (top - base)


def rectangular_tube_inertia(outer_x: float, outer_y: float, t: float) -> Tuple[float, float]:
    """
    Return Ix, Iy of closed rectangular tube by outer minus inner rectangle.
    """
    ix_outer = outer_x * outer_y**3 / 12.0
    iy_outer = outer_y * outer_x**3 / 12.0

    inner_x = max(outer_x - 2 * t, 0.10)
    inner_y = max(outer_y - 2 * t, 0.10)

    ix_inner = inner_x * inner_y**3 / 12.0
    iy_inner = inner_y * inner_x**3 / 12.0

    return max(ix_outer - ix_inner, 1e-9), max(iy_outer - iy_inner, 1e-9)


def perimeter_column_global_inertia(geom: TowerGeometry, mat: MaterialAndMass, col_dim: float) -> Tuple[float, float]:
    """
    Approximate global bending contribution of perimeter columns:
        I_global = sum(I_local + A*r^2)
    """
    n = max(geom.n_perimeter_columns, 4)
    n_side = max(n // 4, 1)

    pts = []

    for i in range(n_side):
        x = -geom.plan_x_m / 2 + i * geom.plan_x_m / max(n_side - 1, 1)
        pts.append((x, -geom.plan_y_m / 2))
        pts.append((x, geom.plan_y_m / 2))

    for i in range(n_side):
        y = -geom.plan_y_m / 2 + i * geom.plan_y_m / max(n_side - 1, 1)
        pts.append((-geom.plan_x_m / 2, y))
        pts.append((geom.plan_x_m / 2, y))

    pts = pts[:n]

    A = col_dim**2
    I_local = col_dim**4 / 12.0

    Ix = 0.0
    Iy = 0.0

    for x, y in pts:
        Ix += I_local + A * y**2
        Iy += I_local + A * x**2

    return mat.column_effective_I_factor * Ix, mat.column_effective_I_factor * Iy


def effective_EI_profile(inp: ModelInput, direction: Direction) -> np.ndarray:
    geom = inp.geometry
    mat = inp.material
    E = mat.E_N_m2

    EI = []

    for story in range(1, geom.n_story + 1):
        t = interpolate(geom.wall_t_base_m, geom.wall_t_top_m, story, geom.n_story)
        col_dim = interpolate(geom.column_dim_base_m, geom.column_dim_top_m, story, geom.n_story)

        Ix_core, Iy_core = rectangular_tube_inertia(geom.core_x_m, geom.core_y_m, t)
        Ix_core *= mat.wall_effective_I_factor
        Iy_core *= mat.wall_effective_I_factor

        Ix_col, Iy_col = perimeter_column_global_inertia(geom, mat, col_dim)

        # X translation -> bending about Y axis.
        if direction == Direction.X:
            I_eff = Iy_core + Iy_col
        else:
            I_eff = Ix_core + Ix_col

        EI.append(E * I_eff)

    return np.array(EI, dtype=float)


def outrigger_eta(typ: OutriggerType) -> float:
    if typ == OutriggerType.BELT_TRUSS:
        return 1.00
    if typ == OutriggerType.PIPE_BRACE:
        return 0.65
    return 0.0


def outrigger_rotational_spring(inp: ModelInput, story: int, direction: Direction) -> float:
    out = inp.outrigger
    geom = inp.geometry

    if not out.use_outrigger:
        return 0.0
    if out.outrigger_type == OutriggerType.NONE:
        return 0.0
    if story not in [out.story_1, out.story_2]:
        return 0.0

    E = inp.material.E_N_m2
    eta = outrigger_eta(out.outrigger_type) * out.connection_efficiency

    if direction == Direction.X:
        arm = max((geom.plan_x_m - geom.core_x_m) / 2.0, 1.0)
    else:
        arm = max((geom.plan_y_m - geom.core_y_m) / 2.0, 1.0)

    L_diag = sqrt(arm**2 + out.depth_m**2)

    k_chord = E * out.chord_area_m2 / arm
    k_diag = E * out.diagonal_area_m2 / L_diag
    k_axial = 2.0 * (k_chord + k_diag)

    # Ktheta = axial stiffness * lever arm^2
    return eta * k_axial * arm**2


def beam_element_stiffness(EI: float, L: float) -> np.ndarray:
    """
    Euler-Bernoulli flexural beam element stiffness matrix.
    DOFs: [u_i, theta_i, u_j, theta_j]
    """
    return EI / L**3 * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2],
        ],
        dtype=float,
    )


def assemble_mk(inp: ModelInput, direction: Direction) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Assemble M and K matrices.

    Node 0 is fixed at base.
    Each node has u and theta DOFs.
    Free DOFs exclude base u and theta.
    """
    n = inp.geometry.n_story
    L = inp.geometry.story_height_m
    ndof = 2 * (n + 1)

    K = np.zeros((ndof, ndof), dtype=float)
    M = np.zeros((ndof, ndof), dtype=float)

    EI = effective_EI_profile(inp, direction)

    for e in range(n):
        ke = beam_element_stiffness(EI[e], L)
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]

        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]

    # Floor masses
    for node in range(1, n + 1):
        u = 2 * node
        theta = 2 * node + 1

        M[u, u] += inp.material.floor_mass_kg

        # Rotational inertia regularization. Very small, only to prevent singular M.
        M[theta, theta] += inp.material.floor_mass_kg * inp.geometry.story_height_m**2 * 1e-5

    # Outrigger rotational springs
    for story in range(1, n + 1):
        ktheta = outrigger_rotational_spring(inp, story, direction)
        if ktheta > 0:
            theta_dof = 2 * story + 1
            K[theta_dof, theta_dof] += ktheta

    free = list(range(2, ndof))
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]

    return Mff, Kff, free


# ============================================================
# MODAL ANALYSIS
# ============================================================

def modal_analysis(inp: ModelInput, direction: Direction) -> Dict:
    M, K, free = assemble_mk(inp, direction)
    n_floor = inp.geometry.n_story
    n_modes_req = min(inp.n_modes, 2 * n_floor)

    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    keep = eigvals > 1e-8
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_modes = min(n_modes_req, len(eigvals))
    eigvals = eigvals[:n_modes]
    eigvecs = eigvecs[:, :n_modes]

    omega = np.sqrt(eigvals)
    periods = 2 * pi / omega
    freqs = omega / (2 * pi)

    # Influence vector: ground acceleration excites translational floor DOFs.
    r = np.zeros((2 * n_floor, 1))
    r[0::2, 0] = 1.0

    total_mass = inp.material.floor_mass_kg * n_floor

    modes = []
    gammas = []
    meff_ratios = []
    cumulative = []

    cum = 0.0

    for i in range(n_modes):
        phi = eigvecs[:, i].reshape(-1, 1)

        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ r) / denom).item()
        meff = gamma**2 * denom
        ratio = meff / total_mass
        cum += ratio

        floor_shape = phi.flatten()[0::2]
        if abs(floor_shape[-1]) > 1e-12:
            floor_shape = floor_shape / floor_shape[-1]
        if floor_shape[-1] < 0:
            floor_shape *= -1

        modes.append(floor_shape)
        gammas.append(gamma)
        meff_ratios.append(ratio)
        cumulative.append(cum)

    return {
        "direction": direction,
        "M": M,
        "K": K,
        "periods": periods,
        "frequencies": freqs,
        "omegas": omega,
        "eigvecs": eigvecs,
        "floor_shapes": modes,
        "gammas": np.array(gammas),
        "effective_mass_ratios": np.array(meff_ratios),
        "cumulative_mass_ratios": np.array(cumulative),
    }


# ============================================================
# MODAL RESPONSE SPECTRUM
# ============================================================

def cqc_rho(omega_i: float, omega_j: float, zeta: float) -> float:
    """
    CQC modal correlation coefficient.
    """
    if abs(omega_i - omega_j) < 1e-12:
        return 1.0

    beta = omega_j / omega_i
    num = 8 * zeta**2 * beta**1.5
    den = (1 - beta**2)**2 + 4 * zeta**2 * beta * (1 + beta)**2
    return num / max(den, 1e-12)


def combine_modal_values(values: np.ndarray, omegas: np.ndarray, method: CombineMethod, zeta: float) -> np.ndarray:
    """
    Combine modal responses.

    values shape:
        n_modes x n_response
    """
    if values.ndim == 1:
        values = values.reshape((-1, 1))

    n_modes, n_resp = values.shape

    if method == CombineMethod.SRSS:
        return np.sqrt(np.sum(values**2, axis=0))

    # CQC
    result = np.zeros(n_resp)
    for k in range(n_resp):
        s = 0.0
        for i in range(n_modes):
            for j in range(n_modes):
                rho = cqc_rho(omegas[i], omegas[j], zeta)
                s += rho * values[i, k] * values[j, k]
        result[k] = sqrt(max(s, 0.0))

    return result


def response_spectrum_analysis(inp: ModelInput, direction: Direction) -> Dict:
    modal = modal_analysis(inp, direction)

    n = inp.geometry.n_story
    n_modes = len(modal["periods"])

    modal_floor_forces = []
    modal_story_shear = []
    modal_overturning = []
    modal_disp = []
    modal_interstory_drift = []
    modal_base_shear = []

    heights = np.arange(1, n + 1) * inp.geometry.story_height_m
    m = inp.material.floor_mass_kg

    for i in range(n_modes):
        T = modal["periods"][i]
        omega = modal["omegas"][i]
        gamma = modal["gammas"][i]

        phi_full = modal["eigvecs"][:, i].reshape(-1, 1)
        phi_u = phi_full.flatten()[0::2]

        Sa_design_g = design_spectrum_for_rsa_g(T, inp.spectrum)
        Sa_design = Sa_design_g * G

        # Modal displacement:
        # u_i = phi_i * Gamma_i * Sa / omega^2
        u_modal = phi_u * gamma * Sa_design / omega**2

        # Modal equivalent lateral floor forces:
        # F_j,i = m_j * phi_j,i * Gamma_i * Sa
        f_modal = m * phi_u * gamma * Sa_design

        story_shear = np.zeros(n)
        for j in range(n - 1, -1, -1):
            story_shear[j] = f_modal[j] + (story_shear[j + 1] if j < n - 1 else 0.0)

        overturning = np.zeros(n)
        for j in range(n):
            overturning[j] = np.sum(f_modal[j:] * (heights[j:] - (heights[j] - inp.geometry.story_height_m)))

        drift = np.zeros(n)
        drift[0] = u_modal[0]
        drift[1:] = np.diff(u_modal)

        modal_floor_forces.append(f_modal)
        modal_story_shear.append(story_shear)
        modal_overturning.append(overturning)
        modal_disp.append(u_modal)
        modal_interstory_drift.append(drift)
        modal_base_shear.append(np.sum(f_modal))

    modal_floor_forces = np.array(modal_floor_forces)
    modal_story_shear = np.array(modal_story_shear)
    modal_overturning = np.array(modal_overturning)
    modal_disp = np.array(modal_disp)
    modal_interstory_drift = np.array(modal_interstory_drift)
    modal_base_shear = np.array(modal_base_shear)

    zeta = inp.spectrum.damping_ratio
    method = inp.combine_method

    floor_forces_comb = combine_modal_values(modal_floor_forces, modal["omegas"], method, zeta)
    story_shear_comb = combine_modal_values(modal_story_shear, modal["omegas"], method, zeta)
    overturning_comb = combine_modal_values(modal_overturning, modal["omegas"], method, zeta)
    disp_comb = combine_modal_values(modal_disp, modal["omegas"], method, zeta)
    drift_comb = combine_modal_values(modal_interstory_drift, modal["omegas"], method, zeta)
    base_shear_comb = combine_modal_values(modal_base_shear, modal["omegas"], method, zeta).item()

    # Drift amplification by Cd/Ie for design drift.
    drift_design = drift_comb * inp.spectrum.Cd / inp.spectrum.Ie
    disp_design = disp_comb * inp.spectrum.Cd / inp.spectrum.Ie

    total_mass = inp.material.floor_mass_kg * n
    T1 = modal["periods"][0]
    V_elf, Cs, T_elf = elf_base_shear_N(total_mass, T1, inp)

    V_required = inp.spectrum.rsa_scale_ratio_to_elf * V_elf
    scale_factor = 1.0
    if base_shear_comb < V_required and base_shear_comb > 1e-9:
        scale_factor = V_required / base_shear_comb

    # Scale force quantities, not necessarily drift unless user chooses.
    floor_forces_scaled = floor_forces_comb * scale_factor
    story_shear_scaled = story_shear_comb * scale_factor
    overturning_scaled = overturning_comb * scale_factor
    base_shear_scaled = base_shear_comb * scale_factor

    if inp.spectrum.scale_drifts_with_base_shear:
        drift_design_scaled = drift_design * scale_factor
        disp_design_scaled = disp_design * scale_factor
    else:
        drift_design_scaled = drift_design
        disp_design_scaled = disp_design

    return {
        "modal": modal,
        "floor_forces_N": floor_forces_scaled,
        "story_shear_N": story_shear_scaled,
        "overturning_Nm": overturning_scaled,
        "disp_m": disp_design_scaled,
        "drift_m": drift_design_scaled,
        "base_shear_rsa_unscaled_N": base_shear_comb,
        "base_shear_scaled_N": base_shear_scaled,
        "elf_base_shear_N": V_elf,
        "elf_Cs": Cs,
        "elf_period_used_s": T_elf,
        "rsa_scale_factor": scale_factor,
    }


# ============================================================
# TABLES
# ============================================================

def spectrum_table(inp: ModelInput) -> pd.DataFrame:
    T_values = np.linspace(0.0, max(10.0, inp.spectrum.TL * 1.25), 160)
    rows = []

    for T in T_values:
        rows.append(
            {
                "T (s)": T,
                "ASCE Sa design spectrum (g)": asce7_design_spectrum_sa_g(T, inp.spectrum),
                "RSA force spectrum Sa*Ie/R (g)": design_spectrum_for_rsa_g(T, inp.spectrum),
            }
        )

    return pd.DataFrame(rows)


def modal_table(rsa: Dict) -> pd.DataFrame:
    modal = rsa["modal"]
    sp = []

    for i, T in enumerate(modal["periods"]):
        sp.append(
            {
                "Mode": i + 1,
                "Period (s)": T,
                "Frequency (Hz)": modal["frequencies"][i],
                "Gamma": modal["gammas"][i],
                "Effective mass (%)": 100 * modal["effective_mass_ratios"][i],
                "Cumulative mass (%)": 100 * modal["cumulative_mass_ratios"][i],
                "Sa design (g)": design_spectrum_for_rsa_g(T, current_input.spectrum) if "current_input" in globals() else None,
            }
        )

    return pd.DataFrame(sp)


def modal_table_with_input(inp: ModelInput, rsa: Dict) -> pd.DataFrame:
    modal = rsa["modal"]
    rows = []

    for i, T in enumerate(modal["periods"]):
        rows.append(
            {
                "Mode": i + 1,
                "Period (s)": T,
                "Frequency (Hz)": modal["frequencies"][i],
                "Gamma": modal["gammas"][i],
                "Effective mass (%)": 100 * modal["effective_mass_ratios"][i],
                "Cumulative mass (%)": 100 * modal["cumulative_mass_ratios"][i],
                "Sa ASCE (g)": asce7_design_spectrum_sa_g(T, inp.spectrum),
                "Sa used for RSA (g)": design_spectrum_for_rsa_g(T, inp.spectrum),
            }
        )

    return pd.DataFrame(rows)


def story_response_table(inp: ModelInput, rsa: Dict) -> pd.DataFrame:
    n = inp.geometry.n_story
    rows = []

    for i in range(n):
        rows.append(
            {
                "Story": i + 1,
                "Elevation (m)": (i + 1) * inp.geometry.story_height_m,
                "Floor force (kN)": rsa["floor_forces_N"][i] / 1000.0,
                "Story shear (kN)": rsa["story_shear_N"][i] / 1000.0,
                "Overturning (kN.m)": rsa["overturning_Nm"][i] / 1000.0,
                "Design displacement (m)": rsa["disp_m"][i],
                "Design interstory drift (m)": rsa["drift_m"][i],
                "Drift ratio": rsa["drift_m"][i] / inp.geometry.story_height_m,
            }
        )

    return pd.DataFrame(rows)


def section_property_table(inp: ModelInput) -> pd.DataFrame:
    rows = []
    EI_x = effective_EI_profile(inp, Direction.X)
    EI_y = effective_EI_profile(inp, Direction.Y)

    for story in range(1, inp.geometry.n_story + 1):
        t = interpolate(inp.geometry.wall_t_base_m, inp.geometry.wall_t_top_m, story, inp.geometry.n_story)
        c = interpolate(inp.geometry.column_dim_base_m, inp.geometry.column_dim_top_m, story, inp.geometry.n_story)

        rows.append(
            {
                "Story": story,
                "Wall t (m)": t,
                "Column dim (m)": c,
                "EI for X translation (GN.m²)": EI_x[story - 1] / 1e9,
                "EI for Y translation (GN.m²)": EI_y[story - 1] / 1e9,
                "Outrigger Ktheta X (GN.m/rad)": outrigger_rotational_spring(inp, story, Direction.X) / 1e9,
                "Outrigger Ktheta Y (GN.m/rad)": outrigger_rotational_spring(inp, story, Direction.Y) / 1e9,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_spectrum(inp: ModelInput):
    df = spectrum_table(inp)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["T (s)"], df["ASCE Sa design spectrum (g)"], label="ASCE design spectrum Sa")
    ax.plot(df["T (s)"], df["RSA force spectrum Sa*Ie/R (g)"], label="RSA force spectrum")
    ax.set_xlabel("Period T (s)")
    ax.set_ylabel("Spectral acceleration (g)")
    ax.set_title("ASCE 7 design response spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_modes(inp: ModelInput, rsa: Dict):
    modal = rsa["modal"]
    y = np.arange(1, inp.geometry.n_story + 1) * inp.geometry.story_height_m
    n_modes = min(5, len(modal["floor_shapes"]))

    fig, axes = plt.subplots(1, n_modes, figsize=(16, 6))
    if n_modes == 1:
        axes = [axes]

    for i in range(n_modes):
        ax = axes[i]
        phi = modal["floor_shapes"][i]
        ax.plot(phi, y, linewidth=2)
        ax.scatter(phi, y, s=12)
        ax.axvline(0, linestyle="--", linewidth=0.8)

        if inp.outrigger.use_outrigger:
            for st in [inp.outrigger.story_1, inp.outrigger.story_2]:
                if 1 <= st <= inp.geometry.n_story:
                    ax.axhline(st * inp.geometry.story_height_m, linestyle=":", alpha=0.6)

        ax.set_title(f"Mode {i+1}\nT={modal['periods'][i]:.2f}s")
        ax.set_xlabel("Normalized displacement")
        if i == 0:
            ax.set_ylabel("Height (m)")
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Mode shapes - Direction {modal['direction'].value}")
    fig.tight_layout()
    return fig


def plot_story_response(inp: ModelInput, rsa: Dict, response: str):
    df = story_response_table(inp, rsa)

    fig, ax = plt.subplots(figsize=(7, 8))
    if response == "Story shear":
        ax.plot(df["Story shear (kN)"], df["Story"])
        ax.set_xlabel("Story shear (kN)")
    elif response == "Drift ratio":
        ax.plot(df["Drift ratio"], df["Story"])
        ax.set_xlabel("Drift ratio")
    elif response == "Displacement":
        ax.plot(df["Design displacement (m)"], df["Story"])
        ax.set_xlabel("Design displacement (m)")
    else:
        ax.plot(df["Overturning (kN.m)"], df["Story"])
        ax.set_xlabel("Overturning (kN.m)")

    ax.set_ylabel("Story")
    ax.set_title(response)
    ax.grid(True, alpha=0.3)
    return fig


def plot_EI(inp: ModelInput):
    df = section_property_table(inp)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(df["EI for X translation (GN.m²)"], df["Story"], label="EI X")
    ax.plot(df["EI for Y translation (GN.m²)"], df["Story"], label="EI Y")
    ax.set_xlabel("EI (GN.m²)")
    ax.set_ylabel("Story")
    ax.set_title("Effective flexural stiffness profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_plan(inp: ModelInput):
    g = inp.geometry

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [-g.plan_x_m/2, g.plan_x_m/2, g.plan_x_m/2, -g.plan_x_m/2, -g.plan_x_m/2],
        [-g.plan_y_m/2, -g.plan_y_m/2, g.plan_y_m/2, g.plan_y_m/2, -g.plan_y_m/2],
        color="black",
    )

    ax.add_patch(
        plt.Rectangle(
            (-g.core_x_m/2, -g.core_y_m/2),
            g.core_x_m,
            g.core_y_m,
            fill=False,
            edgecolor="green",
            linewidth=2.5,
        )
    )

    n = max(g.n_perimeter_columns, 4)
    n_side = max(n // 4, 1)
    pts = []

    for i in range(n_side):
        x = -g.plan_x_m/2 + i * g.plan_x_m / max(n_side - 1, 1)
        pts.append((x, -g.plan_y_m/2))
        pts.append((x, g.plan_y_m/2))

    for i in range(n_side):
        y = -g.plan_y_m/2 + i * g.plan_y_m / max(n_side - 1, 1)
        pts.append((-g.plan_x_m/2, y))
        pts.append((g.plan_x_m/2, y))

    pts = pts[:n]

    for x, y in pts:
        dim = g.column_dim_base_m
        ax.add_patch(
            plt.Rectangle(
                (x-dim/2, y-dim/2),
                dim,
                dim,
                facecolor="darkred",
                alpha=0.85,
            )
        )

    if inp.outrigger.use_outrigger and inp.outrigger.outrigger_type != OutriggerType.NONE:
        ax.plot([-g.plan_x_m/2, -g.core_x_m/2], [0, 0], color="orange", linewidth=4)
        ax.plot([g.core_x_m/2, g.plan_x_m/2], [0, 0], color="orange", linewidth=4)
        ax.plot([0, 0], [-g.plan_y_m/2, -g.core_y_m/2], color="orange", linewidth=4)
        ax.plot([0, 0], [g.core_y_m/2, g.plan_y_m/2], color="orange", linewidth=4)

    ax.set_aspect("equal")
    ax.set_title("Simplified structural plan")
    ax.grid(True, alpha=0.2)
    return fig


# ============================================================
# REPORT
# ============================================================

def build_report(inp: ModelInput, rsa_x: Dict, rsa_y: Dict) -> str:
    T1x = rsa_x["modal"]["periods"][0]
    T1y = rsa_y["modal"]["periods"][0]
    gov_dir = "X" if T1x >= T1y else "Y"
    gov_T = max(T1x, T1y)

    df_x = story_response_table(inp, rsa_x)
    df_y = story_response_table(inp, rsa_y)

    max_drift_x = df_x["Drift ratio"].max()
    max_drift_y = df_y["Drift ratio"].max()

    T0, Ts = asce7_corner_periods(inp.spectrum)

    lines = []
    lines.append("=" * 90)
    lines.append("ASCE 7 MODAL RESPONSE SPECTRUM PRELIMINARY TOWER DESIGN REPORT")
    lines.append("=" * 90)
    lines.append(f"Version: {APP_VERSION}")
    lines.append("")
    lines.append("1. ASCE 7 DESIGN SPECTRUM")
    lines.append("-" * 90)
    lines.append(f"SDS = {inp.spectrum.SDS:.4f} g")
    lines.append(f"SD1 = {inp.spectrum.SD1:.4f} g")
    lines.append(f"TL  = {inp.spectrum.TL:.4f} s")
    lines.append(f"T0  = {T0:.4f} s")
    lines.append(f"Ts  = {Ts:.4f} s")
    lines.append(f"R   = {inp.spectrum.R:.3f}")
    lines.append(f"Ie  = {inp.spectrum.Ie:.3f}")
    lines.append(f"Cd  = {inp.spectrum.Cd:.3f}")
    lines.append("")
    lines.append("2. MODAL PERIODS")
    lines.append("-" * 90)
    lines.append(f"T1 X = {T1x:.4f} s")
    lines.append(f"T1 Y = {T1y:.4f} s")
    lines.append(f"Governing direction = {gov_dir}")
    lines.append(f"Governing period = {gov_T:.4f} s")
    lines.append("")
    lines.append("3. MODAL MASS PARTICIPATION")
    lines.append("-" * 90)
    lines.append(f"X cumulative mass used = {100*rsa_x['modal']['cumulative_mass_ratios'][-1]:.2f} %")
    lines.append(f"Y cumulative mass used = {100*rsa_y['modal']['cumulative_mass_ratios'][-1]:.2f} %")
    lines.append("")
    lines.append("4. BASE SHEAR SCALING")
    lines.append("-" * 90)
    lines.append(f"X RSA base shear unscaled = {rsa_x['base_shear_rsa_unscaled_N']/1000:.2f} kN")
    lines.append(f"X ELF base shear = {rsa_x['elf_base_shear_N']/1000:.2f} kN")
    lines.append(f"X scale factor = {rsa_x['rsa_scale_factor']:.3f}")
    lines.append(f"Y RSA base shear unscaled = {rsa_y['base_shear_rsa_unscaled_N']/1000:.2f} kN")
    lines.append(f"Y ELF base shear = {rsa_y['elf_base_shear_N']/1000:.2f} kN")
    lines.append(f"Y scale factor = {rsa_y['rsa_scale_factor']:.3f}")
    lines.append("")
    lines.append("5. DRIFT")
    lines.append("-" * 90)
    lines.append(f"Maximum design drift ratio X = {max_drift_x:.6f}")
    lines.append(f"Maximum design drift ratio Y = {max_drift_y:.6f}")
    lines.append("")
    lines.append("6. OUTRIGGER MODEL")
    lines.append("-" * 90)
    lines.append(f"Outrigger used = {inp.outrigger.use_outrigger}")
    lines.append(f"Outrigger type = {inp.outrigger.outrigger_type.value}")
    lines.append(f"Outrigger stories = {inp.outrigger.story_1}, {inp.outrigger.story_2}")
    lines.append("The outrigger is modeled as rotational restraint at the actual outrigger levels.")
    lines.append("")
    lines.append("7. ENGINEERING LIMITATION")
    lines.append("-" * 90)
    lines.append(
        "This is a preliminary modal response spectrum framework. Final design requires "
        "3D finite element modeling, torsion, accidental eccentricity, diaphragm behavior, "
        "P-Delta effects, load combinations, foundation flexibility, and detailed member design."
    )

    return "\n".join(lines)


# ============================================================
# STREAMLIT UI
# ============================================================

def read_sidebar() -> ModelInput:
    st.sidebar.header("1. ASCE 7 Spectrum")

    sp = ASCE7SpectrumInput(
        SDS=st.sidebar.number_input("SDS (g)", 0.01, 3.00, 0.70),
        SD1=st.sidebar.number_input("SD1 (g)", 0.01, 3.00, 0.35),
        TL=st.sidebar.number_input("TL (s)", 2.0, 20.0, 8.0),
        S1=st.sidebar.number_input("S1 (g)", 0.00, 3.00, 0.30),
        R=st.sidebar.number_input("R", 1.0, 10.0, 5.0),
        Ie=st.sidebar.number_input("Ie", 0.5, 2.0, 1.0),
        Cd=st.sidebar.number_input("Cd", 1.0, 10.0, 5.0),
        damping_ratio=st.sidebar.number_input("Damping ratio", 0.01, 0.20, 0.05),
        Ct=st.sidebar.number_input("Ct for Ta", 0.001, 0.100, 0.016, format="%.4f"),
        x=st.sidebar.number_input("x for Ta", 0.50, 1.00, 0.90),
        Cu=st.sidebar.number_input("Cu", 1.0, 2.0, 1.40),
        rsa_scale_ratio_to_elf=st.sidebar.number_input("RSA/ELF minimum ratio", 0.50, 1.00, 0.85),
        use_period_cap_for_elf=st.sidebar.checkbox("Use CuTa cap for ELF", value=True),
        scale_drifts_with_base_shear=st.sidebar.checkbox("Scale drifts with RSA base shear factor", value=False),
    )

    st.sidebar.header("2. Geometry")

    geom = TowerGeometry(
        n_story=st.sidebar.number_input("Stories", 10, 120, 60),
        story_height_m=st.sidebar.number_input("Story height (m)", 2.8, 5.0, 3.2),
        plan_x_m=st.sidebar.number_input("Plan X (m)", 20.0, 200.0, 48.0),
        plan_y_m=st.sidebar.number_input("Plan Y (m)", 20.0, 200.0, 42.0),
        core_x_m=st.sidebar.number_input("Core X (m)", 4.0, 80.0, 12.0),
        core_y_m=st.sidebar.number_input("Core Y (m)", 4.0, 80.0, 10.0),
        wall_t_base_m=st.sidebar.number_input("Wall t base (m)", 0.20, 2.50, 0.80),
        wall_t_top_m=st.sidebar.number_input("Wall t top (m)", 0.15, 2.00, 0.35),
        n_perimeter_columns=st.sidebar.number_input("Perimeter column count", 4, 160, 28),
        column_dim_base_m=st.sidebar.number_input("Column dim base (m)", 0.30, 4.00, 1.40),
        column_dim_top_m=st.sidebar.number_input("Column dim top (m)", 0.20, 3.00, 0.80),
    )

    st.sidebar.header("3. Material and mass")

    mat = MaterialAndMass(
        Ec_MPa=st.sidebar.number_input("Ec (MPa)", 20000.0, 60000.0, 34000.0),
        wall_effective_I_factor=st.sidebar.number_input("Wall effective I factor", 0.05, 1.00, 0.35),
        column_effective_I_factor=st.sidebar.number_input("Column effective I factor", 0.05, 1.00, 0.70),
        floor_mass_ton=st.sidebar.number_input("Floor mass (ton)", 100.0, 30000.0, 2500.0),
    )

    st.sidebar.header("4. Outrigger")

    out_type = st.sidebar.selectbox(
        "Outrigger type",
        [OutriggerType.BELT_TRUSS.value, OutriggerType.PIPE_BRACE.value, OutriggerType.NONE.value],
        index=0,
    )

    out = OutriggerInput(
        use_outrigger=st.sidebar.checkbox("Use outrigger", value=True),
        outrigger_type=OutriggerType(out_type),
        story_1=st.sidebar.number_input("Outrigger story 1", 1, int(geom.n_story), int(round(0.50 * geom.n_story))),
        story_2=st.sidebar.number_input("Outrigger story 2", 1, int(geom.n_story), int(round(0.70 * geom.n_story))),
        depth_m=st.sidebar.number_input("Outrigger depth (m)", 1.0, 10.0, 3.0),
        chord_area_m2=st.sidebar.number_input("Chord area (m²)", 0.001, 1.000, 0.08, format="%.4f"),
        diagonal_area_m2=st.sidebar.number_input("Diagonal area (m²)", 0.001, 1.000, 0.04, format="%.4f"),
        connection_efficiency=st.sidebar.number_input("Connection efficiency", 0.10, 1.00, 0.75),
    )

    st.sidebar.header("5. Modal settings")

    method = st.sidebar.selectbox(
        "Modal combination",
        [CombineMethod.CQC.value, CombineMethod.SRSS.value],
        index=0,
    )

    n_modes = st.sidebar.number_input("Number of modes", 1, 40, 12)

    return ModelInput(
        spectrum=sp,
        geometry=geom,
        material=mat,
        outrigger=out,
        combine_method=CombineMethod(method),
        n_modes=int(n_modes),
    )


def main():
    st.set_page_config(page_title="ASCE 7 Modal Tower Pre-Design", layout="wide")

    st.title("ASCE 7 Modal Response Spectrum Tower Pre-Design")
    st.caption(APP_VERSION)

    st.markdown(
        """
        This framework is based on **modal response spectrum analysis**, not ETABS calibration.
        It builds structural stiffness, solves modal periods, applies the ASCE 7 design spectrum,
        combines modal responses, scales base shear, and reports drift/force quantities.
        """
    )

    inp = read_sidebar()

    rsa_x = response_spectrum_analysis(inp, Direction.X)
    rsa_y = response_spectrum_analysis(inp, Direction.Y)

    T1x = rsa_x["modal"]["periods"][0]
    T1y = rsa_y["modal"]["periods"][0]
    gov_T = max(T1x, T1y)
    gov_dir = "X" if T1x >= T1y else "Y"

    max_drift_x = story_response_table(inp, rsa_x)["Drift ratio"].max()
    max_drift_y = story_response_table(inp, rsa_y)["Drift ratio"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T1 X (s)", f"{T1x:.2f}")
    c2.metric("T1 Y (s)", f"{T1y:.2f}")
    c3.metric("Governing T1 (s)", f"{gov_T:.2f}")
    c4.metric("Gov. direction", gov_dir)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("X base shear (kN)", f"{rsa_x['base_shear_scaled_N']/1000:.1f}")
    c6.metric("Y base shear (kN)", f"{rsa_y['base_shear_scaled_N']/1000:.1f}")
    c7.metric("Max drift X", f"{max_drift_x:.5f}")
    c8.metric("Max drift Y", f"{max_drift_y:.5f}")

    tabs = st.tabs(
        [
            "ASCE Spectrum",
            "Modes X",
            "Modes Y",
            "Story Response X",
            "Story Response Y",
            "Section Stiffness",
            "Plan",
            "Tables",
            "Report",
        ]
    )

    with tabs[0]:
        st.pyplot(plot_spectrum(inp), use_container_width=True)
        st.dataframe(spectrum_table(inp), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.pyplot(plot_modes(inp, rsa_x), use_container_width=True)
        st.dataframe(modal_table_with_input(inp, rsa_x), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.pyplot(plot_modes(inp, rsa_y), use_container_width=True)
        st.dataframe(modal_table_with_input(inp, rsa_y), use_container_width=True, hide_index=True)

    with tabs[3]:
        p1, p2 = st.columns(2)
        with p1:
            st.pyplot(plot_story_response(inp, rsa_x, "Story shear"), use_container_width=True)
        with p2:
            st.pyplot(plot_story_response(inp, rsa_x, "Drift ratio"), use_container_width=True)
        st.dataframe(story_response_table(inp, rsa_x), use_container_width=True, hide_index=True)

    with tabs[4]:
        p1, p2 = st.columns(2)
        with p1:
            st.pyplot(plot_story_response(inp, rsa_y, "Story shear"), use_container_width=True)
        with p2:
            st.pyplot(plot_story_response(inp, rsa_y, "Drift ratio"), use_container_width=True)
        st.dataframe(story_response_table(inp, rsa_y), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.pyplot(plot_EI(inp), use_container_width=True)
        st.dataframe(section_property_table(inp), use_container_width=True, hide_index=True)

    with tabs[6]:
        st.pyplot(plot_plan(inp), use_container_width=True)

    with tabs[7]:
        st.markdown("### X Modal Table")
        st.dataframe(modal_table_with_input(inp, rsa_x), use_container_width=True, hide_index=True)
        st.markdown("### Y Modal Table")
        st.dataframe(modal_table_with_input(inp, rsa_y), use_container_width=True, hide_index=True)

    with tabs[8]:
        report = build_report(inp, rsa_x, rsa_y)
        st.text_area("Report", report, height=560)
        st.download_button(
            "Download report",
            data=report.encode("utf-8"),
            file_name="asce7_modal_tower_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
