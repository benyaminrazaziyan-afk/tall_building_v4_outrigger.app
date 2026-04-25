
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import pi, sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.linalg import eigh as scipy_eigh
except Exception:
    scipy_eigh = None


APP_VERSION = "v22.0-checked-brace-plan-MDOF"
G = 9.81
RHO_STEEL = 7850.0
ES_MPA = 200000.0


class Direction(str, Enum):
    X = "X"
    Y = "Y"


class CombinationMethod(str, Enum):
    CQC = "CQC"
    SRSS = "SRSS"


class OutriggerSystem(str, Enum):
    NONE = "None"
    TUBULAR_BRACE = "Tubular Bracing"


@dataclass
class ASCE7Params:
    SDS: float = 0.70
    SD1: float = 0.35
    S1: float = 0.30
    TL: float = 8.0
    R: float = 5.0
    Ie: float = 1.0
    Cd: float = 5.0
    damping_ratio: float = 0.05
    Ct: float = 0.016
    x_exp: float = 0.90
    Cu: float = 1.40
    rsa_min_ratio_to_elf: float = 0.85


@dataclass
class BuildingInput:
    n_story: int = 60
    story_height: float = 3.2
    plan_x: float = 80.0
    plan_y: float = 80.0
    n_bays_x: int = 8
    n_bays_y: int = 8

    core_ratio_x: float = 0.24
    core_ratio_y: float = 0.22
    core_wall_t_base: float = 0.85
    core_wall_t_top: float = 0.35
    side_wall_ratio: float = 0.20
    side_wall_t_base: float = 0.55
    side_wall_t_top: float = 0.25

    fck: float = 70.0
    Ec: float = 36000.0
    DL: float = 3.0
    LL: float = 2.5
    live_load_mass_factor: float = 0.25
    finish_load: float = 1.5
    facade_line_load: float = 1.0

    # Important calibration: tall tower cracked/global effective stiffness
    wall_effective_I_factor: float = 0.18
    column_effective_I_factor: float = 0.35
    side_wall_effective_I_factor: float = 0.18
    global_flexural_modifier: float = 0.60

    column_dim_base: float = 1.80
    column_dim_top: float = 0.80
    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.25

    beam_h_base: float = 1.20
    beam_h_top: float = 0.75
    beam_b_ratio: float = 0.45
    slab_t_base: float = 0.32
    slab_t_top: float = 0.24

    outrigger_system: OutriggerSystem = OutriggerSystem.TUBULAR_BRACE
    outrigger_story_levels: Tuple[int, ...] = (30, 42, 54)
    brace_lines_each_side: int = 2
    brace_D: float = 0.80
    brace_t: float = 0.030
    brace_vertical_depth: float = 3.2
    belt_beam_factor: float = 1.50
    exterior_column_participation: float = 0.70
    connection_efficiency: float = 0.75

    # equivalent lateral transfer factor for simplified stick model
    lateral_transfer_factor: float = 1.00

    drift_limit_ratio: float = 0.015
    minimum_modal_mass_ratio: float = 0.90
    n_modes: int = 12
    combination: CombinationMethod = CombinationMethod.CQC
    asce7: ASCE7Params = None

    auto_redesign: bool = True
    max_iterations: int = 10

    design_wall_scale: float = 1.0
    design_column_scale: float = 1.0
    design_beam_scale: float = 1.0
    design_brace_scale: float = 1.0

    def __post_init__(self):
        if self.asce7 is None:
            self.asce7 = ASCE7Params()

    @property
    def height(self) -> float:
        return self.n_story * self.story_height

    @property
    def bay_x(self) -> float:
        return self.plan_x / max(self.n_bays_x, 1)

    @property
    def bay_y(self) -> float:
        return self.plan_y / max(self.n_bays_y, 1)

    @property
    def floor_area(self) -> float:
        return self.plan_x * self.plan_y


@dataclass
class StorySection:
    story: int
    elevation: float
    core_x: float
    core_y: float
    core_t: float
    side_wall_len_x: float
    side_wall_len_y: float
    side_wall_t: float
    col_int: float
    col_per: float
    col_cor: float
    beam_b: float
    beam_h: float
    slab_t: float


@dataclass
class StoryProperty:
    story: int
    mass_kg: float
    weight_kN: float
    EI_x: float
    EI_y: float
    ktheta_x: float
    ktheta_y: float
    klat_x: float
    klat_y: float
    concrete_m3: float
    steel_kg: float


@dataclass
class ModalResult:
    direction: Direction
    periods_s: List[float]
    frequencies_hz: List[float]
    omegas: np.ndarray
    mode_shapes: List[np.ndarray]
    gammas: List[float]
    eff_mass: List[float]
    cum_mass: List[float]


@dataclass
class RSAResult:
    direction: Direction
    modal: ModalResult
    floor_force_kN: np.ndarray
    story_shear_kN: np.ndarray
    displacement_m: np.ndarray
    drift_m: np.ndarray
    drift_ratio: np.ndarray
    base_shear_scaled_kN: float
    base_shear_unscaled_kN: float
    elf_base_shear_kN: float
    rsa_scale: float
    Ta_s: float
    CuTa_s: float
    T_used_s: float
    Cs: float


@dataclass
class DesignResult:
    inp: BuildingInput
    sections: List[StorySection]
    props: List[StoryProperty]
    modal_x: ModalResult
    modal_y: ModalResult
    rsa_x: RSAResult
    rsa_y: RSAResult
    iteration_table: pd.DataFrame


def interp(base: float, top: float, story: int, n_story: int) -> float:
    r = (story - 1) / max(n_story - 1, 1)
    return base + r * (top - base)


def tube_area(D: float, t: float) -> float:
    return pi / 4.0 * (D**2 - max(D - 2.0*t, 0.001)**2)


def column_counts(inp: BuildingInput) -> Tuple[int, int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2*(inp.n_bays_x-1) + 2*(inp.n_bays_y-1))
    interior = max(0, total - perimeter - corner)
    return total, interior, perimeter, corner


def column_grid(inp: BuildingInput):
    pts = []
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x - inp.plan_x/2
            y = j * inp.bay_y - inp.plan_y/2
            atx = i == 0 or i == inp.n_bays_x
            aty = j == 0 or j == inp.n_bays_y
            typ = "corner" if atx and aty else ("perimeter" if atx or aty else "interior")
            pts.append((x, y, typ))
    return pts


def build_sections(inp: BuildingInput) -> List[StorySection]:
    core_x = inp.core_ratio_x * inp.plan_x
    core_y = inp.core_ratio_y * inp.plan_y
    sections = []

    for st in range(1, inp.n_story + 1):
        hfac = 1.0 - (st - 1) / max(inp.n_story - 1, 1)
        out_floor = st in inp.outrigger_story_levels and inp.outrigger_system != OutriggerSystem.NONE

        core_t = interp(inp.core_wall_t_base, inp.core_wall_t_top, st, inp.n_story) * inp.design_wall_scale
        side_t = interp(inp.side_wall_t_base, inp.side_wall_t_top, st, inp.n_story) * inp.design_wall_scale

        col = interp(inp.column_dim_base, inp.column_dim_top, st, inp.n_story) * inp.design_column_scale
        col = max(col, inp.column_dim_top)

        beam_h = interp(inp.beam_h_base, inp.beam_h_top, st, inp.n_story) * inp.design_beam_scale
        if out_floor:
            beam_h *= inp.belt_beam_factor
        beam_b = max(0.35, inp.beam_b_ratio * beam_h)

        slab_t = interp(inp.slab_t_base, inp.slab_t_top, st, inp.n_story)
        if out_floor:
            slab_t *= 1.10

        sections.append(StorySection(
            story=st,
            elevation=st*inp.story_height,
            core_x=core_x,
            core_y=core_y,
            core_t=core_t,
            side_wall_len_x=inp.side_wall_ratio * inp.plan_x,
            side_wall_len_y=inp.side_wall_ratio * inp.plan_y,
            side_wall_t=side_t,
            col_int=col,
            col_per=col * inp.perimeter_column_factor,
            col_cor=col * inp.corner_column_factor,
            beam_b=beam_b,
            beam_h=beam_h,
            slab_t=slab_t,
        ))

    return enforce_monotonic(sections)


def enforce_monotonic(sections: List[StorySection]) -> List[StorySection]:
    out = list(sections)
    max_core_t = max_side_t = max_int = max_per = max_cor = 0.0
    for i in range(len(out)-1, -1, -1):
        s = out[i]
        max_core_t = max(max_core_t, s.core_t)
        max_side_t = max(max_side_t, s.side_wall_t)
        max_int = max(max_int, s.col_int)
        max_per = max(max_per, s.col_per)
        max_cor = max(max_cor, s.col_cor)
        out[i] = replace(s, core_t=max_core_t, side_wall_t=max_side_t,
                         col_int=max_int, col_per=max_per, col_cor=max_cor)
    return out


def rect_tube_I(outer_x, outer_y, t):
    Ix_o = outer_x * outer_y**3 / 12
    Iy_o = outer_y * outer_x**3 / 12
    ix = max(outer_x - 2*t, 0.1)
    iy = max(outer_y - 2*t, 0.1)
    Ix_i = ix * iy**3 / 12
    Iy_i = iy * ix**3 / 12
    return max(Ix_o - Ix_i, 1e-9), max(Iy_o - Iy_i, 1e-9)


def sidewall_I(inp: BuildingInput, s: StorySection):
    lx, ly, t = s.side_wall_len_x, s.side_wall_len_y, s.side_wall_t
    Ix = 2*(lx*t**3/12 + lx*t*(inp.plan_y/2)**2) + 2*(t*ly**3/12)
    Iy = 2*(t*lx**3/12) + 2*(ly*t**3/12 + ly*t*(inp.plan_x/2)**2)
    return inp.side_wall_effective_I_factor * Ix, inp.side_wall_effective_I_factor * Iy


def column_I(inp: BuildingInput, s: StorySection):
    Ix = Iy = 0.0
    for x, y, typ in column_grid(inp):
        d = s.col_cor if typ == "corner" else (s.col_per if typ == "perimeter" else s.col_int)
        A = d*d
        Iloc = d**4/12
        Ix += Iloc + A*y*y
        Iy += Iloc + A*x*x
    return inp.column_effective_I_factor * Ix, inp.column_effective_I_factor * Iy


def brace_plan_lines(inp: BuildingInput, s: StorySection):
    """
    Generate brace lines from core face to perimeter brace bays.
    The number of lines each side controls stiffness and plan output.
    """
    lines = []
    n = max(inp.brace_lines_each_side, 1)
    cx0 = -s.core_x/2
    cx1 = s.core_x/2
    cy0 = -s.core_y/2
    cy1 = s.core_y/2
    xL = -inp.plan_x/2
    xR = inp.plan_x/2
    yB = -inp.plan_y/2
    yT = inp.plan_y/2

    ys = np.linspace(cy0, cy1, n+2)[1:-1] if n > 1 else np.array([0.0])
    xs = np.linspace(cx0, cx1, n+2)[1:-1] if n > 1 else np.array([0.0])

    for y in ys:
        lines.append(("X", xL, y, cx0, y))
        lines.append(("X", xR, y, cx1, y))
    for x in xs:
        lines.append(("Y", x, yB, x, cy0))
        lines.append(("Y", x, yT, x, cy1))
    return lines


def outrigger_stiffness(inp: BuildingInput, s: StorySection, direction: Direction):
    if inp.outrigger_system == OutriggerSystem.NONE or s.story not in inp.outrigger_story_levels:
        return 0.0, 0.0, 0.0

    E = ES_MPA * 1e6
    A = tube_area(inp.brace_D * inp.design_brace_scale, inp.brace_t * inp.design_brace_scale)
    lines = brace_plan_lines(inp, s)
    k_sum = 0.0
    arm2_sum = 0.0

    for d, x0, y0, x1, y1 in lines:
        if direction == Direction.X and d != "X":
            continue
        if direction == Direction.Y and d != "Y":
            continue

        horizontal_len = sqrt((x1-x0)**2 + (y1-y0)**2)
        L = sqrt(horizontal_len**2 + inp.brace_vertical_depth**2)
        cos2 = (horizontal_len / max(L, 1e-9))**2
        k = E * A / L * cos2
        arm = horizontal_len
        k_sum += k
        arm2_sum += k * arm**2

    # Belt/collector beam limits the brace load path.
    E_c = inp.Ec * 1e6
    I_beam = s.beam_b * s.beam_h**3 / 12
    bay = inp.bay_x if direction == Direction.X else inp.bay_y
    k_collector = 12 * E_c * I_beam / max(bay**3, 1e-9) * max(inp.brace_lines_each_side, 1)

    # exterior columns participate in the axial couple
    participation = inp.exterior_column_participation
    k_brace_effective = k_sum * participation

    # series + partial parallel path to avoid fake infinity and to represent belt continuity
    k_transfer = 1.0 / (1.0/max(k_brace_effective,1e-9) + 1.0/max(k_collector,1e-9))
    k_transfer += 0.25 * min(k_brace_effective, k_collector)

    ktheta = inp.connection_efficiency * arm2_sum * (k_transfer / max(k_sum, 1e-9))
    klat = inp.lateral_transfer_factor * ktheta / max(s.elevation**2, 1e-9)
    return ktheta, klat, k_transfer


def story_quantities(inp: BuildingInput, s: StorySection):
    total, nint, nper, ncor = column_counts(inp)
    slab_vol = inp.floor_area * s.slab_t
    beam_lines = inp.n_bays_y*(inp.n_bays_x+1) + inp.n_bays_x*(inp.n_bays_y+1)
    beam_vol = beam_lines * 0.5*(inp.bay_x+inp.bay_y) * s.beam_b * s.beam_h
    core_vol = 2*(s.core_x+s.core_y)*s.core_t*inp.story_height
    side_vol = 2*(s.side_wall_len_x+s.side_wall_len_y)*s.side_wall_t*inp.story_height
    col_vol = (nint*s.col_int**2 + nper*s.col_per**2 + ncor*s.col_cor**2)*inp.story_height
    concrete = slab_vol + beam_vol + core_vol + side_vol + col_vol

    q_super = (inp.DL + inp.finish_load + inp.live_load_mass_factor*inp.LL)*inp.floor_area
    facade = inp.facade_line_load * 2*(inp.plan_x+inp.plan_y)
    weight = concrete*25 + q_super + facade

    # approximate rebar + tube steel
    steel = concrete * 0.010 * RHO_STEEL
    if s.story in inp.outrigger_story_levels and inp.outrigger_system != OutriggerSystem.NONE:
        A = tube_area(inp.brace_D * inp.design_brace_scale, inp.brace_t * inp.design_brace_scale)
        for _, x0,y0,x1,y1 in brace_plan_lines(inp, s):
            L = sqrt((x1-x0)**2 + (y1-y0)**2 + inp.brace_vertical_depth**2)
            steel += A * L * RHO_STEEL

    weight += steel*G/1000
    mass = weight*1000/G
    return mass, weight, concrete, steel


def build_props(inp: BuildingInput, sections: List[StorySection]) -> List[StoryProperty]:
    E = inp.Ec * 1e6
    props = []
    for s in sections:
        Ix_core, Iy_core = rect_tube_I(s.core_x, s.core_y, s.core_t)
        Ix_core *= inp.wall_effective_I_factor
        Iy_core *= inp.wall_effective_I_factor
        Ix_side, Iy_side = sidewall_I(inp, s)
        Ix_col, Iy_col = column_I(inp, s)

        EI_x = E * (Iy_core + Iy_side + Iy_col) * inp.global_flexural_modifier
        EI_y = E * (Ix_core + Ix_side + Ix_col) * inp.global_flexural_modifier

        ktx, klx, _ = outrigger_stiffness(inp, s, Direction.X)
        kty, kly, _ = outrigger_stiffness(inp, s, Direction.Y)
        mass, weight, conc, steel = story_quantities(inp, s)

        props.append(StoryProperty(
            story=s.story, mass_kg=mass, weight_kN=weight, EI_x=EI_x, EI_y=EI_y,
            ktheta_x=ktx, ktheta_y=kty, klat_x=klx, klat_y=kly,
            concrete_m3=conc, steel_kg=steel
        ))
    return props


def beam_ke(EI, L):
    return EI/L**3 * np.array([[12,6*L,-12,6*L],
                               [6*L,4*L**2,-6*L,2*L**2],
                               [-12,-6*L,12,-6*L],
                               [6*L,2*L**2,-6*L,4*L**2]], dtype=float)


def assemble_mk(inp: BuildingInput, props: List[StoryProperty], direction: Direction):
    n = inp.n_story
    ndof = 2*(n+1)
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    L = inp.story_height

    for e in range(n):
        EI = props[e].EI_x if direction == Direction.X else props[e].EI_y
        ke = beam_ke(EI, L)
        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a,b]

    for node in range(1, n+1):
        p = props[node-1]
        u = 2*node
        th = 2*node+1
        M[u,u] += p.mass_kg
        M[th,th] += p.mass_kg * L**2 * 1e-5
        if direction == Direction.X:
            K[th,th] += p.ktheta_x
            K[u,u] += p.klat_x
        else:
            K[th,th] += p.ktheta_y
            K[u,u] += p.klat_y

    free = list(range(2, ndof))
    return M[np.ix_(free, free)], K[np.ix_(free, free)]


def solve_modal(inp: BuildingInput, props: List[StoryProperty], direction: Direction) -> ModalResult:
    M, K = assemble_mk(inp, props, direction)
    M = 0.5*(M+M.T)
    K = 0.5*(K+K.T)

    if scipy_eigh is not None:
        vals, vecs = scipy_eigh(K, M, check_finite=False)
    else:
        L = np.linalg.cholesky(M)
        Li = np.linalg.inv(L)
        A = Li @ K @ Li.T
        vals, y = np.linalg.eigh(0.5*(A+A.T))
        vecs = Li.T @ y

    keep = vals > 1e-8
    vals = np.real(vals[keep])
    vecs = np.real(vecs[:, keep])
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    nm = min(inp.n_modes, len(vals))
    vals = vals[:nm]
    vecs = vecs[:, :nm]
    omegas = np.sqrt(vals)
    periods = 2*pi/omegas
    freqs = omegas/(2*pi)

    n = inp.n_story
    r = np.zeros((2*n,1))
    r[0::2,0] = 1.0
    total_mass = sum(p.mass_kg for p in props)

    gammas, eff, cum, shapes = [], [], [], []
    running = 0.0
    for i in range(nm):
        phi = vecs[:,i].reshape(-1,1)
        den = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ r)/den).item()
        meff = gamma**2 * den
        ratio = meff/max(total_mass,1e-9)
        running += ratio
        shape = phi.flatten()[0::2]
        if abs(shape[-1]) > 1e-12:
            shape = shape/shape[-1]
        if shape[-1] < 0:
            shape = -shape
        gammas.append(gamma); eff.append(ratio); cum.append(running); shapes.append(shape)

    return ModalResult(direction, list(map(float,periods)), list(map(float,freqs)), omegas, shapes, gammas, eff, cum)


def asce_sa(T, a: ASCE7Params):
    T = max(float(T), 1e-9)
    Ts = a.SD1/max(a.SDS,1e-9)
    T0 = 0.2*Ts
    if T < T0:
        return a.SDS*(0.4+0.6*T/max(T0,1e-9))
    if T <= Ts:
        return a.SDS
    if T <= a.TL:
        return a.SD1/T
    return a.SD1*a.TL/T**2


def force_sa(T, a: ASCE7Params):
    return asce_sa(T,a)*a.Ie/a.R


def asce_Cs(T, a: ASCE7Params):
    R_I = a.R/a.Ie
    Cs = min(a.SDS/R_I, a.SD1/(max(T,1e-9)*R_I))
    Cs_min = max(0.044*a.SDS*a.Ie, 0.01)
    if a.S1 >= 0.6:
        Cs_min = max(Cs_min, 0.5*a.S1/R_I)
    return max(Cs, Cs_min)


def Ta(inp: BuildingInput):
    h_ft = inp.height*3.28084
    return inp.asce7.Ct * h_ft**inp.asce7.x_exp


def combine_modal(vals, omegas, method, zeta):
    vals = np.array(vals)
    if vals.ndim == 1:
        vals = vals.reshape((-1,1))
    if method == CombinationMethod.SRSS:
        return np.sqrt(np.sum(vals**2, axis=0))
    nm, nr = vals.shape
    out = np.zeros(nr)
    for k in range(nr):
        s = 0.0
        for i in range(nm):
            for j in range(nm):
                if abs(omegas[i]-omegas[j]) < 1e-12:
                    rho = 1.0
                else:
                    beta = omegas[j]/omegas[i]
                    rho = (8*zeta**2*beta**1.5)/max((1-beta**2)**2 + 4*zeta**2*beta*(1+beta)**2,1e-12)
                s += rho*vals[i,k]*vals[j,k]
        out[k] = sqrt(max(s,0.0))
    return out


def response_spectrum(inp: BuildingInput, props: List[StoryProperty], modal: ModalResult) -> RSAResult:
    """
    Response spectrum using normalized floor mode shapes.

    Important correction in v22:
    The plotted mode shapes are normalized to roof displacement = 1. Therefore,
    the participation factor used in RSA must be recomputed from these normalized
    floor shapes. Using the old participation factor from the full eigenvector
    together with normalized shapes can distort displacements and drifts.
    """
    n = inp.n_story
    h = inp.story_height
    masses = np.array([p.mass_kg for p in props], dtype=float)

    floor_forces, story_shears, disps, drifts, bases = [], [], [], [], []

    for i, T in enumerate(modal.periods_s):
        omega = modal.omegas[i]
        phi = np.array(modal.mode_shapes[i], dtype=float)

        # Consistent participation factor for normalized floor displacement shape.
        gamma = np.sum(masses * phi) / max(np.sum(masses * phi * phi), 1e-12)

        Sa = force_sa(T, inp.asce7) * G
        u = phi * gamma * Sa / omega**2
        f = masses * phi * gamma * Sa / 1000.0

        V = np.zeros(n)
        for j in range(n-1, -1, -1):
            V[j] = f[j] + (V[j+1] if j < n-1 else 0.0)

        drift = np.zeros(n)
        drift[0] = u[0]
        drift[1:] = np.diff(u)

        floor_forces.append(f)
        story_shears.append(V)
        disps.append(u)
        drifts.append(drift)
        bases.append(np.sum(f))

    floor = combine_modal(np.array(floor_forces), modal.omegas, inp.combination, inp.asce7.damping_ratio)
    shear = combine_modal(np.array(story_shears), modal.omegas, inp.combination, inp.asce7.damping_ratio)
    disp = combine_modal(np.array(disps), modal.omegas, inp.combination, inp.asce7.damping_ratio)
    drift = combine_modal(np.array(drifts), modal.omegas, inp.combination, inp.asce7.damping_ratio)
    base_un = combine_modal(np.array(bases), modal.omegas, inp.combination, inp.asce7.damping_ratio).item()

    total_mass = sum(p.mass_kg for p in props)
    ta = Ta(inp)
    cuta = inp.asce7.Cu * ta
    Tused = min(modal.periods_s[0], cuta)
    Cs = asce_Cs(Tused, inp.asce7)
    Velf = Cs * total_mass * G / 1000.0
    required = inp.asce7.rsa_min_ratio_to_elf * Velf
    scale = required / base_un if base_un < required and base_un > 1e-9 else 1.0

    floor *= scale
    shear *= scale
    base_sc = base_un * scale

    # ASCE displacement/drift amplification.
    disp_d = disp * inp.asce7.Cd / inp.asce7.Ie
    drift_d = drift * inp.asce7.Cd / inp.asce7.Ie

    return RSAResult(
        modal.direction,
        modal,
        floor,
        shear,
        disp_d,
        drift_d,
        drift_d / h,
        base_sc,
        base_un,
        Velf,
        scale,
        ta,
        cuta,
        Tused,
        Cs,
    )


def evaluate(inp: BuildingInput) -> DesignResult:
    sections = build_sections(inp)
    props = build_props(inp, sections)
    mx = solve_modal(inp, props, Direction.X)
    my = solve_modal(inp, props, Direction.Y)
    rx = response_spectrum(inp, props, mx)
    ry = response_spectrum(inp, props, my)
    return DesignResult(inp, sections, props, mx, my, rx, ry, pd.DataFrame())


def run_design(inp: BuildingInput) -> DesignResult:
    if not inp.auto_redesign:
        return evaluate(inp)
    wall = inp.design_wall_scale
    col = inp.design_column_scale
    beam = inp.design_beam_scale
    brace = inp.design_brace_scale
    logs = []
    best = None
    best_score = 1e99
    for it in range(1, inp.max_iterations+1):
        cur = replace(inp, design_wall_scale=wall, design_column_scale=col, design_beam_scale=beam, design_brace_scale=brace)
        res = evaluate(cur)
        md = max(float(np.max(res.rsa_x.drift_ratio)), float(np.max(res.rsa_y.drift_ratio)))
        mass = min(res.modal_x.cum_mass[-1], res.modal_y.cum_mass[-1])
        conc = sum(p.concrete_m3 for p in res.props)
        steel = sum(p.steel_kg for p in res.props)
        over = md/max(inp.drift_limit_ratio,1e-12)
        score = 2000*max(over-1,0)**2 + 300*max(inp.minimum_modal_mass_ratio-mass,0)**2 + 0.000001*conc + 0.0000002*steel
        logs.append({
            "Iteration": it, "Wall scale": wall, "Column scale": col, "Beam scale": beam, "Brace scale": brace,
            "T X (s)": res.modal_x.periods_s[0], "T Y (s)": res.modal_y.periods_s[0],
            "Max drift X": float(np.max(res.rsa_x.drift_ratio)), "Max drift Y": float(np.max(res.rsa_y.drift_ratio)),
            "Drift limit": inp.drift_limit_ratio, "Mass X (%)": 100*res.modal_x.cum_mass[-1], "Mass Y (%)": 100*res.modal_y.cum_mass[-1],
            "Base shear X (kN)": res.rsa_x.base_shear_scaled_kN, "Base shear Y (kN)": res.rsa_y.base_shear_scaled_kN,
            "Concrete (m3)": conc, "Steel (kg)": steel, "Beam h base": res.sections[0].beam_h,
            "Column base": res.sections[0].col_int, "Core wall base": res.sections[0].core_t,
        })
        if score < best_score:
            best_score, best = score, res
        if md <= inp.drift_limit_ratio*1.03 and mass >= inp.minimum_modal_mass_ratio:
            best = res
            break
        if over > 1:
            wall *= min(1.15, over**0.18)
            col *= min(1.12, over**0.12)
            beam *= min(1.10, over**0.10)
            brace *= min(1.25, over**0.20)
        elif over < 0.35:
            wall *= 0.96
            col *= 0.96
            beam *= 0.98
            brace *= 0.96
        wall = float(np.clip(wall,0.4,3.5))
        col = float(np.clip(col,0.4,3.5))
        beam = float(np.clip(beam,0.5,3.0))
        brace = float(np.clip(brace,0.4,4.0))
    best.iteration_table = pd.DataFrame(logs)
    return best


def no_outrigger_same_mass(inp: BuildingInput) -> BuildingInput:
    return replace(inp, connection_efficiency=0.0, auto_redesign=False)


def outrigger_effect_table(res: DesignResult):
    with_r = evaluate(res.inp)
    no_r = evaluate(no_outrigger_same_mass(res.inp))
    rows = []
    for d, rw, rn, mw, mn in [
        ("X", with_r.rsa_x, no_r.rsa_x, with_r.modal_x, no_r.modal_x),
        ("Y", with_r.rsa_y, no_r.rsa_y, with_r.modal_y, no_r.modal_y),
    ]:
        Tw, Tn = mw.periods_s[0], mn.periods_s[0]
        dw, dn = float(np.max(rw.drift_ratio)), float(np.max(rn.drift_ratio))
        rows.append({
            "Direction": d,
            "T with outrigger (s)": Tw,
            "T without outrigger stiffness (s)": Tn,
            "Period reduction (%)": 100*(Tn-Tw)/max(Tn,1e-9),
            "Drift with": dw,
            "Drift without outrigger stiffness": dn,
            "Drift reduction (%)": 100*(dn-dw)/max(dn,1e-12),
        })
    return pd.DataFrame(rows)


def outrigger_design_table(res: DesignResult):
    rows = []
    for s, p in zip(res.sections, res.props):
        if s.story in res.inp.outrigger_story_levels and res.inp.outrigger_system != OutriggerSystem.NONE:
            ktx, klx, ktr = outrigger_stiffness(res.inp, s, Direction.X)
            kty, kly, _ = outrigger_stiffness(res.inp, s, Direction.Y)
            rows.append({
                "Story": s.story,
                "Brace lines each side": res.inp.brace_lines_each_side,
                "Brace D (m)": res.inp.brace_D*res.inp.design_brace_scale,
                "Brace t (m)": res.inp.brace_t*res.inp.design_brace_scale,
                "Brace area (m2)": tube_area(res.inp.brace_D*res.inp.design_brace_scale, res.inp.brace_t*res.inp.design_brace_scale),
                "Ktheta X (GN.m/rad)": ktx/1e9,
                "Ktheta Y (GN.m/rad)": kty/1e9,
                "Klat X (MN/m)": klx/1e6,
                "Klat Y (MN/m)": kly/1e6,
                "Collector transfer stiffness (MN/m)": ktr/1e6,
                "Beam h at outrigger (m)": s.beam_h,
            })
    return pd.DataFrame(rows)



def sanity_check_table(res: DesignResult):
    H = res.inp.height
    T_x = res.modal_x.periods_s[0]
    T_y = res.modal_y.periods_s[0]
    Ta_val = Ta(res.inp)
    max_drift_x = float(np.max(res.rsa_x.drift_ratio))
    max_drift_y = float(np.max(res.rsa_y.drift_ratio))
    out_eff = outrigger_effect_table(res)

    # Very broad preliminary expectation, not a code limit.
    # Tall RC/core-outrigger towers can vary widely.
    lower_watch = 0.015 * H
    upper_watch = 0.12 * H

    rows = [
        {
            "Check": "T1 X rough engineering range",
            "Value": T_x,
            "Reference": f"{lower_watch:.2f} to {upper_watch:.2f} s",
            "Status": "WATCH" if T_x < lower_watch or T_x > upper_watch else "OK",
        },
        {
            "Check": "T1 Y rough engineering range",
            "Value": T_y,
            "Reference": f"{lower_watch:.2f} to {upper_watch:.2f} s",
            "Status": "WATCH" if T_y < lower_watch or T_y > upper_watch else "OK",
        },
        {
            "Check": "ASCE Ta",
            "Value": Ta_val,
            "Reference": "used only for ELF period cap, not actual period",
            "Status": "INFO",
        },
        {
            "Check": "Max drift X",
            "Value": max_drift_x,
            "Reference": res.inp.drift_limit_ratio,
            "Status": "OK" if max_drift_x <= res.inp.drift_limit_ratio else "NOT OK",
        },
        {
            "Check": "Max drift Y",
            "Value": max_drift_y,
            "Reference": res.inp.drift_limit_ratio,
            "Status": "OK" if max_drift_y <= res.inp.drift_limit_ratio else "NOT OK",
        },
    ]

    if not out_eff.empty:
        rows.append({
            "Check": "Outrigger period reduction X",
            "Value": float(out_eff.loc[out_eff["Direction"] == "X", "Period reduction (%)"].iloc[0]),
            "Reference": "should be visible if brace lines/area/collector are significant",
            "Status": "INFO",
        })

    return pd.DataFrame(rows)


def summary_table(res: DesignResult):
    return pd.DataFrame({
        "Item": ["Height","T1 X","T1 Y","Mass X","Mass Y","Max drift X","Max drift Y","Base shear X","Base shear Y","Concrete","Steel"],
        "Value": [res.inp.height,res.modal_x.periods_s[0],res.modal_y.periods_s[0],100*res.modal_x.cum_mass[-1],100*res.modal_y.cum_mass[-1],float(np.max(res.rsa_x.drift_ratio)),float(np.max(res.rsa_y.drift_ratio)),res.rsa_x.base_shear_scaled_kN,res.rsa_y.base_shear_scaled_kN,sum(p.concrete_m3 for p in res.props),sum(p.steel_kg for p in res.props)],
        "Unit": ["m","s","s","%","%","-","-","kN","kN","m3","kg"]
    })


def final_dimensions_table(res: DesignResult):
    idxs = [0, len(res.sections)//2, len(res.sections)-1]
    names = ["Lower", "Middle", "Upper"]
    rows = []
    for name, i in zip(names, idxs):
        s = res.sections[i]
        rows.append({
            "Zone": name, "Story": s.story, "Core": f"{s.core_x:.2f} x {s.core_y:.2f}",
            "Core wall t": s.core_t, "Side wall t": s.side_wall_t,
            "Interior column": s.col_int, "Perimeter column": s.col_per, "Corner column": s.col_cor,
            "Beam": f"{s.beam_b:.2f} x {s.beam_h:.2f}", "Slab t": s.slab_t
        })
    return pd.DataFrame(rows)


def modal_table(m: ModalResult):
    return pd.DataFrame({
        "Mode": list(range(1,len(m.periods_s)+1)),
        "Period (s)": m.periods_s,
        "Frequency (Hz)": m.frequencies_hz,
        "Effective mass (%)": [100*x for x in m.eff_mass],
        "Cumulative mass (%)": [100*x for x in m.cum_mass],
    })


def story_table(res: DesignResult, direction: Direction):
    rsa = res.rsa_x if direction == Direction.X else res.rsa_y
    return pd.DataFrame({
        "Story": [s.story for s in res.sections],
        "Elevation": [s.elevation for s in res.sections],
        "Story shear (kN)": rsa.story_shear_kN,
        "Displacement (m)": rsa.displacement_m,
        "Drift (m)": rsa.drift_m,
        "Drift ratio": rsa.drift_ratio,
    })


def stiffness_table(res: DesignResult):
    return pd.DataFrame({
        "Story": [p.story for p in res.props],
        "EI X (GN.m2)": [p.EI_x/1e9 for p in res.props],
        "EI Y (GN.m2)": [p.EI_y/1e9 for p in res.props],
        "Ktheta X (GN.m/rad)": [p.ktheta_x/1e9 for p in res.props],
        "Ktheta Y (GN.m/rad)": [p.ktheta_y/1e9 for p in res.props],
        "Klat X (MN/m)": [p.klat_x/1e6 for p in res.props],
        "Klat Y (MN/m)": [p.klat_y/1e6 for p in res.props],
    })


def plot_plan(res: DesignResult, story: int):
    inp = res.inp
    s = res.sections[min(max(story,1),inp.n_story)-1]
    fig, ax = plt.subplots(figsize=(12,9))
    ax.plot([-inp.plan_x/2, inp.plan_x/2, inp.plan_x/2, -inp.plan_x/2, -inp.plan_x/2],
            [-inp.plan_y/2, -inp.plan_y/2, inp.plan_y/2, inp.plan_y/2, -inp.plan_y/2], color="black")
    for i in range(inp.n_bays_x+1):
        x = -inp.plan_x/2 + i*inp.bay_x
        ax.plot([x,x],[-inp.plan_y/2,inp.plan_y/2], color="#dddddd")
    for j in range(inp.n_bays_y+1):
        y = -inp.plan_y/2 + j*inp.bay_y
        ax.plot([-inp.plan_x/2,inp.plan_x/2],[y,y], color="#dddddd")
    for x,y,typ in column_grid(inp):
        d = s.col_cor if typ=="corner" else (s.col_per if typ=="perimeter" else s.col_int)
        c = "#8b0000" if typ=="corner" else ("#cc5500" if typ=="perimeter" else "#4444aa")
        ax.add_patch(plt.Rectangle((x-d/2,y-d/2),d,d,facecolor=c,edgecolor="black",alpha=0.9))
    ax.add_patch(plt.Rectangle((-s.core_x/2,-s.core_y/2),s.core_x,s.core_y,fill=False,edgecolor="#2e8b57",linewidth=3))
    ax.plot([-s.side_wall_len_x/2,s.side_wall_len_x/2],[inp.plan_y/2,inp.plan_y/2],color="#4caf50",linewidth=5)
    ax.plot([-s.side_wall_len_x/2,s.side_wall_len_x/2],[-inp.plan_y/2,-inp.plan_y/2],color="#4caf50",linewidth=5)
    ax.plot([inp.plan_x/2,inp.plan_x/2],[-s.side_wall_len_y/2,s.side_wall_len_y/2],color="#4caf50",linewidth=5)
    ax.plot([-inp.plan_x/2,-inp.plan_x/2],[-s.side_wall_len_y/2,s.side_wall_len_y/2],color="#4caf50",linewidth=5)

    if s.story in inp.outrigger_story_levels and inp.outrigger_system != OutriggerSystem.NONE:
        ax.plot([-inp.plan_x/2, inp.plan_x/2, inp.plan_x/2, -inp.plan_x/2, -inp.plan_x/2],
                [-inp.plan_y/2, -inp.plan_y/2, inp.plan_y/2, inp.plan_y/2, -inp.plan_y/2], color="#ff6b00", linewidth=3)
        for d,x0,y0,x1,y1 in brace_plan_lines(inp,s):
            ax.plot([x0,x1],[y0,y1], color="#b34700", linewidth=3, linestyle="--")
            ax.plot([x0,x1],[y0+0.4,y1+0.4], color="#b34700", linewidth=1.5, linestyle=":")
        ax.text(inp.plan_x/2+2,0,f"OUTRIGGER STORY {s.story}\nbrace lines/side={inp.brace_lines_each_side}\nD={inp.brace_D*inp.design_brace_scale:.2f} m",color="#b34700",fontweight="bold")
    else:
        ax.text(inp.plan_x/2+2,0,"No outrigger at this story", color="#666666")

    ax.set_title(f"Plan and brace load path - Story {s.story}")
    ax.set_aspect("equal")
    ax.set_xlim(-inp.plan_x/2-5, inp.plan_x/2+30)
    ax.set_ylim(inp.plan_y/2+5, -inp.plan_y/2-5)
    return fig


def plot_modes(res: DesignResult, direction: Direction):
    m = res.modal_x if direction == Direction.X else res.modal_y
    y = np.array([s.elevation for s in res.sections])
    nm = min(5, len(m.mode_shapes))
    fig, axes = plt.subplots(1,nm,figsize=(16,6))
    if nm == 1:
        axes = [axes]
    for i in range(nm):
        axes[i].plot(m.mode_shapes[i], y)
        axes[i].scatter(m.mode_shapes[i], y, s=10)
        for lev in res.inp.outrigger_story_levels:
            axes[i].axhline(lev*res.inp.story_height, linestyle=":", alpha=0.5)
        axes[i].set_title(f"Mode {i+1}\nT={m.periods_s[i]:.2f}s")
        axes[i].grid(True, alpha=0.3)
    return fig


def plot_story(res: DesignResult, direction: Direction, kind: str):
    df = story_table(res,direction)
    fig, ax = plt.subplots(figsize=(7,8))
    if kind == "Drift":
        ax.plot(df["Drift ratio"], df["Story"])
        ax.axvline(res.inp.drift_limit_ratio, linestyle="--", label="limit")
        ax.set_xlabel("Drift ratio")
        ax.legend()
    else:
        ax.plot(df["Story shear (kN)"], df["Story"])
        ax.set_xlabel("Story shear (kN)")
    ax.set_ylabel("Story")
    ax.grid(True, alpha=0.3)
    return fig


def plot_iteration(res: DesignResult):
    df = res.iteration_table
    if df.empty:
        return None
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(df["Iteration"],df["T X (s)"],marker="o",label="X")
    axes[0,0].plot(df["Iteration"],df["T Y (s)"],marker="s",label="Y")
    axes[0,0].legend(); axes[0,0].set_title("Period")
    axes[0,1].plot(df["Iteration"],df["Max drift X"],marker="o",label="X")
    axes[0,1].plot(df["Iteration"],df["Max drift Y"],marker="s",label="Y")
    axes[0,1].plot(df["Iteration"],df["Drift limit"],linestyle="--",label="limit")
    axes[0,1].legend(); axes[0,1].set_title("Drift")
    axes[1,0].plot(df["Iteration"],df["Wall scale"],label="wall")
    axes[1,0].plot(df["Iteration"],df["Column scale"],label="column")
    axes[1,0].plot(df["Iteration"],df["Beam scale"],label="beam")
    axes[1,0].plot(df["Iteration"],df["Brace scale"],label="brace")
    axes[1,0].legend(); axes[1,0].set_title("Scale factors")
    axes[1,1].plot(df["Iteration"],df["Concrete (m3)"],marker="d")
    axes[1,1].set_title("Concrete")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def principles_table():
    return pd.DataFrame([
        {"Principle":"Period","Rule":"Computed from Kφ=ω²Mφ; not forced as target."},
        {"Principle":"Outrigger brace plan","Rule":"Brace lines are generated in plan; EA/L is calculated for each line."},
        {"Principle":"Braced bay count","Rule":"brace_lines_each_side directly changes stiffness, steel, and plan drawing."},
        {"Principle":"Load path","Rule":"Brace axial stiffness + collector beam stiffness + exterior column participation."},
        {"Principle":"Redesign","Rule":"Walls, columns, beams, and braces are resized when drift is not acceptable."},
        {"Principle":"Monotonic columns","Rule":"Lower columns are forced to be >= upper columns."},
    ])


def build_report(res: DesignResult):
    parts = [
        "TOWER PREDESIGN REPORT - " + APP_VERSION,
        "\nPRINCIPLES\n" + principles_table().to_string(index=False),
        "\nSANITY CHECKS\n" + sanity_check_table(res).to_string(index=False),
        "\nSUMMARY\n" + summary_table(res).to_string(index=False),
        "\nFINAL DIMENSIONS\n" + final_dimensions_table(res).to_string(index=False),
        "\nOUTRIGGER EFFECT\n" + outrigger_effect_table(res).to_string(index=False),
        "\nOUTRIGGER DESIGN\n" + outrigger_design_table(res).to_string(index=False),
        "\nMODAL X\n" + modal_table(res.modal_x).to_string(index=False),
        "\nMODAL Y\n" + modal_table(res.modal_y).to_string(index=False),
    ]
    return "\n\n".join(parts)


def make_input():
    import streamlit as st
    st.sidebar.header("Geometry")
    n_story = st.sidebar.number_input("Stories", 5, 150, 60)
    story_height = st.sidebar.number_input("Story height (m)", 2.5, 6.0, 3.2)
    plan_x = st.sidebar.number_input("Plan X (m)", 20.0, 200.0, 80.0)
    plan_y = st.sidebar.number_input("Plan Y (m)", 20.0, 200.0, 80.0)
    n_bays_x = st.sidebar.number_input("Bays X", 1, 30, 8)
    n_bays_y = st.sidebar.number_input("Bays Y", 1, 30, 8)

    st.sidebar.header("Core and stiffness")
    core_ratio_x = st.sidebar.number_input("Core ratio X", 0.10, 0.50, 0.24)
    core_ratio_y = st.sidebar.number_input("Core ratio Y", 0.10, 0.50, 0.22)
    wall_eff = st.sidebar.number_input("Wall effective I", 0.02, 1.0, 0.18)
    col_eff = st.sidebar.number_input("Column effective I", 0.02, 1.0, 0.35)
    global_mod = st.sidebar.number_input("Global flexural modifier", 0.05, 1.5, 0.60)

    st.sidebar.header("Sections")
    core_wall_t_base = st.sidebar.number_input("Core wall base t (m)", 0.2, 3.0, 0.85)
    core_wall_t_top = st.sidebar.number_input("Core wall top t (m)", 0.15, 2.0, 0.35)
    column_dim_base = st.sidebar.number_input("Column base dim (m)", 0.3, 5.0, 1.80)
    column_dim_top = st.sidebar.number_input("Column top dim (m)", 0.2, 3.0, 0.80)
    beam_h_base = st.sidebar.number_input("Beam base h (m)", 0.3, 4.0, 1.20)
    beam_h_top = st.sidebar.number_input("Beam top h (m)", 0.3, 3.0, 0.75)

    st.sidebar.header("Outrigger")
    use_out = st.sidebar.checkbox("Use outrigger", True)
    levels_text = st.sidebar.text_input("Outrigger stories comma separated", "30,42,54")
    levels = tuple(int(x.strip()) for x in levels_text.split(",") if x.strip().isdigit())
    brace_lines_each_side = st.sidebar.number_input("Brace lines each side", 1, 10, 2)
    brace_D = st.sidebar.number_input("Brace D (m)", 0.1, 3.0, 0.80)
    brace_t = st.sidebar.number_input("Brace t (m)", 0.005, 0.20, 0.030)
    belt = st.sidebar.number_input("Belt beam factor at outrigger", 1.0, 3.0, 1.50)
    ext_part = st.sidebar.number_input("Exterior column participation", 0.1, 1.5, 0.70)
    conn = st.sidebar.number_input("Connection efficiency", 0.1, 1.0, 0.75)
    latf = st.sidebar.number_input("Lateral transfer factor", 0.0, 5.0, 1.00)

    st.sidebar.header("ASCE and criteria")
    drift_limit = st.sidebar.number_input("Drift limit", 0.001, 0.05, 0.015, format="%.4f")
    n_modes = st.sidebar.number_input("Modes", 1, 60, 12)
    max_iter = st.sidebar.number_input("Max redesign iterations", 1, 30, 10)
    auto = st.sidebar.checkbox("Auto redesign", True)

    system = OutriggerSystem.TUBULAR_BRACE if use_out else OutriggerSystem.NONE

    return BuildingInput(
        n_story=int(n_story), story_height=float(story_height), plan_x=float(plan_x), plan_y=float(plan_y),
        n_bays_x=int(n_bays_x), n_bays_y=int(n_bays_y), core_ratio_x=float(core_ratio_x), core_ratio_y=float(core_ratio_y),
        wall_effective_I_factor=float(wall_eff), column_effective_I_factor=float(col_eff),
        side_wall_effective_I_factor=float(wall_eff), global_flexural_modifier=float(global_mod),
        core_wall_t_base=float(core_wall_t_base), core_wall_t_top=float(core_wall_t_top),
        column_dim_base=float(column_dim_base), column_dim_top=float(column_dim_top),
        beam_h_base=float(beam_h_base), beam_h_top=float(beam_h_top),
        outrigger_system=system, outrigger_story_levels=levels,
        brace_lines_each_side=int(brace_lines_each_side), brace_D=float(brace_D), brace_t=float(brace_t),
        belt_beam_factor=float(belt), exterior_column_participation=float(ext_part),
        connection_efficiency=float(conn), lateral_transfer_factor=float(latf),
        drift_limit_ratio=float(drift_limit), n_modes=int(n_modes), max_iterations=int(max_iter), auto_redesign=bool(auto),
    )


def main():
    import streamlit as st
    st.set_page_config(page_title="Brace-plan Outrigger MDOF v21", layout="wide")
    st.title("Brace-plan Outrigger Tower Predesign - v21")
    st.caption(APP_VERSION)
    st.info("This version calculates outrigger stiffness from the actual brace plan: brace lines, EA/L, collector stiffness, exterior column participation, and equivalent MDOF stiffness.")

    inp = make_input()
    if "v21_res" not in st.session_state:
        st.session_state.v21_res = None

    if st.button("ANALYZE", type="primary"):
        with st.spinner("Running brace-plan MDOF solver..."):
            st.session_state.v21_res = run_design(inp)

    res = st.session_state.v21_res
    if res is None:
        st.info("Set inputs and click ANALYZE.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("T1 X", f"{res.modal_x.periods_s[0]:.2f} s")
    c2.metric("T1 Y", f"{res.modal_y.periods_s[0]:.2f} s")
    c3.metric("Max drift X", f"{np.max(res.rsa_x.drift_ratio):.5f}")
    c4.metric("Max drift Y", f"{np.max(res.rsa_y.drift_ratio):.5f}")

    story_choice = st.selectbox("Plan story", list(range(1,res.inp.n_story+1)), index=min(res.inp.outrigger_story_levels[0]-1 if res.inp.outrigger_story_levels else 0, res.inp.n_story-1))

    tabs = st.tabs(["Principles","Sanity checks","Summary","Final dimensions","Outrigger effect","Outrigger design","Plan","Modes X","Modes Y","Story X","Story Y","Stiffness","Iteration","Report"])

    with tabs[0]:
        st.dataframe(principles_table(), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(sanity_check_table(res), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(summary_table(res), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(final_dimensions_table(res), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(outrigger_effect_table(res), use_container_width=True, hide_index=True)
    with tabs[5]:
        st.dataframe(outrigger_design_table(res), use_container_width=True, hide_index=True)
    with tabs[6]:
        st.pyplot(plot_plan(res, int(story_choice)), use_container_width=True)
    with tabs[7]:
        st.pyplot(plot_modes(res, Direction.X), use_container_width=True)
        st.dataframe(modal_table(res.modal_x), use_container_width=True, hide_index=True)
    with tabs[8]:
        st.pyplot(plot_modes(res, Direction.Y), use_container_width=True)
        st.dataframe(modal_table(res.modal_y), use_container_width=True, hide_index=True)
    with tabs[9]:
        p1,p2 = st.columns(2)
        with p1: st.pyplot(plot_story(res, Direction.X, "Shear"), use_container_width=True)
        with p2: st.pyplot(plot_story(res, Direction.X, "Drift"), use_container_width=True)
        st.dataframe(story_table(res, Direction.X), use_container_width=True, hide_index=True)
    with tabs[10]:
        p1,p2 = st.columns(2)
        with p1: st.pyplot(plot_story(res, Direction.Y, "Shear"), use_container_width=True)
        with p2: st.pyplot(plot_story(res, Direction.Y, "Drift"), use_container_width=True)
        st.dataframe(story_table(res, Direction.Y), use_container_width=True, hide_index=True)
    with tabs[11]:
        st.dataframe(stiffness_table(res), use_container_width=True, hide_index=True)
    with tabs[12]:
        fig = plot_iteration(res)
        if fig: st.pyplot(fig, use_container_width=True)
        st.dataframe(res.iteration_table, use_container_width=True, hide_index=True)
    with tabs[13]:
        rep = build_report(res)
        st.text_area("Report", rep, height=600)
        st.download_button("Download report", rep.encode("utf-8"), "v22_report.txt")


if __name__ == "__main__":
    main()
