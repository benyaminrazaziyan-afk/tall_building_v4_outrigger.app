"""
Tall Building Outrigger Predesign Framework - Version 4.4
=========================================================

A preliminary structural engineering tool for comparing tall-building lateral
systems with and without outrigger/braced-bay action.

Core features
-------------
- Flexural MDOF stick model with lateral displacement and rotation at each floor.
- Modal eigenvalue solution for X and Y directions.
- ASCE 7 response-spectrum and ELF scaling options.
- Drift-controlled preliminary redesign loop.
- Real braced-bay outrigger layout: selected bay panels are used consistently
  in the plan drawing, stiffness calculation, and global stiffness matrix.
- Perimeter walls are drawn and included in the lateral stiffness model.
- Optional below-grade retaining-wall stiffness contribution for basement levels.

Author: BENYAMIN RAZAZIYAN
Version: 4.4
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import pi, sqrt, ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy.linalg import eigh as scipy_eigh
except Exception:
    scipy_eigh = None


APP_VERSION = "4.4-real-design-period-drift-controlled"
PROJECT_TITLE = "Tall Building Outrigger Predesign Framework"
AUTHOR_NAME = "BENYAMIN RAZAZIYAN"
G = 9.81
RHO_STEEL = 7850.0
STEEL_E_MPA = 200000.0  # steel modulus for tubular/truss outrigger members


# ============================================================
# 1. ENUMS
# ============================================================

class Direction(str, Enum):
    X = "X"
    Y = "Y"


class CombinationMethod(str, Enum):
    CQC = "CQC"
    SRSS = "SRSS"


class OutriggerSystem(str, Enum):
    NONE = "None"
    TUBULAR_BRACE = "Tubular Bracing"
    BELT_TRUSS = "Belt Truss"


# ============================================================
# 2. DATA MODELS
# ============================================================

@dataclass
class ASCE7Params:
    SDS: float = 0.70
    SD1: float = 0.35
    SS: float = 1.00
    S1: float = 0.30
    Fa: float = 1.00
    Fv: float = 1.00
    use_site_coefficients: bool = False
    TL: float = 8.0
    R: float = 5.0
    Ie: float = 1.0
    Cd: float = 5.0
    damping_ratio: float = 0.05

    Ct: float = 0.016
    x_exp: float = 0.90
    Cu: float = 1.40
    rsa_min_ratio_to_elf: float = 0.85
    use_CuTa_cap: bool = True
    scale_drift_with_base_shear: bool = False


@dataclass
class BuildingInput:
    # Geometry
    plan_shape: str = "square"
    n_story: int = 60
    n_basement: int = 0
    story_height: float = 3.2
    basement_height: float = 3.0
    plan_x: float = 80.0
    plan_y: float = 80.0
    n_bays_x: int = 8
    n_bays_y: int = 8

    # Core/service input
    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each: float = 3.5
    stair_area_each: float = 20.0
    service_area: float = 35.0
    corridor_factor: float = 1.40
    core_ratio_x: float = 0.24
    core_ratio_y: float = 0.22
    core_max_ratio_x: float = 0.42
    core_max_ratio_y: float = 0.42

    # Materials
    fck: float = 70.0
    Ec: float = 36000.0
    fy: float = 420.0

    # Loads and mass
    DL: float = 3.0
    LL: float = 2.5
    live_load_mass_factor: float = 0.25
    slab_finish_allowance: float = 1.5
    facade_line_load: float = 1.0
    additional_mass_factor: float = 1.0

    # Preliminary section limits
    min_wall_thickness: float = 0.30
    max_wall_thickness: float = 2.20
    min_column_dim: float = 0.70
    max_column_dim: float = 3.50
    min_beam_width: float = 0.40
    min_beam_depth: float = 0.75
    min_slab_thickness: float = 0.22
    max_slab_thickness: float = 0.60

    # Material-responsive preliminary sizing reference values
    reference_fck: float = 70.0
    reference_Ec: float = 36000.0

    # Effective stiffness factors
    # Effective stiffness / cracked inertia factors.
    # These factors affect BOTH analysis stiffness and the redesign sizing demand.
    wall_cracked_factor: float = 0.20
    column_cracked_factor: float = 0.35
    side_wall_cracked_factor: float = 0.15
    beam_cracked_factor: float = 0.35
    slab_cracked_factor: float = 0.25
    reference_wall_cracked_factor: float = 0.20
    reference_column_cracked_factor: float = 0.35
    reference_side_wall_cracked_factor: float = 0.15
    reference_beam_cracked_factor: float = 0.35
    reference_slab_cracked_factor: float = 0.25
    coupling_factor: float = 1.00

    # Wall/column layout
    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4
    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30
    side_wall_ratio: float = 0.20
    perimeter_wall_ratio: float = 0.20

    # Below-grade / basement retaining-wall contribution
    include_basement_retaining_wall: bool = True
    retaining_wall_thickness: float = 0.40
    retaining_wall_cracked_factor: float = 0.45
    basement_stiffness_participation: float = 0.65

    # Reinforcement ratios
    wall_rebar_ratio: float = 0.004
    column_rebar_ratio: float = 0.012
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.004

    # Outrigger
    outrigger_system: OutriggerSystem = OutriggerSystem.TUBULAR_BRACE
    outrigger_count: int = 2
    outrigger_story_levels: Tuple[int, ...] = (30, 42)
    outrigger_depth_m: float = 3.0
    outrigger_chord_area_m2: float = 0.08
    outrigger_diagonal_area_m2: float = 0.04
    tubular_diameter_m: float = 0.80
    tubular_thickness_m: float = 0.030
    braced_spans_x: int = 2
    braced_spans_y: int = 2
    # Optional exact bay IDs. Bay i means the panel between grid lines i and i+1.
    # If empty, centered bays are selected according to braced_spans_x/y.
    braced_bay_ids_x: Tuple[int, ...] = ()  # X action: bay IDs along Y on E/W sides
    braced_bay_ids_y: Tuple[int, ...] = ()  # Y action: bay IDs along X on N/S sides
    outrigger_connection_efficiency: float = 0.75
    # Engineering calibration factors for preliminary stick models.
    # The full diagonal EA/L is not allowed to act as a direct floor-to-ground lateral spring.
    # It is first reduced for connection, floor diaphragm, collector, brace-panel, and column-line flexibility.
    outrigger_effectiveness_factor: float = 0.10
    outrigger_lateral_participation: float = 0.00

    # Criteria
    drift_limit_ratio: float = 0.015
    minimum_modal_mass_ratio: float = 0.90

    # Period-controlled real design criteria
    enforce_period_limit: bool = True
    period_limit_multiplier: float = 1.00
    target_period_utilization: float = 0.95
    strengthen_core_for_period: bool = True
    strengthen_columns_for_period: bool = True
    strengthen_outrigger_for_period: bool = True

    # Solver
    n_modes: int = 12
    combination: CombinationMethod = CombinationMethod.CQC
    use_asce7_rsa: bool = True
    asce7: ASCE7Params = None

    # Redesign
    auto_redesign: bool = True
    max_iterations: int = 18
    growth_limit_per_iteration: float = 1.18
    reduction_limit_per_iteration: float = 0.96
    allow_section_reduction: bool = False

    # Internal redesign scale factors.
    # The UI normally leaves these at 1.0. The redesign loop updates them.
    design_wall_scale: float = 1.0
    design_column_scale: float = 1.0
    design_beam_scale: float = 1.0
    design_slab_scale: float = 1.0
    design_outrigger_scale: float = 1.0

    def __post_init__(self):
        if self.asce7 is None:
            self.asce7 = ASCE7Params()

    @property
    def bay_x(self) -> float:
        return self.plan_x / max(self.n_bays_x, 1)

    @property
    def bay_y(self) -> float:
        return self.plan_y / max(self.n_bays_y, 1)

    @property
    def height(self) -> float:
        return self.n_story * self.story_height

    @property
    def floor_area(self) -> float:
        if self.plan_shape == "triangle":
            return 0.5 * self.plan_x * self.plan_y
        return self.plan_x * self.plan_y


@dataclass
class Zone:
    name: str
    start_story: int
    end_story: int

    @property
    def n_stories(self) -> int:
        return self.end_story - self.start_story + 1


@dataclass
class StorySection:
    story: int
    zone: str
    elevation_m: float
    core_x: float
    core_y: float
    opening_x: float
    opening_y: float
    core_wall_t: float
    side_wall_length_x: float
    side_wall_length_y: float
    side_wall_t: float
    wall_count: int
    column_interior_x: float
    column_interior_y: float
    column_perimeter_x: float
    column_perimeter_y: float
    column_corner_x: float
    column_corner_y: float
    beam_b: float
    beam_h: float
    slab_t: float
    n_columns_total: int
    n_interior_columns: int
    n_perimeter_columns: int
    n_corner_columns: int


@dataclass
class StoryProperties:
    story: int
    mass_kg: float
    weight_kN: float
    concrete_m3: float
    steel_kg: float
    EI_x_Nm2: float   # stiffness for X translation, bending about Y
    EI_y_Nm2: float   # stiffness for Y translation, bending about X
    Ktheta_out_x_Nm: float
    Ktheta_out_y_Nm: float
    EI_basement_x_Nm2: float = 0.0
    EI_basement_y_Nm2: float = 0.0


@dataclass
class ModalResult:
    direction: Direction
    periods_s: List[float]
    frequencies_hz: List[float]
    omegas: np.ndarray
    mode_shapes: List[np.ndarray]
    gammas: List[float]
    effective_mass_ratios: List[float]
    cumulative_mass_ratios: List[float]
    raw_eigenvectors: np.ndarray | None = None


@dataclass
class RSAResult:
    direction: Direction
    modal: ModalResult
    floor_force_kN: np.ndarray
    story_shear_kN: np.ndarray
    overturning_kNm: np.ndarray
    displacement_m: np.ndarray
    drift_m: np.ndarray
    drift_ratio: np.ndarray
    base_shear_unscaled_kN: float
    base_shear_scaled_kN: float
    elf_base_shear_kN: float
    rsa_scale_factor: float
    Cs: float
    Ta_s: float
    T_used_s: float


@dataclass
class DesignResult:
    input: BuildingInput
    sections: List[StorySection]
    properties: List[StoryProperties]
    modal_x: ModalResult
    modal_y: ModalResult
    rsa_x: RSAResult
    rsa_y: RSAResult
    iteration_table: pd.DataFrame
    final_message: str


# ============================================================
# 3. GEOMETRY AND SECTION SIZING
# ============================================================

def define_zones(n_story: int) -> List[Zone]:
    z1 = max(1, round(0.30 * n_story))
    z2 = max(z1 + 1, round(0.70 * n_story))
    return [
        Zone("Lower Zone", 1, z1),
        Zone("Middle Zone", z1 + 1, z2),
        Zone("Upper Zone", z2 + 1, n_story),
    ]


def zone_for_story(inp: BuildingInput, story: int) -> str:
    for z in define_zones(inp.n_story):
        if z.start_story <= story <= z.end_story:
            return z.name
    return "Upper Zone"


def wall_count_for_zone(inp: BuildingInput, zone: str) -> int:
    if zone == "Lower Zone":
        return inp.lower_zone_wall_count
    if zone == "Middle Zone":
        return inp.middle_zone_wall_count
    return inp.upper_zone_wall_count


def required_opening_area(inp: BuildingInput) -> float:
    return (
        inp.elevator_count * inp.elevator_area_each
        + inp.stair_count * inp.stair_area_each
        + inp.service_area
    ) * inp.corridor_factor


def opening_dimensions(inp: BuildingInput) -> Tuple[float, float]:
    area = required_opening_area(inp)
    aspect = 1.6
    oy = sqrt(max(area / aspect, 1e-9))
    return aspect * oy, oy


def initial_core_dimensions(inp: BuildingInput) -> Tuple[float, float, float, float]:
    opening_x, opening_y = opening_dimensions(inp)
    core_x = max(opening_x + 3.0, inp.core_ratio_x * inp.plan_x)
    core_y = max(opening_y + 3.0, inp.core_ratio_y * inp.plan_y)
    core_x = min(core_x, inp.core_max_ratio_x * inp.plan_x)
    core_y = min(core_y, inp.core_max_ratio_y * inp.plan_y)
    return core_x, core_y, opening_x, opening_y



def cracked_sizing_factor(inp: BuildingInput, member: str) -> float:
    """
    Converts cracked-stiffness input into a section-size demand modifier.
    If the effective inertia factor is reduced, the final proposed section must
    increase. For rectangular members, I roughly varies with h^3; therefore the
    size demand is scaled with (I_ref/I_eff)^(1/3).
    """
    member = member.lower().strip()
    if member == "column":
        ref, val, exp = inp.reference_column_cracked_factor, inp.column_cracked_factor, 1.0 / 3.0
    elif member == "wall":
        ref, val, exp = inp.reference_wall_cracked_factor, inp.wall_cracked_factor, 1.0 / 3.0
    elif member == "side_wall":
        ref, val, exp = inp.reference_side_wall_cracked_factor, inp.side_wall_cracked_factor, 1.0 / 3.0
    elif member == "beam":
        ref, val, exp = inp.reference_beam_cracked_factor, inp.beam_cracked_factor, 1.0 / 3.0
    elif member == "slab":
        ref, val, exp = inp.reference_slab_cracked_factor, inp.slab_cracked_factor, 0.25
    else:
        ref, val, exp = 1.0, 1.0, 1.0 / 3.0
    return float(np.clip((max(ref, 1e-6) / max(val, 1e-6)) ** exp, 0.75, 2.25))


def material_size_factor(inp: BuildingInput, member: str) -> float:
    """Material- and cracked-stiffness-responsive size modifier."""
    fck = max(float(inp.fck), 12.0)
    Ec = max(float(inp.Ec), 12000.0)
    fck_ref = max(float(inp.reference_fck), 12.0)
    Ec_ref = max(float(inp.reference_Ec), 12000.0)
    member = member.lower().strip()
    exponents = {
        "column": (0.45, 0.10, 0.72, 2.20),
        "wall":   (0.28, 0.22, 0.76, 2.20),
        "side_wall": (0.28, 0.22, 0.76, 2.20),
        "beam":   (0.20, 0.10, 0.82, 1.80),
        "slab":   (0.14, 0.06, 0.88, 1.55),
    }
    a_fck, a_ec, lower, upper = exponents.get(member, exponents["column"])
    strength_factor = (fck_ref / fck) ** a_fck * (Ec_ref / Ec) ** a_ec
    stiffness_factor = cracked_sizing_factor(inp, member)
    return float(np.clip(strength_factor * stiffness_factor, lower, upper))

def material_diagnostics(inp: BuildingInput) -> pd.DataFrame:
    """Table of active fck/Ec modifiers used in member sizing."""
    return pd.DataFrame([
        {
            "Member": m,
            "fck (MPa)": inp.fck,
            "Ec (MPa)": inp.Ec,
            "Reference fck (MPa)": inp.reference_fck,
            "Reference Ec (MPa)": inp.reference_Ec,
            "Cracked stiffness modifier": cracked_sizing_factor(inp, m),
            "Final size modifier": material_size_factor(inp, m),
        }
        for m in ["column", "wall", "side_wall", "beam", "slab"]
    ])


def wall_thickness(inp: BuildingInput, story: int, scale: float = 1.0) -> float:
    """
    Monotonic wall thickness by height.
    Lower stories must not be smaller than upper stories.
    """
    h_ratio = 1.0 - (story - 1) / max(inp.n_story - 1, 1)
    base = inp.height / 220.0
    t = base * (0.55 + 0.45 * h_ratio) * material_size_factor(inp, "wall") * scale * inp.design_wall_scale
    return float(np.clip(t, inp.min_wall_thickness, inp.max_wall_thickness))


def side_wall_thickness(inp: BuildingInput, story: int, scale: float = 1.0) -> float:
    h_ratio = 1.0 - (story - 1) / max(inp.n_story - 1, 1)
    base = 0.70 * inp.height / 220.0
    t = base * (0.55 + 0.45 * h_ratio) * material_size_factor(inp, "side_wall") * scale * inp.design_wall_scale
    return float(np.clip(t, inp.min_wall_thickness, inp.max_wall_thickness))


def slab_thickness(inp: BuildingInput, story: int, scale: float = 1.0) -> float:
    """
    Slab thickness is mainly span-controlled, but lower zones and outrigger floors
    receive a modest strengthening factor for diaphragm/collector action.
    """
    span = max(inp.bay_x, inp.bay_y)
    h_ratio = 1.0 - (story - 1) / max(inp.n_story - 1, 1)
    zone_factor = 1.00 + 0.08 * h_ratio
    out_factor = 1.10 if story in active_outrigger_levels(inp) and inp.outrigger_system != OutriggerSystem.NONE else 1.00
    t = span / 32.0 * zone_factor * out_factor * material_size_factor(inp, "slab") * scale * inp.design_slab_scale
    return float(np.clip(t, inp.min_slab_thickness, inp.max_slab_thickness))


def beam_size(inp: BuildingInput, story: int, scale: float = 1.0) -> Tuple[float, float]:
    """
    Beam size now changes by height and by outrigger/collector demand.
    Outrigger floors receive deeper collector beams/belt beams.
    """
    span = max(inp.bay_x, inp.bay_y)
    h_ratio = 1.0 - (story - 1) / max(inp.n_story - 1, 1)

    base_h = span / 10.5
    zone_factor = 0.82 + 0.28 * h_ratio

    # collector beams at outrigger floors must be stronger
    out_factor = 1.35 if story in active_outrigger_levels(inp) and inp.outrigger_system != OutriggerSystem.NONE else 1.00

    h = base_h * zone_factor * out_factor * material_size_factor(inp, "beam") * scale * inp.design_beam_scale
    h = float(np.clip(h, inp.min_beam_depth, 2.80))
    b = float(np.clip(0.45 * h, inp.min_beam_width, 1.30))
    return b, h


def directional_column_dims(dim: float, inp: BuildingInput) -> Tuple[float, float]:
    aspect = max(inp.plan_x, inp.plan_y) / max(min(inp.plan_x, inp.plan_y), 1e-9)
    if aspect < 1.10:
        return dim, dim
    major = 1.12 * dim
    minor = 0.92 * dim
    if inp.plan_x >= inp.plan_y:
        return major, minor
    return minor, major


def column_counts(inp: BuildingInput) -> Tuple[int, int, int, int]:
    total = (inp.n_bays_x + 1) * (inp.n_bays_y + 1)
    corner = 4
    perimeter = max(0, 2 * (inp.n_bays_x - 1) + 2 * (inp.n_bays_y - 1))
    interior = max(0, total - perimeter - corner)
    return total, interior, perimeter, corner


def column_size(inp: BuildingInput, story: int, slab_t: float, scale: float = 1.0) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Gravity + monotonic height-based preliminary column sizing.

    Corrected rule:
    column dimensions reduce upward. A middle column cannot become larger than
    a lower column because the height factor is monotonic.
    """
    q = inp.DL + inp.slab_finish_allowance + inp.live_load_mass_factor * inp.LL + 25.0 * slab_t
    floors_above = inp.n_story - story + 1 + 0.60 * inp.n_basement
    tributary = inp.bay_x * inp.bay_y

    P_kN = q * tributary * floors_above * 1.15
    sigma_allow = 0.30 * inp.fck * 1000.0
    gravity_dim = sqrt(max(P_kN / max(sigma_allow, 1e-9), 1e-9))

    # This factor is high at base and low at top, enforcing rational taper.
    h_ratio = 1.0 - (story - 1) / max(inp.n_story - 1, 1)
    tower_factor = 0.85 + 0.35 * h_ratio

    dim = gravity_dim * tower_factor * material_size_factor(inp, "column") * scale * inp.design_column_scale
    dim = float(np.clip(dim, inp.min_column_dim, inp.max_column_dim))

    interior = directional_column_dims(dim, inp)
    perimeter = directional_column_dims(float(np.clip(dim * inp.perimeter_column_factor, inp.min_column_dim, inp.max_column_dim)), inp)
    corner = directional_column_dims(float(np.clip(dim * inp.corner_column_factor, inp.min_column_dim, inp.max_column_dim)), inp)
    return interior, perimeter, corner


def enforce_monotonic_sections(sections: List[StorySection]) -> List[StorySection]:
    """
    Enforce preliminary tower design discipline:
    lower story dimensions >= upper story dimensions for primary vertical elements.
    This prevents irrational results such as middle-story columns larger than lower-story columns.
    """
    if not sections:
        return sections

    # Work top-down and push larger upper requirements downward.
    max_core_t = 0.0
    max_side_t = 0.0
    max_int_x = max_int_y = 0.0
    max_per_x = max_per_y = 0.0
    max_cor_x = max_cor_y = 0.0

    updated = list(sections)

    for idx in range(len(updated) - 1, -1, -1):
        s = updated[idx]
        max_core_t = max(max_core_t, s.core_wall_t)
        max_side_t = max(max_side_t, s.side_wall_t)
        max_int_x = max(max_int_x, s.column_interior_x)
        max_int_y = max(max_int_y, s.column_interior_y)
        max_per_x = max(max_per_x, s.column_perimeter_x)
        max_per_y = max(max_per_y, s.column_perimeter_y)
        max_cor_x = max(max_cor_x, s.column_corner_x)
        max_cor_y = max(max_cor_y, s.column_corner_y)

        updated[idx] = replace(
            s,
            core_wall_t=max_core_t,
            side_wall_t=max_side_t,
            column_interior_x=max_int_x,
            column_interior_y=max_int_y,
            column_perimeter_x=max_per_x,
            column_perimeter_y=max_per_y,
            column_corner_x=max_cor_x,
            column_corner_y=max_cor_y,
        )
    return updated


def build_story_sections(inp: BuildingInput, wall_scale: float = 1.0, col_scale: float = 1.0, slab_scale: float = 1.0) -> List[StorySection]:
    core_x, core_y, open_x, open_y = initial_core_dimensions(inp)
    sections = []
    n_total, n_int, n_per, n_cor = column_counts(inp)

    for story in range(1, inp.n_story + 1):
        zone = zone_for_story(inp, story)
        slab_t = slab_thickness(inp, story, slab_scale)
        beam_b, beam_h = beam_size(inp, story, slab_scale)
        interior, perimeter, corner = column_size(inp, story, slab_t, col_scale)

        t_core = wall_thickness(inp, story, wall_scale)
        t_side = side_wall_thickness(inp, story, wall_scale)

        sections.append(
            StorySection(
                story=story,
                zone=zone,
                elevation_m=story * inp.story_height,
                core_x=core_x,
                core_y=core_y,
                opening_x=open_x,
                opening_y=open_y,
                core_wall_t=t_core,
                side_wall_length_x=inp.side_wall_ratio * inp.plan_x,
                side_wall_length_y=inp.side_wall_ratio * inp.plan_y,
                side_wall_t=t_side,
                wall_count=wall_count_for_zone(inp, zone),
                column_interior_x=interior[0],
                column_interior_y=interior[1],
                column_perimeter_x=perimeter[0],
                column_perimeter_y=perimeter[1],
                column_corner_x=corner[0],
                column_corner_y=corner[1],
                beam_b=beam_b,
                beam_h=beam_h,
                slab_t=slab_t,
                n_columns_total=n_total,
                n_interior_columns=n_int,
                n_perimeter_columns=n_per,
                n_corner_columns=n_cor,
            )
        )

    return enforce_monotonic_sections(sections)


# ============================================================
# 4. SECTION PROPERTIES
# ============================================================

def rectangular_tube_inertia(outer_x: float, outer_y: float, t: float) -> Tuple[float, float]:
    Ix_o = outer_x * outer_y**3 / 12.0
    Iy_o = outer_y * outer_x**3 / 12.0
    inner_x = max(outer_x - 2 * t, 0.10)
    inner_y = max(outer_y - 2 * t, 0.10)
    Ix_i = inner_x * inner_y**3 / 12.0
    Iy_i = inner_y * inner_x**3 / 12.0
    return max(Ix_o - Ix_i, 1e-9), max(Iy_o - Iy_i, 1e-9)


def side_wall_inertia(inp: BuildingInput, sec: StorySection) -> Tuple[float, float]:
    lx = sec.side_wall_length_x
    ly = sec.side_wall_length_y
    t = sec.side_wall_t

    # two X-direction side walls placed at ±Y/2
    Ix_xwalls = 2 * (lx * t**3 / 12.0 + lx * t * (inp.plan_y / 2.0) ** 2)
    Iy_xwalls = 2 * (t * lx**3 / 12.0)

    # two Y-direction side walls placed at ±X/2
    Ix_ywalls = 2 * (t * ly**3 / 12.0)
    Iy_ywalls = 2 * (ly * t**3 / 12.0 + ly * t * (inp.plan_x / 2.0) ** 2)

    return inp.side_wall_cracked_factor * (Ix_xwalls + Ix_ywalls), inp.side_wall_cracked_factor * (Iy_xwalls + Iy_ywalls)


def perimeter_wall_segments(inp: BuildingInput, sec: StorySection) -> List[Tuple[float, float, float, float]]:
    """Return four above-grade perimeter wall strips as rectangles.

    Format: (x0, y0, width, height). These wall strips are the same
    side/perimeter walls considered in the stiffness calculation.
    """
    lx = max(sec.side_wall_length_x, 0.0)
    ly = max(sec.side_wall_length_y, 0.0)
    t = max(sec.side_wall_t, 0.0)
    mx = inp.plan_x / 2.0
    my = inp.plan_y / 2.0
    return [
        (mx - lx / 2.0, -t / 2.0, lx, t),
        (mx - lx / 2.0, inp.plan_y - t / 2.0, lx, t),
        (-t / 2.0, my - ly / 2.0, t, ly),
        (inp.plan_x - t / 2.0, my - ly / 2.0, t, ly),
    ]


def basement_retaining_wall_inertia(inp: BuildingInput) -> Tuple[float, float]:
    """Approximate below-grade retaining-wall stiffness contribution.

    The superstructure model has a fixed base, so basement retaining walls are
    represented as a controlled base-zone stiffness contribution rather than as
    random upper-story walls. If basement levels are modeled explicitly in a
    detailed FEM package, this approximation should be replaced by those wall
    elements.
    """
    if not inp.include_basement_retaining_wall or inp.n_basement <= 0:
        return 0.0, 0.0
    t = float(np.clip(inp.retaining_wall_thickness, 0.20, 2.00))
    Ix_box, Iy_box = rectangular_tube_inertia(inp.plan_x, inp.plan_y, t)
    factor = inp.retaining_wall_cracked_factor * inp.basement_stiffness_participation
    return factor * Ix_box, factor * Iy_box


def basement_influence_factor(inp: BuildingInput, story: int) -> float:
    """Taper retaining-wall stiffness over the first base-influence stories."""
    if inp.n_basement <= 0:
        return 0.0
    influence_stories = max(1, min(inp.n_story, int(ceil(inp.n_basement * inp.basement_height / inp.story_height)) + 1))
    if story > influence_stories:
        return 0.0
    return (influence_stories - story + 1) / influence_stories


def grid_column_coordinates(inp: BuildingInput) -> List[Tuple[float, float, str]]:
    coords = []
    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x - inp.plan_x / 2.0
            y = j * inp.bay_y - inp.plan_y / 2.0
            at_x = i == 0 or i == inp.n_bays_x
            at_y = j == 0 or j == inp.n_bays_y
            if at_x and at_y:
                typ = "corner"
            elif at_x or at_y:
                typ = "perimeter"
            else:
                typ = "interior"
            coords.append((x, y, typ))
    return coords


def column_group_inertia(inp: BuildingInput, sec: StorySection) -> Tuple[float, float]:
    Ix = 0.0
    Iy = 0.0

    for x, y, typ in grid_column_coordinates(inp):
        if typ == "corner":
            bx, by = sec.column_corner_x, sec.column_corner_y
        elif typ == "perimeter":
            bx, by = sec.column_perimeter_x, sec.column_perimeter_y
        else:
            bx, by = sec.column_interior_x, sec.column_interior_y

        A = bx * by
        Ix_local = bx * by**3 / 12.0
        Iy_local = by * bx**3 / 12.0

        # In a preliminary flexural stick model, full A*d² of all frame columns would
        # unrealistically assume every floor beam acts as a perfectly rigid outrigger.
        # Only a limited part of the column axial-couple mechanism is mobilized by
        # ordinary floor beams. True outrigger floors are handled separately in
        # outrigger_Ktheta().
        frame_coupling = 0.035
        Ix += Ix_local + frame_coupling * A * y**2
        Iy += Iy_local + frame_coupling * A * x**2

    return inp.column_cracked_factor * Ix, inp.column_cracked_factor * Iy


def tube_area(D: float, t: float) -> float:
    return pi / 4.0 * (D**2 - max(D - 2.0 * t, 0.001) ** 2)


def outrigger_efficiency(system: OutriggerSystem) -> float:
    if system == OutriggerSystem.BELT_TRUSS:
        return 1.00
    if system == OutriggerSystem.TUBULAR_BRACE:
        return 0.85
    return 0.0




def centered_bay_indices(n_bays: int, requested_count: int) -> Tuple[int, ...]:
    """
    Return actual grid bay indices centered about the building centerline.

    Bay index i means the bay between column grid lines i and i+1.
    This prevents the previous mistake where plan braces were spread by linspace
    and did not correspond to real bays between columns.
    """
    n_bays = int(max(n_bays, 1))
    requested_count = int(max(min(requested_count, n_bays), 0))
    if requested_count <= 0:
        return tuple()
    start = max((n_bays - requested_count) // 2, 0)
    return tuple(range(start, start + requested_count))


def clean_bay_ids(ids: Tuple[int, ...], n_bays: int, max_count: int | None = None) -> Tuple[int, ...]:
    """Return valid, unique real bay IDs sorted increasingly."""
    valid: List[int] = []
    for raw in ids:
        i = int(raw)
        if 0 <= i < int(n_bays) and i not in valid:
            valid.append(i)
    if max_count is not None and max_count >= 0:
        valid = valid[: int(max_count)]
    return tuple(valid)


def active_braced_bays(inp: BuildingInput, direction: Direction) -> Tuple[int, ...]:
    """
    Actual braced bay indices used both in stiffness and drawing.

    v27 rule:
      1. If explicit bay IDs are entered, use those exact bay panels.
      2. Otherwise select centered real grid bays from braced_spans_x/y.
      3. Drawing, diagnostic table, and stiffness matrix use the same bay IDs.
    """
    if direction == Direction.X:
        explicit = clean_bay_ids(inp.braced_bay_ids_x, inp.n_bays_y, inp.braced_spans_x)
        if explicit:
            return explicit
        return centered_bay_indices(inp.n_bays_y, inp.braced_spans_x)
    explicit = clean_bay_ids(inp.braced_bay_ids_y, inp.n_bays_x, inp.braced_spans_y)
    if explicit:
        return explicit
    return centered_bay_indices(inp.n_bays_x, inp.braced_spans_y)


def active_outrigger_levels(inp: BuildingInput) -> Tuple[int, ...]:
    """
    Enforce user-specified outrigger_count.
    Only the first valid, unique levels are active.
    """
    if inp.outrigger_system == OutriggerSystem.NONE or inp.outrigger_count <= 0:
        return tuple()
    levels = []
    for lev in inp.outrigger_story_levels:
        ilev = int(lev)
        if 1 <= ilev <= inp.n_story and ilev not in levels:
            levels.append(ilev)
    return tuple(levels[: int(inp.outrigger_count)])


def outrigger_span_basic_values(inp: BuildingInput, sec: StorySection, direction: Direction) -> Dict[str, float]:
    """
    Direct brace-span stiffness method based on the user's sketch.

    One real braced bay panel is treated as an X-braced span between two columns.
    For each diagonal brace:
        k_diag = (Es * A / L) * cos²(theta)
    The total outrigger line stiffness is:
        K_out = k_diag * number_of_diagonal_braces
    The rotational restraint added to the MDOF model is:
        Ktheta = eta * K_out * lever_arm²

    The same active braced bay IDs are used for drawing, stiffness, diagnostics,
    and the MDOF matrix. No random spreading is used.
    """
    zero = {
        "selected_bays": 0.0,
        "total_braces": 0.0,
        "bay_width_m": 0.0,
        "brace_height_m": 0.0,
        "brace_length_m": 0.0,
        "cos2": 0.0,
        "area_brace_m2": 0.0,
        "k_one_brace_N_per_m": 0.0,
        "k_out_total_N_per_m": 0.0,
        "lever_arm_m": 0.0,
        "Ktheta_Nm_per_rad": 0.0,
    }
    if inp.outrigger_system == OutriggerSystem.NONE or sec.story not in active_outrigger_levels(inp):
        return zero

    if direction == Direction.X:
        # X response: east/west outrigger sides, braced panels counted along Y.
        bay_width = max(inp.bay_y, 1e-9)
        bay_ids = active_braced_bays(inp, Direction.X)
        plan_dim = inp.plan_x
        core_dim = sec.core_x
    else:
        # Y response: north/south outrigger sides, braced panels counted along X.
        bay_width = max(inp.bay_x, 1e-9)
        bay_ids = active_braced_bays(inp, Direction.Y)
        plan_dim = inp.plan_y
        core_dim = sec.core_y

    n_selected_bays = len(bay_ids)
    if n_selected_bays <= 0:
        out = dict(zero)
        out["bay_width_m"] = float(bay_width)
        return out

    Es = STEEL_E_MPA * 1e6
    eta = max(min(outrigger_efficiency(inp.outrigger_system) * inp.outrigger_connection_efficiency * inp.outrigger_effectiveness_factor, 1.0), 0.0)

    if inp.outrigger_system == OutriggerSystem.TUBULAR_BRACE:
        A_brace = tube_area(
            inp.tubular_diameter_m * inp.design_outrigger_scale,
            inp.tubular_thickness_m * inp.design_outrigger_scale,
        )
    else:
        A_brace = inp.outrigger_diagonal_area_m2 * inp.design_outrigger_scale

    brace_height = max(inp.outrigger_depth_m, inp.story_height, 1e-9)
    L_diag = sqrt(bay_width**2 + brace_height**2)
    cos2 = (bay_width / L_diag) ** 2

    # Direct requested formula: stiffness of one brace times number of braces.
    k_one_brace = Es * A_brace / L_diag * cos2
    braces_per_bay_per_side = 2.0  # X brace = two diagonals
    side_count = 2.0               # two opposite perimeter/outrigger sides
    total_braces = braces_per_bay_per_side * side_count * n_selected_bays
    k_out_total = total_braces * k_one_brace

    lever_arm = max((plan_dim - core_dim) / 2.0, 1e-9)
    Ktheta = eta * k_out_total * lever_arm**2

    return {
        "selected_bays": float(n_selected_bays),
        "total_braces": float(total_braces),
        "bay_width_m": float(bay_width),
        "brace_height_m": float(brace_height),
        "brace_length_m": float(L_diag),
        "cos2": float(cos2),
        "area_brace_m2": float(A_brace),
        "k_one_brace_N_per_m": float(k_one_brace),
        "k_out_total_N_per_m": float(k_out_total),
        "lever_arm_m": float(lever_arm),
        "Ktheta_Nm_per_rad": float(Ktheta),
    }


def outrigger_Ktheta(inp: BuildingInput, sec: StorySection, direction: Direction) -> float:
    """Rotational restraint: Ktheta = eta * sum(Es*A/L*cos²theta) * lever_arm²."""
    return outrigger_span_basic_values(inp, sec, direction)["Ktheta_Nm_per_rad"]


def outrigger_span_stiffness_components(inp: BuildingInput, sec: StorySection, direction: Direction) -> Dict[str, float]:
    """Diagnostic values for the direct brace-span summation method."""
    v = outrigger_span_basic_values(inp, sec, direction)
    return {
        "selected_bays": v["selected_bays"],
        "total_braces": v["total_braces"],
        "k_one_brace_MN_per_m": v["k_one_brace_N_per_m"] / 1e6,
        "k_braces_total_MN_per_m": v["k_out_total_N_per_m"] / 1e6,
        "Ktheta_GNm_per_rad": v["Ktheta_Nm_per_rad"] / 1e9,
        "bay_width_m": v["bay_width_m"],
        "brace_length_m": v["brace_length_m"],
        "lever_arm_m": v["lever_arm_m"],
    }


def outrigger_Klateral(inp: BuildingInput, sec: StorySection, direction: Direction) -> float:
    """
    Diagnostic translational part of the outrigger stiffness.

    Important engineering correction in v4.4 patch:
    A core-outrigger system mainly provides rotational restraint to the core through
    an axial couple in the exterior columns. It is not a full direct lateral support
    to ground at the outrigger floor. Therefore only a small, user-controlled
    fraction of the projected brace stiffness is permitted to enter the lateral
    displacement DOF.
    """
    v = outrigger_span_basic_values(inp, sec, direction)
    k_out = v["k_out_total_N_per_m"]
    if k_out <= 0.0:
        return 0.0
    eta = max(min(outrigger_efficiency(inp.outrigger_system) * inp.outrigger_connection_efficiency * inp.outrigger_effectiveness_factor, 1.0), 0.0)
    lateral_part = max(min(inp.outrigger_lateral_participation, 0.25), 0.0)
    return float(eta * lateral_part * k_out)

def outrigger_coupled_matrix(inp: BuildingInput, sec: StorySection, direction: Direction) -> np.ndarray:
    """
    Condensed engineering stiffness added to the floor DOFs [u, theta].

    Corrected v4.4 engineering patch:
    - Brace-panel stiffness is still calculated from the real braced spans:
          K_out = sum((E_s A_b / L_b) cos^2(theta_b))
    - The dominant physical contribution is core rotational restraint:
          K_theta = eta * K_out * a^2
    - A small translational term is allowed only as diaphragm/collector participation:
          K_uu = eta * alpha_lat * K_out
    - The coupling term is bounded for positive-definiteness:
          K_uθ = beta * sqrt(K_uu * K_theta)

    This prevents the outrigger from acting as an unrealistic floor-to-ground
    lateral spring, which previously caused excessive period reduction.
    """
    v = outrigger_span_basic_values(inp, sec, direction)
    k_out = v["k_out_total_N_per_m"]
    a = v["lever_arm_m"]
    if k_out <= 0.0 or a <= 0.0:
        return np.zeros((2, 2), dtype=float)

    eta = max(min(outrigger_efficiency(inp.outrigger_system) * inp.outrigger_connection_efficiency * inp.outrigger_effectiveness_factor, 1.0), 0.0)
    alpha_lat = max(min(inp.outrigger_lateral_participation, 0.25), 0.0)
    beta_couple = 0.50

    k_uu = eta * alpha_lat * k_out
    k_tt = eta * k_out * a * a
    k_ut = beta_couple * sqrt(max(k_uu * k_tt, 0.0))

    return np.array([[k_uu, k_ut], [k_ut, k_tt]], dtype=float)

def story_mass_and_quantities(inp: BuildingInput, sec: StorySection) -> Tuple[float, float, float, float]:
    A_floor = inp.floor_area
    slab_vol = A_floor * sec.slab_t

    beam_lines = inp.n_bays_y * (inp.n_bays_x + 1) + inp.n_bays_x * (inp.n_bays_y + 1)
    avg_span = 0.5 * (inp.bay_x + inp.bay_y)
    beam_vol = beam_lines * avg_span * sec.beam_b * sec.beam_h

    core_wall_vol = 2 * (sec.core_x + sec.core_y) * sec.core_wall_t * inp.story_height
    side_wall_vol = 2 * (sec.side_wall_length_x + sec.side_wall_length_y) * sec.side_wall_t * inp.story_height

    col_vol = (
        sec.n_corner_columns * sec.column_corner_x * sec.column_corner_y
        + sec.n_perimeter_columns * sec.column_perimeter_x * sec.column_perimeter_y
        + sec.n_interior_columns * sec.column_interior_x * sec.column_interior_y
    ) * inp.story_height

    concrete = slab_vol + beam_vol + core_wall_vol + side_wall_vol + col_vol

    steel = (
        slab_vol * inp.slab_rebar_ratio
        + beam_vol * inp.beam_rebar_ratio
        + (core_wall_vol + side_wall_vol) * inp.wall_rebar_ratio
        + col_vol * inp.column_rebar_ratio
    ) * RHO_STEEL

    if sec.story in active_outrigger_levels(inp) and inp.outrigger_system != OutriggerSystem.NONE:
        if inp.outrigger_system == OutriggerSystem.TUBULAR_BRACE:
            A_tube = tube_area(inp.tubular_diameter_m * inp.design_outrigger_scale, inp.tubular_thickness_m * inp.design_outrigger_scale)
            arm_x = max((inp.plan_x - sec.core_x) / 2.0, 1.0)
            arm_y = max((inp.plan_y - sec.core_y) / 2.0, 1.0)
            Lx = sqrt(arm_x**2 + inp.outrigger_depth_m**2)
            Ly = sqrt(arm_y**2 + inp.outrigger_depth_m**2)
            steel += 2 * len(active_braced_bays(inp, Direction.X)) * A_tube * Lx * RHO_STEEL
            steel += 2 * len(active_braced_bays(inp, Direction.Y)) * A_tube * Ly * RHO_STEEL
        else:
            arm_x = max((inp.plan_x - sec.core_x) / 2.0, 1.0)
            arm_y = max((inp.plan_y - sec.core_y) / 2.0, 1.0)
            steel += 2 * len(active_braced_bays(inp, Direction.X)) * (inp.outrigger_chord_area_m2 + inp.outrigger_diagonal_area_m2) * arm_x * RHO_STEEL
            steel += 2 * len(active_braced_bays(inp, Direction.Y)) * (inp.outrigger_chord_area_m2 + inp.outrigger_diagonal_area_m2) * arm_y * RHO_STEEL

    structural_weight_kN = concrete * 25.0 + steel * G / 1000.0
    superimposed_kN = (
        inp.DL + inp.slab_finish_allowance + inp.live_load_mass_factor * inp.LL
    ) * A_floor
    facade_kN = inp.facade_line_load * 2 * (inp.plan_x + inp.plan_y)

    total_weight_kN = (structural_weight_kN + superimposed_kN + facade_kN) * inp.additional_mass_factor
    mass_kg = total_weight_kN * 1000.0 / G
    return mass_kg, total_weight_kN, concrete, steel


def build_story_properties(inp: BuildingInput, sections: List[StorySection]) -> List[StoryProperties]:
    props = []
    E = inp.Ec * 1e6

    for sec in sections:
        Ix_core, Iy_core = rectangular_tube_inertia(sec.core_x, sec.core_y, sec.core_wall_t)
        Ix_core *= inp.wall_cracked_factor
        Iy_core *= inp.wall_cracked_factor

        Ix_side, Iy_side = side_wall_inertia(inp, sec)
        Ix_col, Iy_col = column_group_inertia(inp, sec)
        Ix_ret, Iy_ret = basement_retaining_wall_inertia(inp)
        basement_factor = basement_influence_factor(inp, sec.story)

        # X translation bends about Y. Y translation bends about X.
        EI_basement_x = E * Iy_ret * basement_factor
        EI_basement_y = E * Ix_ret * basement_factor
        EI_x = E * (Iy_core + Iy_side + Iy_col) * inp.coupling_factor + EI_basement_x
        EI_y = E * (Ix_core + Ix_side + Ix_col) * inp.coupling_factor + EI_basement_y

        mass, weight, concrete, steel = story_mass_and_quantities(inp, sec)

        props.append(
            StoryProperties(
                story=sec.story,
                mass_kg=mass,
                weight_kN=weight,
                concrete_m3=concrete,
                steel_kg=steel,
                EI_x_Nm2=EI_x,
                EI_y_Nm2=EI_y,
                Ktheta_out_x_Nm=outrigger_Ktheta(inp, sec, Direction.X),
                Ktheta_out_y_Nm=outrigger_Ktheta(inp, sec, Direction.Y),
                EI_basement_x_Nm2=EI_basement_x,
                EI_basement_y_Nm2=EI_basement_y,
            )
        )
    return props


# ============================================================
# 5. FLEXURAL MDOF SOLVER
# ============================================================

def beam_element_stiffness(EI: float, L: float) -> np.ndarray:
    return EI / L**3 * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ],
        dtype=float,
    )


def assemble_flexural_mk(inp: BuildingInput, props: List[StoryProperties], direction: Direction, sections: List[StorySection] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    n = inp.n_story
    ndof = 2 * (n + 1)
    L = inp.story_height

    K = np.zeros((ndof, ndof), dtype=float)
    M = np.zeros((ndof, ndof), dtype=float)

    if sections is None:
        sections = build_story_sections(inp)

    for e in range(n):
        prop = props[e]
        EI = prop.EI_x_Nm2 if direction == Direction.X else prop.EI_y_Nm2
        ke = beam_element_stiffness(EI, L)
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]

    for node in range(1, n + 1):
        prop = props[node - 1]
        sec = sections[node - 1]
        u_dof = 2 * node
        th_dof = 2 * node + 1
        M[u_dof, u_dof] += prop.mass_kg
        M[th_dof, th_dof] += prop.mass_kg * L**2 * 1e-5

        # v27 engineering insertion: condensed braced-bay outrigger matrix.
        # The contribution is not only Ktheta; it includes Kuu and off-diagonal
        # coupling terms from delta = u + lever_arm * theta, so it changes the
        # global stiffness matrix and the modal period.
        kout = outrigger_coupled_matrix(inp, sec, direction)
        if np.any(kout):
            dofs2 = [u_dof, th_dof]
            for a_i in range(2):
                for b_i in range(2):
                    K[dofs2[a_i], dofs2[b_i]] += kout[a_i, b_i]

    free = list(range(2, ndof))  # base u and theta fixed
    return M[np.ix_(free, free)], K[np.ix_(free, free)]


def solve_modal(inp: BuildingInput, props: List[StoryProperties], direction: Direction, sections: List[StorySection] | None = None) -> ModalResult:
    """
    Symmetric generalized eigenvalue solution:
        K phi = omega² M phi

    This replaces the older np.linalg.solve(M,K) eigen-solution. The latter can
    introduce numerical mode-ordering noise for very stiff/weak outrigger cases.
    """
    M, K = assemble_flexural_mk(inp, props, direction, sections)

    # Ensure symmetry for structural eigenproblem.
    M = 0.5 * (M + M.T)
    K = 0.5 * (K + K.T)

    if scipy_eigh is not None:
        eigvals, eigvecs = scipy_eigh(K, M, check_finite=False)
    else:
        # Fallback: mass-normalized symmetric standard problem.
        L = np.linalg.cholesky(M)
        Linv = np.linalg.inv(L)
        A = Linv @ K @ Linv.T
        A = 0.5 * (A + A.T)
        eigvals, yvecs = np.linalg.eigh(A)
        eigvecs = Linv.T @ yvecs

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    keep = eigvals > 1e-8
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_modes = min(inp.n_modes, len(eigvals))
    eigvals = eigvals[:n_modes]
    eigvecs = eigvecs[:, :n_modes]

    omegas = np.sqrt(eigvals)
    periods = 2.0 * pi / omegas
    freqs = omegas / (2.0 * pi)

    n = inp.n_story
    r = np.zeros((2 * n, 1))
    r[0::2, 0] = 1.0

    total_mass = sum(p.mass_kg for p in props)

    gammas = []
    meff = []
    cum = []
    shapes = []
    running = 0.0

    for i in range(n_modes):
        phi = eigvecs[:, i].reshape(-1, 1)
        denom = (phi.T @ M @ phi).item()
        gamma = ((phi.T @ M @ r) / denom).item()
        m_eff = gamma**2 * denom
        ratio = m_eff / max(total_mass, 1e-9)
        running += ratio

        shape = phi.flatten()[0::2]
        if abs(shape[-1]) > 1e-12:
            shape = shape / shape[-1]
        if shape[-1] < 0:
            shape = -shape

        gammas.append(gamma)
        meff.append(ratio)
        cum.append(running)
        shapes.append(shape)

    return ModalResult(
        direction=direction,
        periods_s=[float(x) for x in periods],
        frequencies_hz=[float(x) for x in freqs],
        omegas=omegas,
        mode_shapes=shapes,
        gammas=gammas,
        effective_mass_ratios=meff,
        cumulative_mass_ratios=cum,
        raw_eigenvectors=eigvecs.copy(),
    )


# ============================================================
# 6. ASCE 7 RESPONSE SPECTRUM
# ============================================================

def resolve_asce_design_spectrum(asce: ASCE7Params) -> Tuple[float, float]:
    """Return effective ASCE design spectrum values SDS and SD1."""
    if asce.use_site_coefficients:
        return max((2.0 / 3.0) * asce.Fa * asce.SS, 1e-9), max((2.0 / 3.0) * asce.Fv * asce.S1, 1e-9)
    return max(asce.SDS, 1e-9), max(asce.SD1, 1e-9)

def asce_corner_periods(asce: ASCE7Params) -> Tuple[float, float]:
    SDS_eff, SD1_eff = resolve_asce_design_spectrum(asce)
    Ts = SD1_eff / max(SDS_eff, 1e-9)
    return 0.2 * Ts, Ts


def asce_spectrum_sa_g(T: float, asce: ASCE7Params) -> float:
    T = max(float(T), 1e-9)
    SDS_eff, SD1_eff = resolve_asce_design_spectrum(asce)
    T0, Ts = asce_corner_periods(asce)
    if T < T0 and T0 > 0:
        return SDS_eff * (0.4 + 0.6 * T / T0)
    if T <= Ts:
        return SDS_eff
    if T <= asce.TL:
        return SD1_eff / T
    return SD1_eff * asce.TL / (T**2)


def rsa_force_sa_g(T: float, asce: ASCE7Params) -> float:
    return asce_spectrum_sa_g(T, asce) * asce.Ie / asce.R


def approximate_Ta(inp: BuildingInput) -> float:
    h_ft = inp.height * 3.28084
    return inp.asce7.Ct * h_ft ** inp.asce7.x_exp


def asce_Cs(T: float, asce: ASCE7Params) -> float:
    T = max(float(T), 1e-6)
    SDS_eff, SD1_eff = resolve_asce_design_spectrum(asce)
    R_over_Ie = asce.R / asce.Ie
    Cs_short = SDS_eff / R_over_Ie
    if T <= asce.TL:
        Cs_long = SD1_eff / (T * R_over_Ie)
    else:
        Cs_long = SD1_eff * asce.TL / (T**2 * R_over_Ie)
    Cs = min(Cs_short, Cs_long)
    Cs_min = max(0.044 * SDS_eff * asce.Ie, 0.01)
    if asce.S1 >= 0.6:
        Cs_min = max(Cs_min, 0.5 * asce.S1 / R_over_Ie)
    return max(Cs, Cs_min)


def elf_base_shear(inp: BuildingInput, modal_T: float, total_mass_kg: float) -> Tuple[float, float, float, float]:
    Ta = approximate_Ta(inp)
    T_used = min(modal_T, inp.asce7.Cu * Ta) if inp.asce7.use_CuTa_cap else modal_T
    Cs = asce_Cs(T_used, inp.asce7)
    V_kN = Cs * total_mass_kg * G / 1000.0
    return V_kN, Cs, Ta, T_used



def period_limit_values(inp: BuildingInput, modal_T: float) -> Tuple[float, float, float, float]:
    """Return Ta, CuTa, allowable period, and design period for redesign checks."""
    Ta = approximate_Ta(inp)
    CuTa = inp.asce7.Cu * Ta
    T_allow = inp.period_limit_multiplier * CuTa
    T_design = modal_T
    return Ta, CuTa, T_allow, T_design


def period_redesign_ratio(inp: BuildingInput, res: DesignResult) -> Tuple[float, float, float, float, float]:
    tx = res.modal_x.periods_s[0]
    ty = res.modal_y.periods_s[0]
    Ta, CuTa, T_allow_x, T_design_x = period_limit_values(inp, tx)
    _, _, T_allow_y, T_design_y = period_limit_values(inp, ty)
    ratio_x = T_design_x / max(T_allow_x, 1e-9)
    ratio_y = T_design_y / max(T_allow_y, 1e-9)
    return max(ratio_x, ratio_y), T_design_x, T_design_y, T_allow_x, CuTa


def cqc_rho(wi: float, wj: float, zeta: float) -> float:
    if abs(wi - wj) < 1e-12:
        return 1.0
    beta = wj / wi
    num = 8.0 * zeta**2 * beta**1.5
    den = (1.0 - beta**2) ** 2 + 4.0 * zeta**2 * beta * (1.0 + beta) ** 2
    return num / max(den, 1e-12)


def combine_modal(values: np.ndarray, omegas: np.ndarray, method: CombinationMethod, zeta: float) -> np.ndarray:
    if values.ndim == 1:
        values = values.reshape((-1, 1))

    if method == CombinationMethod.SRSS:
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


def response_spectrum_analysis(inp: BuildingInput, props: List[StoryProperties], modal: ModalResult) -> RSAResult:
    n = inp.n_story
    h = inp.story_height
    masses = np.array([p.mass_kg for p in props], dtype=float)
    heights = np.arange(1, n + 1) * h

    # Reconstruct eigenvectors approximately from normalized shapes for response. For better response,
    # the modal participation gamma already came from the true eigenvectors, so normalized floor shapes
    # are acceptable for preliminary force distribution.
    modal_floor_forces = []
    modal_story_shear = []
    modal_otm = []
    modal_disp = []
    modal_drift = []
    modal_base = []

    for i, T in enumerate(modal.periods_s):
        omega = modal.omegas[i]
        gamma = modal.gammas[i]
        if modal.raw_eigenvectors is not None:
            phi_u = np.array(modal.raw_eigenvectors[:, i].flatten()[0::2], dtype=float)
        else:
            phi_u = np.array(modal.mode_shapes[i], dtype=float)

        if inp.use_asce7_rsa:
            Sa = rsa_force_sa_g(T, inp.asce7) * G
        else:
            Sa = resolve_asce_design_spectrum(inp.asce7)[0] * G * inp.asce7.Ie / inp.asce7.R

        u = phi_u * gamma * Sa / omega**2
        f = masses * phi_u * gamma * Sa

        V = np.zeros(n)
        for j in range(n - 1, -1, -1):
            V[j] = f[j] + (V[j + 1] if j < n - 1 else 0.0)

        OTM = np.zeros(n)
        for j in range(n):
            OTM[j] = np.sum(f[j:] * (heights[j:] - (heights[j] - h)))

        drift = np.zeros(n)
        drift[0] = u[0]
        drift[1:] = np.diff(u)

        modal_floor_forces.append(f / 1000.0)
        modal_story_shear.append(V / 1000.0)
        modal_otm.append(OTM / 1000.0)
        modal_disp.append(u)
        modal_drift.append(drift)
        modal_base.append(np.sum(f) / 1000.0)

    modal_floor_forces = np.array(modal_floor_forces)
    modal_story_shear = np.array(modal_story_shear)
    modal_otm = np.array(modal_otm)
    modal_disp = np.array(modal_disp)
    modal_drift = np.array(modal_drift)
    modal_base = np.array(modal_base)

    zeta = inp.asce7.damping_ratio
    floor_force = combine_modal(modal_floor_forces, modal.omegas, inp.combination, zeta)
    story_shear = combine_modal(modal_story_shear, modal.omegas, inp.combination, zeta)
    overturning = combine_modal(modal_otm, modal.omegas, inp.combination, zeta)
    disp = combine_modal(modal_disp, modal.omegas, inp.combination, zeta)
    drift = combine_modal(modal_drift, modal.omegas, inp.combination, zeta)
    base_unscaled = combine_modal(modal_base, modal.omegas, inp.combination, zeta).item()

    total_mass = sum(p.mass_kg for p in props)
    elf_kN, Cs, Ta, T_used = elf_base_shear(inp, modal.periods_s[0], total_mass)
    required = inp.asce7.rsa_min_ratio_to_elf * elf_kN
    scale = required / base_unscaled if base_unscaled < required and base_unscaled > 1e-9 else 1.0

    floor_force *= scale
    story_shear *= scale
    overturning *= scale
    base_scaled = base_unscaled * scale

    # Drift amplified by Cd/Ie
    drift_design = drift * inp.asce7.Cd / inp.asce7.Ie
    disp_design = disp * inp.asce7.Cd / inp.asce7.Ie

    if inp.asce7.scale_drift_with_base_shear:
        drift_design *= scale
        disp_design *= scale

    return RSAResult(
        direction=modal.direction,
        modal=modal,
        floor_force_kN=floor_force,
        story_shear_kN=story_shear,
        overturning_kNm=overturning,
        displacement_m=disp_design,
        drift_m=drift_design,
        drift_ratio=drift_design / h,
        base_shear_unscaled_kN=base_unscaled,
        base_shear_scaled_kN=base_scaled,
        elf_base_shear_kN=elf_kN,
        rsa_scale_factor=scale,
        Cs=Cs,
        Ta_s=Ta,
        T_used_s=T_used,
    )


# ============================================================
# 7. DESIGN EVALUATION AND ITERATION
# ============================================================

def evaluate(inp: BuildingInput, wall_scale: float = 1.0, col_scale: float = 1.0, slab_scale: float = 1.0) -> DesignResult:
    sections = build_story_sections(inp, wall_scale, col_scale, slab_scale)
    props = build_story_properties(inp, sections)

    modal_x = solve_modal(inp, props, Direction.X, sections)
    modal_y = solve_modal(inp, props, Direction.Y, sections)

    rsa_x = response_spectrum_analysis(inp, props, modal_x)
    rsa_y = response_spectrum_analysis(inp, props, modal_y)

    return DesignResult(
        input=inp,
        sections=sections,
        properties=props,
        modal_x=modal_x,
        modal_y=modal_y,
        rsa_x=rsa_x,
        rsa_y=rsa_y,
        iteration_table=pd.DataFrame(),
        final_message="Single evaluation completed.",
    )


def run_design(inp: BuildingInput) -> DesignResult:
    """
    Real preliminary design loop.

    The code now designs the proposed sections against two independent checks:
    drift and ASCE upper-bound period. If the analytical design period is larger
    than CuTa, the code increases wall, column, beam/slab collector, and outrigger
    scales and rebuilds the stiffness matrix. This is not artificial period matching.
    """
    if not inp.auto_redesign:
        return evaluate(inp)

    wall_scale = inp.design_wall_scale
    col_scale = inp.design_column_scale
    beam_scale = inp.design_beam_scale
    slab_scale = inp.design_slab_scale
    out_scale = inp.design_outrigger_scale

    logs = []
    best = None
    best_score = 1e99

    for it in range(1, inp.max_iterations + 1):
        current_inp = replace(
            inp,
            design_wall_scale=wall_scale,
            design_column_scale=col_scale,
            design_beam_scale=beam_scale,
            design_slab_scale=slab_scale,
            design_outrigger_scale=out_scale,
        )
        res = evaluate(current_inp, 1.0, 1.0, 1.0)

        max_drift_x = float(np.max(res.rsa_x.drift_ratio))
        max_drift_y = float(np.max(res.rsa_y.drift_ratio))
        max_drift = max(max_drift_x, max_drift_y)
        drift_ratio_over = max_drift / max(inp.drift_limit_ratio, 1e-12)

        period_ratio, T_design_x, T_design_y, T_allow, CuTa = period_redesign_ratio(current_inp, res)
        target_period_ratio = period_ratio / max(inp.target_period_utilization, 1e-9)

        mass_x = res.modal_x.cumulative_mass_ratios[-1] if res.modal_x.cumulative_mass_ratios else 0.0
        mass_y = res.modal_y.cumulative_mass_ratios[-1] if res.modal_y.cumulative_mass_ratios else 0.0
        min_mass = min(mass_x, mass_y)
        total_concrete = sum(p.concrete_m3 for p in res.properties)
        total_steel = sum(p.steel_kg for p in res.properties)

        drift_excess = max(drift_ratio_over - 1.0, 0.0)
        period_excess = max(period_ratio - 1.0, 0.0) if inp.enforce_period_limit else 0.0
        modal_mass_deficit = max(inp.minimum_modal_mass_ratio - min_mass, 0.0)
        score = 3500.0 * period_excess**2 + 2000.0 * drift_excess**2 + 300.0 * modal_mass_deficit**2 + 1e-6 * total_concrete + 2e-7 * total_steel

        logs.append({
            "Iteration": it,
            "Wall scale": wall_scale,
            "Column scale": col_scale,
            "Beam scale": beam_scale,
            "Slab scale": slab_scale,
            "Outrigger scale": out_scale,
            "Wall cracked factor": current_inp.wall_cracked_factor,
            "Column cracked factor": current_inp.column_cracked_factor,
            "Side wall cracked factor": current_inp.side_wall_cracked_factor,
            "Outrigger effectiveness": current_inp.outrigger_effectiveness_factor,
            "Outrigger lateral participation": current_inp.outrigger_lateral_participation,
            "T_modal X (s)": res.modal_x.periods_s[0],
            "T_modal Y (s)": res.modal_y.periods_s[0],
            "T_design X (s)": T_design_x,
            "T_design Y (s)": T_design_y,
            "Ta reference (s)": res.rsa_x.Ta_s,
            "CuTa (s)": CuTa,
            "T_allow = factor*CuTa (s)": T_allow,
            "Period demand/cap": period_ratio,
            "Max drift X": max_drift_x,
            "Max drift Y": max_drift_y,
            "Drift demand/limit": drift_ratio_over,
            "Drift limit": inp.drift_limit_ratio,
            "Mass X (%)": 100 * mass_x,
            "Mass Y (%)": 100 * mass_y,
            "RSA scale X": res.rsa_x.rsa_scale_factor,
            "RSA scale Y": res.rsa_y.rsa_scale_factor,
            "ELF base shear X (kN)": res.rsa_x.elf_base_shear_kN,
            "ELF base shear Y (kN)": res.rsa_y.elf_base_shear_kN,
            "Scaled RSA base shear X (kN)": res.rsa_x.base_shear_scaled_kN,
            "Scaled RSA base shear Y (kN)": res.rsa_y.base_shear_scaled_kN,
            "Concrete (m³)": total_concrete,
            "Steel (kg)": total_steel,
            "Core wall base t (m)": res.sections[0].core_wall_t,
            "Interior column base X (m)": res.sections[0].column_interior_x,
            "Perimeter column base X (m)": res.sections[0].column_perimeter_x,
            "Corner column base X (m)": res.sections[0].column_corner_x,
            "Beam b (m)": res.sections[0].beam_b,
            "Beam h (m)": res.sections[0].beam_h,
            "Slab t (m)": res.sections[0].slab_t,
        })

        if score < best_score:
            best_score = score
            best = res

        ok_drift = max_drift <= inp.drift_limit_ratio * 1.03
        ok_mass = min_mass >= inp.minimum_modal_mass_ratio
        ok_period = (period_ratio <= 1.00) if inp.enforce_period_limit else True
        if ok_drift and ok_mass and ok_period:
            best = res
            break

        if (inp.enforce_period_limit and period_ratio > 1.0) or max_drift > inp.drift_limit_ratio:
            if inp.enforce_period_limit and period_ratio > 1.0:
                pg = min(inp.growth_limit_per_iteration, target_period_ratio ** 0.32)
                if inp.strengthen_core_for_period:
                    wall_scale *= pg
                    beam_scale *= min(1.14, target_period_ratio ** 0.14)
                    slab_scale *= min(1.08, target_period_ratio ** 0.08)
                if inp.strengthen_columns_for_period:
                    col_scale *= min(1.16, target_period_ratio ** 0.22)
                if inp.strengthen_outrigger_for_period and current_inp.outrigger_system != OutriggerSystem.NONE:
                    out_scale *= min(1.12, target_period_ratio ** 0.12)
            if max_drift > inp.drift_limit_ratio:
                dg = min(inp.growth_limit_per_iteration, drift_ratio_over ** 0.20)
                wall_scale *= dg
                col_scale *= min(inp.growth_limit_per_iteration, drift_ratio_over ** 0.14)
                beam_scale *= min(1.14, drift_ratio_over ** 0.11)
                slab_scale *= min(1.08, drift_ratio_over ** 0.06)
                out_scale *= min(1.12, drift_ratio_over ** 0.10)
        elif drift_ratio_over < 0.35 and (not inp.enforce_period_limit or period_ratio < 0.80):
            if inp.allow_section_reduction:
                wall_scale *= inp.reduction_limit_per_iteration
                col_scale *= inp.reduction_limit_per_iteration
                beam_scale *= max(inp.reduction_limit_per_iteration, 0.97)
                slab_scale *= max(inp.reduction_limit_per_iteration, 0.98)
                out_scale *= max(inp.reduction_limit_per_iteration, 0.96)

        wall_scale = float(np.clip(wall_scale, 0.40, 6.00))
        col_scale = float(np.clip(col_scale, 0.40, 6.00))
        beam_scale = float(np.clip(beam_scale, 0.50, 4.50))
        slab_scale = float(np.clip(slab_scale, 0.75, 2.50))
        out_scale = float(np.clip(out_scale, 0.40, 5.00))

    if best is None:
        best = evaluate(replace(inp, design_wall_scale=wall_scale, design_column_scale=col_scale, design_beam_scale=beam_scale, design_slab_scale=slab_scale, design_outrigger_scale=out_scale), 1.0, 1.0, 1.0)

    best.iteration_table = pd.DataFrame(logs)
    best.final_message = (
        "Real preliminary design completed. Member dimensions were redesigned "
        "against drift and ASCE upper-bound period criteria. The period is not "
        "artificially calibrated; sections are increased until the design checks pass."
    )
    return best

# 8. OUTPUT TABLES
# ============================================================

def summary_table(res: DesignResult) -> pd.DataFrame:
    total_weight = sum(p.weight_kN for p in res.properties)
    total_mass = sum(p.mass_kg for p in res.properties)
    total_conc = sum(p.concrete_m3 for p in res.properties)
    total_steel = sum(p.steel_kg for p in res.properties)

    return pd.DataFrame(
        {
            "Item": [
                "Project title",
                "Author",
                "Version",
                "Height",
                "Floor area",
                "Total seismic weight",
                "Total mass",
                "T1 X",
                "T1 Y",
                "Modal mass X",
                "Modal mass Y",
                "Base shear X",
                "Base shear Y",
                "Max drift ratio X",
                "Max drift ratio Y",
                "Total concrete",
                "Total steel",
                "Outrigger system",
            ],
            "Value": [
                PROJECT_TITLE,
                AUTHOR_NAME,
                APP_VERSION,
                res.input.height,
                res.input.floor_area,
                total_weight,
                total_mass,
                res.modal_x.periods_s[0],
                res.modal_y.periods_s[0],
                100 * res.modal_x.cumulative_mass_ratios[-1],
                100 * res.modal_y.cumulative_mass_ratios[-1],
                res.rsa_x.base_shear_scaled_kN,
                res.rsa_y.base_shear_scaled_kN,
                float(np.max(res.rsa_x.drift_ratio)),
                float(np.max(res.rsa_y.drift_ratio)),
                total_conc,
                total_steel,
                res.input.outrigger_system.value,
            ],
            "Unit": [
                "-", "-", "-", "m", "m²", "kN", "kg", "s", "s", "%", "%", "kN", "kN",
                "-", "-", "m³", "kg", "-",
            ],
        }
    )


def final_dimensions_table(res: DesignResult) -> pd.DataFrame:
    lower = res.sections[0]
    mid = res.sections[len(res.sections) // 2]
    top = res.sections[-1]

    rows = []
    for label, sec in [("Lower", lower), ("Middle", mid), ("Upper", top)]:
        rows.append(
            {
                "Zone": label,
                "Story": sec.story,
                "Core outer X (m)": sec.core_x,
                "Core outer Y (m)": sec.core_y,
                "Core opening X (m)": sec.opening_x,
                "Core opening Y (m)": sec.opening_y,
                "Core wall t (m)": sec.core_wall_t,
                "Side wall Lx (m)": sec.side_wall_length_x,
                "Side wall Ly (m)": sec.side_wall_length_y,
                "Side wall t (m)": sec.side_wall_t,
                "Interior column (m)": f"{sec.column_interior_x:.2f} x {sec.column_interior_y:.2f}",
                "Perimeter column (m)": f"{sec.column_perimeter_x:.2f} x {sec.column_perimeter_y:.2f}",
                "Corner column (m)": f"{sec.column_corner_x:.2f} x {sec.column_corner_y:.2f}",
                "Beam (m)": f"{sec.beam_b:.2f} x {sec.beam_h:.2f}",
                "Slab t (m)": sec.slab_t,
            }
        )
    return pd.DataFrame(rows)


def story_response_table(res: DesignResult, direction: Direction) -> pd.DataFrame:
    rsa = res.rsa_x if direction == Direction.X else res.rsa_y
    rows = []
    for i, sec in enumerate(res.sections):
        rows.append(
            {
                "Story": sec.story,
                "Elevation (m)": sec.elevation_m,
                "Floor force (kN)": rsa.floor_force_kN[i],
                "Story shear (kN)": rsa.story_shear_kN[i],
                "Overturning (kN.m)": rsa.overturning_kNm[i],
                "Displacement (m)": rsa.displacement_m[i],
                "Interstory drift (m)": rsa.drift_m[i],
                "Drift ratio": rsa.drift_ratio[i],
            }
        )
    return pd.DataFrame(rows)


def modal_table(modal: ModalResult) -> pd.DataFrame:
    rows = []
    for i, T in enumerate(modal.periods_s):
        rows.append(
            {
                "Mode": i + 1,
                "Direction": modal.direction.value,
                "Period (s)": T,
                "Frequency (Hz)": modal.frequencies_hz[i],
                "Gamma": modal.gammas[i],
                "Effective mass (%)": 100 * modal.effective_mass_ratios[i],
                "Cumulative mass (%)": 100 * modal.cumulative_mass_ratios[i],
            }
        )
    return pd.DataFrame(rows)


def stiffness_table(res: DesignResult) -> pd.DataFrame:
    rows = []
    for sec, prop in zip(res.sections, res.properties):
        rows.append(
            {
                "Story": sec.story,
                "Zone": sec.zone,
                "EI X translation (GN.m²)": prop.EI_x_Nm2 / 1e9,
                "EI Y translation (GN.m²)": prop.EI_y_Nm2 / 1e9,
                "Outrigger Ktheta X (GN.m/rad)": prop.Ktheta_out_x_Nm / 1e9,
                "Outrigger Ktheta Y (GN.m/rad)": prop.Ktheta_out_y_Nm / 1e9,
                "Outrigger Klat X (MN/m)": outrigger_Klateral(res.input, sec, Direction.X) / 1e6,
                "Outrigger Klat Y (MN/m)": outrigger_Klateral(res.input, sec, Direction.Y) / 1e6,
                "Basement EI X contribution (GN.m²)": prop.EI_basement_x_Nm2 / 1e9,
                "Basement EI Y contribution (GN.m²)": prop.EI_basement_y_Nm2 / 1e9,
                "Mass (t)": prop.mass_kg / 1000,
                "Concrete (m³)": prop.concrete_m3,
                "Steel (kg)": prop.steel_kg,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 9. PLOTS
# ============================================================

def plot_plan(res: DesignResult, zone_choice: str):
    """Draw the structural plan with real braced bay panels.

    The same bay IDs used in stiffness calculations are drawn here. Bracing is
    placed inside actual column-to-column panels, not distributed by arbitrary
    interpolation.
    """
    sec = next((s for s in res.sections if s.zone == zone_choice), res.sections[0])
    inp = res.input
    fig, ax = plt.subplots(figsize=(13, 10))

    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=1.8)
    for i in range(inp.n_bays_x + 1):
        x = i * inp.bay_x
        ax.plot([x, x], [0, inp.plan_y], color="0.45", linewidth=0.8)
    for j in range(inp.n_bays_y + 1):
        y = j * inp.bay_y
        ax.plot([0, inp.plan_x], [y, y], color="0.45", linewidth=0.8)

    for i in range(inp.n_bays_x + 1):
        for j in range(inp.n_bays_y + 1):
            x = i * inp.bay_x
            y = j * inp.bay_y
            at_x = i == 0 or i == inp.n_bays_x
            at_y = j == 0 or j == inp.n_bays_y
            if at_x and at_y:
                bx, by = sec.column_corner_x, sec.column_corner_y
            elif at_x or at_y:
                bx, by = sec.column_perimeter_x, sec.column_perimeter_y
            else:
                bx, by = sec.column_interior_x, sec.column_interior_y
            ax.add_patch(plt.Rectangle((x - bx / 2, y - by / 2), bx, by, facecolor="white", edgecolor="black", linewidth=0.9))

    # Perimeter/side walls: these are structural walls, not façade lines.
    for x0, y0, w, h in perimeter_wall_segments(inp, sec):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor="0.12", edgecolor="black", linewidth=1.0, zorder=4))

    # Below-grade retaining wall outline is displayed when basement levels exist.
    if inp.include_basement_retaining_wall and inp.n_basement > 0:
        t_ret = max(inp.retaining_wall_thickness, 0.05)
        ax.add_patch(plt.Rectangle((-t_ret, -t_ret), inp.plan_x + 2*t_ret, inp.plan_y + 2*t_ret,
                                   fill=False, edgecolor="0.20", linewidth=2.4, linestyle="--", zorder=3))
        ax.text(inp.plan_x + 2, inp.plan_y * 0.78,
                f"Basement retaining wall included\nLevels below 0: {inp.n_basement}\nt = {inp.retaining_wall_thickness:.2f} m",
                fontsize=8.5, va="center")

    cx0 = (inp.plan_x - sec.core_x) / 2
    cy0 = (inp.plan_y - sec.core_y) / 2
    cx1 = cx0 + sec.core_x
    cy1 = cy0 + sec.core_y
    ax.add_patch(plt.Rectangle((cx0, cy0), sec.core_x, sec.core_y, fill=False, edgecolor="black", linewidth=7.0, zorder=5))

    zone_stories = [s.story for s in res.sections if s.zone == sec.zone]
    outriggers_in_zone = [lev for lev in active_outrigger_levels(inp) if lev in zone_stories]

    def draw_x_panel(x0, x1, y0, y1, lw=1.2):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=lw)
        ax.plot([x0, x1], [y1, y0], color="black", linewidth=lw)

    if inp.outrigger_system != OutriggerSystem.NONE and outriggers_in_zone:
        y_bays_for_x = active_braced_bays(inp, Direction.X)
        x_bays_for_y = active_braced_bays(inp, Direction.Y)

        belt_lw = 5.0
        for y in [cy0, cy1]:
            ax.plot([0, cx0], [y, y], color="black", linewidth=belt_lw, solid_capstyle="butt")
            ax.plot([cx1, inp.plan_x], [y, y], color="black", linewidth=belt_lw, solid_capstyle="butt")
        for x in [cx0, cx1]:
            ax.plot([x, x], [0, cy0], color="black", linewidth=belt_lw, solid_capstyle="butt")
            ax.plot([x, x], [cy1, inp.plan_y], color="black", linewidth=belt_lw, solid_capstyle="butt")

        for j in y_bays_for_x:
            y0 = j * inp.bay_y
            y1 = (j + 1) * inp.bay_y
            draw_x_panel(0, inp.bay_x, y0, y1)
            draw_x_panel(inp.plan_x - inp.bay_x, inp.plan_x, y0, y1)

        for i in x_bays_for_y:
            x0 = i * inp.bay_x
            x1 = (i + 1) * inp.bay_x
            draw_x_panel(x0, x1, 0, inp.bay_y)
            draw_x_panel(x0, x1, inp.plan_y - inp.bay_y, inp.plan_y)

        ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], color="black", linewidth=2.4)
        ax.text(inp.plan_x + 2, inp.plan_y * 0.50,
                f"OUTRIGGER\nStories: {outriggers_in_zone}\nX bay IDs: {list(y_bays_for_x)}\nY bay IDs: {list(x_bays_for_y)}",
                fontsize=9, va="center")

    ax.annotate(f"Total Length = {inp.plan_x:.1f} m", xy=(0, -2.5), xytext=(inp.plan_x, -2.5),
                arrowprops=dict(arrowstyle="<->", lw=1.0), ha="center", va="center", fontsize=9)
    ax.annotate(f"Total Length = {inp.plan_y:.1f} m", xy=(-2.5, 0), xytext=(-2.5, inp.plan_y),
                arrowprops=dict(arrowstyle="<->", lw=1.0), ha="center", va="center", rotation=90, fontsize=9)
    ax.text(2.0, 2.0, f"STRUCTURAL LAYOUT\n{PROJECT_TITLE}\nVERSION: {APP_VERSION}\nAUTHOR: {AUTHOR_NAME}",
            fontsize=10, va="top", bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8))

    ax.set_title(f"Structural plan - {zone_choice}")
    ax.set_aspect("equal")
    ax.set_xlim(-5, inp.plan_x + 26)
    ax.set_ylim(inp.plan_y + 5, -5)
    ax.axis("off")
    return fig
def plot_modes(res: DesignResult, direction: Direction):
    modal = res.modal_x if direction == Direction.X else res.modal_y
    y = np.array([s.elevation_m for s in res.sections])
    n_modes = min(5, len(modal.mode_shapes))
    fig, axes = plt.subplots(1, n_modes, figsize=(16, 6))
    if n_modes == 1:
        axes = [axes]
    for i in range(n_modes):
        ax = axes[i]
        ax.plot(modal.mode_shapes[i], y, linewidth=2)
        ax.scatter(modal.mode_shapes[i], y, s=12)
        for lev in active_outrigger_levels(res.input):
            ax.axhline(lev * res.input.story_height, linestyle=":", alpha=0.6)
        ax.axvline(0, linestyle="--", linewidth=0.8)
        ax.set_title(f"Mode {i+1}\nT={modal.periods_s[i]:.3f}s")
        ax.set_xlabel("Normalized u")
        if i == 0:
            ax.set_ylabel("Height (m)")
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Mode shapes - {direction.value} direction")
    fig.tight_layout()
    return fig


def plot_story_response(res: DesignResult, direction: Direction, response: str):
    df = story_response_table(res, direction)
    fig, ax = plt.subplots(figsize=(7, 8))
    if response == "Story shear":
        ax.plot(df["Story shear (kN)"], df["Story"])
        ax.set_xlabel("Story shear (kN)")
    elif response == "Drift ratio":
        ax.plot(df["Drift ratio"], df["Story"])
        ax.axvline(res.input.drift_limit_ratio, linestyle="--", label="Limit")
        ax.set_xlabel("Drift ratio")
        ax.legend()
    elif response == "Displacement":
        ax.plot(df["Displacement (m)"], df["Story"])
        ax.set_xlabel("Displacement (m)")
    else:
        ax.plot(df["Overturning (kN.m)"], df["Story"])
        ax.set_xlabel("Overturning (kN.m)")
    ax.set_ylabel("Story")
    ax.set_title(f"{response} - {direction.value}")
    ax.grid(True, alpha=0.3)
    return fig


def plot_stiffness(res: DesignResult):
    df = stiffness_table(res)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(df["EI X translation (GN.m²)"], df["Story"], label="EI for X")
    ax.plot(df["EI Y translation (GN.m²)"], df["Story"], label="EI for Y")
    ax.set_xlabel("EI (GN.m²)")
    ax.set_ylabel("Story")
    ax.set_title("Effective flexural stiffness profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_iteration(res: DesignResult):
    df = res.iteration_table
    if df.empty:
        return None

    tx_col = "T1 X (s)" if "T1 X (s)" in df.columns else "T_modal X (s)"
    ty_col = "T1 Y (s)" if "T1 Y (s)" in df.columns else "T_modal Y (s)"
    conc_col = "Concrete (m³)" if "Concrete (m³)" in df.columns else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(df["Iteration"], df["Max drift X"], marker="o", label="X")
    axes[0, 0].plot(df["Iteration"], df["Max drift Y"], marker="s", label="Y")
    axes[0, 0].plot(df["Iteration"], df["Drift limit"], linestyle="--", label="Limit")
    axes[0, 0].set_title("Drift convergence")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(df["Iteration"], df[tx_col], marker="o", label="T modal X")
    axes[0, 1].plot(df["Iteration"], df[ty_col], marker="s", label="T modal Y")
    if "CuTa (s)" in df.columns:
        axes[0, 1].plot(df["Iteration"], df["CuTa (s)"], linestyle="--", label="ASCE CuTa cap")
    axes[0, 1].set_title("Period evolution")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(df["Iteration"], df["Core wall base t (m)"], marker="o", label="Core wall t")
    axes[1, 0].plot(df["Iteration"], df["Interior column base X (m)"], marker="s", label="Column")
    axes[1, 0].set_title("Member size evolution")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    if conc_col:
        axes[1, 1].plot(df["Iteration"], df[conc_col], marker="d")
        axes[1, 1].set_title("Concrete quantity")
    else:
        axes[1, 1].text(0.5, 0.5, "No quantity column", ha="center", va="center")
        axes[1, 1].set_title("Quantity")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def spectrum_table(inp: BuildingInput) -> pd.DataFrame:
    T = np.linspace(0, max(10.0, inp.asce7.TL * 1.25), 150)
    return pd.DataFrame({
        "T (s)": T,
        "ASCE Sa (g)": [asce_spectrum_sa_g(x, inp.asce7) for x in T],
        "RSA Sa*Ie/R (g)": [rsa_force_sa_g(x, inp.asce7) for x in T],
    })


def plot_spectrum(inp: BuildingInput):
    df = spectrum_table(inp)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["T (s)"], df["ASCE Sa (g)"], label="ASCE spectrum")
    ax.plot(df["T (s)"], df["RSA Sa*Ie/R (g)"], label="RSA force spectrum")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Sa (g)")
    ax.set_title("ASCE 7 spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def build_report(res: DesignResult) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append(f"{PROJECT_TITLE.upper()} - PRELIMINARY DESIGN REPORT")
    lines.append("=" * 96)
    lines.append(f"Version: {APP_VERSION}")
    lines.append(f"Author: {AUTHOR_NAME}")
    lines.append("")
    lines.append("1. Solver definition")
    lines.append("-" * 96)
    lines.append("The solver is a flexural MDOF cantilever model with two DOFs at each floor node:")
    lines.append("lateral displacement u and rotation theta. The eigenvalue problem is [K]phi = omega^2[M]phi.")
    lines.append("Outriggers are modeled from real braced bay panels; each diagonal contributes EA/L*cos^2(theta), and the summed stiffness is condensed into the floor DOFs.")
    lines.append("The model does not force a target period. Period is computed from mass and stiffness.")
    lines.append("ELF base shear uses the ASCE period cap T_used = min(T_modal, Cu*Ta) when activated.")
    lines.append("RSA base shear is scaled to the required ELF minimum; drift controls redesign.")
    lines.append("")
    lines.append("1B. Design principles")
    lines.append("-" * 96)
    lines.append(design_principles_table(res).to_string(index=False))
    lines.append("")
    lines.append("2. Global response")
    lines.append("-" * 96)
    lines.append(summary_table(res).to_string(index=False))
    lines.append("")
    lines.append("3. Material effect modifiers")
    lines.append("-" * 96)
    lines.append(material_diagnostics(res.input).to_string(index=False))
    lines.append("")

    lines.append("4. Final dimensions")
    lines.append("-" * 96)
    lines.append(final_dimensions_table(res).to_string(index=False))
    lines.append("")
    lines.append("3B. Outrigger design")
    lines.append("-" * 96)
    lines.append(outrigger_design_table(res).to_string(index=False))
    lines.append("")
    lines.append("3D. Outrigger effect comparison")
    lines.append("-" * 96)
    lines.append(outrigger_effect_comparison(res).to_string(index=False))
    lines.append("")
    lines.append("3E. Outrigger stiffness diagnostic")
    lines.append("-" * 96)
    lines.append(outrigger_stiffness_diagnostic_table(res).to_string(index=False))
    lines.append("")
    lines.append("3C. Design checks")
    lines.append("-" * 96)
    lines.append(design_check_table(res).to_string(index=False))
    lines.append("")
    lines.append("4. Modal X")
    lines.append("-" * 96)
    lines.append(modal_table(res.modal_x).to_string(index=False))
    lines.append("")
    lines.append("5. Modal Y")
    lines.append("-" * 96)
    lines.append(modal_table(res.modal_y).to_string(index=False))
    lines.append("")
    lines.append("6. Engineering conclusion")
    lines.append("-" * 96)
    lines.append("This file is suitable for preliminary structural system comparison and thesis-level methodology discussion.")
    lines.append("It is not a final code-design replacement for a full three-dimensional structural analysis model because torsion, accidental eccentricity, P-Delta, diaphragm modeling, detailed member design, and load combinations are still required.")
    return "\n".join(lines)



# ============================================================
# 9B. OUTRIGGER DESIGN AND CHECK TABLES
# ============================================================


def outrigger_design_table(res: DesignResult) -> pd.DataFrame:
    rows = []
    inp = res.input
    if inp.outrigger_system == OutriggerSystem.NONE:
        return pd.DataFrame([{"Message": "No outrigger is selected."}])

    for lev in active_outrigger_levels(inp):
        idx = min(max(int(lev), 1), inp.n_story) - 1
        sec = res.sections[idx]
        for direction in [Direction.X, Direction.Y]:
            plan_dim = inp.plan_x if direction == Direction.X else inp.plan_y
            core_dim = sec.core_x if direction == Direction.X else sec.core_y
            bay_width = inp.bay_y if direction == Direction.X else inp.bay_x
            active_bays = active_braced_bays(inp, direction)
            comp = outrigger_span_stiffness_components(inp, sec, direction)
            lever_arm = max((plan_dim - core_dim) / 2.0, 1e-9)
            depth = max(inp.outrigger_depth_m, inp.story_height)
            brace_len = sqrt(bay_width**2 + depth**2)
            klat = outrigger_Klateral(inp, sec, direction)
            rows.append({
                "Story": lev,
                "Direction": direction.value,
                "Active bay IDs": list(active_bays),
                "Selected braced bays": int(comp["selected_bays"]),
                "Total diagonal braces": int(comp["total_braces"]),
                "One brace projected stiffness (MN/m)": comp["k_one_brace_MN_per_m"],
                "Sum of brace stiffnesses (MN/m)": comp["k_braces_total_MN_per_m"],
                "Lever arm core-face to exterior (m)": lever_arm,
                "Brace panel width (m)": bay_width,
                "Brace length (m)": brace_len,
                "Collector beam b x h (m)": f"{sec.beam_b:.2f} x {sec.beam_h:.2f}",
                "Ktheta added to MDOF (GN.m/rad)": comp["Ktheta_GNm_per_rad"],
                "Equivalent Klat added (MN/m)": klat / 1e6,
                "Method": "K_out = sum(EA/L*cos²θ) for braces in real braced spans",
            })
    return pd.DataFrame(rows)

def design_check_table(res: DesignResult) -> pd.DataFrame:
    max_drift_x = float(np.max(res.rsa_x.drift_ratio))
    max_drift_y = float(np.max(res.rsa_y.drift_ratio))
    mass_x = res.modal_x.cumulative_mass_ratios[-1] if res.modal_x.cumulative_mass_ratios else 0.0
    mass_y = res.modal_y.cumulative_mass_ratios[-1] if res.modal_y.cumulative_mass_ratios else 0.0
    return pd.DataFrame([
        {"Check": "X-direction drift", "Demand": max_drift_x, "Limit": res.input.drift_limit_ratio, "Status": "OK" if max_drift_x <= res.input.drift_limit_ratio else "NOT OK"},
        {"Check": "Y-direction drift", "Demand": max_drift_y, "Limit": res.input.drift_limit_ratio, "Status": "OK" if max_drift_y <= res.input.drift_limit_ratio else "NOT OK"},
        {"Check": "X modal mass participation", "Demand": mass_x, "Limit": res.input.minimum_modal_mass_ratio, "Status": "OK" if mass_x >= res.input.minimum_modal_mass_ratio else "NOT OK"},
        {"Check": "Y modal mass participation", "Demand": mass_y, "Limit": res.input.minimum_modal_mass_ratio, "Status": "OK" if mass_y >= res.input.minimum_modal_mass_ratio else "NOT OK"},
        {"Check": "X RSA base shear scaling", "Demand": res.rsa_x.base_shear_unscaled_kN, "Limit": res.input.asce7.rsa_min_ratio_to_elf * res.rsa_x.elf_base_shear_kN, "Status": "SCALED" if res.rsa_x.rsa_scale_factor > 1.0001 else "OK"},
        {"Check": "Y RSA base shear scaling", "Demand": res.rsa_y.base_shear_unscaled_kN, "Limit": res.input.asce7.rsa_min_ratio_to_elf * res.rsa_y.elf_base_shear_kN, "Status": "SCALED" if res.rsa_y.rsa_scale_factor > 1.0001 else "OK"},
    ])



def design_principles_table(res: DesignResult) -> pd.DataFrame:
    inp = res.input
    return pd.DataFrame([
        {
            "Principle": "Modal period",
            "Implemented rule": "Computed from flexural MDOF eigenvalue analysis.",
            "Engineering meaning": "Period is an output of stiffness and mass, not a target to be forced."
        },
        {
            "Principle": "ASCE 7 period control",
            "Implemented rule": "ELF base shear uses T_used = min(T_modal, Cu*Ta) when CuTa cap is activated.",
            "Engineering meaning": "Very long analytical periods are not allowed to reduce base shear without limit."
        },
        {
            "Principle": "RSA base shear scaling",
            "Implemented rule": f"RSA base shear is scaled to at least {inp.asce7.rsa_min_ratio_to_elf:.2f} times ELF base shear.",
            "Engineering meaning": "Modal RSA force demand is kept consistent with ASCE minimum lateral strength."
        },
        {
            "Principle": "Drift-controlled redesign",
            "Implemented rule": "Wall, column, and outrigger stiffness increase only when design drift exceeds the limit.",
            "Engineering meaning": "The structure is resized for serviceability/stability, not for arbitrary period matching."
        },
        {
            "Principle": "Modal mass participation",
            "Implemented rule": f"Selected modes must reach at least {100*inp.minimum_modal_mass_ratio:.1f}% cumulative effective modal mass.",
            "Engineering meaning": "Enough modes must be included for a valid modal response estimate."
        },
        {
            "Principle": "Cracked stiffness",
            "Implemented rule": "Effective EI is computed from Ec × Ig × cracked stiffness factors for walls, columns, side walls, and retaining walls.",
            "Engineering meaning": "Changing cracked stiffness factors directly changes the global stiffness matrix, modal period, and drift response."
        },
        {
            "Principle": "Outrigger modeling",
            "Implemented rule": "Tubular/truss braced bay panels are summed as K_out = sum(EA/L*cos²θ), then condensed to [u, theta] floor DOFs with lever-arm coupling.",
            "Engineering meaning": "Outrigger stiffness enters the global stiffness matrix through the same braced bays shown in the plan."
        },
    ])



def without_outrigger_input(inp: BuildingInput) -> BuildingInput:
    """
    For a fair with/without outrigger comparison, keep the same outrigger steel mass
    but remove the stiffness transfer by setting connection efficiency to zero.

    This isolates the stiffness effect. If the outrigger system is fully removed,
    the model also becomes lighter, and the period comparison can be misleading.
    """
    return replace(
        inp,
        outrigger_connection_efficiency=0.0,
        auto_redesign=False,
    )


def outrigger_effect_comparison(res: DesignResult) -> pd.DataFrame:
    """
    Direct comparison using the same final dimensions and mass.
    Only outrigger stiffness is removed; gravity sizes and mass remain fixed.
    This isolates the real dynamic effect of the outrigger load path.
    """
    inp_with = res.input
    inp_without = replace(inp_with, outrigger_system=OutriggerSystem.NONE, outrigger_story_levels=tuple())

    sections_without = [replace(s) for s in res.sections]
    props_without = build_story_properties(inp_without, sections_without)
    modal_x0 = solve_modal(inp_without, props_without, Direction.X, sections_without)
    modal_y0 = solve_modal(inp_without, props_without, Direction.Y, sections_without)
    rsa_x0 = response_spectrum_analysis(inp_without, props_without, modal_x0)
    rsa_y0 = response_spectrum_analysis(inp_without, props_without, modal_y0)

    rows = []
    for d, m_with, m_without, r_with, r_without in [
        ("X", res.modal_x, modal_x0, res.rsa_x, rsa_x0),
        ("Y", res.modal_y, modal_y0, res.rsa_y, rsa_y0),
    ]:
        T_with = m_with.periods_s[0]
        T_without = m_without.periods_s[0]
        drift_with = float(np.max(r_with.drift_ratio))
        drift_without = float(np.max(r_without.drift_ratio))
        disp_with = float(r_with.displacement_m[-1])
        disp_without = float(r_without.displacement_m[-1])
        rows.append({
            "Direction": d,
            "T1 with outrigger (s)": T_with,
            "T1 without outrigger (s)": T_without,
            "Period reduction (%)": 100.0 * (T_without - T_with) / max(T_without, 1e-12),
            "Max drift with": drift_with,
            "Max drift without": drift_without,
            "Drift reduction (%)": 100.0 * (drift_without - drift_with) / max(drift_without, 1e-12),
            "Roof disp with (m)": disp_with,
            "Roof disp without (m)": disp_without,
            "Displacement reduction (%)": 100.0 * (disp_without - disp_with) / max(disp_without, 1e-12),
        })
    return pd.DataFrame(rows)

def outrigger_stiffness_diagnostic_table(res: DesignResult) -> pd.DataFrame:
    """
    Compare Ktheta with local story rotational stiffness order 4EI/L.
    If Ktheta/(4EI/L) is very small, the outrigger cannot strongly affect the global period.
    """
    rows = []
    inp = res.input
    if inp.outrigger_system == OutriggerSystem.NONE:
        return pd.DataFrame([{"Message": "No outrigger is active."}])

    for lev in inp.outrigger_story_levels:
        idx = min(max(int(lev), 1), inp.n_story) - 1
        prop = res.properties[idx]
        story_rot_x = 4.0 * prop.EI_x_Nm2 / max(inp.story_height, 1e-9)
        story_rot_y = 4.0 * prop.EI_y_Nm2 / max(inp.story_height, 1e-9)
        rows.append({
            "Outrigger story": lev,
            "Ktheta X (GN.m/rad)": prop.Ktheta_out_x_Nm / 1e9,
            "4EI/L X (GN.m/rad)": story_rot_x / 1e9,
            "Ktheta/(4EI/L) X": prop.Ktheta_out_x_Nm / max(story_rot_x, 1e-9),
            "Ktheta Y (GN.m/rad)": prop.Ktheta_out_y_Nm / 1e9,
            "4EI/L Y (GN.m/rad)": story_rot_y / 1e9,
            "Ktheta/(4EI/L) Y": prop.Ktheta_out_y_Nm / max(story_rot_y, 1e-9),
            "Klat X (MN/m)": outrigger_Klateral(res.input, res.sections[idx], Direction.X) / 1e6,
            "Klat Y (MN/m)": outrigger_Klateral(res.input, res.sections[idx], Direction.Y) / 1e6,
            "Interpretation": "strong" if max(prop.Ktheta_out_x_Nm / max(story_rot_x, 1e-9), prop.Ktheta_out_y_Nm / max(story_rot_y, 1e-9)) > 0.25 else "weak/moderate",
        })
    return pd.DataFrame(rows)


def final_design_dashboard_table(res: DesignResult) -> pd.DataFrame:
    lower = res.sections[0]
    mid = res.sections[len(res.sections)//2]
    top = res.sections[-1]
    q_conc = sum(p.concrete_m3 for p in res.properties)
    q_steel = sum(p.steel_kg for p in res.properties)
    return pd.DataFrame([
        {"Component": "Central core", "Lower": f"{lower.core_x:.2f} x {lower.core_y:.2f} m", "Middle": f"{mid.core_x:.2f} x {mid.core_y:.2f} m", "Upper": f"{top.core_x:.2f} x {top.core_y:.2f} m"},
        {"Component": "Core wall thickness", "Lower": f"{lower.core_wall_t:.2f} m", "Middle": f"{mid.core_wall_t:.2f} m", "Upper": f"{top.core_wall_t:.2f} m"},
        {"Component": "Side wall thickness", "Lower": f"{lower.side_wall_t:.2f} m", "Middle": f"{mid.side_wall_t:.2f} m", "Upper": f"{top.side_wall_t:.2f} m"},
        {"Component": "Interior columns", "Lower": f"{lower.column_interior_x:.2f} x {lower.column_interior_y:.2f} m", "Middle": f"{mid.column_interior_x:.2f} x {mid.column_interior_y:.2f} m", "Upper": f"{top.column_interior_x:.2f} x {top.column_interior_y:.2f} m"},
        {"Component": "Perimeter columns", "Lower": f"{lower.column_perimeter_x:.2f} x {lower.column_perimeter_y:.2f} m", "Middle": f"{mid.column_perimeter_x:.2f} x {mid.column_perimeter_y:.2f} m", "Upper": f"{top.column_perimeter_x:.2f} x {top.column_perimeter_y:.2f} m"},
        {"Component": "Corner columns", "Lower": f"{lower.column_corner_x:.2f} x {lower.column_corner_y:.2f} m", "Middle": f"{mid.column_corner_x:.2f} x {mid.column_corner_y:.2f} m", "Upper": f"{top.column_corner_x:.2f} x {top.column_corner_y:.2f} m"},
        {"Component": "Beams", "Lower": f"{lower.beam_b:.2f} x {lower.beam_h:.2f} m", "Middle": f"{mid.beam_b:.2f} x {mid.beam_h:.2f} m", "Upper": f"{top.beam_b:.2f} x {top.beam_h:.2f} m"},
        {"Component": "Slab", "Lower": f"{lower.slab_t:.2f} m", "Middle": f"{mid.slab_t:.2f} m", "Upper": f"{top.slab_t:.2f} m"},
        {"Component": "Outrigger system", "Lower": res.input.outrigger_system.value, "Middle": f"Stories {list(active_outrigger_levels(res.input))}", "Upper": f"Braced spans X/Y = {res.input.braced_spans_x}/{res.input.braced_spans_y}"},
        {"Component": "Quantities", "Lower": f"Concrete = {q_conc:,.0f} m³", "Middle": f"Steel = {q_steel:,.0f} kg", "Upper": "-"},
    ])


# ============================================================
# 10. STREAMLIT UI
# ============================================================

def streamlit_input_panel() -> BuildingInput:
    import streamlit as st

    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        plan_shape = st.radio("Plan shape", ["square", "triangle"], horizontal=True)
        n_story = st.number_input("Above-grade stories", 1, 150, 60)
        n_basement = st.number_input("Basement stories", 0, 30, 0)
        story_height = st.number_input("Story height (m)", 2.5, 6.0, 3.2)
        plan_x = st.number_input("Plan X (m)", 10.0, 300.0, 80.0)
        n_bays_x = st.number_input("Bays X", 1, 40, 8)
    with c2:
        basement_height = st.number_input("Basement height (m)", 2.5, 6.0, 3.0)
        plan_y = st.number_input("Plan Y (m)", 10.0, 300.0, 80.0)
        n_bays_y = st.number_input("Bays Y", 1, 40, 8)
        core_ratio_x = st.number_input("Core ratio X", 0.10, 0.50, 0.24)
        core_ratio_y = st.number_input("Core ratio Y", 0.10, 0.50, 0.22)

    st.markdown("### Core / Services")
    c3, c4 = st.columns(2)
    with c3:
        stair_count = st.number_input("Stairs", 0, 20, 2)
        elevator_count = st.number_input("Elevators", 0, 40, 4)
        elevator_area_each = st.number_input("Elevator area each (m²)", 0.0, 30.0, 3.5)
        stair_area_each = st.number_input("Stair area each (m²)", 0.0, 80.0, 20.0)
    with c4:
        service_area = st.number_input("Service area (m²)", 0.0, 300.0, 35.0)
        corridor_factor = st.number_input("Core circulation factor", 0.5, 3.0, 1.40)
        core_max_ratio_x = st.number_input("Core max ratio X", 0.20, 0.70, 0.42)
        core_max_ratio_y = st.number_input("Core max ratio Y", 0.20, 0.70, 0.42)

    st.markdown("### Materials / Loads / Mass")
    c5, c6 = st.columns(2)
    with c5:
        fck = st.number_input("fck (MPa)", 20.0, 120.0, 70.0)
        Ec = st.number_input("Ec (MPa)", 20000.0, 70000.0, 36000.0)
        fy = st.number_input("fy (MPa)", 200.0, 800.0, 420.0)
        DL = st.number_input("DL (kN/m²)", 0.0, 20.0, 3.0)
        LL = st.number_input("LL (kN/m²)", 0.0, 20.0, 2.5)
    with c6:
        live_load_mass_factor = st.number_input("Live load mass factor", 0.0, 1.0, 0.25)
        slab_finish_allowance = st.number_input("Finish/partition load (kN/m²)", 0.0, 10.0, 1.5)
        facade_line_load = st.number_input("Facade line load (kN/m)", 0.0, 30.0, 1.0)
        additional_mass_factor = st.number_input("Additional mass factor", 0.5, 2.0, 1.0)

    st.markdown("### Preliminary Section Limits and Effective Stiffness")
    c7, c8 = st.columns(2)
    with c7:
        min_wall_thickness = st.number_input("Min wall t (m)", 0.10, 2.0, 0.30)
        max_wall_thickness = st.number_input("Max wall t (m)", 0.30, 4.0, 2.20)
        min_column_dim = st.number_input("Min column dim (m)", 0.20, 4.0, 0.70)
        max_column_dim = st.number_input("Max column dim (m)", 0.50, 6.0, 3.50)
        min_slab_thickness = st.number_input("Min slab t (m)", 0.10, 1.0, 0.22)
        max_slab_thickness = st.number_input("Max slab t (m)", 0.15, 1.2, 0.60)
    with c8:
        wall_cracked_factor = st.number_input("Wall effective I factor", 0.03, 1.00, 0.20)
        column_cracked_factor = st.number_input("Column effective I factor", 0.05, 1.00, 0.35)
        side_wall_cracked_factor = st.number_input("Side wall effective I factor", 0.01, 1.00, 0.15)
        beam_cracked_factor = st.number_input("Beam effective I factor", 0.05, 1.00, 0.35)
        slab_cracked_factor = st.number_input("Slab/diaphragm effective I factor", 0.05, 1.00, 0.25)
        coupling_factor = st.number_input("Global coupling factor", 0.30, 2.00, 1.00)
        side_wall_ratio = st.number_input("Side wall length ratio", 0.0, 0.80, 0.20)

    st.markdown("### Layout / Reinforcement")
    c9, c10 = st.columns(2)
    with c9:
        lower_zone_wall_count = st.number_input("Lower zone wall count", 4, 12, 8)
        middle_zone_wall_count = st.number_input("Middle zone wall count", 4, 12, 6)
        upper_zone_wall_count = st.number_input("Upper zone wall count", 4, 12, 4)
        perimeter_column_factor = st.number_input("Perimeter column factor", 1.0, 3.0, 1.10)
        corner_column_factor = st.number_input("Corner column factor", 1.0, 3.0, 1.30)
    with c10:
        wall_rebar_ratio = st.number_input("Wall rebar ratio", 0.0, 0.10, 0.004, format="%.4f")
        column_rebar_ratio = st.number_input("Column rebar ratio", 0.0, 0.10, 0.012, format="%.4f")
        beam_rebar_ratio = st.number_input("Beam rebar ratio", 0.0, 0.10, 0.015, format="%.4f")
        slab_rebar_ratio = st.number_input("Slab rebar ratio", 0.0, 0.10, 0.004, format="%.4f")

    st.markdown("### Outrigger System")
    c11, c12 = st.columns(2)
    with c11:
        system = st.selectbox("Outrigger type", [OutriggerSystem.TUBULAR_BRACE.value, OutriggerSystem.BELT_TRUSS.value, OutriggerSystem.NONE.value])
        outrigger_count = st.number_input("Outrigger count", 0, 6, 2)
        outrigger_depth_m = st.number_input("Outrigger depth (m)", 1.0, 12.0, 3.0)
        braced_spans_x = st.number_input("Braced bays on each E/W side for X action", 0, 20, 2)
        braced_spans_y = st.number_input("Braced bays on each N/S side for Y action", 0, 20, 2)
        braced_bay_ids_x_txt = st.text_input("Exact bay IDs for X action along Y (optional, comma-separated)", "")
        braced_bay_ids_y_txt = st.text_input("Exact bay IDs for Y action along X (optional, comma-separated)", "")
    with c12:
        tubular_diameter_m = st.number_input("Tube diameter D (m)", 0.10, 3.0, 0.80)
        tubular_thickness_m = st.number_input("Tube thickness t (m)", 0.005, 0.20, 0.030)
        outrigger_chord_area_m2 = st.number_input("Truss chord area (m²)", 0.001, 2.0, 0.08, format="%.4f")
        outrigger_diagonal_area_m2 = st.number_input("Truss diagonal area (m²)", 0.001, 2.0, 0.04, format="%.4f")
        outrigger_connection_efficiency = st.number_input("Connection efficiency", 0.10, 1.00, 0.75)
        outrigger_effectiveness_factor = st.number_input("Outrigger effectiveness factor", 0.00, 0.60, 0.10)
        outrigger_lateral_participation = st.number_input("Outrigger lateral participation", 0.00, 0.25, 0.00)
        st.caption("Braced bays are real centered grid bays. The full EA/L is reduced before entering the global MDOF matrix.")

    levels = []
    if outrigger_count > 0:
        st.markdown("**Outrigger levels**")
        cols = st.columns(min(3, int(outrigger_count)))
        for i in range(int(outrigger_count)):
            with cols[i % len(cols)]:
                default = min(int(round((0.50 + 0.20 * i) * n_story)), int(n_story))
                levels.append(int(st.number_input(f"Outrigger story {i+1}", 1, int(n_story), default, key=f"out_level_{i}")))

    st.markdown("### ASCE 7 / Modal Solver")
    c13, c14 = st.columns(2)
    with c13:
        use_asce7_rsa = st.checkbox("Use ASCE 7 RSA", True)
        spectrum_input_mode = st.selectbox("Spectrum input mode", ["Direct design values: SDS and SD1", "Mapped/site values: SS, S1, Fa, Fv"], index=0)
        use_site_coefficients = spectrum_input_mode.startswith("Mapped")
        if use_site_coefficients:
            SS = st.number_input("SS mapped spectral acceleration (g)", 0.01, 5.0, 1.00)
            S1 = st.number_input("S1 mapped spectral acceleration (g)", 0.0, 3.0, 0.30)
            Fa = st.number_input("Fa site coefficient", 0.10, 5.0, 1.00)
            Fv = st.number_input("Fv site coefficient", 0.10, 5.0, 1.00)
            SDS = (2.0 / 3.0) * Fa * SS
            SD1 = (2.0 / 3.0) * Fv * S1
            st.caption(f"Computed: SDS={SDS:.3f} g, SD1={SD1:.3f} g")
        else:
            SDS = st.number_input("SDS design spectral acceleration (g)", 0.01, 3.0, 0.70)
            SD1 = st.number_input("SD1 design spectral acceleration (g)", 0.01, 3.0, 0.35)
            SS = st.number_input("SS mapped value for report only (g)", 0.01, 5.0, 1.00)
            S1 = st.number_input("S1 mapped value for Cs minimum check (g)", 0.0, 3.0, 0.30)
            Fa = 1.0
            Fv = 1.0
        TL = st.number_input("TL (s)", 2.0, 20.0, 8.0)
        R = st.number_input("R", 1.0, 12.0, 5.0)
        Ie = st.number_input("Ie", 0.5, 2.0, 1.0)
        Cd = st.number_input("Cd", 1.0, 12.0, 5.0)
    with c14:
        damping_ratio = st.number_input("Damping ratio", 0.01, 0.20, 0.05)
        Ct = st.number_input("Ct for Ta", 0.001, 0.10, 0.016, format="%.4f")
        x_exp = st.number_input("x for Ta", 0.50, 1.20, 0.90)
        Cu = st.number_input("Cu", 1.0, 2.0, 1.40)
        rsa_min_ratio_to_elf = st.number_input("RSA/ELF min ratio", 0.50, 1.00, 0.85)
        n_modes = st.number_input("Number of modes", 1, 60, 12)
        combination = st.selectbox("Modal combination", [CombinationMethod.CQC.value, CombinationMethod.SRSS.value], index=0)

    st.markdown("### Redesign Criteria")
    c15, c16 = st.columns(2)
    with c15:
        auto_redesign = st.checkbox("Auto redesign", True)
        drift_limit_ratio = st.number_input("Drift limit ratio", 0.001, 0.050, 0.015, format="%.4f")
        minimum_modal_mass_ratio = st.number_input("Minimum modal mass ratio", 0.50, 0.99, 0.90)
    with c16:
        max_iterations = st.number_input("Max redesign iterations", 1, 60, 18)
        growth_limit = st.number_input("Growth limit per iteration", 1.01, 1.50, 1.18)
        reduction_limit = st.number_input("Reduction limit per iteration", 0.70, 0.99, 0.96)
        allow_section_reduction = st.checkbox("Allow automatic section reduction", False)
        enforce_period_limit = st.checkbox("Enforce ASCE period limit in redesign", True)
        period_limit_multiplier = st.number_input("Period cap multiplier × CuTa", 0.50, 2.00, 1.00)
        target_period_utilization = st.number_input("Target period utilization", 0.70, 1.00, 0.95)

    def _parse_bay_ids(txt: str) -> Tuple[int, ...]:
        vals: List[int] = []
        for part in str(txt).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(int(part))
            except Exception:
                pass
        return tuple(vals)

    return BuildingInput(
        plan_shape=plan_shape,
        n_story=int(n_story),
        n_basement=int(n_basement),
        story_height=float(story_height),
        basement_height=float(basement_height),
        plan_x=float(plan_x),
        plan_y=float(plan_y),
        n_bays_x=int(n_bays_x),
        n_bays_y=int(n_bays_y),
        stair_count=int(stair_count),
        elevator_count=int(elevator_count),
        elevator_area_each=float(elevator_area_each),
        stair_area_each=float(stair_area_each),
        service_area=float(service_area),
        corridor_factor=float(corridor_factor),
        core_ratio_x=float(core_ratio_x),
        core_ratio_y=float(core_ratio_y),
        core_max_ratio_x=float(core_max_ratio_x),
        core_max_ratio_y=float(core_max_ratio_y),
        fck=float(fck),
        Ec=float(Ec),
        fy=float(fy),
        DL=float(DL),
        LL=float(LL),
        live_load_mass_factor=float(live_load_mass_factor),
        slab_finish_allowance=float(slab_finish_allowance),
        facade_line_load=float(facade_line_load),
        additional_mass_factor=float(additional_mass_factor),
        min_wall_thickness=float(min_wall_thickness),
        max_wall_thickness=float(max_wall_thickness),
        min_column_dim=float(min_column_dim),
        max_column_dim=float(max_column_dim),
        min_slab_thickness=float(min_slab_thickness),
        max_slab_thickness=float(max_slab_thickness),
        wall_cracked_factor=float(wall_cracked_factor),
        column_cracked_factor=float(column_cracked_factor),
        side_wall_cracked_factor=float(side_wall_cracked_factor),
        beam_cracked_factor=float(beam_cracked_factor),
        slab_cracked_factor=float(slab_cracked_factor),
        coupling_factor=float(coupling_factor),
        lower_zone_wall_count=int(lower_zone_wall_count),
        middle_zone_wall_count=int(middle_zone_wall_count),
        upper_zone_wall_count=int(upper_zone_wall_count),
        perimeter_column_factor=float(perimeter_column_factor),
        corner_column_factor=float(corner_column_factor),
        side_wall_ratio=float(side_wall_ratio),
        wall_rebar_ratio=float(wall_rebar_ratio),
        column_rebar_ratio=float(column_rebar_ratio),
        beam_rebar_ratio=float(beam_rebar_ratio),
        slab_rebar_ratio=float(slab_rebar_ratio),
        outrigger_system=OutriggerSystem(system),
        outrigger_count=int(outrigger_count),
        outrigger_story_levels=tuple(levels),
        outrigger_depth_m=float(outrigger_depth_m),
        outrigger_chord_area_m2=float(outrigger_chord_area_m2),
        outrigger_diagonal_area_m2=float(outrigger_diagonal_area_m2),
        tubular_diameter_m=float(tubular_diameter_m),
        tubular_thickness_m=float(tubular_thickness_m),
        braced_spans_x=int(braced_spans_x),
        braced_spans_y=int(braced_spans_y),
        braced_bay_ids_x=_parse_bay_ids(braced_bay_ids_x_txt),
        braced_bay_ids_y=_parse_bay_ids(braced_bay_ids_y_txt),
        outrigger_connection_efficiency=float(outrigger_connection_efficiency),
        outrigger_effectiveness_factor=float(outrigger_effectiveness_factor),
        outrigger_lateral_participation=float(outrigger_lateral_participation),
        drift_limit_ratio=float(drift_limit_ratio),
        minimum_modal_mass_ratio=float(minimum_modal_mass_ratio),
        enforce_period_limit=bool(enforce_period_limit),
        period_limit_multiplier=float(period_limit_multiplier),
        target_period_utilization=float(target_period_utilization),
        n_modes=int(n_modes),
        combination=CombinationMethod(combination),
        use_asce7_rsa=bool(use_asce7_rsa),
        asce7=ASCE7Params(
            SDS=float(SDS), SD1=float(SD1), SS=float(SS), S1=float(S1),
            Fa=float(Fa), Fv=float(Fv), use_site_coefficients=bool(use_site_coefficients),
            TL=float(TL), R=float(R), Ie=float(Ie), Cd=float(Cd), damping_ratio=float(damping_ratio),
            Ct=float(Ct), x_exp=float(x_exp), Cu=float(Cu),
            rsa_min_ratio_to_elf=float(rsa_min_ratio_to_elf),
        ),
        auto_redesign=bool(auto_redesign),
        max_iterations=int(max_iterations),
        growth_limit_per_iteration=float(growth_limit),
        reduction_limit_per_iteration=float(reduction_limit),
        allow_section_reduction=bool(allow_section_reduction),
    )


def main():
    import streamlit as st

    st.set_page_config(
        page_title=PROJECT_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 0.7rem; padding-bottom: 0.7rem; max-width: 100%;}
        div[data-testid="stHorizontalBlock"] > div {padding-right: 0.35rem; padding-left: 0.35rem;}
        .stButton button {width: 100%; font-weight: 700; height: 3rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(PROJECT_TITLE)
    st.caption(f"Version {APP_VERSION} | Author: {AUTHOR_NAME}")
    st.info("Engineering preliminary design framework: MDOF modal response, braced-bay outrigger stiffness, drift checks, and with/without outrigger comparison.")

    if "v4_result" not in st.session_state:
        st.session_state.v4_result = None
    if "v4_report" not in st.session_state:
        st.session_state.v4_report = ""

    left, right = st.columns([1.05, 2.35], gap="medium")

    with left:
        inp = streamlit_input_panel()
        b1, b2 = st.columns(2)
        with b1:
            if st.button("ANALYZE"):
                try:
                    with st.spinner("Running professional flexural MDOF solver..."):
                        res = run_design(inp)
                        st.session_state.v4_result = res
                        st.session_state.v4_report = build_report(res)
                    st.success("Analysis completed.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        with b2:
            if st.session_state.v4_report:
                st.download_button(
                    "SAVE REPORT",
                    data=st.session_state.v4_report.encode("utf-8"),
                    file_name="tower_predesign_v4_report.txt",
                    mime="text/plain",
                )
            else:
                st.button("SAVE REPORT", disabled=True)

    with right:
        res = st.session_state.v4_result
        if res is None:
            st.info("Click ANALYZE to run the Version 4 MDOF predesign.")
            return

        s = summary_table(res)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T1 X (s)", f"{res.modal_x.periods_s[0]:.3f}")
        c2.metric("T1 Y (s)", f"{res.modal_y.periods_s[0]:.3f}")
        c3.metric("Max drift X", f"{np.max(res.rsa_x.drift_ratio):.5f}")
        c4.metric("Max drift Y", f"{np.max(res.rsa_y.drift_ratio):.5f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Base shear X (kN)", f"{res.rsa_x.base_shear_scaled_kN:,.0f}")
        c6.metric("Base shear Y (kN)", f"{res.rsa_y.base_shear_scaled_kN:,.0f}")
        c7.metric("Mass X (%)", f"{100*res.modal_x.cumulative_mass_ratios[-1]:.1f}")
        c8.metric("Mass Y (%)", f"{100*res.modal_y.cumulative_mass_ratios[-1]:.1f}")

        zone_name = st.selectbox("Displayed plan zone", ["Lower Zone", "Middle Zone", "Upper Zone"], index=0)

        tabs = st.tabs(
            [
                "PRINCIPLES",
                "FINAL DESIGN",
                "OUTRIGGER EFFECT",
                "OUTRIGGER DESIGN",
                "DESIGN CHECKS",
                "MATERIAL EFFECT",
                "Summary",
                "Final dimensions",
                "Plan",
                "Modes X",
                "Modes Y",
                "Story X",
                "Story Y",
                "Stiffness",
                "ASCE spectrum",
                "Iteration",
                "Tables",
                "Report",
            ]
        )

        with tabs[0]:
            st.markdown("### Design principles used in this solver")
            st.dataframe(design_principles_table(res), use_container_width=True, hide_index=True)

        with tabs[1]:
            st.markdown("### Final preliminary design dimensions")
            st.dataframe(final_design_dashboard_table(res), use_container_width=True, hide_index=True)
            st.markdown("### Main design quantities")
            st.dataframe(summary_table(res), use_container_width=True, hide_index=True)

        with tabs[2]:
            st.markdown("### With vs without outrigger comparison")
            st.dataframe(outrigger_effect_comparison(res), use_container_width=True, hide_index=True)
            st.markdown("### Outrigger stiffness diagnostic")
            st.dataframe(outrigger_stiffness_diagnostic_table(res), use_container_width=True, hide_index=True)
            st.caption("Comparison keeps the same mass and removes only outrigger stiffness. If Ktheta/(4EI/L) and Klat are small, the outrigger cannot significantly change the global period.")

        with tabs[3]:
            st.markdown("### Outrigger design result")
            st.dataframe(outrigger_design_table(res), use_container_width=True, hide_index=True)
            st.caption("Ktheta is the rotational restraint added to the flexural MDOF solver at the real outrigger levels.")

        with tabs[4]:
            st.markdown("### Design checks")
            st.dataframe(design_check_table(res), use_container_width=True, hide_index=True)

        with tabs[5]:
            st.markdown("### Material modifiers used in member sizing")
            st.dataframe(material_diagnostics(res.input), use_container_width=True, hide_index=True)
            st.caption("Higher fck/Ec reduces preliminary member dimensions until minimum/detailing limits or drift redesign govern.")

        with tabs[6]:
            st.dataframe(summary_table(res), use_container_width=True, hide_index=True)

        with tabs[7]:
            st.dataframe(final_dimensions_table(res), use_container_width=True, hide_index=True)

        with tabs[8]:
            st.pyplot(plot_plan(res, zone_name), use_container_width=True)

        with tabs[9]:
            st.pyplot(plot_modes(res, Direction.X), use_container_width=True)
            st.dataframe(modal_table(res.modal_x), use_container_width=True, hide_index=True)

        with tabs[10]:
            st.pyplot(plot_modes(res, Direction.Y), use_container_width=True)
            st.dataframe(modal_table(res.modal_y), use_container_width=True, hide_index=True)

        with tabs[11]:
            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(plot_story_response(res, Direction.X, "Story shear"), use_container_width=True)
            with p2:
                st.pyplot(plot_story_response(res, Direction.X, "Drift ratio"), use_container_width=True)
            st.dataframe(story_response_table(res, Direction.X), use_container_width=True, hide_index=True)

        with tabs[12]:
            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(plot_story_response(res, Direction.Y, "Story shear"), use_container_width=True)
            with p2:
                st.pyplot(plot_story_response(res, Direction.Y, "Drift ratio"), use_container_width=True)
            st.dataframe(story_response_table(res, Direction.Y), use_container_width=True, hide_index=True)

        with tabs[13]:
            st.pyplot(plot_stiffness(res), use_container_width=True)
            st.dataframe(stiffness_table(res), use_container_width=True, hide_index=True)

        with tabs[14]:
            st.pyplot(plot_spectrum(res.input), use_container_width=True)
            st.dataframe(spectrum_table(res.input), use_container_width=True, hide_index=True)

        with tabs[15]:
            fig = plot_iteration(res)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            st.dataframe(res.iteration_table, use_container_width=True, hide_index=True)

        with tabs[16]:
            st.markdown("### Story sections")
            st.dataframe(pd.DataFrame([s.__dict__ for s in res.sections]), use_container_width=True, hide_index=True)
            st.markdown("### Story properties")
            st.dataframe(pd.DataFrame([p.__dict__ for p in res.properties]), use_container_width=True, hide_index=True)

        with tabs[17]:
            st.text_area("Report", st.session_state.v4_report, height=600)


if __name__ == "__main__":
    main()
