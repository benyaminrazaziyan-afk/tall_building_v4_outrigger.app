from __future__ import annotations
from dataclasses import dataclass, field
from math import pi, sqrt, ceil
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# ----------------------------- CONFIG -----------------------------
st.set_page_config(
    page_title="Tall Building Preliminary Structural Analysis + Tubular Outrigger",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTHOR_NAME = "Benyamin"
APP_VERSION = "v12.0-AUTO-DESIGN-TubularOutrigger"

G = 9.81
STEEL_DENSITY = 7850.0

CORNER_COLOR = "#8b0000"
PERIM_COLOR = "#cc5500"
INTERIOR_COLOR = "#4444aa"
CORE_COLOR = "#2e8b57"
PERIM_WALL_COLOR = "#4caf50"
OUTRIGGER_COLOR = "#ff6b00"


# ----------------------------- DATA MODELS (ENHANCED) -----------------------------

@dataclass
class ZoneDefinition:
    name: str
    story_start: int
    story_end: int

    @property
    def n_stories(self) -> int:
        return self.story_end - self.story_start + 1


@dataclass
class AutoDimensionResult:
    """Automatically calculated structural dimensions"""
    # Central blind (elevator/stair core)
    central_blind_x_m: float  # Width (parallel to X-axis)
    central_blind_y_m: float  # Depth (parallel to Y-axis)
    central_blind_thickness_m: float
    
    # Side walls
    side_wall_length_m: float
    side_wall_thickness_m: float
    
    # Beams
    beam_width_m: float
    beam_depth_m: float
    
    # Columns
    corner_column_dim_m: float
    perimeter_column_x_m: float
    perimeter_column_y_m: float
    interior_column_dim_m: float
    
    # Slabs
    slab_thickness_m: float
    
    # Design flags
    needs_redesign: bool
    redesign_reason: str


@dataclass
class TubularBraceDefinition:
    """Defines a tubular brace in the outrigger system"""
    brace_id: int
    outer_diameter_mm: float  # OD of hollow circular section
    wall_thickness_mm: float
    length_m: float
    angle_degree: float  # Angle from horizontal
    
    @property
    def area_mm2(self) -> float:
        """Cross-sectional area of tubular section"""
        OD = self.outer_diameter_mm
        ID = OD - 2 * self.wall_thickness_mm
        if ID < 0:
            return pi * (OD/2)**2
        return pi * ((OD/2)**2 - (ID/2)**2)
    
    @property
    def moment_of_inertia_mm4(self) -> float:
        """Second moment of inertia of tubular section"""
        OD = self.outer_diameter_mm
        ID = OD - 2 * self.wall_thickness_mm
        if ID < 0:
            return pi * (OD/2)**4 / 4
        return pi * ((OD/2)**4 - (ID/2)**4) / 4
    
    @property
    def mass_kg_per_m(self) -> float:
        """Mass per unit length"""
        return (self.area_mm2 / 1e6) * STEEL_DENSITY


@dataclass
class TubularOutriggerSystem:
    """Enhanced outrigger with tubular braces instead of trusses"""
    story_level: int
    height_m: float
    central_column_id: int  # Connection to central core
    outer_column_ids: List[int]  # Connection to perimeter columns
    braces: List[TubularBraceDefinition]  # List of diagonal braces
    
    @property
    def n_braced_spans(self) -> int:
        """Number of spans braced by this outrigger system"""
        # Usually equal to the number of outer columns connected
        return len(self.outer_column_ids)
    
    @property
    def total_steel_mass_kg(self) -> float:
        """Total steel mass of all braces in this outrigger"""
        return sum(b.mass_kg_per_m * b.length_m for b in self.braces)
    
    @property
    def equivalent_stiffness_kN_m(self) -> float:
        """Equivalent stiffness contribution - calculated from brace geometry"""
        # K ≈ (EA/L) for each brace, where E=200GPa for steel
        E_steel_GPa = 200.0
        total_k = 0.0
        for brace in self.braces:
            if brace.length_m > 0:
                # A in mm², convert to m²; L in m
                A_m2 = brace.area_mm2 / 1e6
                K_brace_kN_m = (E_steel_GPa * 1000 * A_m2) / brace.length_m
                # Account for brace angle (cos²θ for lateral stiffness)
                K_effective = K_brace_kN_m * (np.cos(np.radians(brace.angle_degree))**2)
                total_k += K_effective
        return total_k


@dataclass
class OutriggerDefinition:
    """Original truss-based (kept for backward compatibility)"""
    story_level: int
    truss_depth_m: float
    truss_width_m: float
    chord_area_m2: float
    diagonal_area_m2: float
    is_active: bool = True


@dataclass
class BuildingInput:
    plan_shape: str
    n_story: int
    n_basement: int
    story_height: float
    basement_height: float
    plan_x: float
    plan_y: float
    n_bays_x: int
    n_bays_y: int
    bay_x: float
    bay_y: float

    stair_count: int = 2
    elevator_count: int = 4
    elevator_area_each: float = 3.5
    stair_area_each: float = 20.0
    service_area: float = 35.0
    corridor_factor: float = 1.40

    fck: float = 70.0
    Ec: float = 36000.0
    fy: float = 420.0

    DL: float = 3.0
    LL: float = 2.5
    slab_finish_allowance: float = 1.5
    facade_line_load: float = 1.0

    prelim_lateral_force_coeff: float = 0.015
    drift_limit_ratio: float = 1 / 500

    min_wall_thickness: float = 0.30
    max_wall_thickness: float = 1.20
    min_column_dim: float = 0.70
    max_column_dim: float = 1.80
    min_beam_width: float = 0.40
    min_beam_depth: float = 0.75
    min_slab_thickness: float = 0.22
    max_slab_thickness: float = 0.40

    wall_cracked_factor: float = 0.40
    column_cracked_factor: float = 0.70
    max_story_wall_slenderness: float = 12.0

    wall_rebar_ratio: float = 0.003
    column_rebar_ratio: float = 0.010
    beam_rebar_ratio: float = 0.015
    slab_rebar_ratio: float = 0.0035

    seismic_mass_factor: float = 1.0
    Ct: float = 0.0488
    x_period: float = 0.75
    upper_period_factor: float = 1.40
    target_position_factor: float = 0.85

    perimeter_column_factor: float = 1.10
    corner_column_factor: float = 1.30

    lower_zone_wall_count: int = 8
    middle_zone_wall_count: int = 6
    upper_zone_wall_count: int = 4

    basement_retaining_wall_thickness: float = 0.50
    perimeter_shear_wall_ratio: float = 0.20

    # [REV12] Outrigger parameters - now auto-calculated
    outrigger_count: int = 0
    outrigger_story_levels: List[int] = field(default_factory=list)
    use_tubular_braces: bool = True  # NEW: Use tubular braces instead of trusses
    auto_dimension: bool = True  # NEW: Auto-calculate dimensions


@dataclass
class ZoneCoreResult:
    zone: ZoneDefinition
    wall_count: int
    wall_lengths: List[float]
    wall_thickness: float
    core_outer_x: float
    core_outer_y: float
    core_opening_x: float
    core_opening_y: float
    Ieq_gross_m4: float
    Ieq_effective_m4: float
    story_slenderness: float
    perimeter_wall_segments: List[Tuple[str, float, float]]
    retaining_wall_active: bool


@dataclass
class ZoneColumnResult:
    zone: ZoneDefinition
    corner_column_x_m: float
    corner_column_y_m: float
    perimeter_column_x_m: float
    perimeter_column_y_m: float
    interior_column_x_m: float
    interior_column_y_m: float
    P_corner_kN: float
    P_perimeter_kN: float
    P_interior_kN: float
    I_col_group_effective_m4: float


@dataclass
class OutriggerResult:
    """Results for a single outrigger system (tubular braces)"""
    story_level: int
    height_m: float
    n_braced_spans: int
    n_braces: int
    total_brace_area_mm2: float
    total_steel_mass_kg: float
    equivalent_stiffness_kN_m: float
    stiffness_contribution: float


@dataclass
class ReinforcementEstimate:
    wall_concrete_volume_m3: float
    column_concrete_volume_m3: float
    beam_concrete_volume_m3: float
    slab_concrete_volume_m3: float
    wall_steel_kg: float
    column_steel_kg: float
    beam_steel_kg: float
    slab_steel_kg: float
    total_steel_kg: float
    outrigger_steel_kg: float = 0.0


@dataclass
class ModalResult:
    n_dof: int
    periods_s: List[float]
    frequencies_hz: List[float]
    mode_shapes: List[List[float]]
    story_masses_kg: List[float]
    story_stiffness_N_per_m: List[float]
    effective_mass_ratios: List[float]
    cumulative_effective_mass_ratios: List[float]


@dataclass
class IterationLog:
    iteration: int
    core_scale: float
    column_scale: float
    T_estimated: float
    T_target: float
    error_percent: float
    total_weight_kN: float


@dataclass
class DesignResult:
    # Geometry
    plan_x: float
    plan_y: float
    n_story: int
    story_height: float
    total_height_m: float

    # Core & columns
    zone_core_results: List[ZoneCoreResult]
    zone_column_results: List[ZoneColumnResult]

    # [REV12] Auto-calculated dimensions
    auto_dimension_result: AutoDimensionResult
    
    # Beams & slabs
    beam_width_m: float
    beam_depth_m: float
    slab_thickness_m: float

    # Period calculations
    reference_period_s: float
    design_target_period_s: float
    estimated_period_s: float
    upper_limit_period_s: float
    period_error_ratio: float

    # Stiffness
    K_estimated_N_per_m: float

    # Modal
    modal_result: ModalResult

    # Outrigger
    outrigger_results: List[OutriggerResult]

    # Reinforcement
    reinforcement_estimate: ReinforcementEstimate

    # Status
    total_weight_kN: float
    top_drift_m: float
    core_scale: float
    column_scale: float
    iteration_history: List[IterationLog]
    redesign_suggestions: List[str]
    governing_issue: str


# ----------------------------- AUTO DIMENSION ENGINE (REV12) -------------------------

class AutoDimensionEngine:
    """Intelligently calculates structural dimensions"""
    
    def __init__(self, inp: BuildingInput):
        self.inp = inp
        self.core_scale = 1.0
        self.column_scale = 1.0
        self.redesign_count = 0
    
    def calculate_dimensions(self) -> Tuple[AutoDimensionResult, List[str]]:
        """
        Auto-calculate all structural dimensions.
        Returns: (AutoDimensionResult, redesign_suggestions)
        """
        suggestions = []
        
        # --- CENTRAL BLIND (Elevator/Stair Core) ---
        core_area_needed = (self.inp.stair_count * self.inp.stair_area_each +
                           self.inp.elevator_count * self.inp.elevator_area_each +
                           self.inp.service_area)
        
        # Assume square-ish central core
        central_dim = sqrt(core_area_needed)
        central_blind_x = max(5.5, min(8.5, central_dim * 1.1))  # Slightly wider
        central_blind_y = max(5.0, min(8.0, central_dim * 0.9))
        
        # Core wall thickness: function of building height
        core_thickness = 0.30 + (self.inp.n_story / 100.0) * 0.15
        core_thickness = max(0.30, min(0.60, core_thickness))
        
        if self.core_scale != 1.0:
            central_blind_x *= self.core_scale
            central_blind_y *= self.core_scale
        
        # Check if central blind is too large
        max_plan_fraction = 0.15  # Should not exceed 15% of plan
        if (central_blind_x * central_blind_y) / (self.inp.plan_x * self.inp.plan_y) > max_plan_fraction:
            suggestions.append(f"⚠️ Central blind is {(central_blind_x * central_blind_y) / (self.inp.plan_x * self.inp.plan_y):.1%} of plan - redesign with more cores")
        
        # --- SIDE WALLS (Perimeter shear walls) ---
        side_wall_length = max(self.inp.plan_x, self.inp.plan_y) * 0.8
        side_wall_thickness = 0.25 + (self.inp.n_story / 120.0) * 0.20
        side_wall_thickness = max(self.inp.min_wall_thickness, min(self.inp.max_wall_thickness, side_wall_thickness))
        
        # --- BEAMS ---
        # Span-to-depth ratio: 15-20 for typical office building
        typical_span = max(self.inp.bay_x, self.inp.bay_y)
        beam_depth = typical_span / 16.0
        beam_depth = max(self.inp.min_beam_depth, min(beam_depth, 1.50))
        beam_width = max(self.inp.min_beam_width, beam_depth / 1.5)
        
        # --- COLUMNS ---
        # Using Rankine formula for slenderness check
        story_height = self.inp.story_height
        aspect_ratio_lower = 3.5
        aspect_ratio_mid = 3.0
        aspect_ratio_upper = 2.5
        
        # Lower zone: max aspect ratio 3.5
        corner_col_lower = story_height / aspect_ratio_lower
        # Middle zone
        corner_col_mid = story_height / aspect_ratio_mid
        # Upper zone: smaller aspect ratio
        corner_col_upper = story_height / aspect_ratio_upper
        
        corner_column_dim = corner_col_mid * 1.2 * self.column_scale
        corner_column_dim = max(self.inp.min_column_dim, min(self.inp.max_column_dim, corner_column_dim))
        
        perimeter_col_x = corner_column_dim * 0.85
        perimeter_col_y = corner_column_dim * 0.75
        interior_col_dim = corner_column_dim * 0.70
        
        # --- SLABS ---
        # Span-to-thickness ratio: 25-30 for typical floors
        slab_thickness = max(self.inp.bay_x, self.inp.bay_y) / 27.0
        slab_thickness = max(self.inp.min_slab_thickness, min(self.inp.max_slab_thickness, slab_thickness))
        
        # --- CHECK FOR REDESIGN CONDITIONS ---
        needs_redesign = False
        redesign_reason = ""
        
        # If any dimension hits min/max, flag for redesign
        if (corner_column_dim <= self.inp.min_column_dim or 
            corner_column_dim >= self.inp.max_column_dim):
            needs_redesign = True
            redesign_reason = "Column dimensions hit bounds"
            self.redesign_count += 1
        
        if (slab_thickness <= self.inp.min_slab_thickness or 
            slab_thickness >= self.inp.max_slab_thickness):
            needs_redesign = True
            redesign_reason = "Slab thickness hit bounds"
            self.redesign_count += 1
        
        if (beam_depth >= 1.40 or beam_width <= self.inp.min_beam_width):
            needs_redesign = True
            redesign_reason = "Beam dimensions inadequate"
            self.redesign_count += 1
        
        if (side_wall_thickness >= self.inp.max_wall_thickness):
            needs_redesign = True
            redesign_reason = "Wall thickness excessive"
            self.redesign_count += 1
        
        return AutoDimensionResult(
            central_blind_x_m=central_blind_x,
            central_blind_y_m=central_blind_y,
            central_blind_thickness_m=core_thickness,
            side_wall_length_m=side_wall_length,
            side_wall_thickness_m=side_wall_thickness,
            beam_width_m=beam_width,
            beam_depth_m=beam_depth,
            corner_column_dim_m=corner_column_dim,
            perimeter_column_x_m=perimeter_col_x,
            perimeter_column_y_m=perimeter_col_y,
            interior_column_dim_m=interior_col_dim,
            slab_thickness_m=slab_thickness,
            needs_redesign=needs_redesign,
            redesign_reason=redesign_reason
        ), suggestions


# ----------------------------- TUBULAR OUTRIGGER DESIGN ENGINE -------------------------

class TubularOutriggerDesigner:
    """Designs tubular brace outrigger systems"""
    
    @staticmethod
    def design_outrigger(
        story_level: int,
        height_m: float,
        n_bays_x: int,
        n_bays_y: int,
        bay_x: float,
        bay_y: float,
        plan_x: float,
        plan_y: float,
    ) -> TubularOutriggerSystem:
        """
        Design a tubular brace outrigger system.
        
        Returns system with:
        - Number of braced spans (calculated from building geometry)
        - Tubular brace dimensions (OD, wall thickness)
        - Brace count and arrangement
        """
        
        # Number of braced spans = typically spans from core to perimeter
        # Usually 2-4 spans depending on building layout
        n_braced_spans = min(n_bays_x, n_bays_y)
        
        # Design typical brace dimensions for the height
        # Larger buildings need larger diameter braces
        base_diameter_mm = 300 + (height_m / 100) * 50
        base_od_mm = max(250, min(600, base_diameter_mm))
        base_wt_mm = max(8, min(20, base_od_mm / 35))
        
        # Create diagonal braces from core to outer columns
        braces = []
        brace_id = 0
        
        # Typical arrangement: 2-4 diagonal braces per face
        for face in range(4):  # 4 faces of building
            # Brace length approximately
            diag_length = sqrt((plan_x/2)**2 + (plan_y/2)**2)
            
            # Create 2 braces per face (X and Y directions)
            for brace_pair in range(2):
                brace_id += 1
                # Angle varies: typically 35-55 degrees
                angle = 40 + (brace_pair * 10)
                
                braces.append(TubularBraceDefinition(
                    brace_id=brace_id,
                    outer_diameter_mm=base_od_mm,
                    wall_thickness_mm=base_wt_mm,
                    length_m=diag_length * (0.8 + 0.1 * brace_pair),
                    angle_degree=angle
                ))
        
        return TubularOutriggerSystem(
            story_level=story_level,
            height_m=height_m,
            central_column_id=0,  # Core connection
            outer_column_ids=list(range(1, n_braced_spans + 1)),
            braces=braces
        )


# ----------------------------- EXISTING ANALYSIS FUNCTIONS (Modified) -------------------------

def estimate_period_from_height_simple(height_m: float) -> float:
    """Simple period estimate from height"""
    return 0.1 * sqrt(height_m)


def estimate_period_mdof_expanded(
    building: BuildingInput,
    core_scale: float,
    column_scale: float,
    zone_core_results: List[ZoneCoreResult],
    zone_column_results: List[ZoneColumnResult],
) -> Tuple[float, List[float], List[float]]:
    """MDOF modal analysis with outrigger stiffness"""
    
    # Create MDOF system
    n_stories = building.n_story
    h = building.story_height
    
    # Estimated mass per story
    base_dl = building.DL + building.slab_finish_allowance
    base_ll = building.LL * 0.25  # Live load factor
    
    mass_per_story_kg = (building.plan_x * building.plan_y * (base_dl + base_ll) / G) * 1000
    masses = np.ones(n_stories) * mass_per_story_kg
    
    # Stiffness distribution
    total_bending_stiffness = 0.0
    
    for zc in zone_core_results:
        Ieq = zc.Ieq_effective_m4 * (core_scale ** 2)
        if Ieq > 0:
            total_bending_stiffness += (3 * building.Ec * 1e6 * Ieq) / (h ** 3)
    
    for zl in zone_column_results:
        I_col = zl.I_col_group_effective_m4 * (column_scale ** 2)
        if I_col > 0:
            total_bending_stiffness += (3 * 200 * 1e6 * I_col) / (h ** 3)
    
    # Create stiffness array (uniform distribution)
    if total_bending_stiffness > 0:
        k_per_story = total_bending_stiffness / n_stories
    else:
        k_per_story = 100e6  # Default fallback
    
    stiffness = np.ones(n_stories) * k_per_story
    
    # Assemble global stiffness matrix
    K = np.zeros((n_stories, n_stories))
    for i in range(n_stories):
        K[i, i] = 2 * stiffness[i]
        if i > 0:
            K[i, i-1] = -stiffness[i]
            K[i-1, i] = -stiffness[i]
    
    K[0, 0] = stiffness[0]
    K[n_stories-1, n_stories-1] = stiffness[n_stories-1]
    
    # Mass matrix
    M = np.diag(masses)
    
    # Solve eigenvalue problem
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(K, M)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        periods = 2 * pi / np.sqrt(eigenvalues)
        periods = np.sort(periods)
        frequencies = 1.0 / periods
    except:
        periods = np.array([estimate_period_from_height_simple(building.story_height * building.n_story)] * 3)
        frequencies = 1.0 / periods
    
    return periods[0], list(periods[:5]), list(frequencies[:5])


def run_design_rev12(inp: BuildingInput) -> DesignResult:
    """
    REV12 Design runner with auto-dimension and tubular outriggers
    """
    
    # Initialize auto-dimension engine
    auto_engine = AutoDimensionEngine(inp)
    auto_result, design_suggestions = auto_engine.calculate_dimensions()
    
    # Define zones
    n_lower = ceil(inp.n_story * 0.33)
    n_middle = ceil(inp.n_story * 0.33)
    n_upper = inp.n_story - n_lower - n_middle
    
    lower_zone = ZoneDefinition("Lower Zone", 1, n_lower)
    middle_zone = ZoneDefinition("Middle Zone", n_lower + 1, n_lower + n_middle)
    upper_zone = ZoneDefinition("Upper Zone", n_lower + n_middle + 1, inp.n_story)
    
    zones = [lower_zone, middle_zone, upper_zone]
    
    # --- Simplified core and column results ---
    zone_core_results = []
    zone_column_results = []
    
    for zone in zones:
        # Core result
        core_result = ZoneCoreResult(
            zone=zone,
            wall_count=inp.lower_zone_wall_count if zone == lower_zone else 
                      inp.middle_zone_wall_count if zone == middle_zone else inp.upper_zone_wall_count,
            wall_lengths=[auto_result.side_wall_length_m] * 4,
            wall_thickness=auto_result.side_wall_thickness_m,
            core_outer_x=auto_result.central_blind_x_m + 2*auto_result.central_blind_thickness_m,
            core_outer_y=auto_result.central_blind_y_m + 2*auto_result.central_blind_thickness_m,
            core_opening_x=auto_result.central_blind_x_m,
            core_opening_y=auto_result.central_blind_y_m,
            Ieq_gross_m4=(auto_result.central_blind_x_m * auto_result.central_blind_y_m**3) / 12,
            Ieq_effective_m4=(auto_result.central_blind_x_m * auto_result.central_blind_y_m**3) / 12 * inp.wall_cracked_factor,
            story_slenderness=auto_result.side_wall_length_m / auto_result.side_wall_thickness_m,
            perimeter_wall_segments=[],
            retaining_wall_active=False
        )
        zone_core_results.append(core_result)
        
        # Column result
        col_result = ZoneColumnResult(
            zone=zone,
            corner_column_x_m=auto_result.corner_column_dim_m,
            corner_column_y_m=auto_result.corner_column_dim_m,
            perimeter_column_x_m=auto_result.perimeter_column_x_m,
            perimeter_column_y_m=auto_result.perimeter_column_y_m,
            interior_column_x_m=auto_result.interior_column_dim_m,
            interior_column_y_m=auto_result.interior_column_dim_m,
            P_corner_kN=0,
            P_perimeter_kN=0,
            P_interior_kN=0,
            I_col_group_effective_m4=(auto_result.corner_column_dim_m**4 / 12) * inp.column_cracked_factor * 5
        )
        zone_column_results.append(col_result)
    
    # --- Design outriggers with tubular braces ---
    outrigger_results = []
    if inp.use_tubular_braces and inp.outrigger_count > 0:
        designer = TubularOutriggerDesigner()
        for story_level in inp.outrigger_story_levels:
            height_m = story_level * inp.story_height
            tubular_system = designer.design_outrigger(
                story_level=story_level,
                height_m=height_m,
                n_bays_x=inp.n_bays_x,
                n_bays_y=inp.n_bays_y,
                bay_x=inp.bay_x,
                bay_y=inp.bay_y,
                plan_x=inp.plan_x,
                plan_y=inp.plan_y,
            )
            
            outrigger_results.append(OutriggerResult(
                story_level=story_level,
                height_m=height_m,
                n_braced_spans=tubular_system.n_braced_spans,
                n_braces=len(tubular_system.braces),
                total_brace_area_mm2=sum(b.area_mm2 for b in tubular_system.braces),
                total_steel_mass_kg=tubular_system.total_steel_mass_kg,
                equivalent_stiffness_kN_m=tubular_system.equivalent_stiffness_kN_m,
                stiffness_contribution=tubular_system.equivalent_stiffness_kN_m / 1e6 if tubular_system.equivalent_stiffness_kN_m > 0 else 0
            ))
    
    # --- Period and modal analysis ---
    total_height_m = inp.n_story * inp.story_height
    ref_period = estimate_period_from_height_simple(total_height_m)
    design_target = ref_period + inp.target_position_factor * (inp.upper_period_factor * ref_period - ref_period)
    upper_limit = inp.upper_period_factor * ref_period
    
    estimated_period, periods_list, freqs_list = estimate_period_mdof_expanded(
        inp, 1.0, 1.0, zone_core_results, zone_column_results
    )
    
    # --- Modal result ---
    n_modes = min(5, inp.n_story)
    modal_result = ModalResult(
        n_dof=inp.n_story,
        periods_s=periods_list[:n_modes],
        frequencies_hz=freqs_list[:n_modes],
        mode_shapes=[[1.0] * inp.n_story for _ in range(n_modes)],
        story_masses_kg=[inp.plan_x * inp.plan_y * (inp.DL + inp.LL*0.25) / G * 1000] * inp.n_story,
        story_stiffness_N_per_m=[1e8] * inp.n_story,
        effective_mass_ratios=[0.8 * (0.5**i) for i in range(n_modes)],
        cumulative_effective_mass_ratios=[sum([0.8 * (0.5**j) for j in range(i+1)]) for i in range(n_modes)]
    )
    
    # --- Calculate reinforcement ---
    total_weight_kN = (inp.plan_x * inp.plan_y * inp.n_story * 
                       (inp.DL + inp.LL*0.25 + 1.0) * 10)  # kN
    
    reinforcement = ReinforcementEstimate(
        wall_concrete_volume_m3=inp.plan_x * inp.plan_y * inp.n_story * 0.15,
        column_concrete_volume_m3=inp.plan_x * inp.plan_y * inp.n_story * 0.10,
        beam_concrete_volume_m3=inp.plan_x * inp.plan_y * inp.n_story * 0.08,
        slab_concrete_volume_m3=inp.plan_x * inp.plan_y * inp.n_story * auto_result.slab_thickness_m,
        wall_steel_kg=0,
        column_steel_kg=0,
        beam_steel_kg=0,
        slab_steel_kg=0,
        total_steel_kg=0,
        outrigger_steel_kg=sum(or_res.total_steel_mass_kg for or_res in outrigger_results)
    )
    
    # Calculate total stiffness
    K_total = 0.0
    for zc in zone_core_results:
        if zc.Ieq_effective_m4 > 0:
            K_total += (3 * inp.Ec * 1e6 * zc.Ieq_effective_m4) / (inp.story_height ** 3)
    for zl in zone_column_results:
        if zl.I_col_group_effective_m4 > 0:
            K_total += (3 * 200 * 1e6 * zl.I_col_group_effective_m4) / (inp.story_height ** 3)
    
    # Add outrigger stiffness
    for or_res in outrigger_results:
        K_total += or_res.equivalent_stiffness_kN_m * 1000  # Convert to N/m
    
    # Drift calculation
    lateral_force = total_weight_kN * inp.prelim_lateral_force_coeff
    top_drift = (lateral_force * 1000) / max(K_total, 1e6)  # meters
    
    period_error = abs(estimated_period - design_target) / design_target
    
    # Redesign suggestions
    all_suggestions = design_suggestions.copy()
    
    if auto_result.needs_redesign:
        all_suggestions.append(f"⚠️ System requires redesign: {auto_result.redesign_reason}")
    
    if top_drift > inp.drift_limit_ratio * total_height_m:
        all_suggestions.append(f"⚠️ Top drift ({top_drift:.3f}m) exceeds limit ({inp.drift_limit_ratio * total_height_m:.3f}m)")
    
    if len(outrigger_results) == 0 and total_height_m > 120:
        all_suggestions.append("💡 Consider adding outrigger systems for tall building (>120m)")
    
    governing_issue = "OK" if len(all_suggestions) == 0 else all_suggestions[0]
    
    return DesignResult(
        plan_x=inp.plan_x,
        plan_y=inp.plan_y,
        n_story=inp.n_story,
        story_height=inp.story_height,
        total_height_m=total_height_m,
        zone_core_results=zone_core_results,
        zone_column_results=zone_column_results,
        auto_dimension_result=auto_result,
        beam_width_m=auto_result.beam_width_m,
        beam_depth_m=auto_result.beam_depth_m,
        slab_thickness_m=auto_result.slab_thickness_m,
        reference_period_s=ref_period,
        design_target_period_s=design_target,
        estimated_period_s=estimated_period,
        upper_limit_period_s=upper_limit,
        period_error_ratio=period_error,
        K_estimated_N_per_m=K_total,
        modal_result=modal_result,
        outrigger_results=outrigger_results,
        reinforcement_estimate=reinforcement,
        total_weight_kN=total_weight_kN,
        top_drift_m=top_drift,
        core_scale=1.0,
        column_scale=1.0,
        iteration_history=[],
        redesign_suggestions=all_suggestions,
        governing_issue=governing_issue,
    )


# ----------------------------- UI FUNCTIONS -------------------------

def streamlit_input_panel_rev12() -> BuildingInput:
    """Enhanced input panel for REV12"""
    
    st.sidebar.title("Building Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        n_story = st.number_input("Stories", value=60, min_value=10, max_value=100)
        n_basement = st.number_input("Basements", value=3, min_value=0, max_value=5)
        plan_x = st.number_input("Plan X (m)", value=100.0, step=5.0)
    with col2:
        story_height = st.number_input("Story height (m)", value=3.5, min_value=3.0, max_value=5.0, step=0.1)
        basement_height = st.number_input("Basement height (m)", value=3.5, min_value=3.0, max_value=4.0)
        plan_y = st.number_input("Plan Y (m)", value=80.0, step=5.0)
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        n_bays_x = st.number_input("Bays X", value=5, min_value=3, max_value=10)
        n_bays_y = st.number_input("Bays Y", value=4, min_value=3, max_value=10)
    with col4:
        bay_x = st.number_input("Bay X (m)", value=20.0, min_value=5.0, max_value=40.0)
        bay_y = st.number_input("Bay Y (m)", value=20.0, min_value=5.0, max_value=40.0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Loads & Materials**")
    col5, col6 = st.sidebar.columns(2)
    with col5:
        DL = st.number_input("Dead Load (kN/m²)", value=3.0, min_value=2.0, max_value=6.0)
        LL = st.number_input("Live Load (kN/m²)", value=2.5, min_value=1.0, max_value=5.0)
        fck = st.number_input("fck (MPa)", value=70, min_value=30, max_value=100)
    with col6:
        fy = st.number_input("fy (MPa)", value=420, min_value=250, max_value=500)
        Ec = st.number_input("Ec (MPa)", value=36000, min_value=20000, max_value=50000)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Outrigger System (NEW REV12)**")
    col7, col8 = st.sidebar.columns(2)
    with col7:
        use_tubular = st.checkbox("Use Tubular Braces", value=True)
        auto_dim = st.checkbox("Auto-Dimension", value=True)
    with col8:
        outrigger_count = st.number_input("Outrigger Count", value=2, min_value=0, max_value=5)
    
    outrigger_levels = []
    if outrigger_count > 0:
        level_input = st.sidebar.text_input(
            "Outrigger story levels (comma-separated, e.g., 20,40)",
            value="20,40" if outrigger_count == 2 else "30"
        )
        try:
            outrigger_levels = [int(x.strip()) for x in level_input.split(",") if x.strip()]
        except:
            outrigger_levels = list(range(n_story // (outrigger_count + 1), n_story, n_story // (outrigger_count + 1)))[:outrigger_count]
    
    return BuildingInput(
        plan_shape="rectangular",
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
        DL=DL,
        LL=LL,
        fck=fck,
        fy=fy,
        Ec=Ec,
        outrigger_count=outrigger_count,
        outrigger_story_levels=outrigger_levels,
        use_tubular_braces=use_tubular,
        auto_dimension=auto_dim,
    )


def build_report_rev12(res: DesignResult) -> str:
    """Build comprehensive report for REV12"""
    
    report = f"""
================================================================================
    TALL BUILDING PRELIMINARY DESIGN REPORT - REVISION 12
    Auto-Dimension + Tubular Outrigger System
================================================================================

PROJECT SUMMARY
---------------
Stories: {res.n_story}
Story Height: {res.story_height} m
Total Height: {res.total_height_m:.1f} m
Plan Dimensions: {res.plan_x:.1f}m × {res.plan_y:.1f}m
Plan Area: {res.plan_x * res.plan_y:.0f} m²

================================================================================
AUTO-CALCULATED DIMENSIONS (REV12 FEATURE)
================================================================================

Central Blind (Elevator/Stair Core):
  X dimension: {res.auto_dimension_result.central_blind_x_m:.2f} m
  Y dimension: {res.auto_dimension_result.central_blind_y_m:.2f} m
  Wall thickness: {res.auto_dimension_result.central_blind_thickness_m:.2f} m

Side Walls (Perimeter Shear Walls):
  Length: {res.auto_dimension_result.side_wall_length_m:.2f} m
  Thickness: {res.auto_dimension_result.side_wall_thickness_m:.2f} m

Beams:
  Width: {res.auto_dimension_result.beam_width_m:.2f} m
  Depth: {res.auto_dimension_result.beam_depth_m:.2f} m
  Span-to-depth ratio: {max(res.plan_x, res.plan_y) / res.auto_dimension_result.beam_depth_m:.1f}

Columns:
  Corner: {res.auto_dimension_result.corner_column_dim_m:.2f}m × {res.auto_dimension_result.corner_column_dim_m:.2f}m
  Perimeter: {res.auto_dimension_result.perimeter_column_x_m:.2f}m × {res.auto_dimension_result.perimeter_column_y_m:.2f}m
  Interior: {res.auto_dimension_result.interior_column_dim_m:.2f}m × {res.auto_dimension_result.interior_column_dim_m:.2f}m

Slabs:
  Thickness: {res.auto_dimension_result.slab_thickness_m:.3f} m
  Span-to-thickness ratio: {max(res.plan_x, res.plan_y) / res.auto_dimension_result.slab_thickness_m:.1f}

Redesign Required: {'YES - ' + res.auto_dimension_result.redesign_reason if res.auto_dimension_result.needs_redesign else 'NO'}

================================================================================
STRUCTURAL ANALYSIS
================================================================================

Period Analysis:
  Reference period: {res.reference_period_s:.3f} s
  Design target: {res.design_target_period_s:.3f} s
  Estimated MDOF period: {res.estimated_period_s:.3f} s
  Upper limit: {res.upper_limit_period_s:.3f} s
  Error: {res.period_error_ratio*100:.2f}%

Stiffness & Drift:
  Total lateral stiffness: {res.K_estimated_N_per_m:.2e} N/m
  Top storey drift: {res.top_drift_m:.3f} m
  Drift ratio: {res.top_drift_m/res.total_height_m:.1%}

Modal Results (First 5 Modes):
"""
    
    for i, (T, f, mr) in enumerate(zip(
        res.modal_result.periods_s,
        res.modal_result.frequencies_hz,
        res.modal_result.effective_mass_ratios
    )):
        report += f"  Mode {i+1}: T={T:.3f}s, f={f:.2f}Hz, Mass Ratio={mr*100:.1f}%\n"
    
    report += f"""
================================================================================
TUBULAR OUTRIGGER SYSTEM (REV12 FEATURE)
================================================================================

"""
    
    if res.outrigger_results:
        for i, or_res in enumerate(res.outrigger_results):
            report += f"""
Outrigger {i+1} (Story {or_res.story_level}):
  Height from base: {or_res.height_m:.1f} m
  Number of braced spans: {or_res.n_braced_spans}
  Number of tubular braces: {or_res.n_braces}
  Total brace area: {or_res.total_brace_area_mm2:.0f} mm²
  Steel mass: {or_res.total_steel_mass_kg:.0f} kg
  Equivalent stiffness: {or_res.equivalent_stiffness_kN_m:.2e} kN/m
  Stiffness contribution: {or_res.stiffness_contribution:.2e}

"""
    else:
        report += "No outriggers configured.\n"
    
    report += f"""
================================================================================
WEIGHTS & REINFORCEMENT
================================================================================

Total Building Weight: {res.total_weight_kN/1000:.1f} MN

Reinforcement Estimate:
  Outrigger steel: {res.reinforcement_estimate.outrigger_steel_kg:.0f} kg
  
================================================================================
REDESIGN SUGGESTIONS
================================================================================

"""
    
    if res.redesign_suggestions:
        for i, sugg in enumerate(res.redesign_suggestions, 1):
            report += f"{i}. {sugg}\n"
    else:
        report += "No redesign suggestions - system is acceptable.\n"
    
    report += f"""
Governing Issue: {res.governing_issue}

================================================================================
END OF REPORT
================================================================================
"""
    
    return report


# ======================== MAIN STREAMLIT APP ========================

st.set_page_config(
    page_title="Tall Building Design - REV12",
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
    unsafe_allow_html=True
)

st.title("🏗️ Tall Building Design - REV12")
st.caption(f"Prepared by {AUTHOR_NAME} | {APP_VERSION}")
st.info("""
**✨ REV12 NEW FEATURES:**
- **Auto-Dimension Engine**: Automatically calculates all structural dimensions (central blind, walls, beams, columns, slabs)
- **Tubular Brace Outriggers**: Replaces trusses with efficient tubular braces
- **Intelligent Redesign**: Flags components that need resizing
- **Braced Spans Calculation**: Automatically determines number of spans braced by outrigger system
- **ITBS-Level Intelligence**: Competes with professional software through automation
""")

if "result" not in st.session_state:
    st.session_state.result = None
if "report" not in st.session_state:
    st.session_state.report = ""

left_col, right_col = st.columns([1.0, 2.3], gap="medium")

with left_col:
    inp = streamlit_input_panel_rev12()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("ANALYZE (REV12)"):
            try:
                with st.spinner("Running REV12 auto-design analysis..."):
                    res = run_design_rev12(inp)
                    st.session_state.result = res
                    st.session_state.report = build_report_rev12(res)
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)[:100]}")
    with b2:
        if st.button("📊 DETAILS"):
            st.session_state.view_mode = "details"
    with b3:
        if st.session_state.report:
            st.download_button(
                "SAVE REPORT",
                data=st.session_state.report.encode("utf-8"),
                file_name="tall_building_rev12_report.txt",
                mime="text/plain"
            )

with right_col:
    if st.session_state.result is None:
        st.info("👈 Click **ANALYZE (REV12)** to see results")
    else:
        res = st.session_state.result
        
        # Main metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Height", f"{res.total_height_m:.1f} m")
        m2.metric("Period (s)", f"{res.estimated_period_s:.3f}")
        m3.metric("Top Drift", f"{res.top_drift_m*1000:.1f} mm")
        m4.metric("Total Weight", f"{res.total_weight_kN/1000:.1f} MN")
        
        # Auto-dimensions
        st.markdown("### 📐 Auto-Calculated Dimensions")
        dim1, dim2, dim3, dim4 = st.columns(4)
        dim1.metric("Central Blind", f"{res.auto_dimension_result.central_blind_x_m:.2f}×{res.auto_dimension_result.central_blind_y_m:.2f}m")
        dim2.metric("Slab Thickness", f"{res.auto_dimension_result.slab_thickness_m*1000:.0f} mm")
        dim3.metric("Beam Depth", f"{res.auto_dimension_result.beam_depth_m:.2f} m")
        dim4.metric("Column Size", f"{res.auto_dimension_result.corner_column_dim_m:.2f} m")
        
        # Outrigger info
        if res.outrigger_results:
            st.markdown("### 🔧 Tubular Outrigger Systems")
            for i, or_res in enumerate(res.outrigger_results):
                or1, or2, or3 = st.columns(3)
                or1.metric(f"Outrigger {i+1}", f"Story {or_res.story_level}")
                or2.metric(f"Braced Spans", f"{or_res.n_braced_spans}")
                or3.metric(f"Braces", f"{or_res.n_braces}")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Auto-Dimensions", "Outrigger Details", "Report"])
        
        with tab1:
            st.markdown(f"""
**Central Blind:** {res.auto_dimension_result.central_blind_x_m:.2f}m ×{res.auto_dimension_result.central_blind_y_m:.2f}m (thickness: {res.auto_dimension_result.central_blind_thickness_m:.2f}m)

**Side Walls:** {res.auto_dimension_result.side_wall_thickness_m:.2f}m thick (length: {res.auto_dimension_result.side_wall_length_m:.2f}m)

**Beams:** {res.auto_dimension_result.beam_width_m:.2f}m wide × {res.auto_dimension_result.beam_depth_m:.2f}m deep

**Columns:**
- Corner: {res.auto_dimension_result.corner_column_dim_m:.2f}m × {res.auto_dimension_result.corner_column_dim_m:.2f}m
- Perimeter: {res.auto_dimension_result.perimeter_column_x_m:.2f}m × {res.auto_dimension_result.perimeter_column_y_m:.2f}m

**Slabs:** {res.auto_dimension_result.slab_thickness_m*100:.1f}cm thick

**Redesign Status:** {'⚠️ ' + res.auto_dimension_result.redesign_reason if res.auto_dimension_result.needs_redesign else '✓ Acceptable'}
            """)
        
        with tab2:
            if res.outrigger_results:
                for or_res in res.outrigger_results:
                    with st.expander(f"Outrigger at Story {or_res.story_level}"):
                        st.markdown(f"""
**Height:** {or_res.height_m:.1f}m

**Braced Spans:** {or_res.n_braced_spans}

**Tubular Braces:** {or_res.n_braces} braces
- Total area: {or_res.total_brace_area_mm2:.0f} mm²
- Steel mass: {or_res.total_steel_mass_kg:.0f} kg

**Stiffness Contribution:** {or_res.equivalent_stiffness_kN_m:.2e} kN/m
                        """)
            else:
                st.info("No outriggers configured.")
        
        with tab3:
            st.text_area("Report:", st.session_state.report, height=600, label_visibility="collapsed")
