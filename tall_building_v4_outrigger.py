
from __future__ import annotations

# Jupyter-safe backend
import matplotlib
matplotlib.use("Agg")

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G = 9.81


@dataclass
class Material:
    E: float
    fy: float
    density: float = 7850.0


@dataclass
class CoreWallSection:
    thickness: float
    length_x: float
    length_y: float
    n_web_x: int = 2
    n_web_y: int = 2
    cracked_factor: float = 0.45

    @property
    def area(self) -> float:
        return self.n_web_x * self.length_x * self.thickness + self.n_web_y * self.length_y * self.thickness

    def inertia_y(self) -> float:
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        I_from_ywalls = self.n_web_y * (t * ly**3 / 12.0)
        arm = max(0.25 * lx, t / 2.0)
        A_x = lx * t
        I_from_xwalls = self.n_web_x * (lx * t**3 / 12.0 + A_x * arm**2)
        return self.cracked_factor * (I_from_ywalls + I_from_xwalls)

    def inertia_x(self) -> float:
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        I_from_xwalls = self.n_web_x * (t * lx**3 / 12.0)
        arm = max(0.25 * ly, t / 2.0)
        A_y = ly * t
        I_from_ywalls = self.n_web_y * (ly * t**3 / 12.0 + A_y * arm**2)
        return self.cracked_factor * (I_from_xwalls + I_from_ywalls)


@dataclass
class ColumnSection:
    b: float
    h: float
    cracked_factor: float = 0.70

    @property
    def area(self) -> float:
        return self.b * self.h

    def inertia_x(self) -> float:
        return self.cracked_factor * self.b * self.h**3 / 12.0

    def inertia_y(self) -> float:
        return self.cracked_factor * self.h * self.b**3 / 12.0


@dataclass
class OutriggerLevel:
    story: int
    axis: str = "x"
    truss_depth: float = 4.0
    chord_area: float = 0.12
    diagonal_area: float = 0.06
    truss_E: float = 200e9

    def validate(self) -> None:
        if self.axis.lower() not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")
        if self.story < 1:
            raise ValueError("story must be >= 1")


@dataclass
class TowerInput:
    n_story: int
    story_height: float
    plan_x: float
    plan_y: float
    floor_mass: float
    core: CoreWallSection
    perimeter_column: ColumnSection
    corner_column: ColumnSection
    n_perimeter_columns_x_face: int
    n_perimeter_columns_y_face: int
    outriggers: List[OutriggerLevel] = field(default_factory=list)
    column_material: Material = field(default_factory=lambda: Material(E=32e9, fy=420e6))
    wall_material: Material = field(default_factory=lambda: Material(E=32e9, fy=40e6))
    design_base_shear_ratio: float = 0.02
    core_stiffness_modifier: float = 1.0
    column_stiffness_modifier: float = 1.0

    def height(self) -> float:
        return self.n_story * self.story_height

    def bay_arm(self, axis: str) -> float:
        if axis.lower() == "x":
            return max((self.plan_x - self.core.length_x) / 2.0, 0.5)
        return max((self.plan_y - self.core.length_y) / 2.0, 0.5)

    def n_engaged_columns_per_side(self, axis: str) -> int:
        return self.n_perimeter_columns_y_face if axis.lower() == "x" else self.n_perimeter_columns_x_face

    def validate(self) -> None:
        if self.n_story < 2:
            raise ValueError("n_story must be >= 2")
        if self.story_height <= 0 or self.floor_mass <= 0:
            raise ValueError("story_height and floor_mass must be positive")
        for ou in self.outriggers:
            ou.validate()
            if ou.story > self.n_story:
                raise ValueError(f"outrigger story {ou.story} > n_story {self.n_story}")


@dataclass
class OutriggerMechanics:
    story: int
    axis: str
    arm: float
    k_truss_tip: float
    k_column_tip: float
    k_tip_combined: float
    k_rot: float
    distributed_story_stiffness: float
    axial_force_per_side: float


@dataclass
class AnalysisResult:
    periods: np.ndarray
    frequencies: np.ndarray
    mode_shapes: np.ndarray
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    story_stiffness_base: np.ndarray
    story_stiffness_total: np.ndarray
    outriggers: List[OutriggerMechanics]
    floor_displacements: np.ndarray
    story_drifts: np.ndarray
    outrigger_axial_by_level: dict
    max_drift_ratio: float
    base_shear: float
    input_model: TowerInput


def generalized_eigh(K: np.ndarray, M: np.ndarray):
    m = np.diag(M).astype(float)
    if np.any(m <= 0):
        raise ValueError("Mass matrix must be positive diagonal.")
    A = K / m[:, None]
    vals, vecs = np.linalg.eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def _ensure_vector_mass(inp: TowerInput) -> np.ndarray:
    return np.full(inp.n_story, inp.floor_mass, dtype=float)


def _assemble_shear_building_K(story_k: np.ndarray) -> np.ndarray:
    n = len(story_k)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += story_k[i]
        else:
            K[i, i] += story_k[i]
            K[i, i - 1] -= story_k[i]
            K[i - 1, i] -= story_k[i]
            K[i - 1, i - 1] += story_k[i]
    return K


def _assemble_diagonal_mass(masses: np.ndarray) -> np.ndarray:
    return np.diag(masses)


def _story_stiffness_from_core(inp: TowerInput, axis: str) -> np.ndarray:
    axis = axis.lower()
    I = inp.core.inertia_y() if axis == "x" else inp.core.inertia_x()
    E = inp.wall_material.E
    H = inp.height()
    k_global = inp.core_stiffness_modifier * 3.0 * E * I / max(H**3, 1e-12)
    return np.full(inp.n_story, inp.n_story * k_global, dtype=float)


def _story_stiffness_from_columns(inp: TowerInput, axis: str) -> np.ndarray:
    axis = axis.lower()
    E = inp.column_material.E
    if axis == "x":
        I_per = inp.perimeter_column.inertia_y()
        I_cor = inp.corner_column.inertia_y()
    else:
        I_per = inp.perimeter_column.inertia_x()
        I_cor = inp.corner_column.inertia_x()
    n_per_cols = 2 * inp.n_engaged_columns_per_side(axis)
    n_corner = 4
    I_total = n_per_cols * I_per + n_corner * I_cor
    H = inp.height()
    k_global = inp.column_stiffness_modifier * 3.0 * E * I_total / max(H**3, 1e-12)
    return np.full(inp.n_story, inp.n_story * k_global, dtype=float)


def _engaged_column_axial_stiffness(inp: TowerInput, story: int, axis: str) -> float:
    E = inp.column_material.E
    L = max(story * inp.story_height, inp.story_height)
    n_side = inp.n_engaged_columns_per_side(axis)
    A_per = inp.perimeter_column.area
    A_cor = inp.corner_column.area
    A_side_total = n_side * A_per + 1.0 * A_cor
    return E * A_side_total / L


def _outrigger_truss_tip_stiffness(arm: float, truss_depth: float, chord_area: float, diagonal_area: float, truss_E: float) -> float:
    d = max(truss_depth, 0.25)
    b = max(arm, 0.25)
    Ld = sqrt(b**2 + d**2)
    Ieq = 0.5 * chord_area * d**2
    k_bend = 3.0 * truss_E * Ieq / max(b**3, 1e-12)
    k_diag = 2.0 * truss_E * diagonal_area * d**2 / max(Ld**3, 1e-12)
    return k_bend + k_diag


def outrigger_mechanics(inp: TowerInput, ou: OutriggerLevel) -> OutriggerMechanics:
    axis = ou.axis.lower()
    arm = inp.bay_arm(axis)
    k_truss_tip = _outrigger_truss_tip_stiffness(arm, ou.truss_depth, ou.chord_area, ou.diagonal_area, ou.truss_E)
    k_col_tip = _engaged_column_axial_stiffness(inp, ou.story, axis)
    k_tip = 1.0 / (1.0 / max(k_truss_tip, 1e-12) + 1.0 / max(k_col_tip, 1e-12))
    k_rot = 2.0 * k_tip * arm**2
    distributed_story_stiffness = k_rot / max(ou.story * inp.story_height**2, 1e-12)
    return OutriggerMechanics(ou.story, axis, arm, k_truss_tip, k_col_tip, k_tip, k_rot, distributed_story_stiffness, 0.0)


def _triangular_lateral_force(inp: TowerInput):
    masses = _ensure_vector_mass(inp)
    weights = masses * G
    Vb = inp.design_base_shear_ratio * weights.sum()
    z = np.arange(1, inp.n_story + 1, dtype=float) * inp.story_height
    profile = z / z.sum()
    return Vb * profile, Vb


def analyze_tower(inp: TowerInput, axis: str = "x") -> AnalysisResult:
    inp.validate()
    axis = axis.lower()
    m = _ensure_vector_mass(inp)
    k_core = _story_stiffness_from_core(inp, axis)
    k_cols = _story_stiffness_from_columns(inp, axis)
    k_total = k_core + k_cols

    ou_mech = []
    for ou in inp.outriggers:
        if ou.axis.lower() == axis:
            mech = outrigger_mechanics(inp, ou)
            ou_mech.append(mech)
            k_total[:ou.story] += mech.distributed_story_stiffness

    K = _assemble_shear_building_K(k_total)
    M = _assemble_diagonal_mass(m)
    w2, phi = generalized_eigh(K, M)
    mask = w2 > 1e-9
    w = np.sqrt(w2[mask])
    phi = phi[:, mask]
    periods = 2.0 * pi / w
    freqs = w / (2.0 * pi)

    phi_n = phi.copy()
    for i in range(phi_n.shape[1]):
        scale = phi_n[-1, i] if abs(phi_n[-1, i]) > 1e-12 else np.max(np.abs(phi_n[:, i]))
        phi_n[:, i] /= scale
        if phi_n[-1, i] < 0:
            phi_n[:, i] *= -1.0

    f_lat, Vb = _triangular_lateral_force(inp)
    u = np.linalg.solve(K, f_lat)
    drift = np.diff(np.hstack(([0.0], u))) / inp.story_height
    max_drift = float(np.max(np.abs(drift)))

    ou_axial = {}
    updated = []
    for mech in ou_mech:
        j = mech.story - 1
        theta = u[j] / max(mech.story * inp.story_height, 1e-12)
        delta_tip = theta * mech.arm
        axial = mech.k_tip_combined * delta_tip
        ou_axial[j + 1] = axial
        updated.append(replace(mech, axial_force_per_side=axial))

    return AnalysisResult(periods, freqs, phi_n, M, K, k_core + k_cols, k_total, updated, u, drift, ou_axial, max_drift, Vb, inp)


def example_input() -> TowerInput:
    return TowerInput(
        n_story=60,
        story_height=3.2,
        plan_x=80.0,
        plan_y=80.0,
        floor_mass=9.0e5,
        core=CoreWallSection(thickness=0.60, length_x=18.0, length_y=18.0, n_web_x=2, n_web_y=2, cracked_factor=0.45),
        perimeter_column=ColumnSection(b=0.95, h=0.95, cracked_factor=0.70),
        corner_column=ColumnSection(b=1.10, h=1.10, cracked_factor=0.70),
        n_perimeter_columns_x_face=7,
        n_perimeter_columns_y_face=7,
        outriggers=[],
        column_material=Material(E=32e9, fy=420e6),
        wall_material=Material(E=32e9, fy=40e6),
        design_base_shear_ratio=0.02,
    )


def quick_demo():
    inp = example_input()
    inp.outriggers = [OutriggerLevel(15, "x"), OutriggerLevel(30, "x"), OutriggerLevel(45, "x")]
    res = analyze_tower(inp, axis="x")
    return {
        "T1_s": float(res.periods[0]),
        "roof_disp_mm": float(res.floor_displacements[-1] * 1000.0),
        "max_drift_ratio": float(res.max_drift_ratio),
        "max_outrigger_axial_kN": float(max(res.outrigger_axial_by_level.values(), default=0.0) / 1000.0),
    }


if __name__ == "__main__":
    print(quick_demo())
